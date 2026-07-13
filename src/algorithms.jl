# Multi-limb algorithms built on the kernels: Karatsuba sqrt, powermod, and
# radix conversion. (Multiplication lives in mul.jl, division in div.jl,
# Montgomery reduction in montgomery.jl, the Lehmer gcds in gcd.jl.)

# Karatsuba square root (Zimmermann, INRIA RR-3805): s[1..h] = isqrt(a[1..n]),
# h = (n+1)>>1. Requires n even (or n <= 2) and a[ao+n] >= 2^62 (caller
# normalizes by an even bit shift plus a zero low limb for odd lengths); an
# odd-length high part would leave S' half-normalized, 2S' < β^hh, and the
# quotient step unbounded.
# On return a[ao+1..ao+h] holds the low limbs of the remainder a - s^2; the
# return value is its top limb (0 or 1). scratch needs 5h+8 limbs at sco;
# recursion levels share it (a level touches it only after its child returns).
function sqrtrem!(s::Memory{Limb}, so::Int, a::Memory{Limb}, ao::Int, n::Int,
                  scratch::Memory{Limb}, sco::Int=0)
    if n <= 2
        v = n == 1 ? UInt128(@inbounds a[ao+1]) :
            (UInt128(@inbounds a[ao+2]) << 64) | (@inbounds a[ao+1])
        rt = isqrt(v)
        rm = v - rt * rt
        @inbounds s[so+1] = rt % Limb
        @inbounds a[ao+1] = rm % Limb
        return Int((rm >> 64) % Limb)
    end
    h = (n + 1) >> 1
    lq = h >> 1       # low-root limbs (the quotient Q)
    hh = h - lq       # high-root limbs (S')
    nh = n - 2lq      # limbs of the high part
    # a = Ahi*β^2lq + A1*β^lq + A0; recurse: Ahi = S'^2 + R', R' <= 2S'.
    c1 = sqrtrem!(s, so + lq, a, ao + 2lq, nh, scratch, sco)
    num = sco               # h+1 limbs: (c1, R', A1)
    dd = sco + h + 1        # hh+1 limbs: 2S'
    qq = sco + 2h + 2       # lq+2 limbs: quotient Q
    uu = sco + 3h + 4       # hh+1 limbs: division remainder U
    q2 = sco + 4h + 5       # 2lq limbs: Q^2
    copyto!(scratch, num + 1, a, ao + lq + 1, h)
    numlen = h
    if c1 != 0
        numlen = h + 1
        @inbounds scratch[num+numlen] = c1 % Limb
    end
    dc = lshift!(scratch, dd, s, so + lq, hh, 1)
    dhi = dc != 0            # doubling S' spilled a top limb into 2S' (dl = hh+1)
    dl = hh
    if dhi
        dl = hh + 1
        @inbounds scratch[dd+dl] = dc
    end
    # (Q, U) = divrem(R'*β^lq + A1, 2S'); S = S'*β^lq + Q.
    # Strip zero top limbs (divrem!'s fast path needs a meaningful a[n]) and
    # pre-zero the quotient slot so untouched high limbs read as zero.
    @inbounds while numlen > dl && scratch[num+numlen] == 0
        numlen -= 1
    end
    fill!(view(scratch, qq+1:qq+lq+2), zero(Limb))
    divrem!(scratch, qq, scratch, uu, scratch, num, numlen, scratch, dd, dl)
    qlen = numlen - dl + 1
    rhi = 0
    copyto!(a, ao + lq + 1, scratch, uu + 1, hh)
    dhi && (rhi = Int(@inbounds scratch[uu+dl]))
    # Q <= β^lq; if Q = β^lq exactly, clamp to β^lq - 1 and put 2S' back in U
    # (still >= the true root; the correction loop repairs the remainder).
    toobig = any(!iszero, view(scratch, qq+lq+1:qq+qlen))
    if toobig
        fill!(view(scratch, qq+1:qq+lq), typemax(Limb))
        c = add_n!(a, ao + lq, a, ao + lq, scratch, dd, hh)
        rhi += Int(c) + (dhi ? Int(@inbounds scratch[dd+dl]) : 0)
    end
    copyto!(s, so + 1, scratch, qq + 1, lq)
    # R = U*β^lq + A0 - Q^2, tracked as (rhi, a[ao+1..ao+h]) with rhi signed
    sqr!(scratch, q2, scratch, qq, lq)
    rhi -= Int(sub!(a, ao, a, ao, h, scratch, q2, 2lq))
    while rhi < 0
        # (S+1)^2 overshoots: R += 2S - 1, S -= 1
        tc = lshift!(scratch, num, s, so, h, 1)
        sub_1!(scratch, num, scratch, num, h, one(Limb))
        c = add_n!(a, ao, a, ao, scratch, num, h)
        rhi += Int(c) + Int(tc)
        sub_1!(s, so, s, so, h, one(Limb))
    end
    return rhi
end

# O(1) exponent bit access; NBig overloads live in nbig.jl.
@inline expbit(e::Integer, i::Int) = (e >>> i) % Bool
@inline expbits(e::Integer) = Base.top_set_bit(e)

# b^e mod m on magnitudes: m has k limbs (m[k] ≠ 0, m > 1), 0 < b < m
# (lb limbs), e > 0 any Integer supporting expbit/expbits. Returns a k-limb
# Memory (unnormalized). Sliding-window exponentiation; for odd m the values
# live in Montgomery form with redc! after each mul/sqr, for even m each
# product is reduced with divrem! instead.
function powermod_limbs(b::Memory{Limb}, lb::Int, e::Integer,
                        m::Memory{Limb}, k::Int)
    odd = isodd(@inbounds m[1])
    ninv = odd ? mont_ninv(@inbounds m[1]) : zero(Limb)
    nbits = expbits(e)
    w = nbits <= 8 ? 1 : nbits <= 24 ? 2 : nbits <= 80 ? 3 : nbits <= 240 ? 4 : 5
    tsize = 1 << (w - 1)   # table of odd powers b^1, b^3, …, b^(2^w - 1)
    acc = Memory{Limb}(undef, k)
    prod = Memory{Limb}(undef, 2k + 1)
    qbuf = Memory{Limb}(undef, k + 2)
    table = Memory{Limb}(undef, tsize * k)

    # dst[1..k] = x*y (or x^2) brought back into the working domain
    function mulred!(dst::Memory{Limb}, dsto::Int, x::Memory{Limb}, xo::Int,
                     y::Memory{Limb}, yo::Int)
        if x === y && xo == yo
            sqr!(prod, 0, x, xo, k)
        else
            mul!(prod, 0, x, xo, k, y, yo, k)
        end
        @inbounds prod[2k+1] = 0
        if odd
            redc!(dst, dsto, prod, 0, m, 0, k, ninv)
        else
            lt = normlen(prod, 0, 2k)
            if lt < k || cmp_limbs(prod, 0, lt, m, 0, k) < 0
                copyto!(dst, dsto + 1, prod, 1, lt)
                fill!(view(dst, dsto+lt+1:dsto+k), zero(Limb))
            else
                divrem!(qbuf, 0, dst, dsto, prod, 0, lt, m, 0, k)
            end
        end
    end

    # table slot 0 = base: b·β^k mod m in Montgomery form, else b zero-padded
    if odd
        fill!(view(prod, 1:k), zero(Limb))
        copyto!(prod, k + 1, b, 1, lb)
        divrem!(qbuf, 0, table, 0, prod, 0, k + lb, m, 0, k)
    else
        copyto!(table, 1, b, 1, lb)
        fill!(view(table, lb+1:k), zero(Limb))
    end
    if tsize > 1
        bsq = Memory{Limb}(undef, k)
        mulred!(bsq, 0, table, 0, table, 0)
        for j in 1:tsize-1
            mulred!(table, j * k, table, (j - 1) * k, bsq, 0)
        end
    end

    first = true
    i = nbits - 1
    while i >= 0
        if !expbit(e, i)
            first || mulred!(acc, 0, acc, 0, acc, 0)
            i -= 1
        else
            l = min(w, i + 1)
            while !expbit(e, i - l + 1)   # window must end on a set bit
                l -= 1
            end
            val = 0
            for j in 0:l-1
                val = (val << 1) | Int(expbit(e, i - j))
            end
            if first
                copyto!(acc, 1, table, (val >> 1) * k + 1, k)
                first = false
            else
                for _ in 1:l
                    mulred!(acc, 0, acc, 0, acc, 0)
                end
                mulred!(acc, 0, acc, 0, table, (val >> 1) * k)
            end
            i -= l
        end
    end
    if odd   # leave Montgomery form: one reduction of the bare value
        copyto!(prod, 1, acc, 1, k)
        fill!(view(prod, k+1:2k+1), zero(Limb))
        redc!(acc, 0, prod, 0, m, 0, k, ninv)
    end
    return acc
end

# Largest power of base that fits in a limb: (base^k, k).
function big_base(base::Int)
    bb = Limb(base)
    k = 1
    while true
        nb, ovf = Base.mul_with_overflow(bb, Limb(base))
        ovf && return bb, k
        bb, k = nb, k + 1
    end
end

# Little-endian base^k digit chunks of the n-limb magnitude in a (destroyed:
# a is divided down in place). Repeated divrem_1! — O(n²), fine for v1;
# divide-and-conquer is a post-v1 extension for very large n. A power-of-two
# base^k = 2^c needs no division: each chunk is a fixed-width c-bit window (O(n)).
function radix_chunks!(a::Memory{Limb}, n::Int, bb::Limb)
    ispow2(bb) && return radix_chunks_pow2(a, n, trailing_zeros(bb))
    chunks = Limb[]
    while n > 0
        push!(chunks, divrem_1!(a, 0, a, 0, n, bb))
        n = normlen(a, 0, n)
    end
    return chunks
end

# Little-endian c-bit windows of the n-limb magnitude (a[n] ≠ 0, c ≤ 63):
# identical output to repeatedly dividing by 2^c, but O(n). Does not touch a.
function radix_chunks_pow2(a::Memory{Limb}, n::Int, c::Int)
    n == 0 && return Limb[]
    bitlen = magnitude_bits(a, 0, n)
    nchunks = cld(bitlen, c)
    chunks = Vector{Limb}(undef, nchunks)
    mask = (one(Limb) << c) - one(Limb)
    @inbounds for t in 0:nchunks-1
        b = t * c
        q = b >> 6
        off = b & 63
        w = a[q+1] >> off
        if off != 0 && q+2 <= n   # window straddles a limb boundary
            w |= a[q+2] << (64 - off)
        end
        chunks[t+1] = w & mask
    end
    return chunks
end

