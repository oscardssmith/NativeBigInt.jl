# Multi-limb algorithms built on the kernels: Karatsuba multiplication,
# Knuth Algorithm D division, sqrt, powermod, and radix conversion.
# (Montgomery reduction lives in montgomery.jl, the Lehmer gcds in gcd.jl.)

# Below this operand length (limbs) mul_basecase! wins; benchmark-tuned.
const KARATSUBA_THRESHOLD = 29

# Value comparison of la-limb a vs lb-limb b (la >= lb): strip a's zero top
# limbs (split halves are zero-padded, cmp_limbs trusts lengths) and delegate.
@inline function cmp_padded(a::Memory{Limb}, ao::Int, la::Int, b::Memory{Limb}, bo::Int, lb::Int)
    @inbounds while la > lb && a[ao+la] == 0
        la -= 1
    end
    return cmp_limbs(a, ao, la, b, bo, lb)
end

# d[1..lo_len] = |x_lo - x_hi| where x_lo = x[xo+1 .. xo+lo_len] and
# x_hi = x[xo+lo_len+1 .. xo+lo_len+hi_len], hi_len <= lo_len.
# Returns true iff the difference is negative (x_lo < x_hi).
function abs_diff!(d::Memory{Limb}, dof::Int, x::Memory{Limb}, xo::Int, lo_len::Int, hi_len::Int)
    if cmp_padded(x, xo, lo_len, x, xo + lo_len, hi_len) >= 0
        sub!(d, dof, x, xo, lo_len, x, xo + lo_len, hi_len)
        return false
    else
        # x_lo < x_hi < B^hi_len forces x_lo's limbs above hi_len to be zero
        sub_n!(d, dof, x, xo + lo_len, x, xo, hi_len)
        fill!(view(d, dof+hi_len+1:dof+lo_len), zero(Limb))
        return true
    end
end

# Scratch limbs kar_mul! needs for an n x n product: 4*ceil(n/2) per level.
function kar_scratch_len(n::Int, thr::Int=KARATSUBA_THRESHOLD)
    len = 0
    while n >= thr
        n = (n + 1) >> 1
        len += 4n
    end
    return len
end

# Balanced n x n subtractive Karatsuba: r[1..2n] = a[1..n] * b[1..n].
# a*b = hi*B^(2h2) + (lo + hi - s*mid)*B^h2 + lo where mid = |a_lo-a_hi|*|b_lo-b_hi|.
# The middle term equals a_lo*b_hi + a_hi*b_lo >= 0, so carries never underflow.
# scratch must have kar_scratch_len(n) limbs free at so; r must not alias a/b.
function kar_mul!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int,
                  b::Memory{Limb}, bo::Int, n::Int, scratch::Memory{Limb}, so::Int,
                  thr::Int=KARATSUBA_THRESHOLD)
    if n < thr
        return mul_basecase!(r, ro, a, ao, n, b, bo, n)
    end
    h2 = (n + 1) >> 1   # low-half length
    h = n - h2          # high-half length (h2 or h2-1)
    L = 2h2
    mid = so            # L limbs: |a_lo-a_hi| * |b_lo-b_hi|
    tmp = so + L        # L limbs: the two differences, then lo+hi±mid
    rec = so + 2L       # recursion scratch
    nega = abs_diff!(scratch, tmp, a, ao, h2, h)
    negb = abs_diff!(scratch, tmp + h2, b, bo, h2, h)
    kar_mul!(scratch, mid, scratch, tmp, scratch, tmp + h2, h2, scratch, rec, thr)
    kar_mul!(r, ro, a, ao, b, bo, h2, scratch, rec, thr)                 # lo
    kar_mul!(r, ro + L, a, ao + h2, b, bo + h2, h, scratch, rec, thr)    # hi
    c = add!(scratch, tmp, r, ro, L, r, ro + L, 2h)                 # tmp = lo + hi
    if nega == negb
        c -= sub_n!(scratch, tmp, scratch, tmp, scratch, mid, L)
    else
        c += add_n!(scratch, tmp, scratch, tmp, scratch, mid, L)
    end
    add_into!(r, ro + h2, 2n - h2, scratch, tmp, L)
    add_carry!(r, ro + h2, 2n - h2, L + 1, c)
    return nothing
end

# sqr_basecase! does half the multiplies of mul_basecase!, so squaring stays
# basecase longer than mul; benchmark-tuned separately.
const SQR_KARATSUBA_THRESHOLD = 52

# Balanced Karatsuba squaring: r[1..2n] = a[1..n]^2. Same recursion shape as
# kar_mul! with mid = (a_lo - a_hi)^2 >= 0, so the middle term is always
# lo + hi - mid and there is no sign tracking. Scratch layout matches
# kar_scratch_len(n); r must not alias a.
function kar_sqr!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int,
                  scratch::Memory{Limb}, so::Int, thr::Int=SQR_KARATSUBA_THRESHOLD)
    if n < thr
        return sqr_basecase!(r, ro, a, ao, n)
    end
    h2 = (n + 1) >> 1   # low-half length
    h = n - h2          # high-half length (h2 or h2-1)
    L = 2h2
    mid = so            # L limbs: (a_lo - a_hi)^2
    tmp = so + L        # L limbs: the difference, then lo + hi - mid
    rec = so + 2L       # recursion scratch
    abs_diff!(scratch, tmp, a, ao, h2, h)
    kar_sqr!(scratch, mid, scratch, tmp, h2, scratch, rec, thr)
    kar_sqr!(r, ro, a, ao, h2, scratch, rec, thr)                 # lo
    kar_sqr!(r, ro + L, a, ao + h2, h, scratch, rec, thr)         # hi
    c = add!(scratch, tmp, r, ro, L, r, ro + L, 2h)               # tmp = lo + hi
    c -= sub_n!(scratch, tmp, scratch, tmp, scratch, mid, L)
    add_into!(r, ro + h2, 2n - h2, scratch, tmp, L)
    add_carry!(r, ro + h2, 2n - h2, L + 1, c)
    return nothing
end

# r[1..2n] = a[1..n]^2, n >= 1; r must not alias a.
function sqr!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int,
              thr::Int=SQR_KARATSUBA_THRESHOLD)
    if n < thr
        return sqr_basecase!(r, ro, a, ao, n)
    end
    scratch = Memory{Limb}(undef, kar_scratch_len(n, thr))
    kar_sqr!(r, ro, a, ao, n, scratch, 0, thr)
    return nothing
end

# General product r[1..m+n] = a[1..m] * b[1..n], m >= n >= 1; r must not
# alias a or b. Karatsuba on n-limb chunks of a; each block's low n limbs
# accumulate into r, its high limbs land in fresh territory (plus carry).
function mul!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, m::Int,
              b::Memory{Limb}, bo::Int, n::Int, thr::Int=KARATSUBA_THRESHOLD)
    if n < thr
        return mul_basecase!(r, ro, a, ao, m, b, bo, n)
    end
    scratch = Memory{Limb}(undef, 2n + kar_scratch_len(n, thr))
    kar_mul!(r, ro, a, ao, b, bo, n, scratch, 2n, thr)
    i = n
    while i < m
        chunk = min(n, m - i)
        if chunk == n
            kar_mul!(scratch, 0, a, ao + i, b, bo, n, scratch, 2n, thr)
        else
            mul!(scratch, 0, b, bo, n, a, ao + i, chunk, thr)  # ragged tail, n x chunk
        end
        c = add_n!(r, ro + i, r, ro + i, scratch, 0, n)
        copy_tail!(r, ro + i, scratch, 0, n + 1, n + chunk)
        add_carry!(r, ro + i, m + n - i, n + 1, c)
        i += chunk
    end
    return nothing
end

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
    bitlen = 64 * (n - 1) + Base.top_set_bit(@inbounds a[n])
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

# Quotient/remainder: a (n limbs) ÷ d (m limbs, d[m] ≠ 0), n ≥ m ≥ 1.
# For m ≥ 3, a[n] must be nonzero unless n == m (the small-quotient fast
# path bounds the quotient by a[n]'s bit position).
# Writes n-m+1 quotient limbs (top may be zero) and m remainder limbs
# (unnormalized). One scratch Memory holds the shifted numerator copy (n+1
# limbs) and, for unnormalized d, the shifted divisor; a is not modified.
function divrem!(q::Memory{Limb}, qo::Int, r::Memory{Limb}, ro::Int,
                 a::Memory{Limb}, ao::Int, n::Int, d::Memory{Limb}, do_::Int, m::Int)
    if m == 1
        @inbounds r[ro+1] = divrem_1!(q, qo, a, ao, n, d[do_+1])
        return nothing
    end
    if m == 2
        r1, r0 = @inbounds divrem_2!(q, qo, a, ao, n, d[do_+2], d[do_+1])
        @inbounds r[ro+1] = r0
        @inbounds r[ro+2] = r1
        return nothing
    end
    # Small-quotient fast path: bitlength(a) - bitlength(d) ≤ 2 bounds the
    # quotient below 8 (a/d < 2^(Δ+1)), so at most 7 subtraction sweeps beat
    # the scratch-alloc + normalize + invert + basecase machinery. Δ ≤ 2 also
    # forces n ≤ m+1 with a[n] < 4, so the value fits r plus one register.
    dbits = 64 * (n - m) + Base.top_set_bit(@inbounds a[ao+n]) -
            Base.top_set_bit(@inbounds d[do_+m])
    if dbits <= 2
        copyto!(r, ro + 1, a, ao + 1, m)
        t = n > m ? (@inbounds a[ao+n]) : zero(Limb)
        c = zero(Limb)
        while t != zero(Limb) || cmp_limbs(r, ro, m, d, do_, m) >= 0
            t -= sub_n!(r, ro, r, ro, d, do_, m)
            c += one(Limb)
        end
        @inbounds q[qo+1] = c
        n > m && (@inbounds q[qo+2] = zero(Limb))
        return nothing
    end
    l = leading_zeros(@inbounds d[do_+m])
    nn = n + 1
    scratch = Memory{Limb}(undef, nn + (l > 0 ? m : 0))
    if l == 0
        copyto!(scratch, 1, a, ao + 1, n)
        @inbounds scratch[nn] = zero(Limb)
        dv, dvo = d, do_
    else
        @inbounds scratch[nn] = lshift!(scratch, 0, a, ao, n, l)
        lshift!(scratch, nn, d, do_, m, l)
        dv, dvo = scratch, nn
    end
    v = @inbounds invert_pi1(dv[dvo+m], dv[dvo+m-1])
    divrem_bc!(q, qo, scratch, 0, nn, dv, dvo, m, v)   # qh == 0: Q < β^(nn-m)
    if l == 0
        copyto!(r, ro + 1, scratch, 1, m)
    else
        rshift!(r, ro, scratch, 0, m, l)
    end
    return nothing
end
