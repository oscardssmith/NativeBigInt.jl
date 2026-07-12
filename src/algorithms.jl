# Multi-limb algorithms built on the kernels: Knuth Algorithm D division,
# Karatsuba sqrt, powermod, and radix conversion. (Multiplication lives in
# mul.jl, Montgomery reduction in montgomery.jl, the Lehmer gcds in gcd.jl.)

# Bit length of the n-limb magnitude at a[ao+1..ao+n], n >= 1; requires a
# normalized (nonzero) top limb.
@inline magnitude_bits(a, ao::Int, n::Int) = 64 * (n - 1) + Base.top_set_bit(@inbounds a[ao+n])

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
    dbits = magnitude_bits(a, ao, n) - magnitude_bits(d, do_, m)
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
    # qh == 0 either way: Q < β^(nn-m)
    if m >= DC_DIV_THRESHOLD && nn - m >= DC_DIV_THRESHOLD
        divrem_dc!(q, qo, scratch, 0, nn, dv, dvo, m, v)
    else
        divrem_bc!(q, qo, scratch, 0, nn, dv, dvo, m, v)
    end
    if l == 0
        copyto!(r, ro + 1, scratch, 1, m)
    else
        rshift!(r, ro, scratch, 0, m, l)
    end
    return nothing
end

# Schoolbook → divide-and-conquer division crossover, in limbs (GMP's
# DC_DIV_QR_THRESHOLD analogue; tuned by bench/bench_dc_thr.jl). divrem!
# dispatches to divrem_dc! only when both the divisor and the quotient reach
# it — a short quotient over a long divisor costs O(qn·m) either way.
# Balanced 2m/m sweep: schoolbook wins through m = 96, ties at m = 128, and
# falls behind from m = 192 (1.1-3x GMP by m = 2048 vs dc's 0.8-1.0x); as the
# recursion cutoff, 100-110 also edges out lower values at large m.
const DC_DIV_THRESHOLD = 100

# Balanced 2n/n divide-and-conquer division step (GMP mpn_dcpi1_div_qr_n).
# u[uo+1..uo+2n] ÷ d[do_+1..do_+n], d normalized, v = invert_pi1 of its top
# two limbs. Writes q[qo+1..qo+n], leaves the n-limb remainder in
# u[uo+1..uo+n], returns the extra top quotient bit qh. scratch holds the
# n-limb cross products at so; recursion levels share it (a level touches it
# only after its child returns). thr >= 4 keeps the divrem_bc! basecase at
# m >= 2.
#
# Split n = hi + lo. Dividing the top 2·hi limbs by the top hi limbs of d
# overshoots the true high quotient block by at most 2 (the ignored d_lo only
# makes the divisor larger): subtract the cross product q_hi·d_lo — plus
# qh·d_lo at β^hi for the implicit qh row — and repair each resulting borrow
# by adding d back and decrementing the block. The same step on the remaining
# n + lo live limbs yields the low block. The pi1 inverse stays valid down
# the recursion because every divisor suffix keeps d's top two limbs.
function divrem_dc_n!(q::Memory{Limb}, qo::Int, u::Memory{Limb}, uo::Int,
                      d::Memory{Limb}, do_::Int, n::Int, v::Limb,
                      scratch::Memory{Limb}, so::Int, thr::Int)
    lo = n >> 1
    hi = n - lo
    qh = hi < thr ? divrem_bc!(q, qo + lo, u, uo + 2lo, 2hi, d, do_ + lo, hi, v) :
                    divrem_dc_n!(q, qo + lo, u, uo + 2lo, d, do_ + lo, hi, v, scratch, so, thr)
    mul!(scratch, so, q, qo + lo, hi, d, do_, lo)          # q_hi × d_lo, n limbs
    cy = sub_n!(u, uo + lo, u, uo + lo, scratch, so, n)
    if qh != zero(Limb)
        cy += sub_n!(u, uo + n, u, uo + n, d, do_, lo)
    end
    while cy != zero(Limb)
        qh -= sub_1!(q, qo + lo, q, qo + lo, hi, one(Limb))
        cy -= add_n!(u, uo + lo, u, uo + lo, d, do_, n)
    end
    ql = lo < thr ? divrem_bc!(q, qo, u, uo + hi, 2lo, d, do_ + hi, lo, v) :
                    divrem_dc_n!(q, qo, u, uo + hi, d, do_ + hi, lo, v, scratch, so, thr)
    mul!(scratch, so, d, do_, hi, q, qo, lo)               # q_lo × d_lo', hi >= lo
    cy = sub_n!(u, uo, u, uo, scratch, so, n)
    if ql != zero(Limb)
        cy += sub_n!(u, uo + lo, u, uo + lo, d, do_, hi)
    end
    while cy != zero(Limb)
        sub_1!(q, qo, q, qo, lo, one(Limb))   # borrow folds into ql's correction
        cy -= add_n!(u, uo, u, uo, d, do_, n)
    end
    return qh
end

# Leading partial quotient block: u[uo+1..uo+m+s] ÷ d (full m limbs), s <= m.
# Writes q[qo+1..qo+s], leaves the m-limb remainder in u[uo+1..uo+m], returns
# qh. s == m is the balanced step; small s stays schoolbook (O(s·m), one-off);
# otherwise divide the top 2s limbs by the top s limbs of d, then subtract the
# cross product q·d_lo with the same add-back repair (GMP mpn_dcpi1_div_qr's
# qn < dn arm).
function divrem_dc_partial!(q::Memory{Limb}, qo::Int, u::Memory{Limb}, uo::Int,
                            d::Memory{Limb}, do_::Int, m::Int, s::Int, v::Limb,
                            scratch::Memory{Limb}, so::Int, thr::Int)
    s == m && return divrem_dc_n!(q, qo, u, uo, d, do_, m, v, scratch, so, thr)
    s < thr && return divrem_bc!(q, qo, u, uo, m + s, d, do_, m, v)
    qh = divrem_dc_n!(q, qo, u, uo + (m - s), d, do_ + (m - s), s, v, scratch, so, thr)
    if s >= m - s                                          # q × d_lo, m limbs
        mul!(scratch, so, q, qo, s, d, do_, m - s)
    else
        mul!(scratch, so, d, do_, m - s, q, qo, s)
    end
    cy = sub_n!(u, uo, u, uo, scratch, so, m)
    if qh != zero(Limb)
        cy += sub_n!(u, uo + s, u, uo + s, d, do_, m - s)
    end
    while cy != zero(Limb)
        qh -= sub_1!(q, qo, q, qo, s, one(Limb))
        cy -= add_n!(u, uo, u, uo, d, do_, m)
    end
    return qh
end

# Divide-and-conquer quotient/remainder with divrem_bc!'s exact contract:
# u (nn limbs, destroyed; remainder left in u[uo+1..uo+m]) ÷ normalized
# m-limb d with v = invert_pi1 of its top two limbs; writes q[1..nn-m],
# returns the extra top quotient bit qh. Requires nn > m. The quotient is
# peeled from the top: a leading partial block of s limbs (qn reduced mod m
# into [1, m]), then full balanced 2m/m blocks — each later window tops out
# with the previous remainder (< d), so only the leading block can set qh.
function divrem_dc!(q::Memory{Limb}, qo::Int, u::Memory{Limb}, uo::Int, nn::Int,
                    d::Memory{Limb}, do_::Int, m::Int, v::Limb,
                    thr::Int=DC_DIV_THRESHOLD)
    qn = nn - m
    thr = max(thr, 4)
    scratch = Memory{Limb}(undef, m)
    s = qn <= m ? qn : qn - m * ((qn - 1) ÷ m)
    off = qn - s
    qh = divrem_dc_partial!(q, qo + off, u, uo + off, d, do_, m, s, v, scratch, 0, thr)
    while off > 0
        off -= m
        divrem_dc_n!(q, qo + off, u, uo + off, d, do_, m, v, scratch, 0, thr)
    end
    return qh
end
