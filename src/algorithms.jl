# Multi-limb algorithms built on the kernels: Karatsuba multiplication and the
# Lehmer cofactor-matrix apply. (Montgomery reduction lives in montgomery.jl.)

@inline sterm(c::Int64, x::Limb) =
    c >= 0 ? Int128(widemul(Limb(c), x)) : -Int128(widemul(Limb(-c), x))

# Fused Lehmer matrix apply: r1 = A*U + B*V, r2 = C*U + D*V in one pass over
# the operands via exact two's-complement limb accumulation (signed Int128
# carries; |carry| stays < 2^63 for |cofactor| < 2^62). Valid cofactor
# matrices give nonnegative results, so the final carries are the top limbs.
# Writes n+1 limbs each, n = max(lu, lv); returns n+1. u/v are indexed from
# offset 0 (this is gcd-algorithm glue, not a uniform (mem, offset, len) kernel).
function lehmer_apply!(r1::Memory{Limb}, r1o::Int, r2::Memory{Limb}, r2o::Int,
                       u::Memory{Limb}, lu::Int, v::Memory{Limb}, lv::Int,
                       A::Int64, B::Int64, C::Int64, D::Int64)
    n = max(lu, lv)
    c1 = Int128(0)
    c2 = Int128(0)
    @inbounds for i in 1:n
        ui = i <= lu ? u[i] : zero(Limb)
        vi = i <= lv ? v[i] : zero(Limb)
        a1 = c1 + sterm(A, ui) + sterm(B, vi)
        r1[r1o+i] = a1 % Limb
        c1 = a1 >> 64
        a2 = c2 + sterm(C, ui) + sterm(D, vi)
        r2[r2o+i] = a2 % Limb
        c2 = a2 >> 64
    end
    @inbounds r1[r1o+n+1] = c1 % Limb
    @inbounds r2[r2o+n+1] = c2 % Limb
    return n + 1
end

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

# ⌊X / 2^pos⌋ for the n-limb magnitude x, truncated to 128 bits (callers
# guarantee the true value fits). Limbs above n read as zero.
@inline function extract_window(x::Memory{Limb}, n::Int, pos::Int)
    i = pos >> 6
    r = pos & 63
    w1 = i + 1 <= n ? (@inbounds x[i+1]) : zero(Limb)
    w2 = i + 2 <= n ? (@inbounds x[i+2]) : zero(Limb)
    w3 = i + 3 <= n ? (@inbounds x[i+3]) : zero(Limb)
    if r == 0
        return (UInt128(w2) << 64) | w1
    end
    lo = (w1 >> r) | (w2 << (64 - r))
    hi = (w2 >> r) | (w3 << (64 - r))
    return (UInt128(hi) << 64) | lo
end

# One single-word Lehmer pass on 63-bit window tops: Euclid with nonnegative
# magnitude cofactors (implicit alternating signs, Knuth's formulation) capped
# at 2^30. A quotient is accepted only when both truncation extremes
# (x - s1)/(y + s2) and (x + s3)/(y - s4) agree; `slack` widens the brackets
# to absorb inherited window error (0 for exact tops, 2 after a prior matrix
# application). q <= 3 (~85%, Gauss–Kuzmin) comes from a subtract chain, the
# rest from one hardware 64-bit divide; only the verify widens to 128 bits.
# Returns (A, B, C, D, even, steps); at even parity the signed rows are
# (+A, -B) / (-C, +D), flipped at odd.
@inline function lehmer63(x::UInt64, y::UInt64, slack::UInt64)
    A, B, C, D = UInt64(1), UInt64(0), UInt64(0), UInt64(1)
    even = true
    steps = 0
    while true
        s1, s2, s3, s4 = even ? (B, D, A, C) : (A, C, B, D)
        s1, s2, s3, s4 = s1 + slack, s2 + slack, s3 + slack, s4 + slack
        (s1 > x || s4 >= y) && break
        xl, dh = x - s1, y + s2
        xh, dl = x + s3, y - s4
        xl < dh && break
        r1 = xl - dh
        if r1 < dh
            q = UInt64(1)
        elseif r1 - dh < dh
            q = UInt64(2)
        elseif r1 - 2dh < dh   # reached only when r1 >= 2dh, so 2dh < 2^63
            q = UInt64(3)
        else
            q = div(xl, dh)
        end
        # verify against the other extreme (q*dl can exceed 64 bits)
        w = widemul(q, dl)
        (w <= xh && UInt128(xh) < w + dl) || break
        q > UInt64(2)^30 && break   # would blow the cap; q*C could wrap
        nC = A + q * C
        nD = B + q * D
        (nC > UInt64(2)^30 || nD > UInt64(2)^30) && break
        x, y = y, xl + s1 - q * y   # wrap-exact: true remainder < y
        A, B, C, D = C, D, nC, nD
        even = !even
        steps += 1
    end
    return A, B, C, D, even, steps
end

# gcd of the magnitudes u[1..lu] and v[1..lv]; both buffers are destroyed and
# must have capacity >= max(lu, lv) + 1. Returns (mem, len) with the result in
# one of the two buffers. Lehmer's method (Knuth TAOCP §4.5.2, Algorithm L):
# per 126-bit leading window, two lehmer63 passes build a cofactor matrix that
# one fused lehmer_apply! pass applies to the full operands (buffer rotation,
# no copies); a full divrem! step when the window makes no progress; UInt128
# binary gcd once v fits two limbs.
function gcd!(u::Memory{Limb}, lu::Int, v::Memory{Limb}, lv::Int)
    cap = max(lu, lv) + 1
    w1 = Memory{Limb}(undef, cap)
    w2 = Memory{Limb}(undef, cap)   # also serves as divrem!'s quotient buffer
    while true
        # invariant maintenance: normalized lengths, u >= v
        if lu < lv || (lu == lv && cmp_limbs(u, 0, lu, v, 0, lv) < 0)
            u, v = v, u
            lu, lv = lv, lu
        end
        lv == 0 && return u, lu
        if lv <= 2
            if lu > 2
                divrem!(w2, 0, w1, 0, u, 0, lu, v, 0, lv)
                x = extract_window(w1, lv, 0)
            else
                x = extract_window(u, lu, 0)
            end
            g = gcd(x, extract_window(v, lv, 0))
            @inbounds u[1] = g % Limb
            @inbounds u[2] = (g >> 64) % Limb
            return u, ((g >> 64) != 0 ? 2 : 1)
        end
        ub = 64lu - leading_zeros(@inbounds u[lu])
        vb = 64lv - leading_zeros(@inbounds v[lv])
        steps = 0
        if ub - vb < 64
            # Two single-word phases per 126-bit window (hgcd2-flavoured):
            # phase 1 on the exact 63-bit tops, then the phase-1 matrix is
            # applied to the window (wrap-exact), fresh tops are extracted and
            # phase 2 runs with brackets widened by 2 to absorb the inherited
            # truncation error (< 2^30 window-ulps, < 1 ulp after a >= 31-bit
            # shift). Composed magnitudes stay <= 2^61.
            pos = ub - 126
            x = extract_window(u, lu, pos)
            y = extract_window(v, lv, pos)
            A, B, C, D, even, steps = lehmer63(UInt64(x >> 63), UInt64(y >> 63), UInt64(0))
            if steps > 0
                xn, yn = even ? (A * x - B * y, D * y - C * x) :
                                (B * y - A * x, C * x - D * y)
                nb = 128 - leading_zeros(xn)
                if nb >= 94 && yn != 0
                    sh = nb - 63
                    A2, B2, C2, D2, even2, steps2 =
                        lehmer63(UInt64(xn >> sh), UInt64(yn >> sh), UInt64(2))
                    if steps2 > 0   # compose: M <- M2 * M1, parities add
                        A, B, C, D = A2 * A + B2 * C, A2 * B + B2 * D,
                                     C2 * A + D2 * C, C2 * B + D2 * D
                        even = even == even2
                    end
                end
            end
        end
        if steps == 0
            # window made no progress: one full division step, rotating the
            # remainder buffer in rather than copying
            divrem!(w2, 0, w1, 0, u, 0, lu, v, 0, lv)
            u, v, w1 = v, w1, u
            lu = lv
            lv = normlen(v, 0, lv)
        else
            # (U, V) <- (A*U + B*V, C*U + D*V), one fused pass, then rotate
            sA, sB, sC, sD = Int64(A), Int64(B), Int64(C), Int64(D)
            if even
                sB, sC = -sB, -sC
            else
                sA, sD = -sA, -sD
            end
            n = lehmer_apply!(w1, 0, w2, 0, u, lu, v, lv, sA, sB, sC, sD)
            u, w1 = w1, u
            v, w2 = w2, v
            lu = normlen(u, 0, n)
            lv = normlen(v, 0, n)
        end
    end
end

# One batch of exact extended Euclid on 128-bit values: quotient steps
# accumulate the nonnegative cofactor matrix (Knuth's alternating-sign
# formulation, lehmer63's convention) until an entry would pass 2^62 or y
# hits zero. Values are exact so no bracket verification is needed; q <= 2
# (~80%, Gauss–Kuzmin) comes from a subtract chain. Requires x >= y.
@inline function euclid128(x::UInt128, y::UInt128)
    A, B, C, D = UInt64(1), UInt64(0), UInt64(0), UInt64(1)
    even = true
    steps = 0
    while y != 0
        r1 = x - y
        if r1 < y
            q, r = UInt128(1), r1
        elseif r1 - y < y
            q, r = UInt128(2), r1 - y
        else
            q, r = divrem(x, y)
        end
        (q >> 62) != 0 && break
        nC = widemul(q % UInt64, C) + A
        nD = widemul(q % UInt64, D) + B
        (nC >= UInt128(2)^62 || nD >= UInt128(2)^62) && break
        x, y = y, r
        A, B, C, D = C, D, nC % UInt64, nD % UInt64
        even = !even
        steps += 1
    end
    return x, y, A, B, C, D, even, steps
end

# x1[1..ret] = t_u + q * t_v on cofactor magnitudes (the alternating-sign
# invariant makes a full division step's cofactor update additive).
# x1 must not alias tu/tv/q and needs capacity max(lq + ltv, ltu) + 1.
function cofactor_step!(x1::Memory{Limb}, tu::Memory{Limb}, ltu::Int,
                        tv::Memory{Limb}, ltv::Int, q::Memory{Limb}, lq::Int)
    if ltv == 0 || lq == 0
        copyto!(x1, 1, tu, 1, ltu)
        return ltu
    end
    if lq >= ltv
        mul!(x1, 0, q, 0, lq, tv, 0, ltv)
    else
        mul!(x1, 0, tv, 0, ltv, q, 0, lq)
    end
    lp = normlen(x1, 0, lq + ltv)
    ltu == 0 && return lp
    if lp >= ltu
        c = add!(x1, 0, x1, 0, lp, tu, 0, ltu)
    else
        c = add!(x1, 0, tu, 0, ltu, x1, 0, lp)
        lp = ltu
    end
    if c != 0
        lp += 1
        @inbounds x1[lp] = c
    end
    return lp
end

# Extended gcd of the magnitudes U = u[1..lu], V = v[1..lv]: gcd!'s loop with
# the V-cofactor pair carried in lockstep through every window apply, division
# step, and swap. Both buffers are destroyed (capacity >= max(lu, lv) + 1).
# Returns (g, lg, t, lt, tpos) with g = gcd(U, V) and s*U + (tpos ? t : -t)*V
# == g for some s; |t| <= max(U, V) / gcd. Cofactor signs alternate (Knuth),
# so only magnitudes are stored: sign(t_u) = (tpos ? + : -), sign(t_v) the
# opposite; even-parity matrices preserve the pattern, odd ones and
# division/swap steps flip it. The two-limb tail runs exact 128-bit Euclid
# batches (euclid128) instead of gcd!'s cofactor-blind binary gcd.
function gcdext!(u::Memory{Limb}, lu::Int, v::Memory{Limb}, lv::Int)
    cap = max(lu, lv) + 1
    capt = cap + 2
    w1 = Memory{Limb}(undef, cap)
    w2 = Memory{Limb}(undef, cap)   # also serves as divrem!'s quotient buffer
    tu = Memory{Limb}(undef, capt)
    tv = Memory{Limb}(undef, capt)
    x1 = Memory{Limb}(undef, capt)
    x2 = Memory{Limb}(undef, capt)
    ltu, ltv = 0, 1
    @inbounds tv[1] = one(Limb)
    tpos = false            # t_u = 0 <= 0, t_v = 1 >= 0
    while true
        if lu < lv || (lu == lv && cmp_limbs(u, 0, lu, v, 0, lv) < 0)
            u, v = v, u
            lu, lv = lv, lu
            tu, tv = tv, tu
            ltu, ltv = ltv, ltu
            tpos = !tpos
        end
        lv == 0 && return u, lu, tu, ltu, tpos
        if lv <= 2
            if lu > 2
                # one full division step brings u down to two limbs
                divrem!(w2, 0, w1, 0, u, 0, lu, v, 0, lv)
                lq = normlen(w2, 0, lu - lv + 1)
                lt = cofactor_step!(x1, tu, ltu, tv, ltv, w2, lq)
                u, v, w1 = v, w1, u
                lu, lv = lv, normlen(v, 0, lv)
                tu, tv, x1 = tv, x1, tu
                ltu, ltv = ltv, lt
                tpos = !tpos
            end
            x = extract_window(u, lu, 0)
            y = lv == 0 ? UInt128(0) : extract_window(v, lv, 0)
            while y != 0
                x, y, A, B, C, D, even, steps = euclid128(x, y)
                if steps > 0
                    n = lehmer_apply!(x1, 0, x2, 0, tu, ltu, tv, ltv,
                                      Int64(A), Int64(B), Int64(C), Int64(D))
                    tu, x1 = x1, tu
                    tv, x2 = x2, tv
                    ltu = normlen(tu, 0, n)
                    ltv = normlen(tv, 0, n)
                    even || (tpos = !tpos)
                end
                if y != 0   # oversized quotient stalled the batch: one exact step
                    q, r = divrem(x, y)
                    @inbounds w2[1] = q % Limb
                    @inbounds w2[2] = (q >> 64) % Limb
                    lq = (q >> 64) != 0 ? 2 : Int(q != 0)
                    lt = cofactor_step!(x1, tu, ltu, tv, ltv, w2, lq)
                    tu, tv, x1 = tv, x1, tu
                    ltu, ltv = ltv, lt
                    tpos = !tpos
                    x, y = y, r
                end
            end
            @inbounds u[1] = x % Limb
            @inbounds u[2] = (x >> 64) % Limb
            return u, ((x >> 64) != 0 ? 2 : 1), tu, ltu, tpos
        end
        ub = 64lu - leading_zeros(@inbounds u[lu])
        vb = 64lv - leading_zeros(@inbounds v[lv])
        steps = 0
        if ub - vb < 64
            # identical window construction to gcd! (see comments there)
            pos = ub - 126
            x = extract_window(u, lu, pos)
            y = extract_window(v, lv, pos)
            A, B, C, D, even, steps = lehmer63(UInt64(x >> 63), UInt64(y >> 63), UInt64(0))
            if steps > 0
                xn, yn = even ? (A * x - B * y, D * y - C * x) :
                                (B * y - A * x, C * x - D * y)
                nb = 128 - leading_zeros(xn)
                if nb >= 94 && yn != 0
                    sh = nb - 63
                    A2, B2, C2, D2, even2, steps2 =
                        lehmer63(UInt64(xn >> sh), UInt64(yn >> sh), UInt64(2))
                    if steps2 > 0
                        A, B, C, D = A2 * A + B2 * C, A2 * B + B2 * D,
                                     C2 * A + D2 * C, C2 * B + D2 * D
                        even = even == even2
                    end
                end
            end
        end
        if steps == 0
            # window made no progress: one full division step
            divrem!(w2, 0, w1, 0, u, 0, lu, v, 0, lv)
            lq = normlen(w2, 0, lu - lv + 1)
            lt = cofactor_step!(x1, tu, ltu, tv, ltv, w2, lq)
            u, v, w1 = v, w1, u
            lu, lv = lv, normlen(v, 0, lv)
            tu, tv, x1 = tv, x1, tu
            ltu, ltv = ltv, lt
            tpos = !tpos
        else
            # numbers get the signed matrix, cofactor magnitudes the positive one
            sA, sB, sC, sD = Int64(A), Int64(B), Int64(C), Int64(D)
            if even
                sB, sC = -sB, -sC
            else
                sA, sD = -sA, -sD
            end
            n = lehmer_apply!(w1, 0, w2, 0, u, lu, v, lv, sA, sB, sC, sD)
            nt = lehmer_apply!(x1, 0, x2, 0, tu, ltu, tv, ltv,
                               Int64(A), Int64(B), Int64(C), Int64(D))
            u, w1 = w1, u
            v, w2 = w2, v
            tu, x1 = x1, tu
            tv, x2 = x2, tv
            lu = normlen(u, 0, n)
            lv = normlen(v, 0, n)
            ltu = normlen(tu, 0, nt)
            ltv = normlen(tv, 0, nt)
            even || (tpos = !tpos)
        end
    end
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
    bitlen = 64n - leading_zeros(@inbounds a[n])
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
    dbits = 64n - leading_zeros(@inbounds a[ao+n]) -
            (64m - leading_zeros(@inbounds d[do_+m]))
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
