# Multiplication and squaring above the basecase kernels: benchmark-tuned
# dispatch thresholds, subtractive Karatsuba, Toom-3, and the mul!/sqr!
# entry points that route between them and the NTT (ntt.jl).

# Benchmark-tuned dispatch thresholds for mul!/sqr!, in limbs.
# Basecase → Karatsuba (bench/bench_kar_thr.jl, bench/bench_sqr.jl):
# sqr_basecase! does half the multiplies of mul_basecase!, so squaring stays
# basecase longer.
const MUL_KARATSUBA_THRESHOLD = 29
const SQR_KARATSUBA_THRESHOLD = 52
# Karatsuba → Toom-3 (bench/bench_toom_thr.jl): Toom-3's O(n^1.465) overtakes
# Karatsuba once the extra evaluation/interpolation passes amortize — between
# 200 and 256 balanced limbs (Karatsuba clearly ahead at 200, behind at 256,
# and recursion children forced below ~110 limbs run faster as Karatsuba).
# Squaring crosses later for the same reason as Karatsuba (cheaper
# sub-products): sqr_kar! leads through 320 limbs and falls behind by 448.
# Both must stay ≥ 16: the interpolation's fixed-width buffers (2k+2 limbs at
# offset 3k of r) only fit inside r's 2n limbs for k ≥ 6.
const MUL_TOOM3_THRESHOLD = 240
const SQR_TOOM3_THRESHOLD = 384
# Toom-3 → NTT (bench/bench_mul.jl, bench/bench_sqr.jl): the NTT's stepwise
# transform cost drops it below Toom-3 exactly at the 1024-limb size step
# (Toom still wins both mul and sqr at 960), but for unbalanced operands only
# once the smaller one is substantial (the chunked Toom path is
# ~max·min^0.465 while the NTT pays for the combined length).
const MUL_NTT_MIN = 256         # smaller operand at least this many limbs
const MUL_NTT_THRESHOLD = 1024  # average operand at least this many limbs
const SQR_NTT_THRESHOLD = 1024  # operand at least this many limbs

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

# Scratch limbs mul_kar! needs for an n x n product: 4*ceil(n/2) per level.
function kar_scratch_len(n::Int, thr::Int=MUL_KARATSUBA_THRESHOLD)
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
function mul_kar!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int,
                  b::Memory{Limb}, bo::Int, n::Int, scratch::Memory{Limb}, so::Int,
                  thr::Int=MUL_KARATSUBA_THRESHOLD)
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
    mul_kar!(scratch, mid, scratch, tmp, scratch, tmp + h2, h2, scratch, rec, thr)
    mul_kar!(r, ro, a, ao, b, bo, h2, scratch, rec, thr)                 # lo
    mul_kar!(r, ro + L, a, ao + h2, b, bo + h2, h, scratch, rec, thr)    # hi
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

# Balanced Karatsuba squaring: r[1..2n] = a[1..n]^2. Same recursion shape as
# mul_kar! with mid = (a_lo - a_hi)^2 >= 0, so the middle term is always
# lo + hi - mid and there is no sign tracking. Scratch layout matches
# kar_scratch_len(n); r must not alias a.
function sqr_kar!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int,
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
    sqr_kar!(scratch, mid, scratch, tmp, h2, scratch, rec, thr)
    sqr_kar!(r, ro, a, ao, h2, scratch, rec, thr)                 # lo
    sqr_kar!(r, ro + L, a, ao + h2, h, scratch, rec, thr)         # hi
    c = add!(scratch, tmp, r, ro, L, r, ro + L, 2h)               # tmp = lo + hi
    c -= sub_n!(scratch, tmp, scratch, tmp, scratch, mid, L)
    add_into!(r, ro + h2, 2n - h2, scratch, tmp, L)
    add_carry!(r, ro + h2, 2n - h2, L + 1, c)
    return nothing
end

# d[1..la] = |a[1..la] - b[1..lb]|, lb <= la; returns true iff a < b (then b
# bounds a's value below β^lb, so a's limbs above lb are zero).
function abs_sub!(d::Memory{Limb}, dof::Int, a::Memory{Limb}, ao::Int, la::Int,
                  b::Memory{Limb}, bo::Int, lb::Int)
    if cmp_padded(a, ao, la, b, bo, lb) >= 0
        sub!(d, dof, a, ao, la, b, bo, lb)
        return false
    else
        sub_n!(d, dof, b, bo, a, ao, lb)
        fill!(view(d, dof+lb+1:dof+la), zero(Limb))
        return true
    end
end

# Scratch limbs the balanced dispatchers need for an n x n product/square:
# per Toom-3 level 2(k+1) evaluation slots (mul only) + 4 product slots of
# 2k+2 limbs, then whatever the k+1-limb children need (their sizes k, m are
# ≤ k+1 and both length functions are monotone).
function mul_scratch_len(n::Int, thr::Int=MUL_TOOM3_THRESHOLD)
    n < thr && return kar_scratch_len(n)
    k = (n + 2) ÷ 3
    return 10k + 10 + mul_scratch_len(k + 1, thr)
end
function sqr_scratch_len(n::Int, thr::Int=SQR_TOOM3_THRESHOLD)
    n < thr && return kar_scratch_len(n, SQR_KARATSUBA_THRESHOLD)
    k = (n + 2) ÷ 3
    return 9k + 9 + sqr_scratch_len(k + 1, thr)
end

# d[1..k+1] = a(2) = a0 + 2*a1 + 4*a2 by Horner, for the k/k/m limb split of
# a[1..2k+m]. Max value 7*(β^k - 1) fits k+1 limbs with room to spare.
function toom3_eval2!(d::Memory{Limb}, dof::Int, a::Memory{Limb}, ao::Int, k::Int, m::Int)
    c = lshift!(d, dof, a, ao + 2k, m, 1)          # d = 2*a2
    @inbounds d[dof+m+1] = c
    fill!(view(d, dof+m+2:dof+k+1), zero(Limb))
    add_into!(d, dof, k + 1, a, ao + k, k)         # + a1
    lshift!(d, dof, d, dof, k + 1, 1)              # *2 (< β^(k+1)/2, no spill)
    add_into!(d, dof, k + 1, a, ao, k)             # + a0
    return nothing
end

# Toom-3 interpolation + recombination, shared by mul_toom3!/sqr_toom3!.
# On entry r holds v0 = C(0) in [1..2k], zeros in [2k+1..4k], vinf in
# [4k+1..2n]; scratch holds |v(-1)| at p0, v(1) at p1, v(2) at p2 (each a
# fixed-width L = 2k+2 limb buffer, zero-padded), with L free limbs at t;
# neg says v(-1) < 0. Solves for c1, c2, c3 (all nonnegative, and every
# intermediate below stays nonnegative since |v(-1)| <= v(1)) and adds
# them into r at limb offsets k, 2k, 3k.
function toom3_interp!(r::Memory{Limb}, ro::Int, n::Int, k::Int, m::Int,
                       scratch::Memory{Limb}, p0::Int, p1::Int, p2::Int, t::Int,
                       neg::Bool)
    L = 2k + 2
    # t1 = (v1 + vm1)/2 = c0 + c2 + c4 → t;  t2 = (v1 - vm1)/2 = c1 + c3 → p0
    if neg
        sub_n!(scratch, t, scratch, p1, scratch, p0, L)
        add_n!(scratch, p0, scratch, p1, scratch, p0, L)
    else
        add_n!(scratch, t, scratch, p1, scratch, p0, L)
        sub_n!(scratch, p0, scratch, p1, scratch, p0, L)
    end
    rshift!(scratch, t, scratch, t, L, 1)
    rshift!(scratch, p0, scratch, p0, L, 1)
    # c2 = t1 - v0 - vinf → t
    sub!(scratch, t, scratch, t, L, r, ro, 2k)
    sub!(scratch, t, scratch, t, L, r, ro + 4k, 2m)
    # t3 = (v2 - v0 - 4c2 - 16vinf)/2 = c1 + 4c3 → p2 (p1 is free as a temp)
    sub!(scratch, p2, scratch, p2, L, r, ro, 2k)
    c = lshift!(scratch, p1, r, ro + 4k, 2m, 4)
    @inbounds scratch[p1+2m+1] = c
    sub!(scratch, p2, scratch, p2, L, scratch, p1, 2m + 1)
    lshift!(scratch, p1, scratch, t, L, 2)         # 4c2 < 12β^2k, no spill
    sub_n!(scratch, p2, scratch, p2, scratch, p1, L)
    rshift!(scratch, p2, scratch, p2, L, 1)
    # c3 = (t3 - t2)/3 → p2;  c1 = t2 - c3 → p0
    sub_n!(scratch, p2, scratch, p2, scratch, p0, L)
    divexact_by3!(scratch, p2, scratch, p2, L)
    sub_n!(scratch, p0, scratch, p0, scratch, p2, L)
    # r += c1·β^k + c2·β^2k + c3·β^3k (partial sums stay < β^2n: all cᵢ ≥ 0)
    add_into!(r, ro + k, 2n - k, scratch, p0, L)
    add_into!(r, ro + 2k, 2n - 2k, scratch, t, L)
    add_into!(r, ro + 3k, 2n - 3k, scratch, p2, L)
    return nothing
end

# Balanced n x n Toom-3: r[1..2n] = a[1..n] * b[1..n]. Splits into k/k/m limb
# thirds (k = ⌈n/3⌉), evaluates A, B at {0, 1, -1, 2, ∞}, recurses on five
# ≤(k+1)-limb balanced products, and interpolates the degree-4 result.
# scratch must have mul_scratch_len(n, thr) limbs free at so; r must not
# alias a/b. thr is the Karatsuba → Toom-3 crossover for the whole recursion
# (overridable for threshold tuning, cf. mul_kar!).
function mul_toom3!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int,
                    b::Memory{Limb}, bo::Int, n::Int, scratch::Memory{Limb}, so::Int,
                    thr::Int=MUL_TOOM3_THRESHOLD)
    if n < MUL_KARATSUBA_THRESHOLD
        return mul_basecase!(r, ro, a, ao, n, b, bo, n)
    elseif n < thr
        return mul_kar!(r, ro, a, ao, b, bo, n, scratch, so)
    end
    k = (n + 2) ÷ 3
    m = n - 2k          # top-third length, k-2 <= m <= k
    L = 2k + 2
    oA = so             # k+1 limbs: a evaluations
    oB = so + (k + 1)   # k+1 limbs: b evaluations
    oP0 = so + 2(k + 1)         # L limbs: |v(-1)|, later t2, then c1
    oP1 = oP0 + L               # L limbs: v(1), later a shift temp
    oP2 = oP1 + L               # L limbs: v(2), later c3
    oT = oP2 + L                # L limbs: |A(-1)|·|B(-1)| inputs, later t1/c2
    rec = oT + L                # recursion scratch for the k+1-limb children
    # A = a0 + a2, B = b0 + b2; v(-1) factors |A - a1| packed into oT
    @inbounds scratch[oA+k+1] = add!(scratch, oA, a, ao, k, a, ao + 2k, m)
    @inbounds scratch[oB+k+1] = add!(scratch, oB, b, bo, k, b, bo + 2k, m)
    nega = abs_sub!(scratch, oT, scratch, oA, k + 1, a, ao + k, k)
    negb = abs_sub!(scratch, oT + k + 1, scratch, oB, k + 1, b, bo + k, k)
    mul_toom3!(scratch, oP0, scratch, oT, scratch, oT + k + 1, k + 1, scratch, rec, thr)
    # v(1) = (A + a1)(B + b1)
    add_into!(scratch, oA, k + 1, a, ao + k, k)
    add_into!(scratch, oB, k + 1, b, bo + k, k)
    mul_toom3!(scratch, oP1, scratch, oA, scratch, oB, k + 1, scratch, rec, thr)
    # v(2)
    toom3_eval2!(scratch, oA, a, ao, k, m)
    toom3_eval2!(scratch, oB, b, bo, k, m)
    mul_toom3!(scratch, oP2, scratch, oA, scratch, oB, k + 1, scratch, rec, thr)
    # v(0) and v(∞) straight into r, zero the gap between them
    mul_toom3!(r, ro, a, ao, b, bo, k, scratch, rec, thr)
    mul_toom3!(r, ro + 4k, a, ao + 2k, b, bo + 2k, m, scratch, rec, thr)
    fill!(view(r, ro+2k+1:ro+4k), zero(Limb))
    toom3_interp!(r, ro, n, k, m, scratch, oP0, oP1, oP2, oT, nega != negb)
    return nothing
end

# Balanced Toom-3 squaring: r[1..2n] = a[1..n]^2. Same shape as mul_toom3!
# with the b evaluations dropped and v(-1) = A(-1)^2 >= 0, so no sign
# tracking. scratch needs sqr_scratch_len(n, thr) limbs at so; r must not
# alias a. thr is the Karatsuba → Toom-3 crossover for the whole recursion.
function sqr_toom3!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int,
                    scratch::Memory{Limb}, so::Int, thr::Int=SQR_TOOM3_THRESHOLD)
    if n < SQR_KARATSUBA_THRESHOLD
        return sqr_basecase!(r, ro, a, ao, n)
    elseif n < thr
        return sqr_kar!(r, ro, a, ao, n, scratch, so)
    end
    k = (n + 2) ÷ 3
    m = n - 2k
    L = 2k + 2
    oA = so
    oP0 = so + (k + 1)
    oP1 = oP0 + L
    oP2 = oP1 + L
    oT = oP2 + L
    rec = oT + L
    @inbounds scratch[oA+k+1] = add!(scratch, oA, a, ao, k, a, ao + 2k, m)
    abs_sub!(scratch, oT, scratch, oA, k + 1, a, ao + k, k)
    sqr_toom3!(scratch, oP0, scratch, oT, k + 1, scratch, rec, thr)
    add_into!(scratch, oA, k + 1, a, ao + k, k)
    sqr_toom3!(scratch, oP1, scratch, oA, k + 1, scratch, rec, thr)
    toom3_eval2!(scratch, oA, a, ao, k, m)
    sqr_toom3!(scratch, oP2, scratch, oA, k + 1, scratch, rec, thr)
    sqr_toom3!(r, ro, a, ao, k, scratch, rec, thr)
    sqr_toom3!(r, ro + 4k, a, ao + 2k, m, scratch, rec, thr)
    fill!(view(r, ro+2k+1:ro+4k), zero(Limb))
    toom3_interp!(r, ro, n, k, m, scratch, oP0, oP1, oP2, oT, false)
    return nothing
end

# Squaring dispatch with caller-provided scratch (sqr_scratch_len(n) limbs
# at so); squaring is inherently balanced, so this is sqr! itself.
function sqr!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int,
              scratch::Memory{Limb}, so::Int)
    if n < SQR_KARATSUBA_THRESHOLD
        sqr_basecase!(r, ro, a, ao, n)
    elseif n < SQR_TOOM3_THRESHOLD
        sqr_kar!(r, ro, a, ao, n, scratch, so)
    elseif n >= SQR_NTT_THRESHOLD
        sqr_ntt!(r, ro, a, ao, n)
    else
        sqr_toom3!(r, ro, a, ao, n, scratch, so)
    end
    return nothing
end

# Allocating convenience form: r[1..2n] = a[1..n]^2, n >= 1; r must not
# alias a. Basecase/NTT sizes skip the scratch allocation entirely.
function sqr!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int)
    if n < SQR_KARATSUBA_THRESHOLD
        return sqr_basecase!(r, ro, a, ao, n)
    end
    if n >= SQR_NTT_THRESHOLD
        return sqr_ntt!(r, ro, a, ao, n)
    end
    sqr!(r, ro, a, ao, n, Memory{Limb}(undef, sqr_scratch_len(n)), 0)
    return nothing
end

# Balanced n x n dispatch with caller-provided scratch (mul_scratch_len(n)
# limbs at so; the NTT branch allocates its own and ignores it).
function mul_bal!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int,
                  b::Memory{Limb}, bo::Int, n::Int, scratch::Memory{Limb}, so::Int)
    if n < MUL_KARATSUBA_THRESHOLD
        mul_basecase!(r, ro, a, ao, n, b, bo, n)
    elseif n < MUL_TOOM3_THRESHOLD
        mul_kar!(r, ro, a, ao, b, bo, n, scratch, so)
    elseif n >= MUL_NTT_THRESHOLD
        mul_ntt!(r, ro, a, ao, n, b, bo, n)
    else
        mul_toom3!(r, ro, a, ao, b, bo, n, scratch, so)
    end
    return nothing
end

# General product r[1..m+n] = a[1..m] * b[1..n], m >= n >= 1; r must not
# alias a or b. Balanced Karatsuba/Toom-3 on n-limb chunks of a; each block's low n limbs
# accumulate into r, its high limbs land in fresh territory (plus carry).
function mul!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, m::Int,
              b::Memory{Limb}, bo::Int, n::Int)
    if n < MUL_KARATSUBA_THRESHOLD
        return mul_basecase!(r, ro, a, ao, m, b, bo, n)
    end
    if n >= MUL_NTT_MIN && m + n >= 2MUL_NTT_THRESHOLD
        return mul_ntt!(r, ro, a, ao, m, b, bo, n)
    end
    scratch = Memory{Limb}(undef, 2n + mul_scratch_len(n))
    mul_bal!(r, ro, a, ao, b, bo, n, scratch, 2n)
    i = n
    while i < m
        chunk = min(n, m - i)
        if chunk == n
            mul_bal!(scratch, 0, a, ao + i, b, bo, n, scratch, 2n)
        else
            mul!(scratch, 0, b, bo, n, a, ao + i, chunk)  # ragged tail, n x chunk
        end
        c = add_n!(r, ro + i, r, ro + i, scratch, 0, n)
        copy_tail!(r, ro + i, scratch, 0, n + 1, n + chunk)
        add_carry!(r, ro + i, m + n - i, n + 1, c)
        i += chunk
    end
    return nothing
end
