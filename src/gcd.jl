# Lehmer gcd and extended gcd (Knuth TAOCP §4.5.2, Algorithm L) on magnitude
# buffers: one shared window/apply/division loop (gcd_impl!), with extended
# runs carrying the V-cofactor pair (Cofactors) in lockstep. Built on the
# kernels and on algorithms.jl's mul!/divrem!.

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

# One single-word Euclid pass on 63-bit window tops, shared by the Lehmer and
# hgcd base cases: nonnegative magnitude cofactors (implicit alternating signs,
# Knuth's formulation) capped at 2^30. A quotient is accepted only when both
# truncation extremes (x - s1)/(y + s2) and (x + s3)/(y - s4) agree; `slack`
# widens the brackets to absorb inherited window error (0 for exact tops, 2
# after a prior matrix application). q <= 3 (~85%, Gauss–Kuzmin) comes from a
# subtract chain, the rest from one hardware 64-bit divide; only the verify
# widens to 128 bits.
#
# HGCD (compile-time, so both extras fold away in Lehmer mode) toggles the two
# hgcd-only behaviours:
#   * a floor gate f — a step is accepted only if both members of the new pair
#     stay >= f with margin (cofactor sum) * (slack + 1) covering the
#     truncation error the cofactors can amplify;
#   * det = +1 on exit — a trailing odd step is rolled back (at most one wasted
#     step per window).
# Lehmer mode (HGCD == false) ignores f and returns (A, B, C, D, even, steps);
# at even parity the signed rows are (+A, -B) / (-C, +D), flipped at odd. HGCD
# mode returns (A, B, C, D, steps) with det = +1 (always-even signed rows).
@inline function euclid63(x::UInt64, y::UInt64, slack::UInt64, f::UInt64,
                          ::Val{HGCD}) where {HGCD}
    A, B, C, D = UInt64(1), UInt64(0), UInt64(0), UInt64(1)
    pA, pB, pC, pD = A, B, C, D   # prior matrix, for the det = +1 rollback
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
        r = xl + s1 - q * y   # wrap-exact: true remainder < y
        if HGCD
            (y < f + (C + D) * (slack + 1) ||
             r < f + (nC + nD) * (slack + 1)) && break
        end
        x, y = y, r
        pA, pB, pC, pD = A, B, C, D
        A, B, C, D = C, D, nC, nD
        even = !even
        steps += 1
    end
    if HGCD
        if !even   # det = +1 required: drop the final step
            A, B, C, D = pA, pB, pC, pD
            steps -= 1
        end
        return A, B, C, D, steps
    else
        return A, B, C, D, even, steps
    end
end

# Cofactor matrix for one leading window of (U, V), ub their bit length
# with bits(U) - bits(V) < 64. Two single-word phases per 126-bit window
# (hgcd2-flavoured): phase 1 on the exact 63-bit tops, then the phase-1
# matrix is applied to the window (wrap-exact), fresh tops are extracted and
# phase 2 runs with brackets widened by 2 to absorb the inherited truncation
# error (< 2^30 window-ulps, < 1 ulp after a >= 31-bit shift). Composed
# magnitudes stay <= 2^61. steps == 0 means the window made no progress.
@inline function lehmer_window(u::Memory{Limb}, lu::Int, ub::Int,
                               v::Memory{Limb}, lv::Int)
    pos = ub - 126
    x = extract_window(u, lu, pos)
    y = extract_window(v, lv, pos)
    A, B, C, D, even, steps = euclid63(UInt64(x >> 63), UInt64(y >> 63), UInt64(0), UInt64(0), Val(false))
    if steps > 0
        xn, yn = even ? (A * x - B * y, D * y - C * x) :
                        (B * y - A * x, C * x - D * y)
        nb = Base.top_set_bit(xn)
        if nb >= 94 && yn != 0
            sh = nb - 63
            A2, B2, C2, D2, even2, steps2 =
                euclid63(UInt64(xn >> sh), UInt64(yn >> sh), UInt64(2), UInt64(0), Val(false))
            if steps2 > 0   # compose: M <- M2 * M1, parities add
                A, B, C, D = A2 * A + B2 * C, A2 * B + B2 * D,
                             C2 * A + D2 * C, C2 * B + D2 * D
                even = even == even2
            end
        end
    end
    return A, B, C, D, even, steps
end

# The V-cofactor pair (t_u, t_v) carried alongside gcd_impl!'s (u, v).
# Cofactor signs alternate (Knuth's formulation), so only magnitudes are
# stored: sign(t_u) = (tpos ? + : -) and sign(t_v) is the opposite.
# Even-parity matrices preserve the pattern; odd ones, division steps, and
# swaps flip it. x1/x2 are the rotation buffers.
mutable struct Cofactors
    tu::Memory{Limb}
    tv::Memory{Limb}
    x1::Memory{Limb}
    x2::Memory{Limb}
    ltu::Int
    ltv::Int
    tpos::Bool
end
function Cofactors(cap::Int)
    t = Cofactors(Memory{Limb}(undef, cap), Memory{Limb}(undef, cap),
                  Memory{Limb}(undef, cap), Memory{Limb}(undef, cap),
                  0, 1, false)   # (t_u, t_v) = (0, +1)
    @inbounds t.tv[1] = one(Limb)
    return t
end

function swap!(t::Cofactors)
    t.tu, t.tv = t.tv, t.tu
    t.ltu, t.ltv = t.ltv, t.ltu
    t.tpos = !t.tpos
    return nothing
end

# (t_u, t_v) <- (A*t_u + B*t_v, C*t_u + D*t_v): the positive-magnitude image
# of the signed matrix applied to the numbers.
function apply!(t::Cofactors, A::UInt64, B::UInt64, C::UInt64, D::UInt64, even::Bool)
    n = lehmer_apply!(t.x1, 0, t.x2, 0, t.tu, t.ltu, t.tv, t.ltv,
                      Int64(A), Int64(B), Int64(C), Int64(D))
    t.tu, t.x1 = t.x1, t.tu
    t.tv, t.x2 = t.x2, t.tv
    t.ltu = normlen(t.tu, 0, n)
    t.ltv = normlen(t.tv, 0, n)
    even || (t.tpos = !t.tpos)
    return nothing
end

# One full division step u = q*v + r: (t_u, t_v) <- (t_v, t_u + q*t_v).
# The alternating-sign invariant makes the update additive in magnitudes.
function divstep!(t::Cofactors, q::Memory{Limb}, lq::Int)
    x1, tu, ltu, tv, ltv = t.x1, t.tu, t.ltu, t.tv, t.ltv
    if ltv == 0 || lq == 0
        copyto!(x1, 1, tu, 1, ltu)
        lp = ltu
    else
        lq >= ltv ? mul!(x1, 0, q, 0, lq, tv, 0, ltv) :
                    mul!(x1, 0, tv, 0, ltv, q, 0, lq)
        lp = normlen(x1, 0, lq + ltv)
        if ltu > 0
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
        end
    end
    t.tu, t.tv, t.x1 = tv, x1, tu
    t.ltu, t.ltv = ltv, lp
    t.tpos = !t.tpos
    return nothing
end

# ---------------------------------------------------------------------------
# Subquadratic gcd: HGCD layer (Möller 2008; the control flow mirrors GMP's
# mpn_hgcd/mpn_gcd because the invariant bounds there are what's proven).
#
# hgcd! on an n-limb pair produces a 2x2 matrix M of nonnegative mpn
# magnitudes with (A; B) = M * (a; b), reducing the operands to just above
# s = n÷2 + 1 limbs. All matrices here have det = +1: the signed image of
# (A B; C D) is (+A -B; -C +D) and its inverse (D B; C A) is again
# nonnegative, so every application site uses one fixed sign pattern (no
# parity bit). The base-case step generator euclid63 (with HGCD == true)
# enforces det = +1 by rolling back a trailing odd step, and enforces the
# "stay above s limbs" invariant with a floor check; the floor is what keeps
# hgcd_matrix_adjust! nonnegative, so it is load-bearing, not a tuning knob.
# Matrix-matrix
# products route through mul!, which is where the subquadratic total
# O(M(n) log n) comes from.

const HGCD_THRESHOLD = 120        # bench/bench_gcd_thr.jl (flat 100..130)
const GCD_DC_THRESHOLD = 300      # Lehmer crossover ~300-400 limbs
const GCDEXT_DC_THRESHOLD = 250   # cofactor updates push the crossover up

# hgcd analog of lehmer_window on already-extracted 128-bit windows (bits
# above the window are zero, window values < 2^126). Guarantees det = +1 and
# that both true window values stay >= 2^fexp after the matrix is applied
# (callers pick fexp <= 64). x < y is handled by conjugating with the swap
# J: J (A B; C D) J = (D C; B A), preserving nonnegativity and det = +1.
@inline function hgcd_window(x::UInt128, y::UInt128, fexp::Int)
    swapped = x < y
    if swapped
        x, y = y, x
    end
    f1 = (fexp >= 63 ? UInt64(1) << (fexp - 63) : UInt64(0)) + UInt64(1)
    A, B, C, D, steps = euclid63(UInt64(x >> 63), UInt64(y >> 63), UInt64(0), f1, Val(true))
    if steps > 0
        xn, yn = A * x - B * y, D * y - C * x   # even-parity exact images
        nb = Base.top_set_bit(xn)
        if nb >= 94 && yn != 0
            sh = nb - 63
            f2 = UInt64((UInt128(1) << fexp) >> sh) + UInt64(1)
            A2, B2, C2, D2, steps2 =
                euclid63(UInt64(xn >> sh), UInt64(yn >> sh), UInt64(2), f2, Val(true))
            if steps2 > 0   # compose: M <- M2 * M1, both even
                A, B, C, D = A2 * A + B2 * C, A2 * B + B2 * D,
                             C2 * A + D2 * C, C2 * B + D2 * D
                steps += steps2
            end
        end
    end
    if swapped
        A, B, C, D = D, C, B, A
    end
    return A, B, C, D, steps
end

# 2x2 matrix of nonnegative mpn magnitudes with det = +1, (A; B) = M (a; b).
# All four entries are zero-padded to the common length n: buffers are
# zero-filled at construction and entries only ever grow (diagonal >= 1
# because det = +1 forces AD = 1 + BC >= 1), so the padding invariant is
# maintained for free.
mutable struct HgcdMatrix
    m00::Memory{Limb}
    m01::Memory{Limb}
    m10::Memory{Limb}
    m11::Memory{Limb}
    n::Int
end

# entry capacity for an hgcd of n limbs: GMP's bound is (n+1)/2 + 1; the
# extra margin absorbs the +1-limb growth of the compose operations
hgcd_matrix_cap(n::Int) = ((n + 1) >> 1) + 3

function HgcdMatrix(cap::Int)
    M = HgcdMatrix(fill!(Memory{Limb}(undef, cap), Limb(0)),
                   fill!(Memory{Limb}(undef, cap), Limb(0)),
                   fill!(Memory{Limb}(undef, cap), Limb(0)),
                   fill!(Memory{Limb}(undef, cap), Limb(0)), 1)
    @inbounds M.m00[1] = one(Limb)
    @inbounds M.m11[1] = one(Limb)
    return M
end

# M <- M * (D B; C A): compose with the inverse of a window matrix whose
# signed image (+A -B; -C +D) was applied to the values. Row-wise fused
# apply; in-place is safe because lehmer_apply! reads index i before writing
# it. Grows M.n by at most one limb.
function hgcd_matrix_mul1!(M::HgcdMatrix, A::UInt64, B::UInt64, C::UInt64, D::UInt64)
    n = M.n
    @assert n + 1 <= length(M.m00)
    lehmer_apply!(M.m00, 0, M.m01, 0, M.m00, n, M.m01, n,
                  Int64(D), Int64(C), Int64(B), Int64(A))
    lehmer_apply!(M.m10, 0, M.m11, 0, M.m10, n, M.m11, n,
                  Int64(D), Int64(C), Int64(B), Int64(A))
    M.n = max(normlen(M.m00, 0, n + 1), normlen(M.m01, 0, n + 1),
              normlen(M.m10, 0, n + 1), normlen(M.m11, 0, n + 1), 1)
    return nothing
end

# Division-step compose: M <- M * (1 q; 0 1) when the row-0 value was
# reduced (col = 1: column 1 gains q * column 0), or M * (1 0; q 1) when the
# row-1 value was (col = 0).
function hgcd_matrix_update_q!(M::HgcdMatrix, q::Memory{Limb}, lq::Int, col::Int)
    n = M.n
    scratch = Memory{Limb}(undef, lq + n + 1)
    nn = n
    for (src, dst) in (col == 1 ? ((M.m00, M.m01), (M.m10, M.m11)) :
                                  ((M.m01, M.m00), (M.m11, M.m10)))
        ls = normlen(src, 0, n)
        ls == 0 && continue
        lq >= ls ? mul!(scratch, 0, q, 0, lq, src, 0, ls) :
                   mul!(scratch, 0, src, 0, ls, q, 0, lq)
        lt = normlen(scratch, 0, lq + ls)
        ld = normlen(dst, 0, n)
        l = 0
        if lt >= ld
            c = ld == 0 ? zero(Limb) : add!(scratch, 0, scratch, 0, lt, dst, 0, ld)
            copyto!(dst, 1, scratch, 1, lt)
            l = lt
        else
            c = add!(dst, 0, dst, 0, ld, scratch, 0, lt)
            l = ld
        end
        if c != 0
            l += 1
            @inbounds dst[l] = c
        end
        l > nn && (nn = l)
    end
    @assert nn <= length(M.m00)
    M.n = nn
    return nothing
end

# t = x*u + y*v on zero-padded lengths (n for x,y; n1 for u,v), writing
# n + n1 + 1 limbs of t; w is n + n1 limbs of scratch.
function hgcd_mul2!(t::Memory{Limb}, x::Memory{Limb}, y::Memory{Limb}, n::Int,
                    u::Memory{Limb}, v::Memory{Limb}, n1::Int, w::Memory{Limb})
    n >= n1 ? mul!(t, 0, x, 0, n, u, 0, n1) : mul!(t, 0, u, 0, n1, x, 0, n)
    n >= n1 ? mul!(w, 0, y, 0, n, v, 0, n1) : mul!(w, 0, v, 0, n1, y, 0, n)
    c = add_n!(t, 0, t, 0, w, 0, n + n1)
    @inbounds t[n+n1+1] = c
    return nothing
end

# M <- M * M1 (both nonnegative, det = +1): naive 8 mul! + adds. The muls
# route through Karatsuba/fpNTT — this product is the subquadratic heart.
# (GMP uses a Strassen-style matrix22_mul here; benchmark before bothering.)
function hgcd_matrix_mul!(M::HgcdMatrix, M1::HgcdMatrix)
    n, n1 = M.n, M1.n
    L = n + n1 + 1
    @assert L <= length(M.m00)
    ta = Memory{Limb}(undef, L)
    tb = Memory{Limb}(undef, L)
    w = Memory{Limb}(undef, n + n1)
    for (r0, r1) in ((M.m00, M.m01), (M.m10, M.m11))
        hgcd_mul2!(ta, r0, r1, n, M1.m00, M1.m10, n1, w)
        hgcd_mul2!(tb, r0, r1, n, M1.m01, M1.m11, n1, w)
        copyto!(r0, 1, ta, 1, L)
        copyto!(r1, 1, tb, 1, L)
    end
    M.n = max(normlen(M.m00, 0, L), normlen(M.m01, 0, L),
              normlen(M.m10, 0, L), normlen(M.m11, 0, L), 1)
    return nothing
end

# Un-truncation after running hgcd on the top n - p limbs of (u, v): the
# reduced tops (ra, rb; nn limbs) and the untouched low p limbs recombine
# through M^{-1} = (m11 -m01; -m10 m00):
#   u' = ra*2^(64p) + m11*ulow - m01*vlow
#   v' = rb*2^(64p) + m00*vlow - m10*ulow
# Nonnegativity (asserted) is guaranteed by hgcd's floor invariant. Both
# buffers are rewritten and zero-padded through L <= p + max(M.n+1, nn) + 1
# limbs; returns the new common padded length.
function hgcd_matrix_adjust!(M::HgcdMatrix, u::Memory{Limb}, v::Memory{Limb},
                             p::Int, ra::Memory{Limb}, rb::Memory{Limb}, nn::Int)
    mn = M.n
    L = p + max(mn + 1, nn) + 1
    @assert L <= length(u) && L <= length(v)
    lp = p + mn
    t0 = Memory{Limb}(undef, lp)
    t1 = Memory{Limb}(undef, lp)
    t2 = Memory{Limb}(undef, lp)
    # both products of ulow, before u is overwritten
    p >= mn ? mul!(t0, 0, u, 0, p, M.m11, 0, mn) : mul!(t0, 0, M.m11, 0, mn, u, 0, p)
    p >= mn ? mul!(t1, 0, u, 0, p, M.m10, 0, mn) : mul!(t1, 0, M.m10, 0, mn, u, 0, p)
    p >= mn ? mul!(t2, 0, v, 0, p, M.m01, 0, mn) : mul!(t2, 0, M.m01, 0, mn, v, 0, p)
    # u' = ra*2^(64p) + t0 - t2
    copyto!(u, 1, t0, 1, p)
    @inbounds for i in p+1:L
        u[i] = zero(Limb)
    end
    copyto!(u, p + 1, ra, 1, nn)
    add_into!(u, p, L - p, t0, p, mn)
    lt = normlen(t2, 0, lp)
    if lt > 0
        brw = sub!(u, 0, u, 0, L, t2, 0, lt)
        @assert brw == 0
    end
    # v' = rb*2^(64p) + m00*vlow - t1   (t2 reused for m00*vlow)
    p >= mn ? mul!(t2, 0, v, 0, p, M.m00, 0, mn) : mul!(t2, 0, M.m00, 0, mn, v, 0, p)
    copyto!(v, 1, t2, 1, p)
    @inbounds for i in p+1:L
        v[i] = zero(Limb)
    end
    copyto!(v, p + 1, rb, 1, nn)
    add_into!(v, p, L - p, t2, p, mn)
    lt = normlen(t1, 0, lp)
    if lt > 0
        brw = sub!(v, 0, v, 0, L, t1, 0, lt)
        @assert brw == 0
    end
    return max(normlen(u, 0, L), normlen(v, 0, L), 1)
end

# One hgcd reduction step on the zero-padded pair (a, b; n limbs), keeping
# both values > s limbs: a window matrix step when the leading window makes
# progress, else a guarded division step. Buffers rotate (no copies);
# returns (a, b, w1, w2, nn) with nn = 0 when no step is possible within the
# invariant — hgcd is finished at this level.
function hgcd_step!(a::Memory{Limb}, b::Memory{Limb}, n::Int, s::Int,
                    M::HgcdMatrix, w1::Memory{Limb}, w2::Memory{Limb},
                    qb::Memory{Limb})
    @inbounds mask = a[n] | b[n]
    if !(n == s + 1 && mask < 4)   # at n == s+1 tiny tops go straight to subdiv
        wb = 64 * (n - 1) + Base.top_set_bit(mask)
        # clamp the window so reduced values stay > s limbs: value >= 2^pos
        # * (window floor 2^fexp) and pos + fexp = 64s when the clamp binds
        pos = max(wb - 126, 64 * (s - 1))
        fexp = max(64 * s - pos, 0)
        x = extract_window(a, n, pos)
        y = extract_window(b, n, pos)
        A, B, C, D, steps = hgcd_window(x, y, fexp)
        if steps > 0
            hgcd_matrix_mul1!(M, A, B, C, D)
            nl = lehmer_apply!(w1, 0, w2, 0, a, n, b, n,
                               Int64(A), -Int64(B), -Int64(C), Int64(D))
            a, w1 = w1, a
            b, w2 = w2, b
            la = normlen(a, 0, nl)
            lb = normlen(b, 0, nl)
            @assert la > s && lb > s
            return a, b, w1, w2, max(la, lb)
        end
    end
    # guarded division step: reduce the larger mod the smaller; if the
    # remainder lands at or below s limbs use quotient q - 1 (adding the
    # divisor back), and q - 1 == 0 means the invariant permits no step.
    an = normlen(a, 0, n)
    bn = normlen(b, 0, n)
    c = an == bn ? cmp_limbs(a, 0, an, b, 0, bn) : (an < bn ? -1 : 1)
    c == 0 && return a, b, w1, w2, 0
    big, lbig, small, lsmall, col = c > 0 ? (a, an, b, bn, 1) : (b, bn, a, an, 0)
    lsmall <= s && return a, b, w1, w2, 0
    divrem!(qb, 0, w1, 0, big, 0, lbig, small, 0, lsmall)
    lq = normlen(qb, 0, lbig - lsmall + 1)
    lr = normlen(w1, 0, lsmall)
    if lr <= s
        (lq == 1 && (@inbounds qb[1]) == one(Limb)) && return a, b, w1, w2, 0
        cy = lr == 0 ? (copyto!(w1, 1, small, 1, lsmall); zero(Limb)) :
                       add!(w1, 0, w1, 0, lsmall, small, 0, lsmall)
        sub_1!(qb, 0, qb, 0, lq, one(Limb))
        lq = normlen(qb, 0, lq)
        lr = lsmall
        if cy != 0
            lr += 1
            @inbounds w1[lr] = cy
        end
    end
    hgcd_matrix_update_q!(M, qb, lq, col)
    nn = max(lsmall, lr)
    @inbounds for i in lr+1:nn
        w1[i] = zero(Limb)
    end
    if col == 1
        a, w1 = w1, a
    else
        b, w1 = w1, b
    end
    return a, b, w1, w2, nn
end

# copy of the top n - p limbs into a fresh buffer with one growth limb
function hgcd_tops(x::Memory{Limb}, n::Int, p::Int)
    t = Memory{Limb}(undef, n - p + 1)
    copyto!(t, 1, x, p + 1, n - p)
    @inbounds t[n-p+1] = zero(Limb)
    return t
end

# hgcd on the zero-padded pair (a, b; n limbs, top limbs not both zero,
# capacity >= n + 1 each): reduces toward s = n÷2 + 1 limbs, accumulating
# into M (which must be the identity at entry, capacity hgcd_matrix_cap(n))
# so that (A; B) = M * (a'; b'). Buffers rotate internally — the returned
# references are authoritative. Returns (a, b, nn); nn == 0 means no
# reduction was possible (M and values untouched). Control flow is GMP's
# mpn_hgcd: recursive reduce of the top half, a stray step loop, a second
# recursion at p = 2s - n + 1 merged via hgcd_matrix_mul!, then steps down
# to the invariant boundary. Below thr the body is just the step loop (the
# Lehmer-with-matrix base case). Scratch is allocated per call: depth is
# log n with halving sizes, so the total allocation is O(n).
function hgcd!(a::Memory{Limb}, b::Memory{Limb}, n::Int, M::HgcdMatrix,
               thr::Int=HGCD_THRESHOLD)
    s = n ÷ 2 + 1
    n <= s && return a, b, 0
    success = false
    w1 = Memory{Limb}(undef, n + 1)
    w2 = Memory{Limb}(undef, n + 1)
    qb = Memory{Limb}(undef, n + 2)
    if n > thr
        n2 = 3 * n ÷ 4 + 1
        p = n ÷ 2
        ta, tb = hgcd_tops(a, n, p), hgcd_tops(b, n, p)
        ra, rb, rn = hgcd!(ta, tb, n - p, M, thr)
        if rn > 0
            n = hgcd_matrix_adjust!(M, a, b, p, ra, rb, rn)
            success = true
        end
        while n > n2
            a, b, w1, w2, nn = hgcd_step!(a, b, n, s, M, w1, w2, qb)
            nn == 0 && return a, b, success ? n : 0
            n = nn
            success = true
        end
        if n > s + 2
            p = 2 * s - n + 1
            M1 = HgcdMatrix(hgcd_matrix_cap(n - p))
            ta, tb = hgcd_tops(a, n, p), hgcd_tops(b, n, p)
            ra, rb, rn = hgcd!(ta, tb, n - p, M1, thr)
            if rn > 0
                n = hgcd_matrix_adjust!(M1, a, b, p, ra, rb, rn)
                hgcd_matrix_mul!(M, M1)
                success = true
            end
        end
    end
    while true
        a, b, w1, w2, nn = hgcd_step!(a, b, n, s, M, w1, w2, qb)
        nn == 0 && return a, b, success ? n : 0
        n = nn
        success = true
    end
end

# r = a*x + b*y with matrix entries a, b zero-padded to mn; returns the
# normalized length. Used for the gcdx cofactor update, once per hgcd
# reduction (the fresh scratch is noise at that frequency).
function cof_mul2!(r::Memory{Limb}, a::Memory{Limb}, x::Memory{Limb}, lx::Int,
                   b::Memory{Limb}, y::Memory{Limb}, ly::Int, mn::Int)
    la = normlen(a, 0, mn)
    lb = normlen(b, 0, mn)
    l1 = 0
    if la > 0 && lx > 0
        @assert la + lx <= length(r)
        la >= lx ? mul!(r, 0, a, 0, la, x, 0, lx) : mul!(r, 0, x, 0, lx, a, 0, la)
        l1 = normlen(r, 0, la + lx)
    end
    (lb == 0 || ly == 0) && return l1
    if l1 == 0
        @assert lb + ly <= length(r)
        lb >= ly ? mul!(r, 0, b, 0, lb, y, 0, ly) : mul!(r, 0, y, 0, ly, b, 0, lb)
        return normlen(r, 0, lb + ly)
    end
    w = Memory{Limb}(undef, lb + ly)
    lb >= ly ? mul!(w, 0, b, 0, lb, y, 0, ly) : mul!(w, 0, y, 0, ly, b, 0, lb)
    l2 = normlen(w, 0, lb + ly)
    l = 0
    if l1 >= l2
        c = add!(r, 0, r, 0, l1, w, 0, l2)
        l = l1
    else
        c = add!(w, 0, w, 0, l2, r, 0, l1)
        copyto!(r, 1, w, 1, l2)
        l = l2
    end
    if c != 0
        l += 1
        @inbounds r[l] = c
    end
    return l
end

# Cofactor-pair update for one hgcd reduction: (u'; v') = M^{-1} (u; v) with
# M^{-1} = (m11 -m01; -m10 m00), so on the signed V-cofactor row
#   (t_u, t_v) <- (m11*t_u + m01*t_v, m10*t_u + m00*t_v)
# with tpos preserved (det = +1 keeps the alternating-sign pattern).
function cofactor_matrix_apply!(t::Cofactors, M::HgcdMatrix)
    mn = M.n
    l1 = cof_mul2!(t.x1, M.m11, t.tu, t.ltu, M.m01, t.tv, t.ltv, mn)
    l2 = cof_mul2!(t.x2, M.m10, t.tu, t.ltu, M.m00, t.tv, t.ltv, mn)
    t.tu, t.x1 = t.x1, t.tu
    t.tv, t.x2 = t.x2, t.tv
    t.ltu, t.ltv = l1, l2
    return nothing
end

# Plain-gcd tail: once v fits two limbs, one divrem! collapses u down to v's
# size and a single UInt128 gcd finishes in registers (no cofactors to track,
# so the shortcut is available). Returns (mem, len) with the result in u.
function gcd_small_tail!(u::Memory{Limb}, lu::Int, v::Memory{Limb}, lv::Int,
                         w1::Memory{Limb}, w2::Memory{Limb})
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

# Extended-gcd tail: once u fits two limbs the reduction runs euclid63 batches
# on the in-register values (slack 0: plain truncation, wrap-exact matrix
# apply), carrying the cofactor pair through each apply!/divstep! — which the
# plain gcd tail's binary finish could not track. Returns (mem, len) in u.
function gcdext_small_tail!(u::Memory{Limb}, lu::Int, v::Memory{Limb}, lv::Int,
                            t::Cofactors, w2::Memory{Limb})
    x = extract_window(u, lu, 0)
    y = extract_window(v, lv, 0)
    while y != 0
        sh = max(65 - leading_zeros(x), 0)   # 63-bit tops
        A, B, C, D, even, steps = euclid63(UInt64(x >> sh), UInt64(y >> sh), UInt64(0), UInt64(0), Val(false))
        if steps > 0
            x, y = even ? (A * x - B * y, D * y - C * x) :
                          (B * y - A * x, C * x - D * y)
            apply!(t, A, B, C, D, even)
        else   # window stalled: one exact division step
            q, r = divrem(x, y)
            @inbounds w2[1] = q % Limb
            @inbounds w2[2] = (q >> 64) % Limb
            divstep!(t, w2, (q >> 64) != 0 ? 2 : 1)
            x, y = y, r
        end
    end
    @inbounds u[1] = x % Limb
    @inbounds u[2] = (x >> 64) % Limb
    return u, ((x >> 64) != 0 ? 2 : 1)
end

# One subquadratic reduction step: hgcd on the top third (gcd) or half (gcdx)
# of the limbs, then reconstruct the reduced full operands through the matrix;
# a plain division step when the top window can't be reduced (e.g. v barely
# reaches into it). Rotates buffers; returns (u, v, lu, lv, w1, w2).
function hgcd_reduce!(u::Memory{Limb}, lu::Int, v::Memory{Limb}, lv::Int,
                      t::Union{Cofactors, Nothing}, w1::Memory{Limb},
                      w2::Memory{Limb}, hgcd_thr::Int)
    n = lu
    p = t === nothing ? 2 * n ÷ 3 : n ÷ 2
    np = n - p
    ta = hgcd_tops(u, n, p)
    tb = Memory{Limb}(undef, np + 1)
    nv = max(lv - p, 0)
    nv > 0 && copyto!(tb, 1, v, p + 1, nv)
    @inbounds for i in nv+1:np+1
        tb[i] = zero(Limb)
    end
    M = HgcdMatrix(hgcd_matrix_cap(np))
    ra, rb, rn = hgcd!(ta, tb, np, M, hgcd_thr)
    if rn > 0
        @assert lv > p   # success implies v reached into the window
        nl = hgcd_matrix_adjust!(M, u, v, p, ra, rb, rn)
        t === nothing || cofactor_matrix_apply!(t, M)
        lu = normlen(u, 0, nl)
        lv = normlen(v, 0, nl)
    else
        divrem!(w2, 0, w1, 0, u, 0, lu, v, 0, lv)
        t === nothing || divstep!(t, w2, normlen(w2, 0, lu - lv + 1))
        u, v, w1 = v, w1, u
        lu, lv = lv, normlen(v, 0, lv)
    end
    return u, v, lu, lv, w1, w2
end

# One Lehmer window step: build a cofactor matrix from the leading 126-bit
# window (lehmer_window) and apply it to the full operands in one fused
# lehmer_apply! pass with buffer rotation, else a full divrem! step when the
# window makes no progress. Rotates buffers; returns (u, v, lu, lv, w1, w2).
function lehmer_reduce!(u::Memory{Limb}, lu::Int, v::Memory{Limb}, lv::Int,
                        t::Union{Cofactors, Nothing}, w1::Memory{Limb},
                        w2::Memory{Limb})
    ub = magnitude_bits(u, 0, lu)
    vb = magnitude_bits(v, 0, lv)
    A = B = C = D = UInt64(0)
    even = true
    steps = 0
    if ub - vb < 64
        A, B, C, D, even, steps = lehmer_window(u, lu, ub, v, lv)
    end
    if steps == 0
        # window made no progress: one full division step, rotating the
        # remainder buffer in rather than copying
        divrem!(w2, 0, w1, 0, u, 0, lu, v, 0, lv)
        t === nothing || divstep!(t, w2, normlen(w2, 0, lu - lv + 1))
        u, v, w1 = v, w1, u
        lu, lv = lv, normlen(v, 0, lv)
    else
        # (U, V) <- (A*U + B*V, C*U + D*V), one fused pass, then rotate;
        # the cofactors get the positive-magnitude image of the matrix
        sA, sB, sC, sD = Int64(A), Int64(B), Int64(C), Int64(D)
        if even
            sB, sC = -sB, -sC
        else
            sA, sD = -sA, -sD
        end
        n = lehmer_apply!(w1, 0, w2, 0, u, lu, v, lv, sA, sB, sC, sD)
        t === nothing || apply!(t, A, B, C, D, even)
        u, w1 = w1, u
        v, w2 = w2, v
        lu = normlen(u, 0, n)
        lv = normlen(v, 0, n)
    end
    return u, v, lu, lv, w1, w2
end

# Shared gcd driver over the magnitudes u[1..lu] and v[1..lv]; both buffers are
# destroyed and must have capacity >= max(lu, lv) + 1. Returns (mem, len) with
# the result in one of the two buffers. The loop keeps u >= v and dispatches
# each reduction to the widest applicable engine as the operands shrink:
# subquadratic hgcd_reduce! while lv > dc_thr, then lehmer_reduce! per 126-bit
# window, finishing in gcd_small_tail! / gcdext_small_tail!. t === nothing is
# plain gcd; a Cofactors t carries the V-cofactor pair through every swap,
# apply, and division step (the branches on t are compile-time).
function gcd_impl!(u::Memory{Limb}, lu::Int, v::Memory{Limb}, lv::Int,
                   t::Union{Cofactors, Nothing};
                   dc_thr::Int = t === nothing ? GCD_DC_THRESHOLD : GCDEXT_DC_THRESHOLD,
                   hgcd_thr::Int = HGCD_THRESHOLD)
    cap = max(lu, lv) + 1
    w1 = Memory{Limb}(undef, cap)
    w2 = Memory{Limb}(undef, cap)   # also serves as divrem!'s quotient buffer
    while true
        # invariant maintenance: normalized lengths, u >= v
        if lu < lv || (lu == lv && cmp_limbs(u, 0, lu, v, 0, lv) < 0)
            u, v = v, u
            lu, lv = lv, lu
            t === nothing || swap!(t)
        end
        lv == 0 && return u, lu
        if t === nothing && lv <= 2
            return gcd_small_tail!(u, lu, v, lv, w1, w2)
        end
        if t !== nothing && lu <= 2
            return gcdext_small_tail!(u, lu, v, lv, t, w2)
        end
        if lv > dc_thr
            u, v, lu, lv, w1, w2 = hgcd_reduce!(u, lu, v, lv, t, w1, w2, hgcd_thr)
        else
            u, v, lu, lv, w1, w2 = lehmer_reduce!(u, lu, v, lv, t, w1, w2)
        end
    end
end

gcd!(u::Memory{Limb}, lu::Int, v::Memory{Limb}, lv::Int; kw...) =
    gcd_impl!(u, lu, v, lv, nothing; kw...)

# Extended gcd: additionally returns (t, lt, tpos) with
# s*U + (tpos ? t : -t)*V == gcd for some s; |t| <= max(U, V) / gcd.
function gcdext!(u::Memory{Limb}, lu::Int, v::Memory{Limb}, lv::Int; kw...)
    t = Cofactors(max(lu, lv) + 3)
    g, lg = gcd_impl!(u, lu, v, lv, t; kw...)
    return g, lg, t.tu, t.ltu, t.tpos
end
