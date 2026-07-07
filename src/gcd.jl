# Lehmer gcd and extended gcd (Knuth TAOCP §4.5.2, Algorithm L) on magnitude
# buffers: one shared window/apply/division loop (gcd_core!), with extended
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
    return A, B, C, D, even, steps
end

# The V-cofactor pair (t_u, t_v) carried alongside gcd_core!'s (u, v).
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

# Shared Lehmer loop: gcd of the magnitudes u[1..lu] and v[1..lv]; both
# buffers are destroyed and must have capacity >= max(lu, lv) + 1. Returns
# (mem, len) with the result in one of the two buffers. Per 126-bit leading
# window, lehmer_window builds a cofactor matrix that one fused
# lehmer_apply! pass applies to the full operands (buffer rotation, no
# copies); a full divrem! step when the window makes no progress.
# t === nothing is plain gcd, finishing with UInt128 binary gcd once v fits
# two limbs. A Cofactors t carries the V-cofactor pair through every swap,
# window apply, and division step (the branches on t are compile-time), and
# the tail instead keeps running lehmer63 batches on the in-register values
# (which the binary tail could not track).
function gcd_core!(u::Memory{Limb}, lu::Int, v::Memory{Limb}, lv::Int,
                   t::Union{Cofactors, Nothing})
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
        if t !== nothing && lu <= 2
            x = extract_window(u, lu, 0)
            y = extract_window(v, lv, 0)
            while y != 0
                # lehmer63 on the values' 63-bit tops (slack 0: plain
                # truncation), wrap-exact matrix apply to the values.
                sh = max(65 - leading_zeros(x), 0)   # 63-bit tops
                A, B, C, D, even, steps = lehmer63(UInt64(x >> sh), UInt64(y >> sh), UInt64(0))
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
        ub = 64lu - leading_zeros(@inbounds u[lu])
        vb = 64lv - leading_zeros(@inbounds v[lv])
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
    end
end

gcd!(u::Memory{Limb}, lu::Int, v::Memory{Limb}, lv::Int) =
    gcd_core!(u, lu, v, lv, nothing)

# Extended gcd: additionally returns (t, lt, tpos) with
# s*U + (tpos ? t : -t)*V == gcd for some s; |t| <= max(U, V) / gcd.
function gcdext!(u::Memory{Limb}, lu::Int, v::Memory{Limb}, lv::Int)
    t = Cofactors(max(lu, lv) + 3)
    g, lg = gcd_core!(u, lu, v, lv, t)
    return g, lg, t.tu, t.ltu, t.tpos
end
