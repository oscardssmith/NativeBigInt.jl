# Floating-point NTT over GF(p), p = 2^49 - 2^33 + 1, following FLINT's
# fft_small design: coefficients live in Float64, products use the FMA
# error-free transform, and quotients use the magic-constant round.  See
# docs/superpowers/specs/2026-07-12-fpntt-design.md for the full rationale.
#
# Arithmetic model.  Values are exact integers in balanced representation,
# |x| < a few p.  Every operation is exactly correct (not approximately):
#   - a ± b is exact while |a ± b| < 2^53.
#   - fp_mulmod(x, w, wpinv) returns r ≡ w·x (mod p) exactly:
#     h + l == w·x exactly (FMA error-free transform), q is an integer
#     within (1/2 + |x|·2^-52) of w·x/p, and r = w·x - q·p is an integer
#     small enough to be exact through the final fma and add.
#   - fp_round(v) = (v + 1.5·2^52) - 1.5·2^52 rounds exactly to the nearest
#     integer for |v| <= 2^51.  p < 2^49 makes every quotient in the engine
#     fit: mulmod arguments never exceed 4p < 2^51, and fp_reduce quotients
#     are at most ~5.
#
# Lazy-bounds invariants (c := p/2^52 < 1/8; mulmod output bound is
# p·(1/2 + c·|x|/p)).  One fp_reduce per butterfly keeps everything closed:
#   - the ω^0 output of every butterfly is reduced before storing;
#   - inverse stages reduce the untwiddled input t0 on load.
# Forward stage values then converge to <= ~0.9p (transient <= 1.6p after the
# leftover radix-2), inverse stage values to <= ~3.2p, every mulmod argument
# stays <= 4p, and every intermediate sum stays far below 2^53.  The
# differential tests plus adversarial max-magnitude cases validate these
# bounds empirically.
#
# Hardware FMA and round-to-nearest are assumed.  No @fastmath anywhere near
# this file: contraction would break the fma(x, w, -h) cancellation.

const FP_PI   = 0x0001_FFFE_0000_0001          # 562941363486721 = 2^49 - 2^33 + 1
const FP_P    = 562941363486721.0
const FP_PN   = -FP_P
const FP_PINV = 1.0 / FP_P
const FP_MAGIC = 6755399441055744.0            # 1.5·2^52
const VF8 = SIMD.Vec{8,Float64}

# exact round-to-nearest-integer for |v| <= 2^51 (scalar and VF8)
@inline fp_round(v) = (v + FP_MAGIC) - FP_MAGIC

# r ≡ w·x (mod p), w in [0, p), |x| <= 4p; |r| <= p(1/2 + |x|·2^-52) < p.
# q = fp_round(x * wpinv) has no dependency on h, so both multiplies issue
# together.
@inline function fp_mulmod(x, w, wpinv)
    h = x * w
    l = fma(x, w, -h)
    q = fp_round(x * wpinv)
    return fma(q, FP_PN, h) + l
end

# r ≡ x (mod p), |r| <= p(1/2 + |x|·2^-52/p): exact whenever the quotient
# fits the magic round, |x·pinv| <= 2^51, i.e. |x| <= 2^51·p.  Butterfly
# callers use the easy end of that domain (|x| <= ~8p, quotient <= 8);
# fp_mulmod2 uses the far edge (x = h ~ 4p², quotient ~ 4p ≈ 2^51).  The
# result is exact either way: x is an exact integer (any double >= 2^53 in
# magnitude is one), q·p is an integer, and their difference is small enough
# to represent exactly.
@inline function fp_reduce(x)
    q = fp_round(x * FP_PINV)
    return fma(q, FP_PN, x)
end

# r ≡ x·y (mod p) for two data operands (no precomputed quotient), inputs
# |x|,|y| <= 2p: reduce the rounded high part, add back the exact FMA error
# (an integer <= ulp(h)/2, so the final add is exact too)
@inline function fp_mulmod2(x, y)
    h = x * y
    l = fma(x, y, -h)
    return fp_reduce(h) + l
end

# integer-domain helpers for building twiddle tables
function fpi_pow(a::UInt64, e::Integer)
    r = UInt128(1)
    b = UInt128(a % FP_PI)
    while e != 0
        isodd(e) && (r = r * b % FP_PI)
        b = b * b % FP_PI
        e >>= 1
    end
    return r % UInt64
end
@inline fpi_inv(a::UInt64) = fpi_pow(a, FP_PI - 2)

# p - 1 = 2^33 · 3 · 5 · 17 · 257
function fp_generator()
    for g in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
        all(fpi_pow(UInt64(g), (FP_PI - 1) ÷ q) != 1 for q in (2, 3, 5, 17, 257)) &&
            return UInt64(g)
    end
    error("unreachable: GF(p) has a small generator")
end

# ---------------------------------------------------------------------------
# Transform plan, mirroring NttPlan: N = m·2^k, m in (1, 3, 5, 15), k <= 33.
# Twiddles are stored as (w, w/p) pairs so fp_mulmod's quotient estimate is a
# table load.

struct FpNttStage
    q::Int
    w1::Vector{Float64};  w1p::Vector{Float64}
    w2::Vector{Float64};  w2p::Vector{Float64}
    w3::Vector{Float64};  w3p::Vector{Float64}
end

function build_fp_stage(table::Vector{UInt64}, N2::Int, L::Int)
    q = L >> 2
    st = N2 ÷ L
    len = max(q, 8)
    mk(f) = [Float64(table[f(j % q)*st+1]) for j in 0:len-1]
    w1 = mk(j -> j);  w2 = mk(j -> 2j);  w3 = mk(j -> 3j)
    return FpNttStage(q, w1, w1 ./ FP_P, w2, w2 ./ FP_P, w3, w3 ./ FP_P)
end

struct FpOddStage
    m::Int
    span::Int
    Q::Int
    rot::Vector{Float64};  rotp::Vector{Float64}
    tw::Vector{Vector{Float64}};  twp::Vector{Vector{Float64}}
end

function build_fp_odd(m::Int, span::Int, root::UInt64)
    Q = span ÷ m
    c = fpi_pow(root, Q)
    rot = [Float64(fpi_pow(c, r)) for r in 1:m-1]
    tw = Vector{Vector{Float64}}(undef, m - 1)
    for r in 1:m-1
        ωr = fpi_pow(root, r)
        t = Vector{Float64}(undef, Q)
        f = UInt64(1)
        for j in 1:Q
            t[j] = Float64(f)
            f = UInt64(UInt128(f) * ωr % FP_PI)
        end
        tw[r] = t
    end
    return FpOddStage(m, span, Q, rot, rot ./ FP_P, tw, [t ./ FP_P for t in tw])
end

struct FpNttPlan
    N::Int
    N2::Int
    ninv::Float64;  ninvp::Float64
    wi::Float64;    wip::Float64    # i = ω2^(N2/4); the inverse reuses it (i^-1 == -i)
    oddf::Vector{FpOddStage}
    oddi::Vector{FpOddStage}
    fwd::Vector{FpNttStage}
    inv::Vector{FpNttStage}
end

function build_fp_plan(N::Int)
    k = trailing_zeros(N)
    m = N >> k
    @assert m in (1, 3, 5, 15) && k <= 33 && (m == 1 || k >= 2)
    @assert (FP_PI - 1) % N == 0
    ω = fpi_pow(fp_generator(), (FP_PI - 1) ÷ N)
    @assert fpi_pow(ω, N >> 1) == FP_PI - 1
    m % 3 == 0 && @assert fpi_pow(ω, N ÷ 3) != 1
    m % 5 == 0 && @assert fpi_pow(ω, N ÷ 5) != 1
    ωinv = fpi_inv(ω)

    oddf = FpOddStage[]
    oddi = FpOddStage[]
    span = N
    r, ri = ω, ωinv
    for mf in (3, 5)
        if m % mf == 0
            push!(oddf, build_fp_odd(mf, span, r))
            push!(oddi, build_fp_odd(mf, span, ri))
            r = fpi_pow(r, mf)
            ri = fpi_pow(ri, mf)
            span ÷= mf
        end
    end
    reverse!(oddi)

    N2 = span
    M = max(1, (3N2) >> 2)
    tw = Vector{UInt64}(undef, M)
    twinv = Vector{UInt64}(undef, M)
    fw, fi = UInt64(1), UInt64(1)
    for j in 1:M
        tw[j] = fw
        twinv[j] = fi
        fw = UInt64(UInt128(fw) * r % FP_PI)
        fi = UInt64(UInt128(fi) * ri % FP_PI)
    end
    fwd = FpNttStage[]
    L = N2
    while L >= 4
        push!(fwd, build_fp_stage(tw, N2, L))
        L >>= 2
    end
    inv = FpNttStage[]
    L = isodd(k) ? 8 : 16
    while L <= N2
        push!(inv, build_fp_stage(twinv, N2, L))
        L <<= 2
    end
    wi = N2 >= 4 ? Float64(fpi_pow(r, N2 >> 2)) : 1.0
    ninv = Float64(fpi_inv(N % UInt64))
    return FpNttPlan(N, N2, ninv, ninv / FP_P, wi, wi / FP_P, oddf, oddi, fwd, inv)
end

mutable struct FpNttPlanCache
    @atomic plans::Dict{Int,FpNttPlan}
end
const FPNTT_PLAN_LOCK = ReentrantLock()
const FPNTT_PLAN_CACHE = FpNttPlanCache(Dict{Int,FpNttPlan}())
function fp_ntt_plan(N::Int)
    plan = get(@atomic(:acquire, FPNTT_PLAN_CACHE.plans), N, nothing)
    plan === nothing || return plan
    lock(FPNTT_PLAN_LOCK) do
        plans = @atomic :acquire FPNTT_PLAN_CACHE.plans
        cached = get(plans, N, nothing)
        cached === nothing || return cached
        built = build_fp_plan(N)
        next = copy(plans)
        next[N] = built
        @atomic :release FPNTT_PLAN_CACHE.plans = next
        return built
    end
end

# ---------------------------------------------------------------------------
# Odd-radix stages.  Same Winograd radix-3 / plain radix-5 shapes as ntt.jl;
# the ω^0 output is fp_reduced, and inverse stages fp_reduce t0 on load.

function fp_radix3_fwd!(x::Vector{Float64}, o::Int, st::FpOddStage)
    Q = st.Q
    ω3, ω3p = st.rot[1], st.rotp[1]
    w1, w1p, w2, w2p = st.tw[1], st.twp[1], st.tw[2], st.twp[2]
    j = 0
    if Q >= 8
        vω3, vω3p = VF8(ω3), VF8(ω3p)
        @inbounds while j + 8 <= Q
            i0 = o + j + 1
            a = SIMD.vload(VF8, x, i0)
            b = SIMD.vload(VF8, x, i0 + Q)
            c = SIMD.vload(VF8, x, i0 + 2Q)
            u = fp_mulmod(b - c, vω3, vω3p)
            SIMD.vstore(fp_reduce(a + (b + c)), x, i0)
            SIMD.vstore(fp_mulmod((a - c) + u, SIMD.vload(VF8, w1, j + 1),
                                  SIMD.vload(VF8, w1p, j + 1)), x, i0 + Q)
            SIMD.vstore(fp_mulmod((a - b) - u, SIMD.vload(VF8, w2, j + 1),
                                  SIMD.vload(VF8, w2p, j + 1)), x, i0 + 2Q)
            j += 8
        end
    end
    @inbounds while j < Q
        i0 = o + j + 1
        a = x[i0]
        b = x[i0+Q]
        c = x[i0+2Q]
        u = fp_mulmod(b - c, ω3, ω3p)
        x[i0] = fp_reduce(a + (b + c))
        x[i0+Q] = fp_mulmod((a - c) + u, w1[j+1], w1p[j+1])
        x[i0+2Q] = fp_mulmod((a - b) - u, w2[j+1], w2p[j+1])
        j += 1
    end
    return x
end

function fp_radix3_inv!(x::Vector{Float64}, o::Int, st::FpOddStage)
    Q = st.Q
    λ3, λ3p = st.rot[1], st.rotp[1]
    w1, w1p, w2, w2p = st.tw[1], st.twp[1], st.tw[2], st.twp[2]
    j = 0
    if Q >= 8
        vλ3, vλ3p = VF8(λ3), VF8(λ3p)
        @inbounds while j + 8 <= Q
            i0 = o + j + 1
            t0 = fp_reduce(SIMD.vload(VF8, x, i0))
            t1 = fp_mulmod(SIMD.vload(VF8, x, i0 + Q),
                           SIMD.vload(VF8, w1, j + 1), SIMD.vload(VF8, w1p, j + 1))
            t2 = fp_mulmod(SIMD.vload(VF8, x, i0 + 2Q),
                           SIMD.vload(VF8, w2, j + 1), SIMD.vload(VF8, w2p, j + 1))
            u = fp_mulmod(t1 - t2, vλ3, vλ3p)
            SIMD.vstore(fp_reduce(t0 + (t1 + t2)), x, i0)
            SIMD.vstore((t0 - t2) + u, x, i0 + Q)
            SIMD.vstore((t0 - t1) - u, x, i0 + 2Q)
            j += 8
        end
    end
    @inbounds while j < Q
        i0 = o + j + 1
        t0 = fp_reduce(x[i0])
        t1 = fp_mulmod(x[i0+Q], w1[j+1], w1p[j+1])
        t2 = fp_mulmod(x[i0+2Q], w2[j+1], w2p[j+1])
        u = fp_mulmod(t1 - t2, λ3, λ3p)
        x[i0] = fp_reduce(t0 + (t1 + t2))
        x[i0+Q] = (t0 - t2) + u
        x[i0+2Q] = (t0 - t1) - u
        j += 1
    end
    return x
end

# order-5 DFT combine; y0 comes back unreduced (callers reduce before storing)
@inline function fp_dft5(t0, t1, t2, t3, t4, c1, c2, c3, c4, c1p, c2p, c3p, c4p)
    s = (t1 + t2) + (t3 + t4)
    y0 = t0 + s
    y1 = t0 + ((fp_mulmod(t1, c1, c1p) + fp_mulmod(t2, c2, c2p)) +
               (fp_mulmod(t3, c3, c3p) + fp_mulmod(t4, c4, c4p)))
    y2 = t0 + ((fp_mulmod(t1, c2, c2p) + fp_mulmod(t2, c4, c4p)) +
               (fp_mulmod(t3, c1, c1p) + fp_mulmod(t4, c3, c3p)))
    y3 = t0 + ((fp_mulmod(t1, c3, c3p) + fp_mulmod(t2, c1, c1p)) +
               (fp_mulmod(t3, c4, c4p) + fp_mulmod(t4, c2, c2p)))
    y4 = t0 + ((fp_mulmod(t1, c4, c4p) + fp_mulmod(t2, c3, c3p)) +
               (fp_mulmod(t3, c2, c2p) + fp_mulmod(t4, c1, c1p)))
    return y0, y1, y2, y3, y4
end

function fp_radix5_fwd!(x::Vector{Float64}, o::Int, st::FpOddStage)
    Q = st.Q
    c1, c2, c3, c4 = st.rot[1], st.rot[2], st.rot[3], st.rot[4]
    c1p, c2p, c3p, c4p = st.rotp[1], st.rotp[2], st.rotp[3], st.rotp[4]
    w = st.tw
    wp = st.twp
    j = 0
    if Q >= 8
        v1, v2, v3, v4 = VF8(c1), VF8(c2), VF8(c3), VF8(c4)
        v1p, v2p, v3p, v4p = VF8(c1p), VF8(c2p), VF8(c3p), VF8(c4p)
        @inbounds while j + 8 <= Q
            i0 = o + j + 1
            a = SIMD.vload(VF8, x, i0)
            b = SIMD.vload(VF8, x, i0 + Q)
            c = SIMD.vload(VF8, x, i0 + 2Q)
            d = SIMD.vload(VF8, x, i0 + 3Q)
            e = SIMD.vload(VF8, x, i0 + 4Q)
            y0, y1, y2, y3, y4 = fp_dft5(a, b, c, d, e, v1, v2, v3, v4,
                                         v1p, v2p, v3p, v4p)
            SIMD.vstore(fp_reduce(y0), x, i0)
            SIMD.vstore(fp_mulmod(y1, SIMD.vload(VF8, w[1], j + 1),
                                  SIMD.vload(VF8, wp[1], j + 1)), x, i0 + Q)
            SIMD.vstore(fp_mulmod(y2, SIMD.vload(VF8, w[2], j + 1),
                                  SIMD.vload(VF8, wp[2], j + 1)), x, i0 + 2Q)
            SIMD.vstore(fp_mulmod(y3, SIMD.vload(VF8, w[3], j + 1),
                                  SIMD.vload(VF8, wp[3], j + 1)), x, i0 + 3Q)
            SIMD.vstore(fp_mulmod(y4, SIMD.vload(VF8, w[4], j + 1),
                                  SIMD.vload(VF8, wp[4], j + 1)), x, i0 + 4Q)
            j += 8
        end
    end
    @inbounds while j < Q
        i0 = o + j + 1
        y0, y1, y2, y3, y4 = fp_dft5(x[i0], x[i0+Q], x[i0+2Q], x[i0+3Q], x[i0+4Q],
                                     c1, c2, c3, c4, c1p, c2p, c3p, c4p)
        x[i0] = fp_reduce(y0)
        x[i0+Q] = fp_mulmod(y1, w[1][j+1], wp[1][j+1])
        x[i0+2Q] = fp_mulmod(y2, w[2][j+1], wp[2][j+1])
        x[i0+3Q] = fp_mulmod(y3, w[3][j+1], wp[3][j+1])
        x[i0+4Q] = fp_mulmod(y4, w[4][j+1], wp[4][j+1])
        j += 1
    end
    return x
end

function fp_radix5_inv!(x::Vector{Float64}, o::Int, st::FpOddStage)
    Q = st.Q
    c1, c2, c3, c4 = st.rot[1], st.rot[2], st.rot[3], st.rot[4]
    c1p, c2p, c3p, c4p = st.rotp[1], st.rotp[2], st.rotp[3], st.rotp[4]
    w = st.tw
    wp = st.twp
    j = 0
    if Q >= 8
        v1, v2, v3, v4 = VF8(c1), VF8(c2), VF8(c3), VF8(c4)
        v1p, v2p, v3p, v4p = VF8(c1p), VF8(c2p), VF8(c3p), VF8(c4p)
        @inbounds while j + 8 <= Q
            i0 = o + j + 1
            t0 = fp_reduce(SIMD.vload(VF8, x, i0))
            t1 = fp_mulmod(SIMD.vload(VF8, x, i0 + Q),
                           SIMD.vload(VF8, w[1], j + 1), SIMD.vload(VF8, wp[1], j + 1))
            t2 = fp_mulmod(SIMD.vload(VF8, x, i0 + 2Q),
                           SIMD.vload(VF8, w[2], j + 1), SIMD.vload(VF8, wp[2], j + 1))
            t3 = fp_mulmod(SIMD.vload(VF8, x, i0 + 3Q),
                           SIMD.vload(VF8, w[3], j + 1), SIMD.vload(VF8, wp[3], j + 1))
            t4 = fp_mulmod(SIMD.vload(VF8, x, i0 + 4Q),
                           SIMD.vload(VF8, w[4], j + 1), SIMD.vload(VF8, wp[4], j + 1))
            y0, y1, y2, y3, y4 = fp_dft5(t0, t1, t2, t3, t4, v1, v2, v3, v4,
                                         v1p, v2p, v3p, v4p)
            SIMD.vstore(fp_reduce(y0), x, i0)
            SIMD.vstore(y1, x, i0 + Q)
            SIMD.vstore(y2, x, i0 + 2Q)
            SIMD.vstore(y3, x, i0 + 3Q)
            SIMD.vstore(y4, x, i0 + 4Q)
            j += 8
        end
    end
    @inbounds while j < Q
        i0 = o + j + 1
        t0 = fp_reduce(x[i0])
        t1 = fp_mulmod(x[i0+Q], w[1][j+1], wp[1][j+1])
        t2 = fp_mulmod(x[i0+2Q], w[2][j+1], wp[2][j+1])
        t3 = fp_mulmod(x[i0+3Q], w[3][j+1], wp[3][j+1])
        t4 = fp_mulmod(x[i0+4Q], w[4][j+1], wp[4][j+1])
        y0, y1, y2, y3, y4 = fp_dft5(t0, t1, t2, t3, t4, c1, c2, c3, c4,
                                     c1p, c2p, c3p, c4p)
        x[i0] = fp_reduce(y0)
        x[i0+Q] = y1
        x[i0+2Q] = y2
        x[i0+3Q] = y3
        x[i0+4Q] = y4
        j += 1
    end
    return x
end

# ---------------------------------------------------------------------------
# Radix-4 butterfly cores.  i^-1 == -i, so the inverse keeps the same wi and
# swaps the add/sub pair around w, mirroring dft4_fwd/dft4_inv in ntt.jl.
# y0 is returned unreduced; callers fp_reduce before storing.

@inline function fp_dft4_fwd(a, b, c, d, wi, wip)
    apc = a + c
    amc = a - c
    bpd = b + d
    ibmd = fp_mulmod(b - d, wi, wip)
    return apc + bpd, amc + ibmd, apc - bpd, amc - ibmd
end

@inline function fp_dft4_inv(t0, t1, t2, t3, wi, wip)
    u = t0 + t2
    p_ = t0 - t2
    v = t1 + t3
    w = fp_mulmod(t1 - t3, wi, wip)
    return u + v, p_ - w, u - v, p_ + w
end

# small-q pow2 stages via the shared gather/scatter shuffles from ntt.jl
function fp_smallq_fwd!(x::Vector{Float64}, o::Int, N2::Int, ::Val{Q},
                        stg::FpNttStage, vwi::VF8, vwip::VF8) where {Q}
    vw1 = SIMD.vload(VF8, stg.w1, 1); vw1p = SIMD.vload(VF8, stg.w1p, 1)
    vw2 = SIMD.vload(VF8, stg.w2, 1); vw2p = SIMD.vload(VF8, stg.w2p, 1)
    vw3 = SIMD.vload(VF8, stg.w3, 1); vw3p = SIMD.vload(VF8, stg.w3p, 1)
    @inbounds for i0 in o+1:32:o+N2-31
        v0 = SIMD.vload(VF8, x, i0)
        v1 = SIMD.vload(VF8, x, i0 + 8)
        v2 = SIMD.vload(VF8, x, i0 + 16)
        v3 = SIMD.vload(VF8, x, i0 + 24)
        a, b, c, d = ntt_gather4(Val(Q), v0, v1, v2, v3)
        y0, y1, y2, y3 = fp_dft4_fwd(a, b, c, d, vwi, vwip)
        o0, o1, o2, o3 = ntt_scatter4(Val(Q), fp_reduce(y0),
                                      fp_mulmod(y1, vw1, vw1p),
                                      fp_mulmod(y2, vw2, vw2p),
                                      fp_mulmod(y3, vw3, vw3p))
        SIMD.vstore(o0, x, i0)
        SIMD.vstore(o1, x, i0 + 8)
        SIMD.vstore(o2, x, i0 + 16)
        SIMD.vstore(o3, x, i0 + 24)
    end
    return x
end

function fp_smallq_inv!(x::Vector{Float64}, o::Int, N2::Int, ::Val{Q},
                        stg::FpNttStage, vwi::VF8, vwip::VF8) where {Q}
    vw1 = SIMD.vload(VF8, stg.w1, 1); vw1p = SIMD.vload(VF8, stg.w1p, 1)
    vw2 = SIMD.vload(VF8, stg.w2, 1); vw2p = SIMD.vload(VF8, stg.w2p, 1)
    vw3 = SIMD.vload(VF8, stg.w3, 1); vw3p = SIMD.vload(VF8, stg.w3p, 1)
    @inbounds for i0 in o+1:32:o+N2-31
        v0 = SIMD.vload(VF8, x, i0)
        v1 = SIMD.vload(VF8, x, i0 + 8)
        v2 = SIMD.vload(VF8, x, i0 + 16)
        v3 = SIMD.vload(VF8, x, i0 + 24)
        g0, g1, g2, g3 = ntt_gather4(Val(Q), v0, v1, v2, v3)
        z0, z1, z2, z3 = fp_dft4_inv(fp_reduce(g0),
                                     fp_mulmod(g1, vw1, vw1p),
                                     fp_mulmod(g2, vw2, vw2p),
                                     fp_mulmod(g3, vw3, vw3p), vwi, vwip)
        o0, o1, o2, o3 = ntt_scatter4(Val(Q), fp_reduce(z0), z1, z2, z3)
        SIMD.vstore(o0, x, i0)
        SIMD.vstore(o1, x, i0 + 8)
        SIMD.vstore(o2, x, i0 + 16)
        SIMD.vstore(o3, x, i0 + 24)
    end
    return x
end

# ---------------------------------------------------------------------------
# Power-of-two pipelines, mirroring ntt_fwd_pow2!/ntt_inv_pow2!.

function fp_fwd_pow2!(x::Vector{Float64}, o::Int, plan::FpNttPlan)
    N2 = plan.N2
    wi, wip = plan.wi, plan.wip
    vwi, vwip = VF8(wi), VF8(wip)
    L = N2
    for stg in plan.fwd
        q = stg.q
        L = 4q
        if q >= 8
            w1, w1p, w2, w2p, w3, w3p = stg.w1, stg.w1p, stg.w2, stg.w2p, stg.w3, stg.w3p
            for s in o:L:o+N2-1
                @inbounds for j in 0:8:q-8
                    i0 = s + j + 1
                    a = SIMD.vload(VF8, x, i0)
                    b = SIMD.vload(VF8, x, i0 + q)
                    c = SIMD.vload(VF8, x, i0 + 2q)
                    d = SIMD.vload(VF8, x, i0 + 3q)
                    y0, y1, y2, y3 = fp_dft4_fwd(a, b, c, d, vwi, vwip)
                    SIMD.vstore(fp_reduce(y0), x, i0)
                    SIMD.vstore(fp_mulmod(y1, SIMD.vload(VF8, w1, j + 1),
                                          SIMD.vload(VF8, w1p, j + 1)), x, i0 + q)
                    SIMD.vstore(fp_mulmod(y2, SIMD.vload(VF8, w2, j + 1),
                                          SIMD.vload(VF8, w2p, j + 1)), x, i0 + 2q)
                    SIMD.vstore(fp_mulmod(y3, SIMD.vload(VF8, w3, j + 1),
                                          SIMD.vload(VF8, w3p, j + 1)), x, i0 + 3q)
                end
            end
        elseif N2 >= 32
            q == 4 ? fp_smallq_fwd!(x, o, N2, Val(4), stg, vwi, vwip) :
            q == 2 ? fp_smallq_fwd!(x, o, N2, Val(2), stg, vwi, vwip) :
                     fp_smallq_fwd!(x, o, N2, Val(1), stg, vwi, vwip)
        else
            w1, w1p, w2, w2p, w3, w3p = stg.w1, stg.w1p, stg.w2, stg.w2p, stg.w3, stg.w3p
            for s in o:L:o+N2-1
                @inbounds for j in 0:q-1
                    a = x[s+j+1]
                    b = x[s+j+q+1]
                    c = x[s+j+2q+1]
                    d = x[s+j+3q+1]
                    y0, y1, y2, y3 = fp_dft4_fwd(a, b, c, d, wi, wip)
                    x[s+j+1] = fp_reduce(y0)
                    x[s+j+q+1] = fp_mulmod(y1, w1[j+1], w1p[j+1])
                    x[s+j+2q+1] = fp_mulmod(y2, w2[j+1], w2p[j+1])
                    x[s+j+3q+1] = fp_mulmod(y3, w3[j+1], w3p[j+1])
                end
            end
        end
        L = q
    end
    if L == 2
        # leftover radix-2 stage, twiddle-free; outputs <= 2·0.9p feed the
        # pointwise stage, whose quotient bound tolerates 2p inputs
        @inbounds for s in o:2:o+N2-1
            u = x[s+1]
            v = x[s+2]
            x[s+1] = u + v
            x[s+2] = u - v
        end
    end
    return x
end

function fp_inv_pow2!(x::Vector{Float64}, o::Int, plan::FpNttPlan)
    N2 = plan.N2
    wi, wip = plan.wi, plan.wip
    vwi, vwip = VF8(wi), VF8(wip)
    ninv, ninvp = plan.ninv, plan.ninvp
    if isodd(trailing_zeros(N2))
        @inbounds for s in o:2:o+N2-1
            u = x[s+1]
            t = x[s+2]
            x[s+1] = fp_mulmod(u + t, ninv, ninvp)
            x[s+2] = fp_mulmod(u - t, ninv, ninvp)
        end
    elseif N2 >= 4
        @inbounds for s in o:4:o+N2-1
            t0 = fp_mulmod(x[s+1], ninv, ninvp)
            t1 = fp_mulmod(x[s+2], ninv, ninvp)
            t2 = fp_mulmod(x[s+3], ninv, ninvp)
            t3 = fp_mulmod(x[s+4], ninv, ninvp)
            y0, y1, y2, y3 = fp_dft4_inv(t0, t1, t2, t3, wi, wip)
            x[s+1] = y0
            x[s+2] = y1
            x[s+3] = y2
            x[s+4] = y3
        end
    else
        @inbounds for s in o+1:o+N2
            x[s] = fp_mulmod(x[s], ninv, ninvp)
        end
    end
    for stg in plan.inv
        q = stg.q
        L = 4q
        if q >= 8
            w1, w1p, w2, w2p, w3, w3p = stg.w1, stg.w1p, stg.w2, stg.w2p, stg.w3, stg.w3p
            for s in o:L:o+N2-1
                @inbounds for j in 0:8:q-8
                    i0 = s + j + 1
                    t0 = fp_reduce(SIMD.vload(VF8, x, i0))
                    t1 = fp_mulmod(SIMD.vload(VF8, x, i0 + q),
                                   SIMD.vload(VF8, w1, j + 1), SIMD.vload(VF8, w1p, j + 1))
                    t2 = fp_mulmod(SIMD.vload(VF8, x, i0 + 2q),
                                   SIMD.vload(VF8, w2, j + 1), SIMD.vload(VF8, w2p, j + 1))
                    t3 = fp_mulmod(SIMD.vload(VF8, x, i0 + 3q),
                                   SIMD.vload(VF8, w3, j + 1), SIMD.vload(VF8, w3p, j + 1))
                    y0, y1, y2, y3 = fp_dft4_inv(t0, t1, t2, t3, vwi, vwip)
                    SIMD.vstore(fp_reduce(y0), x, i0)
                    SIMD.vstore(y1, x, i0 + q)
                    SIMD.vstore(y2, x, i0 + 2q)
                    SIMD.vstore(y3, x, i0 + 3q)
                end
            end
        elseif N2 >= 32
            q == 4 ? fp_smallq_inv!(x, o, N2, Val(4), stg, vwi, vwip) :
            q == 2 ? fp_smallq_inv!(x, o, N2, Val(2), stg, vwi, vwip) :
                     fp_smallq_inv!(x, o, N2, Val(1), stg, vwi, vwip)
        else
            w1, w1p, w2, w2p, w3, w3p = stg.w1, stg.w1p, stg.w2, stg.w2p, stg.w3, stg.w3p
            for s in o:L:o+N2-1
                @inbounds for j in 0:q-1
                    t0 = fp_reduce(x[s+j+1])
                    t1 = fp_mulmod(x[s+j+q+1], w1[j+1], w1p[j+1])
                    t2 = fp_mulmod(x[s+j+2q+1], w2[j+1], w2p[j+1])
                    t3 = fp_mulmod(x[s+j+3q+1], w3[j+1], w3p[j+1])
                    y0, y1, y2, y3 = fp_dft4_inv(t0, t1, t2, t3, wi, wip)
                    x[s+j+1] = fp_reduce(y0)
                    x[s+j+q+1] = y1
                    x[s+j+2q+1] = y2
                    x[s+j+3q+1] = y3
                end
            end
        end
    end
    return x
end

function fp_ntt_fwd!(x::Vector{Float64}, plan::FpNttPlan)
    for st in plan.oddf
        for o in 0:st.span:plan.N-1
            st.m == 3 ? fp_radix3_fwd!(x, o, st) : fp_radix5_fwd!(x, o, st)
        end
    end
    for o in 0:plan.N2:plan.N-1
        fp_fwd_pow2!(x, o, plan)
    end
    return x
end

function fp_ntt_inv!(x::Vector{Float64}, plan::FpNttPlan)
    for o in 0:plan.N2:plan.N-1
        fp_inv_pow2!(x, o, plan)
    end
    for st in plan.oddi
        for o in 0:st.span:plan.N-1
            st.m == 3 ? fp_radix3_inv!(x, o, st) : fp_radix5_inv!(x, o, st)
        end
    end
    return x
end

# ---------------------------------------------------------------------------
# Coefficient-domain layer.  Same pipeline as ntt.jl with Float64 points.

# chunk width and transform length: exactness needs every convolution
# coefficient (a sum of min(nca, ncb) chunk products) below p, so b <= 24
function fp_ntt_params(bits_a::Int, bits_b::Int)
    for b in 24:-1:1
        nca = cld(bits_a, b)
        ncb = cld(bits_b, b)
        if UInt128(min(nca, ncb)) * (UInt128(2)^b - 1)^2 < FP_PI
            return b, ntt_len(nca + ncb - 1)
        end
    end
    error("unreachable: b == 1 always satisfies the bound for supported sizes")
end

# ntt_pack with Float64 output; chunks < 2^24 are exact doubles and already
# canonical residues
function fp_ntt_pack(limbs::Memory{Limb}, lo::Int, n::Int, b::Int, nch::Int, N::Int)
    x = Vector{Float64}(undef, N)
    mask = (UInt64(1) << b) - 1
    imax = min(nch - 1, fld(64 * (n - 1) - 1, b))
    @inbounds for i in 0:imax
        bit = i * b
        w = bit >> 6
        sh = bit & 63
        c = (limbs[lo+w+1] >>> sh) | (limbs[lo+w+2] << (64 - sh))
        x[i+1] = Float64(c & mask)
    end
    @inbounds for i in imax+1:nch-1
        bit = i * b
        w = bit >> 6
        sh = bit & 63
        c = limbs[lo+w+1] >>> sh
        if sh + b > 64 && w + 2 <= n
            c |= limbs[lo+w+2] << (64 - sh)
        end
        x[i+1] = Float64(c & mask)
    end
    fill!(view(x, nch+1:N), 0.0)
    return x
end

function fp_ntt_pointwise!(xa::Vector{Float64}, xb::Vector{Float64})
    n = length(xa)
    i = 1
    if n >= 8
        @inbounds while i + 7 <= n
            SIMD.vstore(fp_mulmod2(SIMD.vload(VF8, xa, i), SIMD.vload(VF8, xb, i)), xa, i)
            i += 8
        end
    end
    @inbounds while i <= n
        xa[i] = fp_mulmod2(xa[i], xb[i])
        i += 1
    end
    return xa
end

# canonicalize each coefficient to [0, p) and stream limbs out, as in
# ntt_unpack!.  Coefficients < 2^49 with b <= 24 leave far more accumulator
# margin than the Goldilocks unpack (whose bound was coefficients < 2^64,
# b <= 32), so the same single-UInt128 flush cadence is safe.
function fp_ntt_unpack!(r::Memory{Limb}, ro::Int, rn::Int, x::Vector{Float64},
                        nconv::Int, b::Int)
    acc = UInt128(0)
    outw = 1
    outbit = 0
    @inbounds for i in 0:nconv-1
        v = fp_reduce(x[i+1])
        v < 0.0 && (v += FP_P)
        c = unsafe_trunc(UInt64, v)
        s = i * b - outbit
        if s >= 64
            r[ro+outw] = acc % UInt64
            acc >>= 64
            outw += 1
            outbit += 64
            s -= 64
        end
        acc += UInt128(c) << s
    end
    @inbounds while outw <= rn
        r[ro+outw] = acc % UInt64
        acc >>= 64
        outw += 1
    end
    return r
end

# ---------------------------------------------------------------------------
# mpn-layer entry points, same contracts as mul_ntt!/sqr_ntt!.

function mul_fpntt!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, m::Int,
                    b::Memory{Limb}, bo::Int, n::Int)
    bits_a = magnitude_bits(a, ao, m)
    bits_b = magnitude_bits(b, bo, n)
    bch, N = fp_ntt_params(bits_a, bits_b)
    plan = fp_ntt_plan(N)
    nca, ncb = cld(bits_a, bch), cld(bits_b, bch)
    xa = fp_ntt_pack(a, ao, m, bch, nca, N)
    xb = fp_ntt_pack(b, bo, n, bch, ncb, N)
    fp_ntt_fwd!(xa, plan)
    fp_ntt_fwd!(xb, plan)
    fp_ntt_pointwise!(xa, xb)
    fp_ntt_inv!(xa, plan)
    fp_ntt_unpack!(r, ro, m + n, xa, nca + ncb - 1, bch)
    return nothing
end

function sqr_fpntt!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int)
    bits = magnitude_bits(a, ao, n)
    bch, N = fp_ntt_params(bits, bits)
    plan = fp_ntt_plan(N)
    nca = cld(bits, bch)
    xa = fp_ntt_pack(a, ao, n, bch, nca, N)
    fp_ntt_fwd!(xa, plan)
    fp_ntt_pointwise!(xa, xa)
    fp_ntt_inv!(xa, plan)
    fp_ntt_unpack!(r, ro, 2n, xa, 2nca - 1, bch)
    return nothing
end
