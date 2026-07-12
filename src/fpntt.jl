# Floating-point NTT multiplication following FLINT's fft_small design:
# coefficients live in Float64, products use the FMA error-free transform,
# and quotients use the magic-constant round.  The operands are split into
# b-bit chunks and the convolution is computed by CRT over two ~2^49 primes
# (working modulus p1·p2 ≈ 2^98.97, so b ≈ 41-44); a single-prime pipeline
# over p1 alone is kept at the bottom of the file as an unwired test
# cross-check.
#
# Arithmetic model.  Values are exact integers in balanced representation,
# |x| < a few p.  Every operation is exactly correct (not approximately):
#   - a ± b is exact while |a ± b| < 2^53.
#   - fp_mulmod(x, w, wpinv, F) returns r ≡ w·x (mod p) exactly:
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
# All bounds hold for any prime p < 2^49, so the engine is parametrized by an
# FpCtx carrying the per-prime constants and both residue transforms run
# through the same code.
#
# Hardware FMA and round-to-nearest are assumed.  No @fastmath anywhere near
# this file: contraction would break the fma(x, w, -h) cancellation.

const FP_MAGIC = 6755399441055744.0            # 1.5·2^52
const VF8 = SIMD.Vec{8,Float64}

# Per-prime constants as a type-parameter singleton: every property access
# const-folds to a literal in the compiled code (measured ~3-4% faster than
# carrying them as runtime struct fields), while the engine stays parametric
# over the prime.  Properties: pi (the prime as UInt64), p, pn = -p,
# pinv = 1/p, facs (prime factors of p-1).
struct FpCtx{PI,FACS} end
FpCtx(pi::UInt64, facs::Tuple) = FpCtx{pi,facs}()
@inline function Base.getproperty(::FpCtx{PI,FACS}, s::Symbol) where {PI,FACS}
    s === :pi   && return PI
    s === :p    && return Float64(PI)
    s === :pn   && return -Float64(PI)
    s === :pinv && return 1.0 / Float64(PI)
    s === :facs && return FACS
    error("FpCtx has no property $s")
end

# The primes.  Both < 2^49 (rounding bounds above), 2-adicity >= 33 with
# 15 | p - 1 (the {1,3,5,15}·2^k length family), product ~2^98.97 (the
# two-prime working modulus). FP_CTX1.pi etc. are the canonical accessors.
const FP_CTX1 = FpCtx(UInt(2^49 - 2^33 + 1), (2, 3, 5, 17, 257)) # p1 - 1 = 2^33·3·5·17·257
const FP_CTX2 = FpCtx(UInt(255 * 2^41 + 1), (2, 3, 5, 17))        # p2 - 1 = 2^41·3·5·17

# exact round-to-nearest-integer for |v| <= 2^51 (scalar and VF8)
@inline fp_round(v) = (v + FP_MAGIC) - FP_MAGIC

# r ≡ w·x (mod p), w in [0, p), |x| <= 4p; |r| <= p(1/2 + |x|·2^-52) < p.
# q = fp_round(x * wpinv) has no dependency on h, so both multiplies issue
# together.
@inline function fp_mulmod(x, w, wpinv, F::FpCtx)
    h = x * w
    l = fma(x, w, -h)
    q = fp_round(x * wpinv)
    return fma(q, F.pn, h) + l
end

# r ≡ x (mod p), |r| <= p(1/2 + |x|·2^-52/p): exact whenever the quotient
# fits the magic round, |x·pinv| <= 2^51, i.e. |x| <= 2^51·p.  Butterfly
# callers use the easy end of that domain (|x| <= ~8p, quotient <= 8);
# fp_mulmod2 uses the far edge (x = h ~ 4p², quotient ~ 4p ≈ 2^51).  The
# result is exact either way: x is an exact integer (any double >= 2^53 in
# magnitude is one), q·p is an integer, and their difference is small enough
# to represent exactly.
@inline function fp_reduce(x, F::FpCtx)
    q = fp_round(x * F.pinv)
    return fma(q, F.pn, x)
end

# r ≡ x·y (mod p) for two data operands (no precomputed quotient), inputs
# |x|,|y| <= 2p: reduce the rounded high part, add back the exact FMA error
# (an integer <= ulp(h)/2, so the final add is exact too)
@inline function fp_mulmod2(x, y, F::FpCtx)
    h = x * y
    l = fma(x, y, -h)
    return fp_reduce(h, F) + l
end

# integer-domain helpers for building twiddle tables
function fpi_pow(a::UInt64, e::Integer, p::UInt64)
    r = UInt128(1)
    b = UInt128(a % p)
    while e != 0
        isodd(e) && (r = r * b % p)
        b = b * b % p
        e >>= 1
    end
    return r % UInt64
end
@inline fpi_inv(a::UInt64, p::UInt64) = fpi_pow(a, p - 2, p)

function fp_generator(F::FpCtx)
    for g in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
        all(fpi_pow(UInt64(g), (F.pi - 1) ÷ q, F.pi) != 1 for q in F.facs) &&
            return UInt64(g)
    end
    error("unreachable: GF(p) has a small generator")
end

const FP_P1INV2 =                                  # p1^-1 mod p2 (Garner)
    fpi_inv(FP_CTX1.pi % FP_CTX2.pi, FP_CTX2.pi)

# ---------------------------------------------------------------------------
# Transform plan: N = m·2^k, m in (1, 3, 5, 15), 2^k | p - 1.  Twiddles are
# stored as (w, w/p) pairs so fp_mulmod's quotient estimate is a table load.

struct FpNttStage
    q::Int
    w1::Vector{Float64};  w1p::Vector{Float64}
    w2::Vector{Float64};  w2p::Vector{Float64}
    w3::Vector{Float64};  w3p::Vector{Float64}
end

function build_fp_stage(table::Vector{UInt64}, N2::Int, L::Int, P::Float64)
    q = L >> 2
    st = N2 ÷ L
    len = max(q, 8)
    mk(f) = [Float64(table[f(j % q)*st+1]) for j in 0:len-1]
    w1 = mk(j -> j);  w2 = mk(j -> 2j);  w3 = mk(j -> 3j)
    return FpNttStage(q, w1, w1 ./ P, w2, w2 ./ P, w3, w3 ./ P)
end

struct FpOddStage
    m::Int
    span::Int
    Q::Int
    rot::Vector{Float64};  rotp::Vector{Float64}
    tw::Vector{Vector{Float64}};  twp::Vector{Vector{Float64}}
end

function build_fp_odd(m::Int, span::Int, root::UInt64, F::FpCtx)
    Q = span ÷ m
    c = fpi_pow(root, Q, F.pi)
    rot = [Float64(fpi_pow(c, r, F.pi)) for r in 1:m-1]
    tw = Vector{Vector{Float64}}(undef, m - 1)
    for r in 1:m-1
        ωr = fpi_pow(root, r, F.pi)
        t = Vector{Float64}(undef, Q)
        f = UInt64(1)
        for j in 1:Q
            t[j] = Float64(f)
            f = UInt64(UInt128(f) * ωr % F.pi)
        end
        tw[r] = t
    end
    return FpOddStage(m, span, Q, rot, rot ./ F.p, tw, [t ./ F.p for t in tw])
end

struct FpNttPlan{C<:FpCtx}
    ctx::C
    N::Int
    N2::Int
    ninv::Float64;  ninvp::Float64
    wi::Float64;    wip::Float64    # i = ω2^(N2/4); the inverse reuses it (i^-1 == -i)
    oddf::Vector{FpOddStage}
    oddi::Vector{FpOddStage}
    fwd::Vector{FpNttStage}
    inv::Vector{FpNttStage}
end

function build_fp_plan(N::Int, F::FpCtx)
    k = trailing_zeros(N)
    m = N >> k
    @assert m in (1, 3, 5, 15) && (m == 1 || k >= 2)
    @assert (F.pi - 1) % N == 0
    ω = fpi_pow(fp_generator(F), (F.pi - 1) ÷ N, F.pi)
    @assert fpi_pow(ω, N >> 1, F.pi) == F.pi - 1
    m % 3 == 0 && @assert fpi_pow(ω, N ÷ 3, F.pi) != 1
    m % 5 == 0 && @assert fpi_pow(ω, N ÷ 5, F.pi) != 1
    ωinv = fpi_inv(ω, F.pi)

    oddf = FpOddStage[]
    oddi = FpOddStage[]
    span = N
    r, ri = ω, ωinv
    for mf in (3, 5)
        if m % mf == 0
            push!(oddf, build_fp_odd(mf, span, r, F))
            push!(oddi, build_fp_odd(mf, span, ri, F))
            r = fpi_pow(r, mf, F.pi)
            ri = fpi_pow(ri, mf, F.pi)
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
        fw = UInt64(UInt128(fw) * r % F.pi)
        fi = UInt64(UInt128(fi) * ri % F.pi)
    end
    fwd = FpNttStage[]
    L = N2
    while L >= 4
        push!(fwd, build_fp_stage(tw, N2, L, F.p))
        L >>= 2
    end
    inv = FpNttStage[]
    L = isodd(k) ? 8 : 16
    while L <= N2
        push!(inv, build_fp_stage(twinv, N2, L, F.p))
        L <<= 2
    end
    wi = N2 >= 4 ? Float64(fpi_pow(r, N2 >> 2, F.pi)) : 1.0
    ninv = Float64(fpi_inv(N % UInt64, F.pi))
    return FpNttPlan(F, N, N2, ninv, ninv / F.p, wi, wi / F.p, oddf, oddi, fwd, inv)
end

mutable struct FpNttPlanCache
    @atomic plans::Dict{Tuple{Int,UInt64},FpNttPlan}
end
const FPNTT_PLAN_LOCK = ReentrantLock()
const FPNTT_PLAN_CACHE = FpNttPlanCache(Dict{Tuple{Int,UInt64},FpNttPlan}())
function fp_ntt_plan(N::Int, F::C) where {C<:FpCtx}
    key = (N, F.pi)
    plan = get(@atomic(:acquire, FPNTT_PLAN_CACHE.plans), key, nothing)
    plan === nothing || return plan::FpNttPlan{C}
    r = lock(FPNTT_PLAN_LOCK) do
        plans = @atomic :acquire FPNTT_PLAN_CACHE.plans
        cached = get(plans, key, nothing)
        cached === nothing || return cached
        built = build_fp_plan(N, F)
        next = copy(plans)
        next[key] = built
        @atomic :release FPNTT_PLAN_CACHE.plans = next
        return built
    end
    return r::FpNttPlan{C}
end

# ---------------------------------------------------------------------------
# Odd-radix stages: Winograd radix-3 and plain radix-5.  The ω^0 output is
# fp_reduced before storing, and inverse stages fp_reduce t0 on load.

function fp_radix3_fwd!(x::Vector{Float64}, o::Int, st::FpOddStage, F::FpCtx)
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
            u = fp_mulmod(b - c, vω3, vω3p, F)
            SIMD.vstore(fp_reduce(a + (b + c), F), x, i0)
            SIMD.vstore(fp_mulmod((a - c) + u, SIMD.vload(VF8, w1, j + 1),
                                  SIMD.vload(VF8, w1p, j + 1), F), x, i0 + Q)
            SIMD.vstore(fp_mulmod((a - b) - u, SIMD.vload(VF8, w2, j + 1),
                                  SIMD.vload(VF8, w2p, j + 1), F), x, i0 + 2Q)
            j += 8
        end
    end
    @inbounds while j < Q
        i0 = o + j + 1
        a = x[i0]
        b = x[i0+Q]
        c = x[i0+2Q]
        u = fp_mulmod(b - c, ω3, ω3p, F)
        x[i0] = fp_reduce(a + (b + c), F)
        x[i0+Q] = fp_mulmod((a - c) + u, w1[j+1], w1p[j+1], F)
        x[i0+2Q] = fp_mulmod((a - b) - u, w2[j+1], w2p[j+1], F)
        j += 1
    end
    return x
end

function fp_radix3_inv!(x::Vector{Float64}, o::Int, st::FpOddStage, F::FpCtx)
    Q = st.Q
    λ3, λ3p = st.rot[1], st.rotp[1]
    w1, w1p, w2, w2p = st.tw[1], st.twp[1], st.tw[2], st.twp[2]
    j = 0
    if Q >= 8
        vλ3, vλ3p = VF8(λ3), VF8(λ3p)
        @inbounds while j + 8 <= Q
            i0 = o + j + 1
            t0 = fp_reduce(SIMD.vload(VF8, x, i0), F)
            t1 = fp_mulmod(SIMD.vload(VF8, x, i0 + Q),
                           SIMD.vload(VF8, w1, j + 1), SIMD.vload(VF8, w1p, j + 1), F)
            t2 = fp_mulmod(SIMD.vload(VF8, x, i0 + 2Q),
                           SIMD.vload(VF8, w2, j + 1), SIMD.vload(VF8, w2p, j + 1), F)
            u = fp_mulmod(t1 - t2, vλ3, vλ3p, F)
            SIMD.vstore(fp_reduce(t0 + (t1 + t2), F), x, i0)
            SIMD.vstore((t0 - t2) + u, x, i0 + Q)
            SIMD.vstore((t0 - t1) - u, x, i0 + 2Q)
            j += 8
        end
    end
    @inbounds while j < Q
        i0 = o + j + 1
        t0 = fp_reduce(x[i0], F)
        t1 = fp_mulmod(x[i0+Q], w1[j+1], w1p[j+1], F)
        t2 = fp_mulmod(x[i0+2Q], w2[j+1], w2p[j+1], F)
        u = fp_mulmod(t1 - t2, λ3, λ3p, F)
        x[i0] = fp_reduce(t0 + (t1 + t2), F)
        x[i0+Q] = (t0 - t2) + u
        x[i0+2Q] = (t0 - t1) - u
        j += 1
    end
    return x
end

# order-5 DFT combine; y0 comes back unreduced (callers reduce before storing)
@inline function fp_dft5(t0, t1, t2, t3, t4, c1, c2, c3, c4, c1p, c2p, c3p, c4p, F::FpCtx)
    s = (t1 + t2) + (t3 + t4)
    y0 = t0 + s
    y1 = t0 + ((fp_mulmod(t1, c1, c1p, F) + fp_mulmod(t2, c2, c2p, F)) +
               (fp_mulmod(t3, c3, c3p, F) + fp_mulmod(t4, c4, c4p, F)))
    y2 = t0 + ((fp_mulmod(t1, c2, c2p, F) + fp_mulmod(t2, c4, c4p, F)) +
               (fp_mulmod(t3, c1, c1p, F) + fp_mulmod(t4, c3, c3p, F)))
    y3 = t0 + ((fp_mulmod(t1, c3, c3p, F) + fp_mulmod(t2, c1, c1p, F)) +
               (fp_mulmod(t3, c4, c4p, F) + fp_mulmod(t4, c2, c2p, F)))
    y4 = t0 + ((fp_mulmod(t1, c4, c4p, F) + fp_mulmod(t2, c3, c3p, F)) +
               (fp_mulmod(t3, c2, c2p, F) + fp_mulmod(t4, c1, c1p, F)))
    return y0, y1, y2, y3, y4
end

function fp_radix5_fwd!(x::Vector{Float64}, o::Int, st::FpOddStage, F::FpCtx)
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
                                         v1p, v2p, v3p, v4p, F)
            SIMD.vstore(fp_reduce(y0, F), x, i0)
            SIMD.vstore(fp_mulmod(y1, SIMD.vload(VF8, w[1], j + 1),
                                  SIMD.vload(VF8, wp[1], j + 1), F), x, i0 + Q)
            SIMD.vstore(fp_mulmod(y2, SIMD.vload(VF8, w[2], j + 1),
                                  SIMD.vload(VF8, wp[2], j + 1), F), x, i0 + 2Q)
            SIMD.vstore(fp_mulmod(y3, SIMD.vload(VF8, w[3], j + 1),
                                  SIMD.vload(VF8, wp[3], j + 1), F), x, i0 + 3Q)
            SIMD.vstore(fp_mulmod(y4, SIMD.vload(VF8, w[4], j + 1),
                                  SIMD.vload(VF8, wp[4], j + 1), F), x, i0 + 4Q)
            j += 8
        end
    end
    @inbounds while j < Q
        i0 = o + j + 1
        y0, y1, y2, y3, y4 = fp_dft5(x[i0], x[i0+Q], x[i0+2Q], x[i0+3Q], x[i0+4Q],
                                     c1, c2, c3, c4, c1p, c2p, c3p, c4p, F)
        x[i0] = fp_reduce(y0, F)
        x[i0+Q] = fp_mulmod(y1, w[1][j+1], wp[1][j+1], F)
        x[i0+2Q] = fp_mulmod(y2, w[2][j+1], wp[2][j+1], F)
        x[i0+3Q] = fp_mulmod(y3, w[3][j+1], wp[3][j+1], F)
        x[i0+4Q] = fp_mulmod(y4, w[4][j+1], wp[4][j+1], F)
        j += 1
    end
    return x
end

function fp_radix5_inv!(x::Vector{Float64}, o::Int, st::FpOddStage, F::FpCtx)
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
            t0 = fp_reduce(SIMD.vload(VF8, x, i0), F)
            t1 = fp_mulmod(SIMD.vload(VF8, x, i0 + Q),
                           SIMD.vload(VF8, w[1], j + 1), SIMD.vload(VF8, wp[1], j + 1), F)
            t2 = fp_mulmod(SIMD.vload(VF8, x, i0 + 2Q),
                           SIMD.vload(VF8, w[2], j + 1), SIMD.vload(VF8, wp[2], j + 1), F)
            t3 = fp_mulmod(SIMD.vload(VF8, x, i0 + 3Q),
                           SIMD.vload(VF8, w[3], j + 1), SIMD.vload(VF8, wp[3], j + 1), F)
            t4 = fp_mulmod(SIMD.vload(VF8, x, i0 + 4Q),
                           SIMD.vload(VF8, w[4], j + 1), SIMD.vload(VF8, wp[4], j + 1), F)
            y0, y1, y2, y3, y4 = fp_dft5(t0, t1, t2, t3, t4, v1, v2, v3, v4,
                                         v1p, v2p, v3p, v4p, F)
            SIMD.vstore(fp_reduce(y0, F), x, i0)
            SIMD.vstore(y1, x, i0 + Q)
            SIMD.vstore(y2, x, i0 + 2Q)
            SIMD.vstore(y3, x, i0 + 3Q)
            SIMD.vstore(y4, x, i0 + 4Q)
            j += 8
        end
    end
    @inbounds while j < Q
        i0 = o + j + 1
        t0 = fp_reduce(x[i0], F)
        t1 = fp_mulmod(x[i0+Q], w[1][j+1], wp[1][j+1], F)
        t2 = fp_mulmod(x[i0+2Q], w[2][j+1], wp[2][j+1], F)
        t3 = fp_mulmod(x[i0+3Q], w[3][j+1], wp[3][j+1], F)
        t4 = fp_mulmod(x[i0+4Q], w[4][j+1], wp[4][j+1], F)
        y0, y1, y2, y3, y4 = fp_dft5(t0, t1, t2, t3, t4, c1, c2, c3, c4,
                                     c1p, c2p, c3p, c4p, F)
        x[i0] = fp_reduce(y0, F)
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
# swaps the add/sub pair around w.
# y0 is returned unreduced; callers fp_reduce before storing.

@inline function fp_dft4_fwd(a, b, c, d, wi, wip, F::FpCtx)
    apc = a + c
    amc = a - c
    bpd = b + d
    ibmd = fp_mulmod(b - d, wi, wip, F)
    return apc + bpd, amc + ibmd, apc - bpd, amc - ibmd
end

@inline function fp_dft4_inv(t0, t1, t2, t3, wi, wip, F::FpCtx)
    u = t0 + t2
    p_ = t0 - t2
    v = t1 + t3
    w = fp_mulmod(t1 - t3, wi, wip, F)
    return u + v, p_ - w, u - v, p_ + w
end

# Small-q pow2 stages (q in (1,2,4)): butterfly partners sit closer together
# than a vector, so 32 consecutive elements (four V8 loads) are shuffled
# into quarter-role vectors a,b,c,d, put through the ordinary vector
# butterfly, and shuffled back.  The twiddle tables' repeated 8-entry
# patterns line up with the gathered lane order.
#
# Lane l of role t lives at global position (l÷Q)·4Q + t·Q + l%Q within the
# 32 elements; scatter applies the inverse permutation.  The generators
# concatenate the inputs into two Vec{16}s and emit one index-tuple shuffle
# per output vector; LLVM folds the shuffle trees into the same machine
# shuffles as hand-written per-Q versions.
const IOTA16 = ntuple(i -> i - 1, 16)

@generated function ntt_gather4(::Val{Q}, v0, v1, v2, v3) where {Q}
    role(t) = ntuple(l -> ((l - 1) ÷ Q) * 4Q + t * Q + (l - 1) % Q, 8)
    return quote
        $(Expr(:meta, :inline))
        lo = SIMD.shufflevector(v0, v1, Val(IOTA16))
        hi = SIMD.shufflevector(v2, v3, Val(IOTA16))
        return (SIMD.shufflevector(lo, hi, Val($(role(0)))),
                SIMD.shufflevector(lo, hi, Val($(role(1)))),
                SIMD.shufflevector(lo, hi, Val($(role(2)))),
                SIMD.shufflevector(lo, hi, Val($(role(3)))))
    end
end

@generated function ntt_scatter4(::Val{Q}, a, b, c, d) where {Q}
    # global position g pulls role (g%4Q)÷Q, lane (g÷4Q)·Q + g%Q; roles are
    # concatenated in order, so the combined source index is role·8 + lane
    out(o) = ntuple(8) do l
        g = 8o + l - 1
        ((g % 4Q) ÷ Q) * 8 + (g ÷ 4Q) * Q + g % Q
    end
    return quote
        $(Expr(:meta, :inline))
        ab = SIMD.shufflevector(a, b, Val(IOTA16))
        cd = SIMD.shufflevector(c, d, Val(IOTA16))
        return (SIMD.shufflevector(ab, cd, Val($(out(0)))),
                SIMD.shufflevector(ab, cd, Val($(out(1)))),
                SIMD.shufflevector(ab, cd, Val($(out(2)))),
                SIMD.shufflevector(ab, cd, Val($(out(3)))))
    end
end

function fp_smallq_fwd!(x::Vector{Float64}, o::Int, N2::Int, ::Val{Q},
                        stg::FpNttStage, vwi::VF8, vwip::VF8, F::FpCtx) where {Q}
    vw1 = SIMD.vload(VF8, stg.w1, 1); vw1p = SIMD.vload(VF8, stg.w1p, 1)
    vw2 = SIMD.vload(VF8, stg.w2, 1); vw2p = SIMD.vload(VF8, stg.w2p, 1)
    vw3 = SIMD.vload(VF8, stg.w3, 1); vw3p = SIMD.vload(VF8, stg.w3p, 1)
    @inbounds for i0 in o+1:32:o+N2-31
        v0 = SIMD.vload(VF8, x, i0)
        v1 = SIMD.vload(VF8, x, i0 + 8)
        v2 = SIMD.vload(VF8, x, i0 + 16)
        v3 = SIMD.vload(VF8, x, i0 + 24)
        a, b, c, d = ntt_gather4(Val(Q), v0, v1, v2, v3)
        y0, y1, y2, y3 = fp_dft4_fwd(a, b, c, d, vwi, vwip, F)
        o0, o1, o2, o3 = ntt_scatter4(Val(Q), fp_reduce(y0, F),
                                      fp_mulmod(y1, vw1, vw1p, F),
                                      fp_mulmod(y2, vw2, vw2p, F),
                                      fp_mulmod(y3, vw3, vw3p, F))
        SIMD.vstore(o0, x, i0)
        SIMD.vstore(o1, x, i0 + 8)
        SIMD.vstore(o2, x, i0 + 16)
        SIMD.vstore(o3, x, i0 + 24)
    end
    return x
end

function fp_smallq_inv!(x::Vector{Float64}, o::Int, N2::Int, ::Val{Q},
                        stg::FpNttStage, vwi::VF8, vwip::VF8, F::FpCtx) where {Q}
    vw1 = SIMD.vload(VF8, stg.w1, 1); vw1p = SIMD.vload(VF8, stg.w1p, 1)
    vw2 = SIMD.vload(VF8, stg.w2, 1); vw2p = SIMD.vload(VF8, stg.w2p, 1)
    vw3 = SIMD.vload(VF8, stg.w3, 1); vw3p = SIMD.vload(VF8, stg.w3p, 1)
    @inbounds for i0 in o+1:32:o+N2-31
        v0 = SIMD.vload(VF8, x, i0)
        v1 = SIMD.vload(VF8, x, i0 + 8)
        v2 = SIMD.vload(VF8, x, i0 + 16)
        v3 = SIMD.vload(VF8, x, i0 + 24)
        g0, g1, g2, g3 = ntt_gather4(Val(Q), v0, v1, v2, v3)
        z0, z1, z2, z3 = fp_dft4_inv(fp_reduce(g0, F),
                                     fp_mulmod(g1, vw1, vw1p, F),
                                     fp_mulmod(g2, vw2, vw2p, F),
                                     fp_mulmod(g3, vw3, vw3p, F), vwi, vwip, F)
        o0, o1, o2, o3 = ntt_scatter4(Val(Q), fp_reduce(z0, F), z1, z2, z3)
        SIMD.vstore(o0, x, i0)
        SIMD.vstore(o1, x, i0 + 8)
        SIMD.vstore(o2, x, i0 + 16)
        SIMD.vstore(o3, x, i0 + 24)
    end
    return x
end

# ---------------------------------------------------------------------------
# Power-of-two pipelines: radix-4 DIF forward / DIT inverse.

function fp_fwd_pow2!(x::Vector{Float64}, o::Int, plan::FpNttPlan)
    F = plan.ctx
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
                    y0, y1, y2, y3 = fp_dft4_fwd(a, b, c, d, vwi, vwip, F)
                    SIMD.vstore(fp_reduce(y0, F), x, i0)
                    SIMD.vstore(fp_mulmod(y1, SIMD.vload(VF8, w1, j + 1),
                                          SIMD.vload(VF8, w1p, j + 1), F), x, i0 + q)
                    SIMD.vstore(fp_mulmod(y2, SIMD.vload(VF8, w2, j + 1),
                                          SIMD.vload(VF8, w2p, j + 1), F), x, i0 + 2q)
                    SIMD.vstore(fp_mulmod(y3, SIMD.vload(VF8, w3, j + 1),
                                          SIMD.vload(VF8, w3p, j + 1), F), x, i0 + 3q)
                end
            end
        elseif N2 >= 32
            q == 4 ? fp_smallq_fwd!(x, o, N2, Val(4), stg, vwi, vwip, F) :
            q == 2 ? fp_smallq_fwd!(x, o, N2, Val(2), stg, vwi, vwip, F) :
                     fp_smallq_fwd!(x, o, N2, Val(1), stg, vwi, vwip, F)
        else
            w1, w1p, w2, w2p, w3, w3p = stg.w1, stg.w1p, stg.w2, stg.w2p, stg.w3, stg.w3p
            for s in o:L:o+N2-1
                @inbounds for j in 0:q-1
                    a = x[s+j+1]
                    b = x[s+j+q+1]
                    c = x[s+j+2q+1]
                    d = x[s+j+3q+1]
                    y0, y1, y2, y3 = fp_dft4_fwd(a, b, c, d, wi, wip, F)
                    x[s+j+1] = fp_reduce(y0, F)
                    x[s+j+q+1] = fp_mulmod(y1, w1[j+1], w1p[j+1], F)
                    x[s+j+2q+1] = fp_mulmod(y2, w2[j+1], w2p[j+1], F)
                    x[s+j+3q+1] = fp_mulmod(y3, w3[j+1], w3p[j+1], F)
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
    F = plan.ctx
    N2 = plan.N2
    wi, wip = plan.wi, plan.wip
    vwi, vwip = VF8(wi), VF8(wip)
    ninv, ninvp = plan.ninv, plan.ninvp
    if isodd(trailing_zeros(N2))
        @inbounds for s in o:2:o+N2-1
            u = x[s+1]
            t = x[s+2]
            x[s+1] = fp_mulmod(u + t, ninv, ninvp, F)
            x[s+2] = fp_mulmod(u - t, ninv, ninvp, F)
        end
    elseif N2 >= 4
        @inbounds for s in o:4:o+N2-1
            t0 = fp_mulmod(x[s+1], ninv, ninvp, F)
            t1 = fp_mulmod(x[s+2], ninv, ninvp, F)
            t2 = fp_mulmod(x[s+3], ninv, ninvp, F)
            t3 = fp_mulmod(x[s+4], ninv, ninvp, F)
            y0, y1, y2, y3 = fp_dft4_inv(t0, t1, t2, t3, wi, wip, F)
            x[s+1] = y0
            x[s+2] = y1
            x[s+3] = y2
            x[s+4] = y3
        end
    else
        @inbounds for s in o+1:o+N2
            x[s] = fp_mulmod(x[s], ninv, ninvp, F)
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
                    t0 = fp_reduce(SIMD.vload(VF8, x, i0), F)
                    t1 = fp_mulmod(SIMD.vload(VF8, x, i0 + q),
                                   SIMD.vload(VF8, w1, j + 1), SIMD.vload(VF8, w1p, j + 1), F)
                    t2 = fp_mulmod(SIMD.vload(VF8, x, i0 + 2q),
                                   SIMD.vload(VF8, w2, j + 1), SIMD.vload(VF8, w2p, j + 1), F)
                    t3 = fp_mulmod(SIMD.vload(VF8, x, i0 + 3q),
                                   SIMD.vload(VF8, w3, j + 1), SIMD.vload(VF8, w3p, j + 1), F)
                    y0, y1, y2, y3 = fp_dft4_inv(t0, t1, t2, t3, vwi, vwip, F)
                    SIMD.vstore(fp_reduce(y0, F), x, i0)
                    SIMD.vstore(y1, x, i0 + q)
                    SIMD.vstore(y2, x, i0 + 2q)
                    SIMD.vstore(y3, x, i0 + 3q)
                end
            end
        elseif N2 >= 32
            q == 4 ? fp_smallq_inv!(x, o, N2, Val(4), stg, vwi, vwip, F) :
            q == 2 ? fp_smallq_inv!(x, o, N2, Val(2), stg, vwi, vwip, F) :
                     fp_smallq_inv!(x, o, N2, Val(1), stg, vwi, vwip, F)
        else
            w1, w1p, w2, w2p, w3, w3p = stg.w1, stg.w1p, stg.w2, stg.w2p, stg.w3, stg.w3p
            for s in o:L:o+N2-1
                @inbounds for j in 0:q-1
                    t0 = fp_reduce(x[s+j+1], F)
                    t1 = fp_mulmod(x[s+j+q+1], w1[j+1], w1p[j+1], F)
                    t2 = fp_mulmod(x[s+j+2q+1], w2[j+1], w2p[j+1], F)
                    t3 = fp_mulmod(x[s+j+3q+1], w3[j+1], w3p[j+1], F)
                    y0, y1, y2, y3 = fp_dft4_inv(t0, t1, t2, t3, wi, wip, F)
                    x[s+j+1] = fp_reduce(y0, F)
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
    F = plan.ctx
    for st in plan.oddf
        for o in 0:st.span:plan.N-1
            st.m == 3 ? fp_radix3_fwd!(x, o, st, F) : fp_radix5_fwd!(x, o, st, F)
        end
    end
    for o in 0:plan.N2:plan.N-1
        fp_fwd_pow2!(x, o, plan)
    end
    return x
end

function fp_ntt_inv!(x::Vector{Float64}, plan::FpNttPlan)
    F = plan.ctx
    for o in 0:plan.N2:plan.N-1
        fp_inv_pow2!(x, o, plan)
    end
    for st in plan.oddi
        for o in 0:st.span:plan.N-1
            st.m == 3 ? fp_radix3_inv!(x, o, st, F) : fp_radix5_inv!(x, o, st, F)
        end
    end
    return x
end

# ---------------------------------------------------------------------------
# Coefficient-domain layer, shared by both pipelines: size the transform
# (ntt_len), split limbs into chunk coefficients (fp_ntt_pack), and multiply
# lanewise between the transforms (fp_ntt_pointwise!).

# smallest supported transform length >= T: m·2^k, m in (1, 3, 5, 15), k >= 2
function ntt_len(T::Int)
    best = nextpow(2, max(T, 4))
    for m in (3, 5, 15)
        c = m * nextpow(2, max(cld(T, m), 4))
        c < best && (best = c)
    end
    return best
end

# b-bit chunk extraction into Float64 points; chunks < 2^b <= 2^52 are exact
# doubles and, being smaller than every prime, already canonical residues
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

function fp_ntt_pointwise!(xa::Vector{Float64}, xb::Vector{Float64}, F::FpCtx)
    n = length(xa)
    i = 1
    if n >= 8
        @inbounds while i + 7 <= n
            SIMD.vstore(fp_mulmod2(SIMD.vload(VF8, xa, i), SIMD.vload(VF8, xb, i), F),
                        xa, i)
            i += 8
        end
    end
    @inbounds while i <= n
        xa[i] = fp_mulmod2(xa[i], xb[i], F)
        i += 1
    end
    return xa
end

# ---------------------------------------------------------------------------
# Two-prime CRT pipeline — what dispatch uses.  The working modulus
# p1·p2 ≈ 2^98.97 holds the chunk width at b ≈ 41-44 where a single prime's
# density decays with size (b ≈ (49 - log2 nc)/2).

# chunk width and transform length against the CRT modulus p1·p2.  The bound
# min(nca,ncb)·(2^b-1)^2 < p1·p2 is checked in division form: the product
# overflows UInt128 at large b with large operands, and the descending search
# visits large b first regardless of operand size.  Chunks < 2^48 < p2 < p1
# are already canonical residues mod both primes, so one pack serves both.
function fp_ntt_params2(bits_a::Int, bits_b::Int)
    for b in 48:-1:1
        nca = cld(bits_a, b)
        ncb = cld(bits_b, b)
        if UInt128(min(nca, ncb)) <=
           (UInt128(FP_CTX1.pi) * FP_CTX2.pi - 1) ÷ (UInt128(2)^b - 1)^2
            return b, ntt_len(nca + ncb - 1)
        end
    end
    error("unreachable: b == 1 always satisfies the bound for supported sizes")
end

# Garner recombination + limb streaming.  Each coefficient of the product's
# base-2^b representation is recovered from its residues as
#   c = c1 + p1·u,  u = (c2 - c1)·p1^-1 mod p2,  c ∈ [0, p1·p2) ⊂ [0, 2^99),
# but c is never materialized: by linearity the product is
#   Σ cᵢ·2^(b·i) = Σ c1ᵢ·2^(b·i) + p1 · Σ uᵢ·2^(b·i) = S1 + p1·S2,
# so the loop streams S1 (into r) and S2 (into scratch) as two independent
# sums of 49-bit values, folded at the end by one addmul_1! pass at kernel
# speed.
#
# Per 8 coefficients, all SIMD: residues are canonicalized and u is computed
# in the fp domain (·p1^-1 is a mulmod by a fixed twiddle: c1 must be
# canonicalized FIRST — v1 - p1 is a different value mod p2 — then
# t = v2 - fp_reduce(v1, F2) has |t| < 1.5·p2 <= 4p, so fp_mulmod is exact),
# values are converted to integers, and each is pre-split into the window
# halves lo = v << s, hi = v >> (64 - s) with per-lane variable shifts —
# legal because the flush discipline below makes s = (b·i) mod 64, which is
# data-independent.  The scalar scan is then pure adds-with-carry (a
# variable UInt128 << there would cost a multi-uop shift sequence per
# coefficient, since LLVM cannot prove s < 64).
#
# Accumulator proof (each stream).  (a1, a0) is a 128-bit window holding the
# pending sum of contributions to bits [outbit, outbit+128); coefficient i
# contributes v·2^(i·b - outbit) with v < 2^49 and shift s < 64 (the flush
# keeps s in [0, 64) since b < 64), i.e. each add is < 2^113.  Between two
# flushes s advances by b >= 1, so there are at most 64 adds; with
# P' = P >> 64 at each flush the pending value satisfies
# P < (P >> 64) + 64·2^113 < 2^120 < 2^128 at the fixed point — the window
# never overflows.  A flushed limb is final: after the flush every remaining
# coefficient starts at bit i·b >= outbit + 64, and all contributions are
# nonnegative, so nothing can carry below its own position.  The final fold
# S1 + p1·S2 equals the true product < 2^(64·rn), so addmul_1! carries out 0.
function fp_ntt_unpack2!(r::Memory{Limb}, ro::Int, rn::Int,
                         x1::Vector{Float64}, x2::Vector{Float64},
                         nconv::Int, b::Int)
    s2 = Memory{Limb}(undef, rn)
    # SIMD→scalar staging: the scan reads lanes at a runtime index, and Vec
    # lane extraction with a non-constant index compiles to a stack spill
    # per access (measured ~30% slower than this explicit round-trip).
    # One buffer, quarters: lo1 | hi1 | lo2 | hi2.
    stage = Vector{UInt64}(undef, 32)
    g = Float64(FP_P1INV2)          # the Garner inverse as an fp_mulmod
    gp = g / FP_CTX2.p              # twiddle; const-folds to literals
    vg, vgp = VF8(g), VF8(gp)
    # canonical values are < p < 2^49, so v + 1.5·2^52 pins the exponent and
    # leaves v in the low 51 mantissa bits: int(v) is a reinterpret-and-mask
    vmask = SIMD.Vec{8,UInt64}(UInt64(2)^51 - 1)
    ub = UInt64(b)
    vlane = SIMD.Vec{8,UInt64}(ntuple(k -> UInt64(k - 1) * ub, 8))
    a01 = UInt64(0); a11 = UInt64(0)   # stream 1 window, low/high limb
    a02 = UInt64(0); a12 = UInt64(0)   # stream 2 window
    outw = 1
    s = 0             # bit offset of coefficient i within the window
    i = 0
    @inbounds while i + 8 <= nconv
        v1 = fp_reduce(SIMD.vload(VF8, x1, i + 1), FP_CTX1)
        v1 = SIMD.vifelse(v1 < 0.0, v1 + FP_CTX1.p, v1)
        v2 = fp_reduce(SIMD.vload(VF8, x2, i + 1), FP_CTX2)
        u = fp_mulmod(v2 - fp_reduce(v1, FP_CTX2), vg, vgp, FP_CTX2)
        u = SIMD.vifelse(u < 0.0, u + FP_CTX2.p, u)
        c1v = reinterpret(SIMD.Vec{8,UInt64}, v1 + FP_MAGIC) & vmask
        uv = reinterpret(SIMD.Vec{8,UInt64}, u + FP_MAGIC) & vmask
        sv = (UInt64(i) * ub + vlane) & UInt64(63)
        svc = UInt64(63) - sv
        SIMD.vstore(c1v << sv, stage, 1)
        SIMD.vstore((c1v >> 1) >> svc, stage, 9)
        SIMD.vstore(uv << sv, stage, 17)
        SIMD.vstore((uv >> 1) >> svc, stage, 25)
        for j in 1:8
            if s >= 64
                r[ro+outw] = a01
                s2[outw] = a02
                a01 = a11; a11 = UInt64(0)
                a02 = a12; a12 = UInt64(0)
                outw += 1
                s -= 64
            end
            t1 = a01 + stage[j]
            a11 += stage[j+8] + (t1 < a01)
            a01 = t1
            t2 = a02 + stage[j+16]
            a12 += stage[j+24] + (t2 < a02)
            a02 = t2
            s += b
        end
        i += 8
    end
    @inbounds while i < nconv
        v1 = fp_reduce(x1[i+1], FP_CTX1)
        v1 = ifelse(v1 < 0.0, v1 + FP_CTX1.p, v1)
        v2 = fp_reduce(x2[i+1], FP_CTX2)
        u = fp_mulmod(v2 - fp_reduce(v1, FP_CTX2), g, gp, FP_CTX2)
        c1 = unsafe_trunc(UInt64, v1)
        uu = unsafe_trunc(UInt64, ifelse(u < 0.0, u + FP_CTX2.p, u))
        if s >= 64
            r[ro+outw] = a01
            s2[outw] = a02
            a01 = a11; a11 = UInt64(0)
            a02 = a12; a12 = UInt64(0)
            outw += 1
            s -= 64
        end
        t1 = a01 + (c1 << s)
        a11 += ((c1 >> 1) >> (63 - s)) + (t1 < a01)
        a01 = t1
        t2 = a02 + (uu << s)
        a12 += ((uu >> 1) >> (63 - s)) + (t2 < a02)
        a02 = t2
        s += b
        i += 1
    end
    @inbounds while outw <= rn
        r[ro+outw] = a01
        s2[outw] = a02
        a01 = a11; a11 = UInt64(0)
        a02 = a12; a12 = UInt64(0)
        outw += 1
    end
    addmul_1!(r, ro, s2, 0, rn, FP_CTX1.pi)
    return r
end

# mpn-layer entry points: r[1..m+n] = a[1..m]·b[1..n], r must not alias the
# inputs.  Chunks < 2^48 are already canonical residues mod both primes, so
# each operand is packed once and copied for the second prime.
function mul_fpntt2!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, m::Int,
                     b::Memory{Limb}, bo::Int, n::Int)
    bits_a = magnitude_bits(a, ao, m)
    bits_b = magnitude_bits(b, bo, n)
    bch, N = fp_ntt_params2(bits_a, bits_b)
    plan1 = fp_ntt_plan(N, FP_CTX1)
    plan2 = fp_ntt_plan(N, FP_CTX2)
    nca, ncb = cld(bits_a, bch), cld(bits_b, bch)
    xa1 = fp_ntt_pack(a, ao, m, bch, nca, N)
    xa2 = copy(xa1)
    xb1 = fp_ntt_pack(b, bo, n, bch, ncb, N)
    xb2 = copy(xb1)
    fp_ntt_fwd!(xa1, plan1)
    fp_ntt_fwd!(xb1, plan1)
    fp_ntt_pointwise!(xa1, xb1, FP_CTX1)
    fp_ntt_inv!(xa1, plan1)
    fp_ntt_fwd!(xa2, plan2)
    fp_ntt_fwd!(xb2, plan2)
    fp_ntt_pointwise!(xa2, xb2, FP_CTX2)
    fp_ntt_inv!(xa2, plan2)
    fp_ntt_unpack2!(r, ro, m + n, xa1, xa2, nca + ncb - 1, bch)
    return nothing
end

function sqr_fpntt2!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int)
    bits = magnitude_bits(a, ao, n)
    bch, N = fp_ntt_params2(bits, bits)
    plan1 = fp_ntt_plan(N, FP_CTX1)
    plan2 = fp_ntt_plan(N, FP_CTX2)
    nca = cld(bits, bch)
    xa1 = fp_ntt_pack(a, ao, n, bch, nca, N)
    xa2 = copy(xa1)
    fp_ntt_fwd!(xa1, plan1)
    fp_ntt_pointwise!(xa1, xa1, FP_CTX1)
    fp_ntt_inv!(xa1, plan1)
    fp_ntt_fwd!(xa2, plan2)
    fp_ntt_pointwise!(xa2, xa2, FP_CTX2)
    fp_ntt_inv!(xa2, plan2)
    fp_ntt_unpack2!(r, ro, 2n, xa1, xa2, 2nca - 1, bch)
    return nothing
end

# ---------------------------------------------------------------------------
# Single-prime pipeline over p1 alone (b <= 24).  UNWIRED from dispatch —
# the two-prime engine measures faster at every size above the Karatsuba
# band — but kept as an independent pipeline through the shared transform
# machinery for the differential tests.  Same entry-point contracts as the
# two-prime pipeline above.

# chunk width and transform length: exactness needs every convolution
# coefficient (a sum of min(nca, ncb) chunk products) below p, so b <= 24
function fp_ntt_params(bits_a::Int, bits_b::Int)
    for b in 24:-1:1
        nca = cld(bits_a, b)
        ncb = cld(bits_b, b)
        if UInt128(min(nca, ncb)) * (UInt128(2)^b - 1)^2 < FP_CTX1.pi
            return b, ntt_len(nca + ncb - 1)
        end
    end
    error("unreachable: b == 1 always satisfies the bound for supported sizes")
end

# canonicalize each coefficient to [0, p) and stream limbs out.
# Coefficients < 2^49 at b <= 24 spacing never hold more than 64 + 49 live
# bits, so a single-UInt128 accumulator with one flushed limb per 64 output
# bits never overflows, and (contributions being nonnegative, starting at
# bit i·b >= outbit + 64 after each flush) every flushed limb is final.
function fp_ntt_unpack!(r::Memory{Limb}, ro::Int, rn::Int, x::Vector{Float64},
                        nconv::Int, b::Int, F::FpCtx)
    acc = UInt128(0)
    outw = 1
    outbit = 0
    @inbounds for i in 0:nconv-1
        v = fp_reduce(x[i+1], F)
        v < 0.0 && (v += F.p)
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

function mul_fpntt!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, m::Int,
                    b::Memory{Limb}, bo::Int, n::Int)
    bits_a = magnitude_bits(a, ao, m)
    bits_b = magnitude_bits(b, bo, n)
    bch, N = fp_ntt_params(bits_a, bits_b)
    plan = fp_ntt_plan(N, FP_CTX1)
    nca, ncb = cld(bits_a, bch), cld(bits_b, bch)
    xa = fp_ntt_pack(a, ao, m, bch, nca, N)
    xb = fp_ntt_pack(b, bo, n, bch, ncb, N)
    fp_ntt_fwd!(xa, plan)
    fp_ntt_fwd!(xb, plan)
    fp_ntt_pointwise!(xa, xb, FP_CTX1)
    fp_ntt_inv!(xa, plan)
    fp_ntt_unpack!(r, ro, m + n, xa, nca + ncb - 1, bch, FP_CTX1)
    return nothing
end

function sqr_fpntt!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int)
    bits = magnitude_bits(a, ao, n)
    bch, N = fp_ntt_params(bits, bits)
    plan = fp_ntt_plan(N, FP_CTX1)
    nca = cld(bits, bch)
    xa = fp_ntt_pack(a, ao, n, bch, nca, N)
    fp_ntt_fwd!(xa, plan)
    fp_ntt_pointwise!(xa, xa, FP_CTX1)
    fp_ntt_inv!(xa, plan)
    fp_ntt_unpack!(r, ro, 2n, xa, 2nca - 1, bch, FP_CTX1)
    return nothing
end
