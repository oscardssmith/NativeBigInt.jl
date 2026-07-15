# Floating-point NTT multiplication following FLINT's fft_small design:
# coefficients live in Float64, products use the FMA error-free transform,
# and quotients use the magic-constant round.  The operands are split into
# b-bit chunks and the convolution is computed by CRT over two ~2^49 primes
# (working modulus p1·p2 ≈ 2^98.97, so b ≈ 41-44).
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
# The reverse transform (fp_ntt_rev!) is the transpose of the forward network
# — same twiddle tables, stages in reverse order — which computes the same DFT
# because the matrix is symmetric.  Fed the forward-scrambled pointwise product
# it returns N·c[(N-t) mod N] in natural order; the unpack reads descending and
# folds in the 1/N.
#
# Lazy bounds.  One fp_reduce per butterfly closes the per-stage ranges (the
# ω^0 lane, which skips the twiddle mulmod, is reduced explicitly — see the
# butterfly comments for the per-radix numbers).  The global invariant is that
# forward values converge to <= ~0.9p and reverse to <= ~3.5p, so every mulmod
# argument stays <= 4p and every intermediate sum stays far below 2^53.  The
# differential and adversarial max-magnitude tests validate this empirically.
#
# All bounds hold for any prime p < 2^49, so the engine is parametrized by an
# FpCtx carrying the per-prime constants and both residue transforms run
# through the same code.
#
# Hardware FMA and round-to-nearest are assumed.  No @fastmath anywhere near
# this file: contraction would break the fma(x, w, -h) cancellation.

const VF8 = SIMD.Vec{8,Float64}

# smallest small generator of GF(p)*, given the prime factors of p-1
function fp_generator(p::UInt64, facs)
    for g in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
        all(powermod(UInt64(g), (p - 1) ÷ q, p) != 1 for (q, _) in facs) &&
            return UInt64(g)
    end
    error("unreachable: GF(p) has a small generator")
end

# prime to do a NTT over, the factorization of P-1 as (prime => exponent) pairs,
# and a primitive root of GF(P).  The constructor checks facs is the complete
# factorization of P-1 and supplies the 2/3/5/17 radices, so every valid N | P-1
# has a primitive Nth root — build_fp_plan trusts gen without rechecking order.
struct FpCtx{P}
    facs::Tuple{Vararg{Pair{Int,Int}}}
    gen::UInt64
    function FpCtx{P}(facs::Tuple{Vararg{Pair{Int,Int}}}) where {P}
        @assert prod(q^e for (q, e) in facs) == P - 1
        @assert all(in(map(first, facs)), (2, 3, 5, 17))
        new{P}(sort(facs), fp_generator(UInt64(P), facs))
    end
end
@inline fp_prime(::FpCtx{P}) where {P} = UInt64(P)
# 2-adicity of p-1: the exponent of the (2 => k) pair.
@inline two_adicity(F::FpCtx) = F.facs[1].second

# Both primes < 2^49 (rounding bounds above)
# Of form 2^n*(2^(49-n)-1) since that allows radix 2^n and radix, 3, 5, 17 transforms
const FP_CTX1 = FpCtx{2^33 * (2^16-1) + 1}((2 => 33, 3 => 1, 5 => 1, 17 => 1, 257 => 1))
const FP_CTX2 = FpCtx{2^41 * (2^8 -1) + 1}((2 => 41, 3 => 1, 5 => 1, 17 => 1))

# magic rounding constant. FP_MAGIC + x - FP_MAGIC rounds x to Int.
const FP_MAGIC = 1.5*2^52 
# exact round-to-nearest-integer for |v| <= 2^51 (scalar and VF8)
@inline fp_round(v) = (v + FP_MAGIC) - FP_MAGIC

# r ≡ w·x (mod p), w in [0, p), |x| <= 4p; |r| <= p(1/2 + |x|·2^-52) < p.
# q = fp_round(x * wpinv) not dependent on h, so both multiplies issue together.
@inline function fp_mulmod(x, w, wpinv, ::FpCtx{P}) where {P}
    h = x * w
    l = fma(x, w, -h)
    q = fp_round(x * wpinv)
    return fma(q, -P, h) + l
end

# r ≡ x (mod p) in balanced form. Exact for any exact-integer x in the domain |x| <= 2^51·p.
# The round argument |x·inv(P)| <= 2^51, so q = fp_round(..) is valid
# and r = x - q·p is a small integer (|r| <= p(1/2 + |x|·2^-52/p) < p)
# that the single fma computes without rounding.
@inline function fp_reduce(x, ::FpCtx{P}) where {P}
    q = fp_round(x * inv(P))
    return fma(q, -P, x)
end

# r ≡ x·y (mod p) for two data operands (no precomputed quotient), inputs
# |x|,|y| <= 2p: reduce the rounded high part, add back the exact FMA error
# (an integer <= ulp(h)/2, so the final add is exact too)
@inline function fp_mulmod2(x, y, F::FpCtx)
    h = x * y
    l = fma(x, y, -h)
    return fp_reduce(h, F) + l
end

# ---------------------------------------------------------------------------
# Transform plan: N = m·2^k, m divides 255, 2^k | p - 1.
# Twiddles are stored as (w, w/p) pairs so fp_mulmod's quotient estimate is a table load.
struct FpNttStage
    q::Int
    w1::Vector{Float64};  w1p::Vector{Float64}
    w2::Vector{Float64};  w2p::Vector{Float64}
    w3::Vector{Float64};  w3p::Vector{Float64}
end

function build_fp_stage(table::Vector{Float64}, N2::Int, L::Int, P::UInt)
    q = L >> 2
    st = N2 ÷ L
    len = max(q, 8)
    w1 = Vector{Float64}(undef, len); w1p = Vector{Float64}(undef, len)
    w2 = Vector{Float64}(undef, len); w2p = Vector{Float64}(undef, len)
    w3 = Vector{Float64}(undef, len); w3p = Vector{Float64}(undef, len)
    @inbounds for j in 0:len-1
        jm = j < q ? j : j % q          # len > q only for the tiny q < 8 stages
        v1 = table[jm*st+1]
        v2 = table[2jm*st+1]
        v3 = table[3jm*st+1]
        w1[j+1] = v1; w1p[j+1] = v1 / P
        w2[j+1] = v2; w2p[j+1] = v2 / P
        w3[j+1] = v3; w3p[j+1] = v3 / P
    end
    return FpNttStage(q, w1, w1p, w2, w2p, w3, w3p)
end

# Canonical Float64 powers r^0, r^1, ..., r^(n-1) mod p, generated 8 lanes at a time
function fp_twiddles(r::UInt64, n::Int, F::FpCtx{P}) where P
    np = cld(n, 8) << 3
    t = Vector{Float64}(undef, np)
    r8 = powermod(r, 8, P)
    r8p = r8 / P
    v = VF8(ntuple(k -> powermod(r, k - 1, P), 8))
    @inbounds for j in 1:8:np
        SIMD.vstore(SIMD.vifelse(v < 0.0, v + P, v), t, j) # canonicalize to [0, p)
        v = fp_mulmod(v, r8, r8p, F)
    end
    resize!(t, n)
    return t
end

struct FpOddStage
    m::Int
    span::Int
    Q::Int
    rot::Vector{Float64};  rotp::Vector{Float64}
    tw::Vector{Vector{Float64}};  twp::Vector{Vector{Float64}}
end

# Winograd 5-point constants from a primitive 5th root c.
# With a = c + c^4, b = c^2 + c^3 (a + b = -1, the roots sum to zero):
# k1 = -1/4,  k2 = (a - b)/4,  k3 = (c - c^4)/2,  k4 = (c^2 - c^3)/2,
# all canonical residues mod p (see fp_dft5 for the combine they feed).
function fp_dft5_consts(c::UInt64, F::FpCtx)
    p = fp_prime(F)
    mulm(x, y) = UInt64(UInt128(x) * y % p)
    subm(x, y) = x >= y ? x - y : x + p - y
    c2 = mulm(c, c); c3 = mulm(c2, c); c4 = mulm(c3, c)
    inv2 = invmod(UInt64(2), p)
    inv4 = mulm(inv2, inv2)
    a = (c + c4) % p
    b = (c2 + c3) % p
    return Float64[p - inv4, mulm(subm(a, b), inv4),
                   mulm(subm(c, c4), inv2), mulm(subm(c2, c3), inv2)]
end

# Symmetric-pair constants for the 17-point DFT (see fp_dft17_uv).
# Pairing x_j with x_{17-j} and y_r with y_{17-r} (c^{j(17-r)} = c^{-jr})
# turns the 16x16 DFT matrix into 8x8 quadrants of half-sums/half-differences
# A_{jr} = (c^{jr} + c^{-jr})/2,  B_{jr} = (c^{jr} - c^{-jr})/2,
# stored row-major by r: A_{jr} at (r-1)·16 + j, B_{jr} at (r-1)·16 + 8 + j,
# so each output pair reads 16 contiguous constants.
function fp_dft17_consts(c::UInt64, F::FpCtx)
    p = fp_prime(F)
    mulm(x, y) = UInt64(UInt128(x) * y % p)
    subm(x, y) = x >= y ? x - y : x + p - y
    inv2 = invmod(UInt64(2), p)
    rot = Vector{Float64}(undef, 128)
    for r in 1:8, j in 1:8
        cjr = powermod(c, j * r % 17, p)
        cmjr = powermod(c, 17 - j * r % 17, p)
        rot[(r-1)*16+j] = mulm((cjr + cmjr) % p, inv2)
        rot[(r-1)*16+8+j] = mulm(subm(cjr, cmjr), inv2)
    end
    return rot
end

function build_fp_odd(m::Int, span::Int, root::UInt64, F::FpCtx)
    Q = span ÷ m
    c = powermod(root, Q, fp_prime(F))
    rot = m == 5 ? fp_dft5_consts(c, F) :
          m == 17 ? fp_dft17_consts(c, F) :
          Float64[powermod(c, r, fp_prime(F)) for r in 1:m-1]
    tw = Vector{Vector{Float64}}(undef, m - 1)
    for r in 1:m-1
        ωr = powermod(root, r, fp_prime(F))
        tw[r] = fp_twiddles(ωr, Q, F)
    end
    return FpOddStage(m, span, Q, rot, rot ./ fp_prime(F), tw, [t ./ fp_prime(F) for t in tw])
end

struct FpNttPlan{C<:FpCtx}
    ctx::C
    N::Int
    N2::Int
    ninv::Float64
    ninvp::Float64  # 1/N; applied in the unpack
    wi::Float64
    wip::Float64    # i = ω2^(N2/4), the DFT4 combine's twiddle
    oddf::Vector{FpOddStage}
    fwd::Vector{FpNttStage}
end

function build_fp_plan(N::Int, F::FpCtx)
    k = trailing_zeros(N)
    m = N >> k
    # N = m·2^k with odd part m | 255 (radix 3/5/17 stages) and 2^k | P-1; with a
    # validated FpCtx this is the full precondition, and ω is a primitive Nth root.
    @assert m in (1, 3, 5, 15, 17, 51, 85, 255) && (m == 1 || k >= 2) && k <= two_adicity(F)
    ω = powermod(F.gen, (fp_prime(F) - 1) ÷ N, fp_prime(F))

    oddf = FpOddStage[]
    span = N
    r = ω
    for mf in (3, 5, 17)
        if m % mf == 0
            push!(oddf, build_fp_odd(mf, span, r, F))
            r = powermod(r, mf, fp_prime(F))
            span ÷= mf
        end
    end

    N2 = span
    tw = fp_twiddles(r, N2, F)
    fwd = FpNttStage[]
    L = N2
    while L >= 4
        push!(fwd, build_fp_stage(tw, N2, L, fp_prime(F)))
        L >>= 2
    end
    wi = N2 >= 4 ? powermod(r, N2 >> 2, fp_prime(F)) : 1
    ninv = invmod(N % UInt64, fp_prime(F))
    return FpNttPlan(F, N, N2, Float64(ninv), ninv / fp_prime(F), Float64(wi), wi / fp_prime(F), oddf, fwd)
end

mutable struct FpNttPlanCache
    @atomic plans::Dict{Tuple{Int,UInt64},FpNttPlan}
end
const FPNTT_PLAN_LOCK = ReentrantLock()
const FPNTT_PLAN_CACHE = FpNttPlanCache(Dict{Tuple{Int,UInt64},FpNttPlan}())
function fp_ntt_plan(N::Int, F::C) where {C<:FpCtx}
    key = (N, fp_prime(F))
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
# Odd-radix stages: Winograd radix-3 and radix-5, symmetric-pair radix-17.
# A stage is twiddle+reduce with the combine before it (forward) or after it (reverse).
# Radix-3 and radix-5 share that twiddle+reduce block.
# radix-17 instead interleaves its twiddles into the per-pair r-loop because
# a shared block would keep all 17 lanes live at once.

# Raw Winograd 3-point combine (u = ω3·(b - c) folds the matrix into one
# mulmod). Every output, including ω^0, comes back unreduced: the forward
# reduces it in the shared block; the reverse stores it raw (<= ~2.4p),
# legal because radix-3 runs last in the reverse, so only the unpack's
# 4p-domain 1/N mulmod loads it.
@inline function fp_dft3(a, b, c, ω3, ω3p, F::FpCtx)
    u = fp_mulmod(b - c, ω3, ω3p, F)
    return a + (b + c), (a - c) + u, (a - b) - u
end

function fp_radix3!(x::Vector{Float64}, o::Int, st::FpOddStage, F::FpCtx,
                    ::Val{FWD}) where {FWD}
    Q = st.Q
    ω3, ω3p = st.rot[1], st.rotp[1]
    w1, w1p, w2, w2p = st.tw[1], st.twp[1], st.tw[2], st.twp[2]
    j = 0
    @inbounds while j + 8 <= Q
        i0 = o + j + 1
        y0 = SIMD.vload(VF8, x, i0)
        y1 = SIMD.vload(VF8, x, i0 + Q)
        y2 = SIMD.vload(VF8, x, i0 + 2Q)
        if FWD
            y0, y1, y2 = fp_dft3(y0, y1, y2, ω3, ω3p, F)
        end
        y0 = fp_reduce(y0, F)
        y1 = fp_mulmod(y1, SIMD.vload(VF8, w1, j + 1),
                       SIMD.vload(VF8, w1p, j + 1), F)
        y2 = fp_mulmod(y2, SIMD.vload(VF8, w2, j + 1),
                       SIMD.vload(VF8, w2p, j + 1), F)
        if !FWD
            y0, y1, y2 = fp_dft3(y0, y1, y2, ω3, ω3p, F)
        end
        SIMD.vstore(y0, x, i0)
        SIMD.vstore(y1, x, i0 + Q)
        SIMD.vstore(y2, x, i0 + 2Q)
        j += 8
    end
    @inbounds while j < Q
        i0 = o + j + 1
        y0 = x[i0]
        y1 = x[i0+Q]
        y2 = x[i0+2Q]
        if FWD
            y0, y1, y2 = fp_dft3(y0, y1, y2, ω3, ω3p, F)
        end
        y0 = fp_reduce(y0, F)
        y1 = fp_mulmod(y1, w1[j+1], w1p[j+1], F)
        y2 = fp_mulmod(y2, w2[j+1], w2p[j+1], F)
        if !FWD
            y0, y1, y2 = fp_dft3(y0, y1, y2, ω3, ω3p, F)
        end
        x[i0] = y0
        x[i0+Q] = y1
        x[i0+2Q] = y2
        j += 1
    end
    return x
end

# Winograd order-5 DFT combine, 6 mulmods by ±-pairing the outputs:
# y1+y4 = 2·x0 + a·(x1+x4) + b·(x2+x3) for
# a = ω + ω^4, b = ω^2 + ω^3, and a + b = -1 rewrites that as
# 2·x0 - S/2 + ((a-b)/2)·D — one mulmod, not two. 
# With the halving folded into k1..k4 (fp_dft5_consts), 
# r, q, g1, g2 below are the half-sums and half-differences of the pairs:
#   (y1+y4)/2 = x0 + k1·S + k2·D = r + q     (y1-y4)/2 = k3·t3 + k4·t4 = g1
#   (y2+y3)/2 = r - q                        (y2-y3)/2 = k4·t3 - k3·t4 = g2
# so y1, y4 = (r+q) ± g1 and y2, y3 = (r-q) ± g2.
# Bounds:
# for inputs |xi| <= p, S and D stay < 4p (mulmod-legal); reducing
# x0 + k1·S closes the chain, leaving every output <= ~3.5p — inside the 4p
# mulmod domain of whatever loads it next. y0 comes back unreduced;
# callers reduce it in both directions (the reverse column sum reaches
# ~4.1p raw, past the mulmod domain).
@inline function fp_dft5(x0, x1, x2, x3, x4, k1, k2, k3, k4, k1p, k2p, k3p, k4p, F::FpCtx)
    t1 = x1 + x4; t3 = x1 - x4
    t2 = x2 + x3; t4 = x2 - x3
    S = t1 + t2;  D = t1 - t2
    y0 = x0 + S
    r = fp_reduce(x0 + fp_mulmod(S, k1, k1p, F), F)
    q = fp_mulmod(D, k2, k2p, F)
    g1 = fp_mulmod(t3, k3, k3p, F) + fp_mulmod(t4, k4, k4p, F)
    g2 = fp_mulmod(t3, k4, k4p, F) - fp_mulmod(t4, k3, k3p, F)
    u = r + q; v = r - q
    return y0, u + g1, v + g2, v - g2, u - g1
end

function fp_radix5!(x::Vector{Float64}, o::Int, st::FpOddStage, F::FpCtx,
                    ::Val{FWD}) where {FWD}
    Q = st.Q
    k1, k2, k3, k4 = st.rot[1], st.rot[2], st.rot[3], st.rot[4]
    k1p, k2p, k3p, k4p = st.rotp[1], st.rotp[2], st.rotp[3], st.rotp[4]
    w = st.tw
    wp = st.twp
    j = 0
    @inbounds while j + 8 <= Q
        i0 = o + j + 1
        y0 = SIMD.vload(VF8, x, i0)
        y1 = SIMD.vload(VF8, x, i0 + Q)
        y2 = SIMD.vload(VF8, x, i0 + 2Q)
        y3 = SIMD.vload(VF8, x, i0 + 3Q)
        y4 = SIMD.vload(VF8, x, i0 + 4Q)
        if FWD
            y0, y1, y2, y3, y4 = fp_dft5(y0, y1, y2, y3, y4, k1, k2, k3, k4,
                                         k1p, k2p, k3p, k4p, F)
        end
        y0 = fp_reduce(y0, F)
        y1 = fp_mulmod(y1, SIMD.vload(VF8, w[1], j + 1),
                       SIMD.vload(VF8, wp[1], j + 1), F)
        y2 = fp_mulmod(y2, SIMD.vload(VF8, w[2], j + 1),
                       SIMD.vload(VF8, wp[2], j + 1), F)
        y3 = fp_mulmod(y3, SIMD.vload(VF8, w[3], j + 1),
                       SIMD.vload(VF8, wp[3], j + 1), F)
        y4 = fp_mulmod(y4, SIMD.vload(VF8, w[4], j + 1),
                       SIMD.vload(VF8, wp[4], j + 1), F)
        if !FWD
            y0, y1, y2, y3, y4 = fp_dft5(y0, y1, y2, y3, y4, k1, k2, k3, k4,
                                         k1p, k2p, k3p, k4p, F)
            y0 = fp_reduce(y0, F)
        end
        SIMD.vstore(y0, x, i0)
        SIMD.vstore(y1, x, i0 + Q)
        SIMD.vstore(y2, x, i0 + 2Q)
        SIMD.vstore(y3, x, i0 + 3Q)
        SIMD.vstore(y4, x, i0 + 4Q)
        j += 8
    end
    @inbounds while j < Q
        i0 = o + j + 1
        y0 = x[i0]
        y1 = x[i0+Q]
        y2 = x[i0+2Q]
        y3 = x[i0+3Q]
        y4 = x[i0+4Q]
        if FWD
            y0, y1, y2, y3, y4 = fp_dft5(y0, y1, y2, y3, y4,
                                         k1, k2, k3, k4, k1p, k2p, k3p, k4p, F)
        end
        y0 = fp_reduce(y0, F)
        y1 = fp_mulmod(y1, w[1][j+1], wp[1][j+1], F)
        y2 = fp_mulmod(y2, w[2][j+1], wp[2][j+1], F)
        y3 = fp_mulmod(y3, w[3][j+1], wp[3][j+1], F)
        y4 = fp_mulmod(y4, w[4][j+1], wp[4][j+1], F)
        if !FWD
            y0, y1, y2, y3, y4 = fp_dft5(y0, y1, y2, y3, y4,
                                         k1, k2, k3, k4, k1p, k2p, k3p, k4p, F)
            y0 = fp_reduce(y0, F)
        end
        x[i0] = y0
        x[i0+Q] = y1
        x[i0+2Q] = y2
        x[i0+3Q] = y3
        x[i0+4Q] = y4
        j += 1
    end
    return x
end

# Symmetric-pair 17-point DFT output pair (see fp_dft17_consts): with
# s_j = x_j + x_{17-j}, d_j = x_j - x_{17-j},
#   y_r      = x_0 + u_r + v_r,   u_r = Σ_j s_j·A_{jr}
#   y_{17-r} = x_0 + u_r - v_r,   v_r = Σ_j d_j·B_{jr}
# for r in 1..8 — 128 mulmods per column instead of the DFT matrix's 256.
# Bounds: inputs |x| <= ~0.94p give |s|,|d| <= 2p (mulmod-legal), each
# mulmod out <= 0.75p, so |u|,|v| <= 6p; both are fp_reduced here before the
# ±-combine, so every y is <= |x0| + ~1.02p <= ~2p — legal for whatever
# mulmod loads it next.  The sums are balanced trees purely for latency;
# every order is exact (integers < 2^53).
@inline function fp_dft17_uv(s::NTuple{8,T}, d::NTuple{8,T}, base::Int,
                             rot::Vector{Float64}, rotp::Vector{Float64},
                             F::FpCtx) where {T}
    @inbounds begin
        Base.Cartesian.@nexprs 8 j -> u_j =
            fp_mulmod(s[j], T(rot[base+j]), T(rotp[base+j]), F)
        Base.Cartesian.@nexprs 8 j -> v_j =
            fp_mulmod(d[j], T(rot[base+8+j]), T(rotp[base+8+j]), F)
    end
    u = ((u_1 + u_2) + (u_3 + u_4)) + ((u_5 + u_6) + (u_7 + u_8))
    v = ((v_1 + v_2) + (v_3 + v_4)) + ((v_5 + v_6) + (v_7 + v_8))
    return fp_reduce(u, F), fp_reduce(v, F)
end

function fp_radix17!(x::Vector{Float64}, o::Int, st::FpOddStage, F::FpCtx{P},
                     ::Val{FWD}) where {P, FWD}
    Q = st.Q
    rot, rotp = st.rot, st.rotp
    w = st.tw
    invp = inv(P)   # derive w/p on the fly (one mul) instead of a second table load
    j = 0
    @inbounds while j + 8 <= Q
        i0 = o + j + 1
        x0 = SIMD.vload(VF8, x, i0)
        FWD || (x0 = fp_reduce(x0, F))
        Base.Cartesian.@nexprs 8 jj -> begin
            a_jj = SIMD.vload(VF8, x, i0 + jj * Q)
            b_jj = SIMD.vload(VF8, x, i0 + (17 - jj) * Q)
            if !FWD
                wa_jj = SIMD.vload(VF8, w[jj], j + 1)
                a_jj = fp_mulmod(a_jj, wa_jj, wa_jj * invp, F)
                wb_jj = SIMD.vload(VF8, w[17-jj], j + 1)
                b_jj = fp_mulmod(b_jj, wb_jj, wb_jj * invp, F)
            end
            s_jj = a_jj + b_jj
            d_jj = a_jj - b_jj
        end
        s = (s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8)
        d = (d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8)
        y0 = ((s_1 + s_2) + (s_3 + s_4)) + ((s_5 + s_6) + (s_7 + s_8))
        SIMD.vstore(fp_reduce(x0 + y0, F), x, i0)
        for r in 1:8
            u, v = fp_dft17_uv(s, d, (r - 1) * 16, rot, rotp, F)
            xu = x0 + u
            yr = xu + v
            ym = xu - v
            if FWD
                wr = SIMD.vload(VF8, w[r], j + 1)
                yr = fp_mulmod(yr, wr, wr * invp, F)
                wm = SIMD.vload(VF8, w[17-r], j + 1)
                ym = fp_mulmod(ym, wm, wm * invp, F)
            end
            SIMD.vstore(yr, x, i0 + r * Q)
            SIMD.vstore(ym, x, i0 + (17 - r) * Q)
        end
        j += 8
    end
    @inbounds while j < Q
        i0 = o + j + 1
        x0 = x[i0]
        FWD || (x0 = fp_reduce(x0, F))
        Base.Cartesian.@nexprs 8 jj -> begin
            a_jj = x[i0+jj*Q]
            b_jj = x[i0+(17-jj)*Q]
            if !FWD
                wa_jj = w[jj][j+1]
                a_jj = fp_mulmod(a_jj, wa_jj, wa_jj * invp, F)
                wb_jj = w[17-jj][j+1]
                b_jj = fp_mulmod(b_jj, wb_jj, wb_jj * invp, F)
            end
            s_jj = a_jj + b_jj
            d_jj = a_jj - b_jj
        end
        s = (s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8)
        d = (d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8)
        y0 = ((s_1 + s_2) + (s_3 + s_4)) + ((s_5 + s_6) + (s_7 + s_8))
        x[i0] = fp_reduce(x0 + y0, F)
        for r in 1:8
            u, v = fp_dft17_uv(s, d, (r - 1) * 16, rot, rotp, F)
            xu = x0 + u
            yr = xu + v
            ym = xu - v
            if FWD
                wr = w[r][j+1]
                yr = fp_mulmod(yr, wr, wr * invp, F)
                wm = w[17-r][j+1]
                ym = fp_mulmod(ym, wm, wm * invp, F)
            end
            x[i0+r*Q] = yr
            x[i0+(17-r)*Q] = ym
        end
        j += 1
    end
    return x
end

# ---------------------------------------------------------------------------
# Radix-4 butterfly core, shared by both directions (the DFT4 matrix is
# symmetric).  y0 comes back unreduced: the forward reduces it before
# storing; the reverse stores it raw.

@inline function fp_dft4(a, b, c, d, wi, wip, F::FpCtx)
    apc = a + c
    amc = a - c
    bpd = b + d
    ibmd = fp_mulmod(b - d, wi, wip, F)
    return apc + bpd, amc + ibmd, apc - bpd, amc - ibmd
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

function fp_smallq!(x::Vector{Float64}, o::Int, N2::Int, ::Val{Q},
                    stg::FpNttStage, wi::Float64, wip::Float64, F::FpCtx,
                    ::Val{FWD}) where {Q,FWD}
    vw1 = SIMD.vload(VF8, stg.w1, 1); vw1p = SIMD.vload(VF8, stg.w1p, 1)
    vw2 = SIMD.vload(VF8, stg.w2, 1); vw2p = SIMD.vload(VF8, stg.w2p, 1)
    vw3 = SIMD.vload(VF8, stg.w3, 1); vw3p = SIMD.vload(VF8, stg.w3p, 1)
    @inbounds for i0 in o+1:32:o+N2-31
        v0 = SIMD.vload(VF8, x, i0)
        v1 = SIMD.vload(VF8, x, i0 + 8)
        v2 = SIMD.vload(VF8, x, i0 + 16)
        v3 = SIMD.vload(VF8, x, i0 + 24)
        y0, y1, y2, y3 = ntt_gather4(Val(Q), v0, v1, v2, v3)
        if FWD
            y0, y1, y2, y3 = fp_dft4(y0, y1, y2, y3, wi, wip, F)
        end
        y0 = fp_reduce(y0, F)
        y1 = fp_mulmod(y1, vw1, vw1p, F)
        y2 = fp_mulmod(y2, vw2, vw2p, F)
        y3 = fp_mulmod(y3, vw3, vw3p, F)
        if !FWD
            y0, y1, y2, y3 = fp_dft4(y0, y1, y2, y3, wi, wip, F)
        end
        o0, o1, o2, o3 = ntt_scatter4(Val(Q), y0, y1, y2, y3)
        SIMD.vstore(o0, x, i0)
        SIMD.vstore(o1, x, i0 + 8)
        SIMD.vstore(o2, x, i0 + 16)
        SIMD.vstore(o3, x, i0 + 24)
    end
    return x
end

# ---------------------------------------------------------------------------
# Power-of-two pipelines, radix-4.

# One radix-4 stage (butterfly span L = 4q) over the length-N2 block at
# offset o.  Both directions run the same twiddle+reduce pass (reduce the
# ω^0 lane, twiddle the rest); FWD picks which side of it the DFT4 combine
# sits on — before for the forward (stage = twiddle·DFT4), after for the
# reverse (the transpose, DFT4·twiddle).  Reverse outputs store raw; the
# next stage's ω^0 reduce or the unpack's 1/N mulmod absorbs them.
function fp_pow2_stage!(x::Vector{Float64}, o::Int, N2::Int, stg::FpNttStage,
                        wi::Float64, wip::Float64, F::FpCtx,
                        ::Val{FWD}) where {FWD}
    q = stg.q
    L = 4q
    if q >= 8
        w1, w1p, w2, w2p, w3, w3p = stg.w1, stg.w1p, stg.w2, stg.w2p, stg.w3, stg.w3p
        for s in o:L:o+N2-1
            @inbounds for j in 0:8:q-8
                i0 = s + j + 1
                y0 = SIMD.vload(VF8, x, i0)
                y1 = SIMD.vload(VF8, x, i0 + q)
                y2 = SIMD.vload(VF8, x, i0 + 2q)
                y3 = SIMD.vload(VF8, x, i0 + 3q)
                if FWD
                    y0, y1, y2, y3 = fp_dft4(y0, y1, y2, y3, wi, wip, F)
                end
                y0 = fp_reduce(y0, F)
                y1 = fp_mulmod(y1, SIMD.vload(VF8, w1, j + 1),
                               SIMD.vload(VF8, w1p, j + 1), F)
                y2 = fp_mulmod(y2, SIMD.vload(VF8, w2, j + 1),
                               SIMD.vload(VF8, w2p, j + 1), F)
                y3 = fp_mulmod(y3, SIMD.vload(VF8, w3, j + 1),
                               SIMD.vload(VF8, w3p, j + 1), F)
                if !FWD
                    y0, y1, y2, y3 = fp_dft4(y0, y1, y2, y3, wi, wip, F)
                end
                SIMD.vstore(y0, x, i0)
                SIMD.vstore(y1, x, i0 + q)
                SIMD.vstore(y2, x, i0 + 2q)
                SIMD.vstore(y3, x, i0 + 3q)
            end
        end
    elseif N2 >= 32
        q == 4 ? fp_smallq!(x, o, N2, Val(4), stg, wi, wip, F, Val(FWD)) :
        q == 2 ? fp_smallq!(x, o, N2, Val(2), stg, wi, wip, F, Val(FWD)) :
                 fp_smallq!(x, o, N2, Val(1), stg, wi, wip, F, Val(FWD))
    else
        w1, w1p, w2, w2p, w3, w3p = stg.w1, stg.w1p, stg.w2, stg.w2p, stg.w3, stg.w3p
        for s in o:L:o+N2-1
            @inbounds for j in 0:q-1
                y0 = x[s+j+1]
                y1 = x[s+j+q+1]
                y2 = x[s+j+2q+1]
                y3 = x[s+j+3q+1]
                if FWD
                    y0, y1, y2, y3 = fp_dft4(y0, y1, y2, y3, wi, wip, F)
                end
                y0 = fp_reduce(y0, F)
                y1 = fp_mulmod(y1, w1[j+1], w1p[j+1], F)
                y2 = fp_mulmod(y2, w2[j+1], w2p[j+1], F)
                y3 = fp_mulmod(y3, w3[j+1], w3p[j+1], F)
                if !FWD
                    y0, y1, y2, y3 = fp_dft4(y0, y1, y2, y3, wi, wip, F)
                end
                x[s+j+1] = y0
                x[s+j+q+1] = y1
                x[s+j+2q+1] = y2
                x[s+j+3q+1] = y3
            end
        end
    end
    return x
end

function fp_fwd_pow2!(x::Vector{Float64}, o::Int, plan::FpNttPlan)
    F = plan.ctx
    N2 = plan.N2
    for stg in plan.fwd
        fp_pow2_stage!(x, o, N2, stg, plan.wi, plan.wip, F, Val(true))
    end
    if isodd(trailing_zeros(N2))
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

# fp_fwd_pow2!'s stages in exact reverse order with the same tables.  The
# first stage (span 2 for odd k, else span 4) is twiddle-free and fused
# here; for even k the fwd list's trivial-twiddle L=4 stage is skipped.
function fp_rev_pow2!(x::Vector{Float64}, o::Int, plan::FpNttPlan)
    F = plan.ctx
    N2 = plan.N2
    kodd = isodd(trailing_zeros(N2))
    if kodd
        @inbounds for s in o:2:o+N2-1
            u = x[s+1]
            t = x[s+2]
            x[s+1] = u + t
            x[s+2] = u - t
        end
    else
        wi, wip = plan.wi, plan.wip
        @inbounds for s in o:4:o+N2-1
            y0, y1, y2, y3 = fp_dft4(x[s+1], x[s+2], x[s+3], x[s+4], wi, wip, F)
            x[s+1] = y0
            x[s+2] = y1
            x[s+3] = y2
            x[s+4] = y3
        end
    end
    for si in length(plan.fwd)-(kodd ? 0 : 1):-1:1
        fp_pow2_stage!(x, o, N2, plan.fwd[si], plan.wi, plan.wip, F, Val(false))
    end
    return x
end

function fp_ntt_fwd!(x::Vector{Float64}, plan::FpNttPlan)
    F = plan.ctx
    for st in plan.oddf
        for o in 0:st.span:plan.N-1
            st.m == 3 ? fp_radix3!(x, o, st, F, Val(true)) :
            st.m == 5 ? fp_radix5!(x, o, st, F, Val(true)) :
                        fp_radix17!(x, o, st, F, Val(true))
        end
    end
    for o in 0:plan.N2:plan.N-1
        fp_fwd_pow2!(x, o, plan)
    end
    return x
end

# Transpose of fp_ntt_fwd! (same twiddles, stages reversed): consumes the
# forward's scrambled output order and returns natural order.  Fed the
# pointwise product Ĉ it yields N·c[(N-t) mod N]; the unpack reads
# descending and folds in the 1/N.
function fp_ntt_rev!(x::Vector{Float64}, plan::FpNttPlan)
    F = plan.ctx
    for o in 0:plan.N2:plan.N-1
        fp_rev_pow2!(x, o, plan)
    end
    for st in Iterators.reverse(plan.oddf)
        for o in 0:st.span:plan.N-1
            st.m == 3 ? fp_radix3!(x, o, st, F, Val(false)) :
            st.m == 5 ? fp_radix5!(x, o, st, F, Val(false)) :
                        fp_radix17!(x, o, st, F, Val(false))
        end
    end
    return x
end

# ---------------------------------------------------------------------------
# Coefficient-domain layer, shared by both pipelines: size the transform
# (ntt_len), split limbs into chunk coefficients (fp_ntt_pack!), and multiply
# lanewise between the transforms (fp_ntt_pointwise!).

# Smallest transform length >= T: m·2^k with m in (1, 3, 5) at every size;
# the expensive odd multipliers (15 and the 17 family) join only from
# k >= 14, where their tighter padding beats a cache-spilling pow2
# transform (1.26x at 15·2^18 vs 2^22, 0.80x at 255·2^15 vs 2^23).  The
# uniform floor gives up <= ~5% in narrow bands against the measured per-m
# tie points (bench/bench_radix17.jl), accepted for the simpler rule.  A
# pure 2^k tops out at 2^33 (~18 GB operands) since 2^k must divide p-1
# for both primes; the odd multipliers extend reach to 255·2^33 (~4 TB).
function ntt_len(T::Int, ctxs::FpCtx...)
    maxk = minimum(two_adicity(F) for F in ctxs)
    best = typemax(Int)
    for m in (1, 3, 5, 15, 17, 51, 85, 255)           # multiples of 3, 5, 17 each used <= once
        k = max(Base.top_set_bit(cld(T, m) - 1), 2)   # ceil(log2), >= 4 points
        k <= maxk || continue                         # 2^k must divide p-1
        m <= 5 || k >= 14 || continue                 # odd-multiplier floor
        c = m << k
        c < best && (best = c)
    end
    best == typemax(Int) &&
        throw(ArgumentError("fp NTT length $T exceeds the 255·2^$maxk the primes support"))
    return best
end

# b-bit chunk extraction into Float64 points; chunks < 2^b <= 2^52 are exact
# doubles and, being smaller than every prime, already canonical residues.
# Overwrites all of x (chunks, then zero fill), so callers can recycle a
# spent transform buffer.
function fp_ntt_pack!(x::Vector{Float64}, limbs::Memory{Limb}, lo::Int, n::Int,
                      b::Int, nch::Int)
    N = length(x)
    mask = (UInt64(1) << b) - 1
    imax = min(nch - 1, fld(64 * (n - 1) - 1, b))
    @inbounds for i in 0:imax
        bit = i * b
        w = bit >> 6
        sh = bit & 63
        c = (limbs[lo+w+1] >>> sh) | (limbs[lo+w+2] << (64 - sh))
        x[i+1] = c & mask
    end
    @inbounds for i in imax+1:nch-1
        bit = i * b
        w = bit >> 6
        sh = bit & 63
        c = limbs[lo+w+1] >>> sh
        if sh + b > 64 && w + 2 <= n
            c |= limbs[lo+w+2] << (64 - sh)
        end
        x[i+1] = c & mask
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
# visits large b first regardless of operand size.  The search starts at the
# largest b keeping chunks below both primes (48 here), so one pack serves
# both residues.
function fp_ntt_params2(bits_a::Int, bits_b::Int, F1::FpCtx, F2::FpCtx)
    for b in Base.top_set_bit(min(fp_prime(F1), fp_prime(F2)))-1:-1:1
        nca = cld(bits_a, b)
        ncb = cld(bits_b, b)
        if UInt128(min(nca, ncb)) <=
           (UInt128(fp_prime(F1)) * fp_prime(F2) - 1) ÷ (UInt128(2)^b - 1)^2
            return b, ntt_len(nca + ncb - 1, F1, F2)
        end
    end
    error("unreachable: b == 1 always satisfies the bound for supported sizes")
end

# Garner recombination + limb streaming.  fp_ntt_rev! hands back N·c index-
# reversed (coefficient i at (N-i) mod N); this loop scales each residue by 1/N
# and Garner-combines the two mod-p residues of every base-2^b coefficient:
#   c = c1 + p1·u,  u = (c2 - c1)·p1^-1 mod p2.
# c is never materialized — by linearity Σ cᵢ·2^(b·i) = S1 + p1·S2 with
# S1 = Σ c1ᵢ·2^(b·i) and S2 = Σ uᵢ·2^(b·i), so the two sums stream independently
# (S1 into r, S2 into s2) and one final addmul_1! folds them at kernel speed.
#
# Each c1, u is pre-split into window halves lo = v << s, hi = v >> (64 - s) so
# the scalar accumulate is pure add-with-carry; s = (b·i) mod 64 is data-
# independent because the flush drops s by exactly 64.  The 128-bit window never
# overflows: values are < 2^49, s < 64, and each flushed limb is final (later
# coefficients start at a strictly higher bit).  Note c1 must be canonicalized
# before fp_reduce(v1, F2), since v1 - p1 is a different value mod p2.

# One scalar coefficient: unscale each residue by 1/N and Garner-combine into
# the mod-p1 limb c1 and correction uu (both canonical UInt64).
@inline function fp_unpack_coeff(x1::Vector{Float64}, x2::Vector{Float64}, idx::Int,
                                 n1::Float64, n1p::Float64, n2::Float64, n2p::Float64,
                                 g::Float64, gp::Float64)
    v1 = fp_mulmod(x1[idx], n1, n1p, FP_CTX1)
    v1 = ifelse(v1 < 0.0, v1 + fp_prime(FP_CTX1), v1)
    v2 = fp_mulmod(x2[idx], n2, n2p, FP_CTX2)
    u = fp_mulmod(v2 - fp_reduce(v1, FP_CTX2), g, gp, FP_CTX2)
    c1 = unsafe_trunc(UInt64, v1)
    uu = unsafe_trunc(UInt64, ifelse(u < 0.0, u + fp_prime(FP_CTX2), u))
    return c1, uu
end

# Scalar-pack cnt (<= 8) coefficients starting at coefficient i into the stage
# buffer (quarters lo1|hi1|lo2|hi2), shifting lane j by (s + j·b) & 63.
# Coefficient 0 wraps to x1[1]; the rest read descending x1[N-·+1].  The vector
# loop packs the same layout with SIMD; this is the < 8-wide head/tail fallback.
# @noinline keeps its fp_mulmod chains out of fp_ntt_unpack2!'s hot loop, whose
# adc recognition degrades when the function is bloated.
@noinline function fp_pack_scalar!(stage::Vector{UInt64}, x1::Vector{Float64},
                                 x2::Vector{Float64}, N::Int, i::Int, cnt::Int,
                                 s::Int, b::Int, n1::Float64, n1p::Float64,
                                 n2::Float64, n2p::Float64, g::Float64, gp::Float64)
    @inbounds for j in 0:cnt-1
        c = i + j
        idx = c == 0 ? 1 : N - c + 1
        c1, uu = fp_unpack_coeff(x1, x2, idx, n1, n1p, n2, n2p, g, gp)
        sh = (s + j * b) & 63
        stage[j+1]  = c1 << sh
        stage[j+9]  = (c1 >> 1) >> (63 - sh) # 64 - sh but dodging the expensive case cause cpus are dumb
        stage[j+17] = uu << sh
        stage[j+25] = (uu >> 1) >> (63 - sh)
    end
    return stage
end

# Fold cnt (<= 8) pre-shifted coefficients from the stage buffer into both
# 128-bit windows, flushing a finished limb whenever s crosses 64.  Spelled out
# inline rather than via a per-coefficient helper on purpose: a per-lane
# function boundary drops LLVM's adc carry chain (~60% slower).  At the hot call
# site cnt = 8 is a literal, so this unrolls to the full 16-adc chain.
@inline function fp_emit_stage!(r::Memory{Limb}, ro::Int, s2::Memory{Limb}, outw::Int,
                                a01::UInt64, a11::UInt64, a02::UInt64, a12::UInt64,
                                s::Int, b::Int, stage::Vector{UInt64}, cnt::Int)
    @inbounds for j in 1:cnt
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
    return outw, a01, a11, a02, a12, s
end

function fp_ntt_unpack2!(r::Memory{Limb}, ro::Int, rn::Int,
                         x1::Vector{Float64}, x2::Vector{Float64}, nconv::Int,
                         b::Int, n1::Float64, n1p::Float64, n2::Float64, n2p::Float64)
    N = length(x1)
    s2 = Memory{Limb}(undef, rn)
    # SIMD→scalar staging buffer, quarters lo1 | hi1 | lo2 | hi2.  The scan
    # reads lanes at a runtime index; Vec lane extraction with a non-constant
    # index spills to the stack (~30% slower than this explicit round-trip).
    stage = Vector{UInt64}(undef, 32)
    g = Float64(invmod(fp_prime(FP_CTX1) % fp_prime(FP_CTX2), fp_prime(FP_CTX2))) # p1^-1 mod p2 (Garner)
    gp = g / fp_prime(FP_CTX2) # twiddle
    # canonical values are < p < 2^49, so v + 1.5·2^52 pins the exponent and
    # leaves v in the low 51 mantissa bits: int(v) is a reinterpret-and-mask
    vmask = V8(UInt64(2)^51 - 1)
    ub = UInt64(b)
    vlane = V8(ntuple(k -> UInt64(k - 1) * ub, 8))
    a01 = UInt64(0); a11 = UInt64(0)   # stream 1 window, low/high limb
    a02 = UInt64(0); a12 = UInt64(0)   # stream 2 window
    outw = 1
    s = 0             # bit offset of coefficient i within the window
    # scalar head through coefficient 7: covers the i = 0 wraparound so the
    # vector loop's descending 8-loads stay contiguous
    i = min(8, nconv)
    fp_pack_scalar!(stage, x1, x2, N, 0, i, s, b, n1, n1p, n2, n2p, g, gp)
    outw, a01, a11, a02, a12, s =
        fp_emit_stage!(r, ro, s2, outw, a01, a11, a02, a12, s, b, stage, i)
    @inbounds while i + 8 <= nconv
        REV8 = (7, 6, 5, 4, 3, 2, 1, 0)
        v1 = fp_mulmod(SIMD.shufflevector(SIMD.vload(VF8, x1, N - i - 6), Val(REV8)),
                       n1, n1p, FP_CTX1)
        v1 = SIMD.vifelse(v1 < 0, v1 + fp_prime(FP_CTX1), v1)
        v2 = fp_mulmod(SIMD.shufflevector(SIMD.vload(VF8, x2, N - i - 6), Val(REV8)),
                       n2, n2p, FP_CTX2)
        u = fp_mulmod(v2 - fp_reduce(v1, FP_CTX2), g, gp, FP_CTX2)
        u = SIMD.vifelse(u < 0, u + fp_prime(FP_CTX2), u)
        c1v = reinterpret(V8, v1 + FP_MAGIC) & vmask
        uv = reinterpret(V8, u + FP_MAGIC) & vmask
        sv = (UInt64(i) * ub + vlane) & UInt64(63)
        svc = UInt64(63) - sv
        SIMD.vstore(c1v << sv, stage, 1)
        SIMD.vstore((c1v >> 1) >> svc, stage, 9)
        SIMD.vstore(uv << sv, stage, 17)
        SIMD.vstore((uv >> 1) >> svc, stage, 25)
        outw, a01, a11, a02, a12, s =
            fp_emit_stage!(r, ro, s2, outw, a01, a11, a02, a12, s, b, stage, 8)
        i += 8
    end
    # scalar tail: the final nconv mod 8 coefficients
    tail = nconv - i
    fp_pack_scalar!(stage, x1, x2, N, i, tail, s, b, n1, n1p, n2, n2p, g, gp)
    outw, a01, a11, a02, a12, s =
        fp_emit_stage!(r, ro, s2, outw, a01, a11, a02, a12, s, b, stage, tail)
    @inbounds while outw <= rn
        r[ro+outw] = a01
        s2[outw] = a02
        a01 = a11; a11 = UInt64(0)
        a02 = a12; a12 = UInt64(0)
        outw += 1
    end
    addmul_1!(r, ro, s2, 0, rn, fp_prime(FP_CTX1))
    return r
end

# mpn-layer entry points: r[1..m+n] = a[1..m]·b[1..n], r must not alias the inputs.
# Calls fp_ntt_pack! right before fp_ntt_fwd! to improve locality.
function mul_fpntt2!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, m::Int,
                     b::Memory{Limb}, bo::Int, n::Int)
    bits_a = magnitude_bits(a, ao, m)
    bits_b = magnitude_bits(b, bo, n)
    bch, N = fp_ntt_params2(bits_a, bits_b, FP_CTX1, FP_CTX2)
    plan1 = fp_ntt_plan(N, FP_CTX1)
    plan2 = fp_ntt_plan(N, FP_CTX2)
    nca, ncb = cld(bits_a, bch), cld(bits_b, bch)
    tmp1 = fp_ntt_pack!(Vector{Float64}(undef, N), a, ao, m, bch, nca)
    fp_ntt_fwd!(tmp1, plan1)
    tmp2 = fp_ntt_pack!(Vector{Float64}(undef, N), b, bo, n, bch, ncb)
    fp_ntt_fwd!(tmp2, plan1)
    fp_ntt_pointwise!(tmp1, tmp2, FP_CTX1)
    fp_ntt_rev!(tmp1, plan1)
    tmp3 = fp_ntt_pack!(Vector{Float64}(undef, N), a, ao, m, bch, nca)
    fp_ntt_fwd!(tmp3, plan2)
    fp_ntt_pack!(tmp2, b, bo, n, bch, ncb)
    fp_ntt_fwd!(tmp2, plan2)
    fp_ntt_pointwise!(tmp3, tmp2, FP_CTX2)
    fp_ntt_rev!(tmp3, plan2)
    fp_ntt_unpack2!(r, ro, m + n, tmp1, tmp3, nca + ncb - 1, bch,
                    plan1.ninv, plan1.ninvp, plan2.ninv, plan2.ninvp)
    return nothing
end

function sqr_fpntt2!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int)
    bits = magnitude_bits(a, ao, n)
    bch, N = fp_ntt_params2(bits, bits, FP_CTX1, FP_CTX2)
    plan1 = fp_ntt_plan(N, FP_CTX1)
    plan2 = fp_ntt_plan(N, FP_CTX2)
    nca = cld(bits, bch)
    tmp1 = fp_ntt_pack!(Vector{Float64}(undef, N), a, ao, n, bch, nca)
    fp_ntt_fwd!(tmp1, plan1)
    fp_ntt_pointwise!(tmp1, tmp1, FP_CTX1)
    fp_ntt_rev!(tmp1, plan1)
    tmp2 = fp_ntt_pack!(Vector{Float64}(undef, N), a, ao, n, bch, nca)
    fp_ntt_fwd!(tmp2, plan2)
    fp_ntt_pointwise!(tmp2, tmp2, FP_CTX2)
    fp_ntt_rev!(tmp2, plan2)
    fp_ntt_unpack2!(r, ro, 2n, tmp1, tmp2, 2nca - 1, bch,
                    plan1.ninv, plan1.ninvp, plan2.ninv, plan2.ninvp)
    return nothing
end
