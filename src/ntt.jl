# NTT multiplication prototype over the Goldilocks field GF(p),
# p = 2^64 - 2^32 + 1.  Standalone: not wired into mul! dispatch.
#
# p - 1 = 2^32·(2^32 - 1) and 2^32 - 1 = 3·5·17·257·65537, so power-of-two
# transforms exist up to length 2^32 and lengths m·2^k for m in (3, 5, 15)
# are also available, capping zero-padding waste at ~25% instead of ~100%.
# 2^64 ≡ 2^32 - 1 (mod p) makes 128-bit product reduction multiplication-
# free, and i = 2^48 makes the radix-4 rotation shift-only.

const GF_P = 0xFFFF_FFFF_0000_0001
const GF_EPS = 0x0000_0000_FFFF_FFFF   # 2^64 mod p == 2^32 - 1

@inline function gf_add(a::UInt64, b::UInt64)
    s = a + b
    # on wrap: +2^64 ≡ +EPS; s < EPS there, so the correction can't re-wrap
    s = ifelse(s < a, s + GF_EPS, s)
    return ifelse(s >= GF_P, s - GF_P, s)
end

@inline function gf_sub(a::UInt64, b::UInt64)
    d = a - b
    return ifelse(a < b, d + GF_P, d)
end

# reduce a 128-bit product: with t = x1*2^96 + x0*2^64 + lo (x1, x0 32-bit),
# t ≡ lo - x1 + x0*(2^32 - 1) (mod p) since 2^96 ≡ -1 and 2^64 ≡ 2^32 - 1
@inline function gf_mul(a::UInt64, b::UInt64)
    t = widemul(a, b)
    lo = t % UInt64
    hi = (t >>> 64) % UInt64
    x1 = hi >>> 32
    x0 = hi & GF_EPS
    r = lo - x1
    # on borrow: -2^64 ≡ -EPS; r > 2^64 - 2^32 there, so no re-borrow
    r = ifelse(lo < x1, r - GF_EPS, r)
    m = x0 * GF_EPS                # < 2^64, no overflow
    s = r + m
    # on wrap: s < 2^64 - 2^33 there, so the correction can't re-wrap
    s = ifelse(s < r, s + GF_EPS, s)
    return ifelse(s >= GF_P, s - GF_P, s)
end

# i = 2^48 is a primitive fourth root of unity (2^96 ≡ -1), so the radix-4
# butterflies' rotation x·i needs no widemul: the 112-bit product x·2^48
# reduces with shifts, and x0·(2^32-1) is a shift and subtract.
const GF_I4 = UInt64(1) << 48
@inline function gf_mul_i(x::UInt64)
    lo = x << 48
    hi = x >>> 16
    x1 = hi >>> 32
    x0 = hi & GF_EPS
    r = lo - x1
    r = ifelse(lo < x1, r - GF_EPS, r)
    m = (x0 << 32) - x0
    s = r + m
    s = ifelse(s < r, s + GF_EPS, s)
    return ifelse(s >= GF_P, s - GF_P, s)
end

function gf_pow(a::UInt64, e::Integer)
    r = UInt64(1)
    b = a >= GF_P ? a - GF_P : a
    while e != 0
        isodd(e) && (r = gf_mul(r, b))
        b = gf_mul(b, b)
        e >>= 1
    end
    return r
end

@inline gf_inv(a::UInt64) = gf_pow(a, GF_P - 2)

# Vectorized field ops on V8 lanes.  The 64×64 product is assembled from
# four widening 32×32 multiplies (vpmuludq/umull class — LLVM matches the
# masked-input pattern on AVX2, AVX-512, and NEON), and every fixup is a
# lanewise compare + select.  x0·(2^32-1) is a shift and subtract, so the
# whole reduction is multiply-free.

# reduce lanewise (hi, lo) 128-bit values mod p, canonical output
@inline function gf_reducev(hi::V8, lo::V8)
    x1 = hi >>> 32
    x0 = hi & GF_EPS
    r = lo - x1
    r = SIMD.vifelse(lo < x1, r - GF_EPS, r)
    m = (x0 << 32) - x0
    s = r + m
    s = SIMD.vifelse(s < r, s + GF_EPS, s)
    return SIMD.vifelse(s >= GF_P, s - GF_P, s)
end

@inline function gf_addv(a::V8, b::V8)
    s = a + b
    s = SIMD.vifelse(s < a, s + GF_EPS, s)
    return SIMD.vifelse(s >= GF_P, s - GF_P, s)
end

@inline function gf_subv(a::V8, b::V8)
    d = a - b
    return SIMD.vifelse(a < b, d + GF_P, d)
end

@inline function gf_mulv(a::V8, b::V8)
    alo = a & GF_EPS
    ahi = a >>> 32
    blo = b & GF_EPS
    bhi = b >>> 32
    ll = alo * blo
    mid = alo * bhi + ahi * blo          # may wrap: partials are < 2^64 - 2^33
    midc = SIMD.vifelse(mid < alo * bhi, V8(one(Limb)), V8(zero(Limb)))
    lo = ll + (mid << 32)
    loc = SIMD.vifelse(lo < ll, V8(one(Limb)), V8(zero(Limb)))
    hi = ahi * bhi + (mid >>> 32) + (midc << 32) + loc   # true high word: can't wrap
    return gf_reducev(hi, lo)
end

@inline gf_mul_iv(x::V8) = gf_reducev(x >>> 16, x << 48)

# ---------------------------------------------------------------------------
# Transform plan: N = m·2^k with m in (1, 3, 5, 15).  The odd factors are
# handled by radix-3/radix-5 first stages (DIF) over the whole span, after
# which the m independent power-of-two blocks run the radix-4 pipeline.
# The inverse mirrors the stage sequence exactly.

# Power-of-two stage tables (radix-4 butterflies).  q is the quarter size;
# stages with q >= 8 run the V8 path.  For q < 8 the tables are cyclically
# extended to 8 entries so the shuffle path can vload one V8 twiddle
# pattern; scalar indexing j+1 (j < q) still sees the original values.
struct NttStage
    q::Int
    w1::Vector{UInt64}
    w2::Vector{UInt64}
    w3::Vector{UInt64}
end

function build_stage(table::Vector{UInt64}, N2::Int, L::Int)
    q = L >> 2
    st = N2 ÷ L
    len = max(q, 8)
    return NttStage(q,
                    [table[(j % q)*st+1] for j in 0:len-1],
                    [table[2(j % q)*st+1] for j in 0:len-1],
                    [table[3(j % q)*st+1] for j in 0:len-1])
end

# Odd-radix stage: an order-m DFT (m = 3 or 5) across sub-blocks of length
# Q = span/m, with per-output twiddles ω_span^(j·r).  rot[r] = ω_span^(Q·r)
# are the order-m rotation constants.  Inverse stages carry the tables and
# rotations of the inverse root; the butterfly code is the same shape.
struct OddStage
    m::Int
    span::Int
    Q::Int
    rot::Vector{UInt64}
    tw::Vector{Vector{UInt64}}
end

function build_odd(m::Int, span::Int, root::UInt64)
    Q = span ÷ m
    c = gf_pow(root, Q)
    rot = [gf_pow(c, r) for r in 1:m-1]
    tw = Vector{Vector{UInt64}}(undef, m - 1)
    for r in 1:m-1
        ωr = gf_pow(root, r)
        t = Vector{UInt64}(undef, Q)
        f = UInt64(1)
        for j in 1:Q
            t[j] = f
            f = gf_mul(f, ωr)
        end
        tw[r] = t
    end
    return OddStage(m, span, Q, rot, tw)
end

# fwd pow2 stages descending (L = N2, N2/4, ...); inv pow2 stages ascending,
# excluding the first inverse stage (leftover radix-2 or L == 4), whose
# twiddles are trivial and which the inverse special-cases to fold in N^-1.
# oddf in forward application order (3 before 5); oddi in inverse
# application order (the exact reverse).
struct NttPlan
    N::Int
    N2::Int
    ninv::UInt64
    oddf::Vector{OddStage}
    oddi::Vector{OddStage}
    fwd::Vector{NttStage}
    inv::Vector{NttStage}
end

function build_plan(N::Int)
    k = trailing_zeros(N)
    m = N >> k
    @assert m in (1, 3, 5, 15) && k <= 32 && (m == 1 || k >= 2)
    @assert (GF_P - 1) % N == 0
    ω = gf_pow(UInt64(7), (GF_P - 1) ÷ N)   # 7 generates the group
    # pick the primitive root with ω^(N/4) == +2^48 so the radix-4 rotation
    # can shift; exponentiating by s ≡ 3 (mod 4), gcd(s, N) == 1 flips i's
    # sign and preserves primitivity
    if N % 4 == 0 && gf_pow(ω, N >> 2) != GF_I4
        s = m % 3 == 0 ? 7 : 3
        ω = gf_pow(ω, s)
        @assert gf_pow(ω, N >> 2) == GF_I4
    end
    @assert gf_pow(ω, N >> 1) == GF_P - 1       # order divisible by full 2-part
    m % 3 == 0 && @assert gf_pow(ω, N ÷ 3) != 1
    m % 5 == 0 && @assert gf_pow(ω, N ÷ 5) != 1
    ωinv = gf_inv(ω)

    oddf = OddStage[]
    oddi = OddStage[]
    span = N
    r, ri = ω, ωinv
    for mf in (3, 5)
        if m % mf == 0
            push!(oddf, build_odd(mf, span, r))
            push!(oddi, build_odd(mf, span, ri))
            r = gf_pow(r, mf)
            ri = gf_pow(ri, mf)
            span ÷= mf
        end
    end
    reverse!(oddi)   # inverse applies the odd stages in reverse order

    # power-of-two pipeline tables from ω^m (order N2); radix-4 butterflies
    # index up to ω2^(3(N2/4-1)), so the master tables run to 3N2/4
    N2 = span
    M = max(1, (3N2) >> 2)
    tw = Vector{UInt64}(undef, M)
    twinv = Vector{UInt64}(undef, M)
    fw, fi = UInt64(1), UInt64(1)
    for j in 1:M
        tw[j] = fw
        twinv[j] = fi
        fw = gf_mul(fw, r)
        fi = gf_mul(fi, ri)
    end
    fwd = NttStage[]
    L = N2
    while L >= 4
        push!(fwd, build_stage(tw, N2, L))
        L >>= 2
    end
    inv = NttStage[]
    L = isodd(k) ? 8 : 16
    while L <= N2
        push!(inv, build_stage(twinv, N2, L))
        L <<= 2
    end
    return NttPlan(N, N2, gf_inv(N % UInt64), oddf, oddi, fwd, inv)
end

# plans are pure functions of N, so share them across calls
const NTT_PLAN_LOCK = ReentrantLock()
const NTT_PLAN_CACHE = Dict{Int,NttPlan}()
function ntt_plan(N::Int)
    lock(NTT_PLAN_LOCK) do
        get!(() -> build_plan(N), NTT_PLAN_CACHE, N)
    end
end

# ---------------------------------------------------------------------------
# Odd-radix stages.  Radix-3 uses the Winograd form (ω² == -1 - ω since
# 1 + ω + ω² == 0): with u = ω(b - c),
#   a + ωb + ω²c == (a - c) + u        a + ω²b + ωc == (a - b) - u
# so one rotation multiply serves both outputs.  The same identity applies
# to the inverse stage with its own root.  Radix-5 is the plain DFT matrix.

function radix3_fwd!(x::Vector{UInt64}, o::Int, st::OddStage)
    Q = st.Q
    ω3 = st.rot[1]
    w1, w2 = st.tw[1], st.tw[2]
    j = 0
    if Q >= 8
        ω3v = V8(ω3)
        @inbounds while j + 8 <= Q
            i0 = o + j + 1
            a = SIMD.vload(V8, x, i0)
            b = SIMD.vload(V8, x, i0 + Q)
            c = SIMD.vload(V8, x, i0 + 2Q)
            u = gf_mulv(gf_subv(b, c), ω3v)
            SIMD.vstore(gf_addv(a, gf_addv(b, c)), x, i0)
            y1 = gf_addv(gf_subv(a, c), u)
            y2 = gf_subv(gf_subv(a, b), u)
            SIMD.vstore(gf_mulv(y1, SIMD.vload(V8, w1, j + 1)), x, i0 + Q)
            SIMD.vstore(gf_mulv(y2, SIMD.vload(V8, w2, j + 1)), x, i0 + 2Q)
            j += 8
        end
    end
    @inbounds while j < Q
        i0 = o + j + 1
        a = x[i0]
        b = x[i0+Q]
        c = x[i0+2Q]
        u = gf_mul(gf_sub(b, c), ω3)
        x[i0] = gf_add(a, gf_add(b, c))
        x[i0+Q] = gf_mul(gf_add(gf_sub(a, c), u), w1[j+1])
        x[i0+2Q] = gf_mul(gf_sub(gf_sub(a, b), u), w2[j+1])
        j += 1
    end
    return x
end

function radix3_inv!(x::Vector{UInt64}, o::Int, st::OddStage)
    Q = st.Q
    λ3 = st.rot[1]
    w1, w2 = st.tw[1], st.tw[2]
    j = 0
    if Q >= 8
        λ3v = V8(λ3)
        @inbounds while j + 8 <= Q
            i0 = o + j + 1
            t0 = SIMD.vload(V8, x, i0)
            t1 = gf_mulv(SIMD.vload(V8, x, i0 + Q), SIMD.vload(V8, w1, j + 1))
            t2 = gf_mulv(SIMD.vload(V8, x, i0 + 2Q), SIMD.vload(V8, w2, j + 1))
            u = gf_mulv(gf_subv(t1, t2), λ3v)
            SIMD.vstore(gf_addv(t0, gf_addv(t1, t2)), x, i0)
            SIMD.vstore(gf_addv(gf_subv(t0, t2), u), x, i0 + Q)
            SIMD.vstore(gf_subv(gf_subv(t0, t1), u), x, i0 + 2Q)
            j += 8
        end
    end
    @inbounds while j < Q
        i0 = o + j + 1
        t0 = x[i0]
        t1 = gf_mul(x[i0+Q], w1[j+1])
        t2 = gf_mul(x[i0+2Q], w2[j+1])
        u = gf_mul(gf_sub(t1, t2), λ3)
        x[i0] = gf_add(t0, gf_add(t1, t2))
        x[i0+Q] = gf_add(gf_sub(t0, t2), u)
        x[i0+2Q] = gf_sub(gf_sub(t0, t1), u)
        j += 1
    end
    return x
end

# order-5 DFT combine: given t0..t4 (inputs, twiddles already applied for
# the inverse) produce Σ_t c^(r·t)·t_t for r = 0..4, with c powers in rot
@inline function dft5(t0, t1, t2, t3, t4, c1, c2, c3, c4,
                      addf::F, mulf::G) where {F,G}
    s = addf(addf(t1, t2), addf(t3, t4))
    y0 = addf(t0, s)
    y1 = addf(t0, addf(addf(mulf(t1, c1), mulf(t2, c2)),
                       addf(mulf(t3, c3), mulf(t4, c4))))
    y2 = addf(t0, addf(addf(mulf(t1, c2), mulf(t2, c4)),
                       addf(mulf(t3, c1), mulf(t4, c3))))
    y3 = addf(t0, addf(addf(mulf(t1, c3), mulf(t2, c1)),
                       addf(mulf(t3, c4), mulf(t4, c2))))
    y4 = addf(t0, addf(addf(mulf(t1, c4), mulf(t2, c3)),
                       addf(mulf(t3, c2), mulf(t4, c1))))
    return y0, y1, y2, y3, y4
end

function radix5_fwd!(x::Vector{UInt64}, o::Int, st::OddStage)
    Q = st.Q
    c1, c2, c3, c4 = st.rot[1], st.rot[2], st.rot[3], st.rot[4]
    w1, w2, w3, w4 = st.tw[1], st.tw[2], st.tw[3], st.tw[4]
    j = 0
    if Q >= 8
        v1, v2, v3, v4 = V8(c1), V8(c2), V8(c3), V8(c4)
        @inbounds while j + 8 <= Q
            i0 = o + j + 1
            a = SIMD.vload(V8, x, i0)
            b = SIMD.vload(V8, x, i0 + Q)
            c = SIMD.vload(V8, x, i0 + 2Q)
            d = SIMD.vload(V8, x, i0 + 3Q)
            e = SIMD.vload(V8, x, i0 + 4Q)
            y0, y1, y2, y3, y4 = dft5(a, b, c, d, e, v1, v2, v3, v4,
                                      gf_addv, gf_mulv)
            SIMD.vstore(y0, x, i0)
            SIMD.vstore(gf_mulv(y1, SIMD.vload(V8, w1, j + 1)), x, i0 + Q)
            SIMD.vstore(gf_mulv(y2, SIMD.vload(V8, w2, j + 1)), x, i0 + 2Q)
            SIMD.vstore(gf_mulv(y3, SIMD.vload(V8, w3, j + 1)), x, i0 + 3Q)
            SIMD.vstore(gf_mulv(y4, SIMD.vload(V8, w4, j + 1)), x, i0 + 4Q)
            j += 8
        end
    end
    @inbounds while j < Q
        i0 = o + j + 1
        y0, y1, y2, y3, y4 = dft5(x[i0], x[i0+Q], x[i0+2Q], x[i0+3Q], x[i0+4Q],
                                  c1, c2, c3, c4, gf_add, gf_mul)
        x[i0] = y0
        x[i0+Q] = gf_mul(y1, w1[j+1])
        x[i0+2Q] = gf_mul(y2, w2[j+1])
        x[i0+3Q] = gf_mul(y3, w3[j+1])
        x[i0+4Q] = gf_mul(y4, w4[j+1])
        j += 1
    end
    return x
end

function radix5_inv!(x::Vector{UInt64}, o::Int, st::OddStage)
    Q = st.Q
    c1, c2, c3, c4 = st.rot[1], st.rot[2], st.rot[3], st.rot[4]
    w1, w2, w3, w4 = st.tw[1], st.tw[2], st.tw[3], st.tw[4]
    j = 0
    if Q >= 8
        v1, v2, v3, v4 = V8(c1), V8(c2), V8(c3), V8(c4)
        @inbounds while j + 8 <= Q
            i0 = o + j + 1
            t0 = SIMD.vload(V8, x, i0)
            t1 = gf_mulv(SIMD.vload(V8, x, i0 + Q), SIMD.vload(V8, w1, j + 1))
            t2 = gf_mulv(SIMD.vload(V8, x, i0 + 2Q), SIMD.vload(V8, w2, j + 1))
            t3 = gf_mulv(SIMD.vload(V8, x, i0 + 3Q), SIMD.vload(V8, w3, j + 1))
            t4 = gf_mulv(SIMD.vload(V8, x, i0 + 4Q), SIMD.vload(V8, w4, j + 1))
            y0, y1, y2, y3, y4 = dft5(t0, t1, t2, t3, t4, v1, v2, v3, v4,
                                      gf_addv, gf_mulv)
            SIMD.vstore(y0, x, i0)
            SIMD.vstore(y1, x, i0 + Q)
            SIMD.vstore(y2, x, i0 + 2Q)
            SIMD.vstore(y3, x, i0 + 3Q)
            SIMD.vstore(y4, x, i0 + 4Q)
            j += 8
        end
    end
    @inbounds while j < Q
        i0 = o + j + 1
        t0 = x[i0]
        t1 = gf_mul(x[i0+Q], w1[j+1])
        t2 = gf_mul(x[i0+2Q], w2[j+1])
        t3 = gf_mul(x[i0+3Q], w3[j+1])
        t4 = gf_mul(x[i0+4Q], w4[j+1])
        y0, y1, y2, y3, y4 = dft5(t0, t1, t2, t3, t4, c1, c2, c3, c4,
                                  gf_add, gf_mul)
        x[i0] = y0
        x[i0+Q] = y1
        x[i0+2Q] = y2
        x[i0+3Q] = y3
        x[i0+4Q] = y4
        j += 1
    end
    return x
end

# ---------------------------------------------------------------------------
# Small-q pow2 stages (q in (1,2,4)): butterfly partners sit closer together
# than a vector, so 32 consecutive elements (four V8 loads) are shuffled
# into quarter-role vectors a,b,c,d, put through the ordinary vector
# butterfly, and shuffled back.  The twiddle tables' repeated 8-entry
# patterns line up with the gathered lane order.

# Lane l of role t lives at global position (l÷Q)·4Q + t·Q + l%Q within the
# 32 elements; scatter applies the inverse permutation.  The generators
# concatenate the inputs into two Vec{16}s and emit one index-tuple shuffle
# per output vector; LLVM folds the shuffle trees into the same machine
# shuffles as hand-written per-Q versions.
const IOTA16 = ntuple(i -> i - 1, 16)

@generated function ntt_gather4(::Val{Q}, v0::V8, v1::V8, v2::V8, v3::V8) where {Q}
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

@generated function ntt_scatter4(::Val{Q}, a::V8, b::V8, c::V8, d::V8) where {Q}
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

function ntt_smallq_fwd!(x::Vector{UInt64}, o::Int, N2::Int, ::Val{Q},
                         stg::NttStage) where {Q}
    vw1 = SIMD.vload(V8, stg.w1, 1)
    vw2 = SIMD.vload(V8, stg.w2, 1)
    vw3 = SIMD.vload(V8, stg.w3, 1)
    @inbounds for i0 in o+1:32:o+N2-31
        v0 = SIMD.vload(V8, x, i0)
        v1 = SIMD.vload(V8, x, i0 + 8)
        v2 = SIMD.vload(V8, x, i0 + 16)
        v3 = SIMD.vload(V8, x, i0 + 24)
        a, b, c, d = ntt_gather4(Val(Q), v0, v1, v2, v3)
        apc = gf_addv(a, c)
        amc = gf_subv(a, c)
        bpd = gf_addv(b, d)
        ibmd = gf_mul_iv(gf_subv(b, d))
        y0 = gf_addv(apc, bpd)
        y1 = gf_mulv(gf_addv(amc, ibmd), vw1)
        y2 = gf_mulv(gf_subv(apc, bpd), vw2)
        y3 = gf_mulv(gf_subv(amc, ibmd), vw3)
        o0, o1, o2, o3 = ntt_scatter4(Val(Q), y0, y1, y2, y3)
        SIMD.vstore(o0, x, i0)
        SIMD.vstore(o1, x, i0 + 8)
        SIMD.vstore(o2, x, i0 + 16)
        SIMD.vstore(o3, x, i0 + 24)
    end
    return x
end

function ntt_smallq_inv!(x::Vector{UInt64}, o::Int, N2::Int, ::Val{Q},
                         stg::NttStage) where {Q}
    vw1 = SIMD.vload(V8, stg.w1, 1)
    vw2 = SIMD.vload(V8, stg.w2, 1)
    vw3 = SIMD.vload(V8, stg.w3, 1)
    @inbounds for i0 in o+1:32:o+N2-31
        v0 = SIMD.vload(V8, x, i0)
        v1 = SIMD.vload(V8, x, i0 + 8)
        v2 = SIMD.vload(V8, x, i0 + 16)
        v3 = SIMD.vload(V8, x, i0 + 24)
        y0, y1, y2, y3 = ntt_gather4(Val(Q), v0, v1, v2, v3)
        t1 = gf_mulv(y1, vw1)
        t2 = gf_mulv(y2, vw2)
        t3 = gf_mulv(y3, vw3)
        u = gf_addv(y0, t2)
        p = gf_subv(y0, t2)
        v = gf_addv(t1, t3)
        w = gf_mul_iv(gf_subv(t1, t3))   # i^-1 == -i: add/sub swapped
        o0, o1, o2, o3 = ntt_scatter4(Val(Q), gf_addv(u, v), gf_subv(p, w),
                                      gf_subv(u, v), gf_addv(p, w))
        SIMD.vstore(o0, x, i0)
        SIMD.vstore(o1, x, i0 + 8)
        SIMD.vstore(o2, x, i0 + 16)
        SIMD.vstore(o3, x, i0 + 24)
    end
    return x
end

# ---------------------------------------------------------------------------
# Power-of-two pipeline over one block of length N2 at offset o: radix-4
# Gentleman–Sande DIF, natural order in, digit-reversed out.  Block of L
# splits into quarters a,b,c,d; with i = ω2^(N2/4) == 2^48 and
# w = ω2^(jN2/L) the outputs are
#   y0 = (a+c) + (b+d)          y2 = ((a+c) - (b+d))·w²
#   y1 = ((a-c) + i(b-d))·w     y3 = ((a-c) - i(b-d))·w³
# If log2(N2) is odd a final twiddle-free radix-2 stage remains.  The output
# permutation is only ever consumed by the inverse, which mirrors the stage
# sequence exactly, so its precise form doesn't matter.
function ntt_fwd_pow2!(x::Vector{UInt64}, o::Int, plan::NttPlan)
    N2 = plan.N2
    L = N2
    for stg in plan.fwd
        q = stg.q
        L = 4q
        w1, w2, w3 = stg.w1, stg.w2, stg.w3
        if q >= 8
            for s in o:L:o+N2-1
                @inbounds for j in 0:8:q-8
                    i0 = s + j + 1
                    a = SIMD.vload(V8, x, i0)
                    b = SIMD.vload(V8, x, i0 + q)
                    c = SIMD.vload(V8, x, i0 + 2q)
                    d = SIMD.vload(V8, x, i0 + 3q)
                    apc = gf_addv(a, c)
                    amc = gf_subv(a, c)
                    bpd = gf_addv(b, d)
                    ibmd = gf_mul_iv(gf_subv(b, d))
                    vw1 = SIMD.vload(V8, w1, j + 1)
                    vw2 = SIMD.vload(V8, w2, j + 1)
                    vw3 = SIMD.vload(V8, w3, j + 1)
                    SIMD.vstore(gf_addv(apc, bpd), x, i0)
                    SIMD.vstore(gf_mulv(gf_addv(amc, ibmd), vw1), x, i0 + q)
                    SIMD.vstore(gf_mulv(gf_subv(apc, bpd), vw2), x, i0 + 2q)
                    SIMD.vstore(gf_mulv(gf_subv(amc, ibmd), vw3), x, i0 + 3q)
                end
            end
        elseif N2 >= 32
            q == 4 ? ntt_smallq_fwd!(x, o, N2, Val(4), stg) :
            q == 2 ? ntt_smallq_fwd!(x, o, N2, Val(2), stg) :
                     ntt_smallq_fwd!(x, o, N2, Val(1), stg)
        else
            for s in o:L:o+N2-1
                @inbounds for j in 0:q-1
                    a = x[s+j+1]
                    b = x[s+j+q+1]
                    c = x[s+j+2q+1]
                    d = x[s+j+3q+1]
                    apc = gf_add(a, c)
                    amc = gf_sub(a, c)
                    bpd = gf_add(b, d)
                    ibmd = gf_mul_i(gf_sub(b, d))
                    x[s+j+1] = gf_add(apc, bpd)
                    x[s+j+q+1] = gf_mul(gf_add(amc, ibmd), w1[j+1])
                    x[s+j+2q+1] = gf_mul(gf_sub(apc, bpd), w2[j+1])
                    x[s+j+3q+1] = gf_mul(gf_sub(amc, ibmd), w3[j+1])
                end
            end
        end
        L = q
    end
    if L == 2
        # leftover radix-2 stage: the only twiddle is ω^0 == 1, no multiply
        @inbounds for s in o:2:o+N2-1
            u = x[s+1]
            v = x[s+2]
            x[s+1] = gf_add(u, v)
            x[s+2] = gf_sub(u, v)
        end
    end
    return x
end

# Inverse pow2 pipeline for one block: radix-4 Cooley–Tukey DIT, consuming
# the forward's digit-reversed order, natural order out, scaled by N^-1
# (the full N, so the odd stages come out unscaled).  Undoes each forward
# stage in reverse order; the stage factors 4/2 (and m from the odd stages)
# multiply to N and are cancelled by folding N^-1 into the first stage,
# whose twiddles are trivial.  i^-1 == -i, so the rotation stays gf_mul_i
# with the following add/sub pair swapped.
function ntt_inv_pow2!(x::Vector{UInt64}, o::Int, plan::NttPlan)
    N2 = plan.N2
    ninv = plan.ninv
    if isodd(trailing_zeros(N2))
        # leftover radix-2 stage first, with N^-1 folded in
        @inbounds for s in o:2:o+N2-1
            u = x[s+1]
            t = x[s+2]
            x[s+1] = gf_mul(gf_add(u, t), ninv)
            x[s+2] = gf_mul(gf_sub(u, t), ninv)
        end
    elseif N2 >= 4
        # first radix-4 stage (L == 4, all twiddles 1), N^-1 folded into
        # the stage inputs
        @inbounds for s in o:4:o+N2-1
            t0 = gf_mul(x[s+1], ninv)
            t1 = gf_mul(x[s+2], ninv)
            t2 = gf_mul(x[s+3], ninv)
            t3 = gf_mul(x[s+4], ninv)
            u = gf_add(t0, t2)
            p = gf_sub(t0, t2)
            v = gf_add(t1, t3)
            w = gf_mul_i(gf_sub(t1, t3))
            x[s+1] = gf_add(u, v)
            x[s+2] = gf_sub(p, w)
            x[s+3] = gf_sub(u, v)
            x[s+4] = gf_add(p, w)
        end
    else
        # N2 == 1 or 2 with even log2: only N2 == 1, scale alone
        @inbounds for s in o+1:o+N2
            x[s] = gf_mul(x[s], ninv)
        end
    end
    for stg in plan.inv
        q = stg.q
        L = 4q
        w1, w2, w3 = stg.w1, stg.w2, stg.w3
        if q >= 8
            for s in o:L:o+N2-1
                @inbounds for j in 0:8:q-8
                    i0 = s + j + 1
                    vw1 = SIMD.vload(V8, w1, j + 1)
                    vw2 = SIMD.vload(V8, w2, j + 1)
                    vw3 = SIMD.vload(V8, w3, j + 1)
                    t0 = SIMD.vload(V8, x, i0)
                    t1 = gf_mulv(SIMD.vload(V8, x, i0 + q), vw1)
                    t2 = gf_mulv(SIMD.vload(V8, x, i0 + 2q), vw2)
                    t3 = gf_mulv(SIMD.vload(V8, x, i0 + 3q), vw3)
                    u = gf_addv(t0, t2)
                    p = gf_subv(t0, t2)
                    v = gf_addv(t1, t3)
                    w = gf_mul_iv(gf_subv(t1, t3))   # i^-1 == -i
                    SIMD.vstore(gf_addv(u, v), x, i0)
                    SIMD.vstore(gf_subv(p, w), x, i0 + q)
                    SIMD.vstore(gf_subv(u, v), x, i0 + 2q)
                    SIMD.vstore(gf_addv(p, w), x, i0 + 3q)
                end
            end
        elseif N2 >= 32
            q == 4 ? ntt_smallq_inv!(x, o, N2, Val(4), stg) :
            q == 2 ? ntt_smallq_inv!(x, o, N2, Val(2), stg) :
                     ntt_smallq_inv!(x, o, N2, Val(1), stg)
        else
            for s in o:L:o+N2-1
                @inbounds for j in 0:q-1
                    t0 = x[s+j+1]
                    t1 = gf_mul(x[s+j+q+1], w1[j+1])
                    t2 = gf_mul(x[s+j+2q+1], w2[j+1])
                    t3 = gf_mul(x[s+j+3q+1], w3[j+1])
                    u = gf_add(t0, t2)
                    p = gf_sub(t0, t2)
                    v = gf_add(t1, t3)
                    w = gf_mul_i(gf_sub(t1, t3))   # i^-1 == -i
                    x[s+j+1] = gf_add(u, v)
                    x[s+j+q+1] = gf_sub(p, w)
                    x[s+j+2q+1] = gf_sub(u, v)
                    x[s+j+3q+1] = gf_add(p, w)
                end
            end
        end
    end
    return x
end

function ntt_fwd!(x::Vector{UInt64}, plan::NttPlan)
    for st in plan.oddf
        for o in 0:st.span:plan.N-1
            st.m == 3 ? radix3_fwd!(x, o, st) : radix5_fwd!(x, o, st)
        end
    end
    for o in 0:plan.N2:plan.N-1
        ntt_fwd_pow2!(x, o, plan)
    end
    return x
end

function ntt_inv!(x::Vector{UInt64}, plan::NttPlan)
    for o in 0:plan.N2:plan.N-1
        ntt_inv_pow2!(x, o, plan)
    end
    for st in plan.oddi
        for o in 0:st.span:plan.N-1
            st.m == 3 ? radix3_inv!(x, o, st) : radix5_inv!(x, o, st)
        end
    end
    return x
end

# ---------------------------------------------------------------------------
# smallest supported transform length >= T: m·2^k, m in (1, 3, 5, 15), k >= 2
function ntt_len(T::Int)
    best = nextpow(2, max(T, 4))
    for m in (3, 5, 15)
        c = m * nextpow(2, max(cld(T, m), 4))
        c < best && (best = c)
    end
    return best
end

# Chunk width b and transform length N for multiplying magnitudes of the
# given bit lengths.  Each output coefficient sums at most min(nca, ncb)
# chunk products of (2^b-1)^2 each; exactness needs that to stay below p.
# The largest workable b minimizes chunk count and thus transform length
# (the linear convolution has nca + ncb - 1 coefficients).  Once some b
# passes, all smaller b do too (min(nca,ncb) grows slower than (2^b-1)^2
# shrinks), so the first hit is the largest feasible b.
function ntt_params(bits_a::Int, bits_b::Int)
    for b in 32:-1:1
        nca = cld(bits_a, b)
        ncb = cld(bits_b, b)
        if UInt128(min(nca, ncb)) * (UInt128(2)^b - 1)^2 < GF_P
            return b, ntt_len(nca + ncb - 1)
        end
    end
    error("unreachable: b == 1 always satisfies the bound for supported sizes")
end

# split the first `nch` b-bit chunks of an n-limb magnitude into a
# zero-padded length-N coefficient vector.  The main loop reads a two-limb
# window branch-free (Julia defines x << 64 == 0, so sh == 0 needs no
# special case); only the last few chunks, where limbs[w+2] may not exist,
# take the guarded path.
function ntt_pack(limbs::Memory{Limb}, n::Int, b::Int, nch::Int, N::Int)
    x = zeros(UInt64, N)
    mask = (UInt64(1) << b) - 1
    imax = min(nch - 1, (64 * (n - 1) - 1) ÷ b)   # b*i < 64(n-1) ⟹ w+2 <= n
    @inbounds for i in 0:imax
        bit = i * b
        w = bit >> 6
        sh = bit & 63
        c = (limbs[w+1] >>> sh) | (limbs[w+2] << (64 - sh))
        x[i+1] = c & mask
    end
    @inbounds for i in imax+1:nch-1
        bit = i * b
        w = bit >> 6
        sh = bit & 63
        c = limbs[w+1] >>> sh
        if sh + b > 64 && w + 2 <= n
            c |= limbs[w+2] << (64 - sh)
        end
        x[i+1] = c & mask
    end
    return x
end

# Accumulate coefficients x[1:nconv] (each < 2^63) into r as Σ x[i+1]·2^(b·i).
# Coefficients arrive in increasing bit order, so a streaming 128-bit
# accumulator absorbs all carries: between flushes the pending contributions
# total < 2^128 (each added term is < 2^(63+64) and successive terms shift
# up by b), and each iteration needs at most one flush since b <= 32 < 64.
function ntt_unpack!(r::Memory{Limb}, rn::Int, x::Vector{UInt64}, nconv::Int, b::Int)
    acc = UInt128(0)
    outw = 1
    outbit = 0
    @inbounds for i in 0:nconv-1
        s = i * b - outbit
        if s >= 64
            r[outw] = acc % UInt64
            acc >>= 64
            outw += 1
            outbit += 64
            s -= 64
        end
        acc += UInt128(x[i+1]) << s
    end
    # drain the accumulator and zero-fill the rest (the product fits rn limbs)
    @inbounds while outw <= rn
        r[outw] = acc % UInt64
        acc >>= 64
        outw += 1
    end
    return r
end

# ---------------------------------------------------------------------------
# Dispatch thresholds for Base.:* (benchmark-tuned via bench/bench_ntt.jl):
# the NTT beats Karatsuba from ~800 balanced limbs, and for unbalanced
# operands only once the smaller one is substantial (Karatsuba's unbalanced
# path is ~max·min^0.585 while the NTT pays for the combined length).
const NTT_MUL_MIN = 256    # smaller operand at least this many limbs
const NTT_MUL_SUM = 1792   # combined limb count at least this

# Multiply via the Goldilocks NTT; falls back to `*` below NTT sizes.
function ntt_mul(a::NBig, b::NBig)
    (iszero(a) || iszero(b)) && return NBig(0, EMPTY_LIMBS)
    la, lb = nlimbs(a), nlimbs(b)
    (la < 16 || lb < 16) && return a * b
    bits_a = 64 * (la - 1) + Base.top_set_bit(@inbounds a.limbs[la])
    bits_b = 64 * (lb - 1) + Base.top_set_bit(@inbounds b.limbs[lb])
    bch, N = ntt_params(bits_a, bits_b)
    plan = ntt_plan(N)
    nca, ncb = cld(bits_a, bch), cld(bits_b, bch)
    xa = ntt_pack(a.limbs, la, bch, nca, N)
    xb = ntt_pack(b.limbs, lb, bch, ncb, N)
    ntt_fwd!(xa, plan)
    ntt_fwd!(xb, plan)
    n = length(xa)
    i = 1
    if n >= 8
        @inbounds while i + 7 <= n
            SIMD.vstore(gf_mulv(SIMD.vload(V8, xa, i), SIMD.vload(V8, xb, i)), xa, i)
            i += 8
        end
    end
    @inbounds while i <= n
        xa[i] = gf_mul(xa[i], xb[i])
        i += 1
    end
    ntt_inv!(xa, plan)
    rn = la + lb
    r = Memory{Limb}(undef, rn)
    ntt_unpack!(r, rn, xa, nca + ncb - 1, bch)
    return nbig_from_limbs(sign(a) * sign(b), r, rn)
end

# Square via the Goldilocks NTT: one forward transform instead of two.
# Returns the (nonnegative) square of the magnitude.
function ntt_square(a::NBig)
    iszero(a) && return NBig(0, EMPTY_LIMBS)
    la = nlimbs(a)
    la < 16 && return a * a
    bits = 64 * (la - 1) + Base.top_set_bit(@inbounds a.limbs[la])
    bch, N = ntt_params(bits, bits)
    plan = ntt_plan(N)
    nca = cld(bits, bch)
    xa = ntt_pack(a.limbs, la, bch, nca, N)
    ntt_fwd!(xa, plan)
    n = length(xa)
    i = 1
    if n >= 8
        @inbounds while i + 7 <= n
            v = SIMD.vload(V8, xa, i)
            SIMD.vstore(gf_mulv(v, v), xa, i)
            i += 8
        end
    end
    @inbounds while i <= n
        xa[i] = gf_mul(xa[i], xa[i])
        i += 1
    end
    ntt_inv!(xa, plan)
    rn = 2la
    r = Memory{Limb}(undef, rn)
    ntt_unpack!(r, rn, xa, 2nca - 1, bch)
    return nbig_from_limbs(1, r, rn)
end
