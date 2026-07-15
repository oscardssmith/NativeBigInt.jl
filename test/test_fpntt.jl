# fp NTT multiplication: Float64 field arithmetic, transform, and the
# two-prime CRT engine mul_fpntt2!/sqr_fpntt2!.  The engine's correctness
# rests on rounding-error analysis, so beyond the differential net these
# tests hammer the documented lazy-range bounds (mulmod |x| <= 4p, reduce
# |x| <= 8p) where random in-range inputs would never stress the
# magic-constant round.
using NativeBigInt: FP_CTX1, FpCtx, fp_prime, fp_mulmod, fp_mulmod2,
                    fp_reduce, fp_round, fp_ntt_plan,
                    fp_ntt_fwd!, fp_ntt_rev!, VF8,
                    Limb, nlimbs, nbig_from_limbs
using Random: MersenneTwister

const FP_PI = fp_prime(FP_CTX1)
const FP_P = Float64(fp_prime(FP_CTX1))

# canonical residue in [0, p) of a balanced-representation value
fp_canon(x::Float64) = (v = fp_reduce(x, FP_CTX1); v < 0 && (v += FP_P); UInt64(v))

@testset "fp field arithmetic" begin
    P = big(FP_PI)
    @test FP_PI == 0x0001_FFFE_0000_0001          # 2^49 - 2^33 + 1
    @test Float64(FP_PI) == FP_P                   # p is exactly representable
    @test (FP_PI - 1) % (UInt64(255) << 33) == 0   # 255·2^33 length family

    rng = MersenneTwister(0xf94)
    edgew = UInt64[0, 1, 2, FP_PI - 1, FP_PI - 2, UInt64(2)^33, UInt64(2)^48]
    ws = [edgew; rand(rng, UInt64(0):FP_PI-1, 60)]
    # x sweeps the full lazy range ±4p, including the extremes where the
    # magic round and the q-error bound are tight
    lazy_edges = Float64[0.0, 1.0, -1.0, FP_P - 1, -(FP_P - 1), FP_P, -FP_P,
                         2FP_P, -2FP_P, 4FP_P - 4, -(4FP_P - 4), 4FP_P, -4FP_P]
    xs = [lazy_edges;
          [rand(rng, (-4.0, -2.0, -1.0, 1.0, 2.0, 4.0)) * rand(rng, UInt64(0):FP_PI-1)
           for _ in 1:200]]
    for w in ws, x in xs
        r = fp_mulmod(x, Float64(w), Float64(w) / FP_P, FP_CTX1)
        @test abs(r) < FP_P
        @test big(fp_canon(r)) == mod(big(w) * big(Int128(x)), P)
    end
    for x in xs, y in xs
        abs(x) <= 2FP_P && abs(y) <= 2FP_P || continue
        r = fp_mulmod2(x, y, FP_CTX1)
        @test abs(r) < FP_P
        @test big(fp_canon(r)) == mod(big(Int128(x)) * big(Int128(y)), P)
    end
    for x in xs
        r = fp_reduce(2 * x, FP_CTX1)              # up to ±8p
        @test abs(r) < FP_P
        @test big(fp_canon(r)) == mod(2 * big(Int128(x)), P)
    end
    # generator-derived roots of unity have exact order
    for logN in (1, 5, 20, 33)
        N = UInt64(2)^logN
        ω = powermod(NativeBigInt.fp_generator(FP_CTX1), (FP_PI - 1) ÷ N, FP_PI)
        @test powermod(ω, N, FP_PI) == 1
        @test powermod(ω, N >> 1, FP_PI) == FP_PI - 1
    end
end

@testset "fp vectorized ops match scalar" begin
    rng = MersenneTwister(0x51e)
    for trial in 1:300
        x = [rand(rng, (-4.0, -1.0, 1.0, 4.0)) * rand(rng, UInt64(0):FP_PI-1) for _ in 1:8]
        w = rand(rng, UInt64(0):FP_PI-1, 8)
        wf = Float64.(w)
        wp = wf ./ FP_P
        vr = fp_mulmod(VF8(Tuple(x)), VF8(Tuple(wf)), VF8(Tuple(wp)), FP_CTX1)
        @test [vr[i] for i in 1:8] == fp_mulmod.(x, wf, wp, Ref(FP_CTX1))
        vr2 = fp_reduce(VF8(Tuple(x)), FP_CTX1)
        @test [vr2[i] for i in 1:8] == fp_reduce.(x, Ref(FP_CTX1))
    end
end

@testset "fp ntt transform" begin
    rng = MersenneTwister(0xf18)
    for N in (4, 8, 64, 512, 1024, 12, 20, 48, 96, 160, 240, 1536,
              68, 136, 204, 340, 1020, 2176)   # 17-family: 17,51,85,255 odd parts
        plan = fp_ntt_plan(N, FP_CTX1)
        x = Float64.(rand(rng, UInt64(0):FP_PI-1, N))
        y = copy(x)
        fp_ntt_fwd!(y, plan)
        @test y != x
        # the reverse transform is the transposed forward, so the round trip returns the
        # input scaled by N and index-reversed
        fp_ntt_rev!(y, plan)
        want = [UInt64(mod(big(N) * big(UInt64(x[mod(N - t, N)+1])), big(FP_PI)))
                for t in 0:N-1]
        @test fp_canon.(y) == want
        N > 256 && continue
        a = rand(rng, UInt64(0):FP_PI-1, N)
        b = rand(rng, UInt64(0):FP_PI-1, N)
        c = big.(zeros(Int, N))
        for i in 0:N-1, j in 0:N-1
            c[mod(i + j, N) + 1] += big(a[i+1]) * big(b[j+1])
        end
        fa, fb = Float64.(a), Float64.(b)
        fp_ntt_fwd!(fa, plan); fp_ntt_fwd!(fb, plan)
        fc = fp_mulmod2.(fa, fb, Ref(FP_CTX1))
        fp_ntt_rev!(fc, plan)
        # recover c[j] = ninv · rev(Ĉ)[(N-j) mod N], as the unpack does
        got = [big(fp_canon(fp_mulmod(fc[mod(N - j, N)+1], plan.ninv, plan.ninvp,
                                      FP_CTX1)))
               for j in 0:N-1]
        @test got == mod.(c, big(FP_PI))
    end
end

# adversarial magnitude in [2^(64(n-1)), 2^64n): uniform, all-ones, 2^k ± 1
function fpntt_randbig(rng, n::Int)
    n == 0 && return big(0)
    style = rand(rng, 1:3)
    mag = if style == 1
        rand(rng, big(2)^(64 * (n - 1)):big(2)^(64n)-1)
    elseif style == 2
        big(2)^(64n) - 1
    else
        big(2)^rand(rng, (64 * (n - 1)):(64n - 1)) + rand(rng, -1:1)
    end
    return rand(rng, Bool) ? -mag : mag
end

@testset "fpntt * integration" begin
    rng = MersenneTwister(0x5f2e)
    # `*` dispatches to the fp NTT above the thresholds; sizes straddle
    # MUL_FPNTT_THRESHOLD = SQR_FPNTT_THRESHOLD = 224
    for (na, nb) in ((220, 220), (224, 224), (500, 300), (1200, 800), (5000, 4000))
        a = fpntt_randbig(rng, na)
        b = fpntt_randbig(rng, nb)
        @test BigInt(NBig(a) * NBig(b)) == a * b
    end
    for n in (220, 224, 230, 1024, 3000)
        a = fpntt_randbig(rng, n)
        x = NBig(a)
        @test BigInt(x * x) == a^2
        @test BigInt(x^2) == a^2
        @test BigInt(x * -x) == -(a^2)
    end
end

@testset "mpn-level fpntt dispatch" begin
    rng = MersenneTwister(0x3f17)
    a = abs(fpntt_randbig(rng, 400))
    @test BigInt(NBig(a)^3) == a^3
    d = abs(fpntt_randbig(rng, 1000))
    @test BigInt(isqrt(NBig(d))) == isqrt(d)
end

# --------------------------------------------------------------------------
# Two-prime CRT engine

using NativeBigInt: FP_CTX2, FP_P1INV2, mul_fpntt2!, sqr_fpntt2!,
                    fp_ntt_unpack2!, fp_ntt_params2, ntt_len, two_adicity

const FP_PI2 = fp_prime(FP_CTX2)

fp_canonc(x::Float64, F) = (v = fp_reduce(x, F); v < 0 && (v += Float64(fp_prime(F))); UInt64(v))

function fpntt2_mul(a::NBig, b::NBig)
    (iszero(a) || iszero(b)) && return NBig(0)
    la, lb = nlimbs(a), nlimbs(b)
    r = Memory{Limb}(undef, la + lb)
    mul_fpntt2!(r, 0, a.limbs, 0, la, b.limbs, 0, lb)
    return nbig_from_limbs(sign(a) * sign(b), r, la + lb)
end

function fpntt2_square(a::NBig)
    iszero(a) && return NBig(0)
    la = nlimbs(a)
    r = Memory{Limb}(undef, 2la)
    sqr_fpntt2!(r, 0, a.limbs, 0, la)
    return nbig_from_limbs(1, r, 2la)
end

@testset "fp second prime and transform" begin
    @test FP_PI2 == 0x0001_FE00_0000_0001            # 255·2^41 + 1
    @test Float64(FP_PI2) == Float64(fp_prime(FP_CTX2))   # exactly representable
    @test FP_PI2 - 1 == UInt64(255) << 41
    @test (FP_PI2 - 1) % (UInt64(255) << 41) == 0    # 255·2^41 length family
    @test UInt128(FP_P1INV2) * (FP_PI % FP_PI2) % FP_PI2 == 1   # Garner inverse

    rng = MersenneTwister(0x2b1)
    for N in (4, 8, 64, 512, 12, 20, 48, 240, 1536, 68, 204, 340, 1020)
        plan = fp_ntt_plan(N, FP_CTX2)
        x = Float64.(rand(rng, UInt64(0):FP_PI2-1, N))
        y = copy(x)
        fp_ntt_fwd!(y, plan)
        @test y != x
        fp_ntt_rev!(y, plan)
        want = [UInt64(mod(big(N) * big(UInt64(x[mod(N - t, N)+1])), big(FP_PI2)))
                for t in 0:N-1]
        @test fp_canonc.(y, Ref(FP_CTX2)) == want
    end
end

@testset "ntt_len length selection and 2-adic cap" begin
    ctxs = (FP_CTX1, FP_CTX2)
    maxk = min(two_adicity(FP_CTX1), two_adicity(FP_CTX2))
    @test maxk == 33                                  # p1's 2-adicity is the binding cap

    # brute-force reference: smallest m·2^k >= T over the family, honoring
    # the uniform k >= 14 floor for the expensive odd multipliers
    function want(T)
        best = typemax(Int)
        for m in (1, 3, 5, 15, 17, 51, 85, 255), k in 2:maxk
            m <= 5 || k >= 14 || continue
            c = m << k
            c >= T && c < best && (best = c)
        end
        best
    end

    for T in (1, 4, 5, 100, 1000, 1023, 1025, 13000, 15359, 15360, 20000, 1<<20,
              (17<<14) - 100, 17<<14, (17<<14) + 1, (51<<14) - 1, 51<<14,
              (85<<14) - 1, 85<<14, (255<<14) - 1, 255<<14, 1<<22, 1<<25)
        n = ntt_len(T, ctxs...)
        @test n == want(T)
        @test n >= T
        @test trailing_zeros(n) <= maxk              # 2^k divides p-1 for both primes
        @test (fp_prime(FP_CTX1) - 1) % n == 0
        @test (fp_prime(FP_CTX2) - 1) % n == 0
    end
    # floor boundary: m·2^14 is the first admitted length of each odd
    # multiplier; at k = 13 the pow2/small-odd family is chosen instead
    @test ntt_len((17 << 14) - 100, ctxs...) == 17 << 14
    @test ntt_len(15 << 14, ctxs...) == 15 << 14
    @test ntt_len(15 << 13, ctxs...) == 1 << 17      # 15·2^13 is below the floor

    # just past a pure 2^33: must fall back to an odd multiplier, never
    # exceed the primes' shared 2-adicity
    n = ntt_len((1 << 33) + 1, ctxs...)
    @test n >= (1 << 33) + 1 && trailing_zeros(n) <= maxk
    @test n == 17 << 29                              # smallest admissible odd length
    @test ntt_len(1 << 33, ctxs...) == 1 << 33       # exactly 2^33 is still fine
    @test ntt_len(255 << 33, ctxs...) == 255 << 33   # largest supported length

    # beyond 255·2^33 no valid length exists
    @test_throws ArgumentError ntt_len((255 << 33) + 1, ctxs...)
end

@testset "garner unpack vs BigInt CRT" begin
    rng = MersenneTwister(0x6a3)
    P1, P2 = big(FP_PI), big(FP_PI2)
    # random (possibly unreduced, either-sign) fp representatives of each
    # residue, laid out as fp_ntt_rev! hands them over: coefficient i sits at
    # index (N-i) mod N and carries an extra factor N that unpack's 1/N
    # mulmod removes
    rep(c, P) = Float64(c % P) + rand(rng, -2:2) * Float64(P)
    for b in (33, 40, 44, 48), nconv in (1, 7, 64, 200)
        cs = [rand(rng, big(0):P1*P2-1) for _ in 1:nconv]
        N = nconv + rand(rng, 0:5)
        x1 = zeros(Float64, N)
        x2 = zeros(Float64, N)
        for (i, c) in enumerate(cs)
            idx = mod(N - (i - 1), N) + 1
            x1[idx] = rep(c * N, P1)
            x2[idx] = rep(c * N, P2)
        end
        n1 = Float64(invmod(UInt64(N), FP_PI));  n1p = n1 / Float64(FP_PI)
        n2 = Float64(invmod(UInt64(N), FP_PI2)); n2p = n2 / Float64(FP_PI2)
        rn = cld(99 + b * (nconv - 1), 64) + 1
        r = Memory{Limb}(undef, rn)
        fp_ntt_unpack2!(r, 0, rn, x1, x2, nconv, b, n1, n1p, n2, n2p)
        want = sum(c << (b * (i - 1)) for (i, c) in enumerate(cs))
        @test sum(big(r[i]) << (64 * (i - 1)) for i in 1:rn) == want
    end
end

@testset "differential fpntt2" begin
    rng = MersenneTwister(0x91c4)
    @test iszero(fpntt2_mul(NBig(0), NBig(12345)))
    @test BigInt(fpntt2_mul(NBig(-3), NBig(5))) == -15
    for (na, nb) in ((1, 1), (5, 3), (16, 16), (100, 100), (255, 257),
                     (500, 300), (601, 373), (1024, 1024), (2500, 2500),
                     (5000, 4900))
        for trial in 1:(na * nb > 10^5 ? 2 : 8)
            a = fpntt_randbig(rng, na)
            b = fpntt_randbig(rng, nb)
            @test BigInt(fpntt2_mul(NBig(a), NBig(b))) == a * b
        end
    end
    # feasibility-boundary chunk widths: tiny all-ones operands reach the
    # largest reachable b (48) with max-magnitude convolution coefficients
    for na in 1:8, nb in 1:na
        a = big(2)^(64na) - 1
        b = big(2)^(64nb) - 1
        @test BigInt(fpntt2_mul(NBig(a), NBig(b))) == a * b
    end
    # large enough that ntt_len picks a 17-family length (17·2^15 = 557056
    # points here), exercising the radix-17 stage end to end.
    # BigInt round-trips are too slow at this size, so the product is
    # verified by residues mod random 63-bit moduli (an engine-independent
    # O(n) check: a wrong limb survives one modulus with prob ~2^-63)
    limbmod(x, n, m) = (r = UInt128(0); for i in n:-1:1; r = ((r << 64) | x[i]) % m; end; UInt64(r))
    let n = 168_000
        bch, N = fp_ntt_params2(64n, 64n, FP_CTX1, FP_CTX2)
        @test N >> trailing_zeros(N) == 17
        am = Memory{Limb}(rand(rng, UInt64, n))
        bm = Memory{Limb}(rand(rng, UInt64, n))
        r = Memory{Limb}(undef, 2n)
        mul_fpntt2!(r, 0, am, 0, n, bm, 0, n)
        for m in rand(rng, UInt64(2)^62:UInt64(2)^63-1, 4)
            @test limbmod(r, 2n, m) ==
                  UInt64(UInt128(limbmod(am, n, m)) * limbmod(bm, n, m) % m)
        end
    end
    for n in (1, 7, 100, 1024, 3000)
        a = fpntt_randbig(rng, n)
        @test BigInt(fpntt2_square(NBig(a))) == a^2
    end
    @test BigInt(fpntt2_square(NBig(big(2)^(64 * 6) - 1))) == (big(2)^(64 * 6) - 1)^2
end

@testset "fpntt2 dispatch boundary" begin
    rng = MersenneTwister(0x77aa)
    for nm in (NativeBigInt.MUL_FPNTT_THRESHOLD - 1, NativeBigInt.MUL_FPNTT_THRESHOLD)
        a = rand(rng, big(2)^(64nm - 1):big(2)^(64nm)-1)
        b = rand(rng, big(2)^(64nm - 1):big(2)^(64nm)-1)
        @test BigInt(NBig(a) * NBig(b)) == a * b
    end
    for ns in (NativeBigInt.SQR_FPNTT_THRESHOLD - 1, NativeBigInt.SQR_FPNTT_THRESHOLD)
        s = rand(rng, big(2)^(64ns - 1):big(2)^(64ns)-1)
        @test BigInt(NBig(s)^2) == s^2
    end
end
