# fp NTT multiplication: Float64 field arithmetic, transform, mul_fpntt!/
# sqr_fpntt!.  The engine's correctness rests on rounding-error analysis, so
# beyond the differential net these tests hammer the documented lazy-range
# bounds (mulmod |x| <= 4p, reduce |x| <= 8p) where random in-range inputs
# would never stress the magic-constant round.
using NativeBigInt: FP_PI, FP_P, fp_mulmod, fp_mulmod2, fp_reduce, fp_round,
                    fpi_pow, fpi_inv, fp_ntt_plan, fp_ntt_fwd!, fp_ntt_inv!,
                    mul_fpntt!, sqr_fpntt!, VF8, Limb, nlimbs, nbig_from_limbs
using Random: MersenneTwister

# canonical residue in [0, p) of a balanced-representation value
fp_canon(x::Float64) = (v = fp_reduce(x); v < 0 && (v += FP_P); UInt64(v))

function fpntt_mul(a::NBig, b::NBig)
    (iszero(a) || iszero(b)) && return NBig(0)
    la, lb = nlimbs(a), nlimbs(b)
    r = Memory{Limb}(undef, la + lb)
    mul_fpntt!(r, 0, a.limbs, 0, la, b.limbs, 0, lb)
    return nbig_from_limbs(sign(a) * sign(b), r, la + lb)
end

function fpntt_square(a::NBig)
    iszero(a) && return NBig(0)
    la = nlimbs(a)
    r = Memory{Limb}(undef, 2la)
    sqr_fpntt!(r, 0, a.limbs, 0, la)
    return nbig_from_limbs(1, r, 2la)
end

@testset "fp field arithmetic" begin
    P = big(FP_PI)
    @test FP_PI == 0x0001_FFFE_0000_0001          # 2^49 - 2^33 + 1
    @test Float64(FP_PI) == FP_P                   # p is exactly representable
    @test (FP_PI - 1) % (UInt64(15) << 33) == 0    # 15·2^33 length family

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
        r = fp_mulmod(x, Float64(w), Float64(w) / FP_P)
        @test abs(r) < FP_P
        @test big(fp_canon(r)) == mod(big(w) * big(Int128(x)), P)
    end
    for x in xs, y in xs
        abs(x) <= 2FP_P && abs(y) <= 2FP_P || continue
        r = fp_mulmod2(x, y)
        @test abs(r) < FP_P
        @test big(fp_canon(r)) == mod(big(Int128(x)) * big(Int128(y)), P)
    end
    for x in xs
        r = fp_reduce(2 * x)                       # up to ±8p
        @test abs(r) < FP_P
        @test big(fp_canon(r)) == mod(2 * big(Int128(x)), P)
    end
    # generator-derived roots of unity have exact order
    for logN in (1, 5, 20, 33)
        N = UInt64(2)^logN
        ω = fpi_pow(NativeBigInt.fp_generator(), (FP_PI - 1) ÷ N)
        @test fpi_pow(ω, N) == 1
        @test fpi_pow(ω, N >> 1) == FP_PI - 1
    end
end

@testset "fp vectorized ops match scalar" begin
    rng = MersenneTwister(0x51e)
    for trial in 1:300
        x = [rand(rng, (-4.0, -1.0, 1.0, 4.0)) * rand(rng, UInt64(0):FP_PI-1) for _ in 1:8]
        w = rand(rng, UInt64(0):FP_PI-1, 8)
        wf = Float64.(w)
        wp = wf ./ FP_P
        vr = fp_mulmod(VF8(Tuple(x)), VF8(Tuple(wf)), VF8(Tuple(wp)))
        @test [vr[i] for i in 1:8] == fp_mulmod.(x, wf, wp)
        vr2 = fp_reduce(VF8(Tuple(x)))
        @test [vr2[i] for i in 1:8] == fp_reduce.(x)
    end
end

@testset "fp ntt transform" begin
    rng = MersenneTwister(0xf18)
    for N in (4, 8, 64, 512, 1024, 12, 20, 48, 96, 160, 240, 1536)
        plan = fp_ntt_plan(N)
        x = Float64.(rand(rng, UInt64(0):FP_PI-1, N))
        y = copy(x)
        fp_ntt_fwd!(y, plan)
        @test y != x
        fp_ntt_inv!(y, plan)
        @test fp_canon.(y) == UInt64.(x)
        N > 256 && continue
        a = rand(rng, UInt64(0):FP_PI-1, N)
        b = rand(rng, UInt64(0):FP_PI-1, N)
        c = big.(zeros(Int, N))
        for i in 0:N-1, j in 0:N-1
            c[mod(i + j, N) + 1] += big(a[i+1]) * big(b[j+1])
        end
        fa, fb = Float64.(a), Float64.(b)
        fp_ntt_fwd!(fa, plan); fp_ntt_fwd!(fb, plan)
        fc = fp_mulmod2.(fa, fb)
        fp_ntt_inv!(fc, plan)
        @test big.(fp_canon.(fc)) == mod.(c, big(FP_PI))
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

@testset "differential fpntt_mul" begin
    rng = MersenneTwister(0x9f57)
    @test iszero(fpntt_mul(NBig(0), NBig(12345)))
    @test iszero(fpntt_mul(NBig(-7), NBig(0)))
    @test BigInt(fpntt_mul(NBig(-3), NBig(5))) == -15

    for (na, nb) in ((1, 1), (3, 2), (15, 15), (16, 16), (17, 40), (100, 100),
                     (255, 257), (1024, 1024), (2000, 100), (5000, 5000),
                     (190, 190), (600, 600), (1100, 1050), (2500, 2500))
        for trial in 1:(na * nb > 10^5 ? 2 : 8)
            a = fpntt_randbig(rng, na)
            b = fpntt_randbig(rng, nb)
            @test BigInt(fpntt_mul(NBig(a), NBig(b))) == a * b
        end
    end
end

@testset "fpntt_square and * integration" begin
    rng = MersenneTwister(0x5f2e)
    @test iszero(fpntt_square(NBig(0)))
    for n in (1, 5, 16, 100, 700, 1024, 2500)
        a = fpntt_randbig(rng, n)
        @test BigInt(fpntt_square(NBig(a))) == a^2
    end
    # `*` dispatches to the fp NTT above the (new, lower) thresholds; sizes
    # straddle MUL_FPNTT_THRESHOLD = 352 and SQR_FPNTT_THRESHOLD = 448
    for (na, nb) in ((340, 340), (352, 352), (500, 300), (1200, 800), (5000, 4000))
        a = fpntt_randbig(rng, na)
        b = fpntt_randbig(rng, nb)
        @test BigInt(NBig(a) * NBig(b)) == a * b
    end
    for n in (440, 448, 460, 1024, 3000)
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
