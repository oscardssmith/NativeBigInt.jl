# NTT multiplication: Goldilocks field arithmetic, transform, mul_ntt!/sqr_ntt!
using NativeBigInt: GF_P, gf_add, gf_sub, gf_mul, gf_pow, gf_inv,
                    ntt_plan, ntt_fwd!, ntt_inv!, mul_ntt!, sqr_ntt!,
                    Limb, nlimbs, nbig_from_limbs
using Random: MersenneTwister

# NBig-level harness around the mpn entry points: forces the NTT path at any
# size, so the differential tests also cover the small and odd-shaped
# transforms that production mul!/sqr! dispatch (above the thresholds) never
# reaches
function ntt_mul(a::NBig, b::NBig)
    (iszero(a) || iszero(b)) && return NBig(0)
    la, lb = nlimbs(a), nlimbs(b)
    r = Memory{Limb}(undef, la + lb)
    mul_ntt!(r, 0, a.limbs, 0, la, b.limbs, 0, lb)
    return nbig_from_limbs(sign(a) * sign(b), r, la + lb)
end

# the (nonnegative) square of a's magnitude
function ntt_square(a::NBig)
    iszero(a) && return NBig(0)
    la = nlimbs(a)
    r = Memory{Limb}(undef, 2la)
    sqr_ntt!(r, 0, a.limbs, 0, la)
    return nbig_from_limbs(1, r, 2la)
end

@testset "goldilocks field arithmetic" begin
    P = big(GF_P)
    @test GF_P == 0xFFFF_FFFF_0000_0001
    # adversarial values around the reduction boundaries
    edge = UInt64[0, 1, 2, 0xFFFFFFFF, 0x100000000, GF_P - 2, GF_P - 1,
                  0xFFFFFFFF00000000, 0x00000000FFFFFFFE]
    rng = MersenneTwister(0x60_1d)
    vals = [edge; rand(rng, UInt64(0):GF_P-1, 100)]
    for a in vals, b in vals
        @test big(gf_add(a, b)) == mod(big(a) + big(b), P)
        @test big(gf_sub(a, b)) == mod(big(a) - big(b), P)
        @test big(gf_mul(a, b)) == mod(big(a) * big(b), P)
    end
    for a in vals
        e = rand(rng, UInt64(0):UInt64(2)^40)
        @test big(gf_pow(a, e)) == powermod(big(a), big(e), P)
        if a % GF_P != 0
            @test gf_mul(a, gf_inv(a)) == 1
        end
    end
    # 7 generates the multiplicative group: ω = 7^((p-1)/N) is a primitive
    # N-th root of unity for power-of-two N
    for logN in (1, 5, 20, 32)
        N = UInt64(2)^logN
        ω = gf_pow(UInt64(7), (GF_P - 1) ÷ N)
        @test gf_pow(ω, N) == 1
        @test gf_pow(ω, N >> 1) == GF_P - 1   # ω^(N/2) == -1 ⇒ order is exactly N
    end
end

@testset "ntt transform" begin
    rng = MersenneTwister(0xf17)
    # 2/8: scalar-only; 64/512/1024: vectorized stages, odd and even log2;
    # 12/20/48/96/160/240/1536: composite lengths m·2^k, m in (3, 5, 15)
    for N in (2, 8, 64, 512, 1024, 12, 20, 48, 96, 160, 240, 1536)
        plan = ntt_plan(N)
        # forward ∘ inverse == identity
        x = rand(rng, UInt64(0):GF_P-1, N)
        y = copy(x)
        ntt_fwd!(y, plan)
        N > 2 && @test y != x   # transform actually did something
        ntt_inv!(y, plan)
        @test y == x
        # NTT diagonalizes cyclic convolution: intt(ntt(a) .* ntt(b)) == a ⊛ b
        N > 256 && continue   # the reference convolution is O(N²) in BigInt
        a = rand(rng, UInt64(0):GF_P-1, N)
        b = rand(rng, UInt64(0):GF_P-1, N)
        c = big.(zeros(Int, N))
        for i in 0:N-1, j in 0:N-1
            c[mod(i + j, N) + 1] += big(a[i+1]) * big(b[j+1])
        end
        fa, fb = copy(a), copy(b)
        ntt_fwd!(fa, plan); ntt_fwd!(fb, plan)
        fc = gf_mul.(fa, fb)
        ntt_inv!(fc, plan)
        @test big.(fc) == mod.(c, big(GF_P))
    end
end

@testset "vectorized field ops" begin
    using NativeBigInt: gf_addv, gf_subv, gf_mulv, gf_mul_iv, gf_mul_i, V8
    rng = MersenneTwister(0x51d)
    edge = UInt64[0, 1, 0xFFFFFFFF, 0x100000000, GF_P - 2, GF_P - 1,
                  0xFFFFFFFF00000000, 0x00000000FFFFFFFE]
    pool = [edge; rand(rng, UInt64(0):GF_P-1, 64)]
    for trial in 1:300
        a = rand(rng, pool, 8)
        b = rand(rng, pool, 8)
        va, vb = V8(Tuple(a)), V8(Tuple(b))
        @test [gf_addv(va, vb)[i] for i in 1:8] == gf_add.(a, b)
        @test [gf_subv(va, vb)[i] for i in 1:8] == gf_sub.(a, b)
        @test [gf_mulv(va, vb)[i] for i in 1:8] == gf_mul.(a, b)
        @test [gf_mul_iv(va)[i] for i in 1:8] == gf_mul_i.(a)
    end
end

# adversarial magnitude in [2^(64(n-1)), 2^64n): uniform, all-ones, 2^k ± 1
function ntt_randbig(rng, n::Int)
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

@testset "differential ntt_mul" begin
    rng = MersenneTwister(0x9057)
    @test iszero(ntt_mul(NBig(0), NBig(12345)))
    @test iszero(ntt_mul(NBig(-7), NBig(0)))
    @test BigInt(ntt_mul(NBig(-3), NBig(5))) == -15

    # small, mid, large, unbalanced; straddle pow2 boundaries
    for (na, nb) in ((1, 1), (3, 2), (15, 15), (16, 16), (17, 40), (100, 100),
                     (255, 257), (1024, 1024), (2000, 100), (5000, 5000),
                     (190, 190), (600, 600), (1100, 1050), (2500, 2500))
        for trial in 1:(na * nb > 10^5 ? 2 : 8)
            a = ntt_randbig(rng, na)
            b = ntt_randbig(rng, nb)
            @test BigInt(ntt_mul(NBig(a), NBig(b))) == a * b
        end
    end
end

@testset "gf_mul_i" begin
    # multiplication by the fourth root i = 2^48 via shifts only
    using NativeBigInt: gf_mul_i
    P = big(GF_P)
    edge = UInt64[0, 1, 0xFFFF, 0x10000, 0xFFFFFFFF, 0x100000000, GF_P - 1,
                  0xFFFFFFFF00000000]
    rng = MersenneTwister(0x4edd)
    for x in [edge; rand(rng, UInt64(0):GF_P-1, 500)]
        @test big(gf_mul_i(x)) == mod(big(x) << 48, P)
    end
end

@testset "ntt_square and * integration" begin
    rng = MersenneTwister(0x5a2e)
    @test iszero(ntt_square(NBig(0)))
    for n in (1, 5, 16, 100, 700, 1024, 2500)
        a = ntt_randbig(rng, n)
        @test BigInt(ntt_square(NBig(a))) == a^2
    end
    # `*` dispatches to the NTT above the thresholds (and stays correct
    # across them); x*x and x^2 take the squaring path
    for (na, nb) in ((900, 900), (1024, 1024), (1200, 800), (5000, 4000))
        a = ntt_randbig(rng, na)
        b = ntt_randbig(rng, nb)
        @test BigInt(NBig(a) * NBig(b)) == a * b
    end
    for n in (890, 900, 1024, 3000)
        a = ntt_randbig(rng, n)
        x = NBig(a)
        @test BigInt(x * x) == a^2
        @test BigInt(x^2) == a^2
        @test BigInt(x * -x) == -(a^2)
    end
end

@testset "mpn-level ntt dispatch" begin
    # ^ and isqrt reach the NTT through mul!/sqr! on raw limb buffers, not
    # through Base.:* — this is what NBig-level dispatch would miss
    rng = MersenneTwister(0x3717)
    a = abs(ntt_randbig(rng, 900))
    @test BigInt(NBig(a)^3) == a^3
    d = abs(ntt_randbig(rng, 2400))
    @test BigInt(isqrt(NBig(d))) == isqrt(d)
end
