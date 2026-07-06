# Differential property tests vs Base.BigInt: random + adversarial operands,
# all sign combinations, bit-for-bit agreement.

# Random n-limb magnitude with adversarial patterns: uniform limbs, all-ones
# limbs (carry chains), 2^k ± 1, single high bit.
function diff_randbig(rng, n::Int)
    n == 0 && return big(0)
    style = rand(rng, 1:4)
    mag = if style == 1
        rand(rng, big(2)^(64 * (n - 1)):big(2)^(64n)-1)
    elseif style == 2
        big(2)^(64n) - 1
    elseif style == 3
        big(2)^rand(rng, (64 * (n - 1)):(64n - 1)) + rand(rng, -1:1)
    else
        big(2)^(64n - 1)
    end
    return rand(rng, Bool) ? -mag : mag
end

@testset "differential isqrt" begin
    rng = MersenneTwister(0x1509)
    for x in 0:20
        @test BigInt(isqrt(NBig(x))) == isqrt(big(x))
    end
    @test_throws DomainError isqrt(NBig(-1))
    for trial in 1:300
        n = rand(rng, 1:12)
        trial % 20 == 0 && (n = rand(rng, 30:100))
        a = abs(diff_randbig(rng, n))
        for x in (a, a^2, a^2 - 1, a^2 + 1)
            s = isqrt(NBig(x))
            @test BigInt(s) == isqrt(x)
        end
    end
end

@testset "differential gcd" begin
    rng = MersenneTwister(0x9cd)
    @test iszero(gcd(NBig(0), NBig(0)))
    @test BigInt(gcd(NBig(0), NBig(-6))) == 6
    @test BigInt(gcd(NBig(-4), NBig(0))) == 4
    for trial in 1:300
        la, lb = rand(rng, 1:12), rand(rng, 1:12)
        trial % 20 == 0 && (la = rand(rng, 30:70); lb = rand(rng, 1:70))
        a, b = diff_randbig(rng, la), diff_randbig(rng, lb)
        # plant a large common factor half the time
        if isodd(trial)
            g = abs(diff_randbig(rng, rand(rng, 1:6)))
            a, b = a * g, b * g
        end
        @test BigInt(gcd(NBig(a), NBig(b))) == gcd(a, b)
        @test BigInt(gcd(NBig(a), NBig(a))) == abs(a)
        @test BigInt(gcd(NBig(a * b), NBig(b))) == gcd(a * b, b)
    end
end

@testset "differential powermod" begin
    rng = MersenneTwister(0x90d)
    @test_throws DivideError powermod(NBig(2), NBig(5), NBig(0))
    @test_throws DomainError powermod(NBig(2), NBig(-1), NBig(7))
    @test BigInt(powermod(NBig(0), NBig(0), NBig(7))) == 1
    for trial in 1:150
        lm = rand(rng, 1:8)
        trial % 15 == 0 && (lm = rand(rng, 20:40))
        m = diff_randbig(rng, lm)
        iszero(m) && (m = big(3))
        a = diff_randbig(rng, rand(rng, 0:lm+2))
        n = abs(diff_randbig(rng, rand(rng, 1:3)))
        @test BigInt(powermod(NBig(a), NBig(n), NBig(m))) == powermod(a, n, m)
        @test BigInt(powermod(NBig(a), 17, NBig(m))) == powermod(a, 17, m)
        e = rand(rng, 0:40)
        @test BigInt(powermod(NBig(a), NBig(e), NBig(m))) == powermod(a, big(e), m)
        m2 = big(2)^rand(rng, 1:200) # even / power-of-two modulus path
        @test BigInt(powermod(NBig(a), NBig(n), NBig(m2))) == powermod(a, n, m2)
    end
end

@testset "differential add/sub/mul/divrem" begin
    rng = MersenneTwister(0xd1ff)
    for trial in 1:500
        la = rand(rng, 0:12)
        lb = rand(rng, 0:12)
        # occasionally cross the Karatsuba threshold
        if trial % 25 == 0
            la, lb = rand(rng, 30:100), rand(rng, 1:100)
        end
        a, b = diff_randbig(rng, la), diff_randbig(rng, lb)
        na, nb = NBig(a), NBig(b)
        @test BigInt(na + nb) == a + b
        @test BigInt(na - nb) == a - b
        @test BigInt(na * nb) == a * b
        @test BigInt(na * na) == a * a   # squaring path (a.limbs === b.limbs)
        @test BigInt(na^2) == a^2
        if la <= 4
            e = rand(rng, 0:12)
            @test BigInt(na^e) == a^e
        end
        k = rand(rng, 0:200)
        @test BigInt(na << k) == a << k
        @test BigInt(na >> k) == a >> k
        @test BigInt(na & nb) == a & b
        @test BigInt(na | nb) == a | b
        @test BigInt(xor(na, nb)) == xor(a, b)
        @test BigInt(~na) == ~a
        iszero(a) || @test trailing_zeros(na) == trailing_zeros(a)
        a < 0 || @test count_ones(na) == count_ones(a)
        @test string(na) == string(a)
        @test string(na, base = 16) == string(a, base = 16)
        @test string(na, base = 2, pad = 100) == string(a, base = 2, pad = 100)
        @test string(na, base = 8) == string(a, base = 8)
        @test string(na, base = 4) == string(a, base = 4)
        @test string(na, base = 32) == string(a, base = 32)
        @test parse(NBig, string(a)) == na
        @test hash(na) == hash(a)
        @test Float64(na) == Float64(a)
        if !iszero(b)
            q, r = divrem(na, nb)
            @test BigInt(q) == div(a, b)
            @test BigInt(r) == rem(a, b)
            @test BigInt(div(na, nb)) == div(a, b)
            @test BigInt(rem(na, nb)) == rem(a, b)
            @test BigInt(mod(na, nb)) == mod(a, b)
            @test BigInt(fld(na, nb)) == fld(a, b)
            @test BigInt(cld(na, nb)) == cld(a, b)
        end
    end
end
