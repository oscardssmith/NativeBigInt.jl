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
