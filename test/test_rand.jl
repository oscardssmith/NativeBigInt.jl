@testset "rand on NBig ranges" begin
    rng = Xoshiro(42)

    # regression: this used to StackOverflowError (Random's AbstractArray
    # fallback recursed because length(::UnitRange{NBig}) is an NBig)
    x = rand(rng, NBig(1):100)
    @test x isa NBig
    @test NBig(1) <= x <= NBig(100)

    # small range: every value reachable, endpoints included
    seen = Set{NBig}()
    for _ in 1:200
        push!(seen, rand(rng, NBig(-2):NBig(2)))
    end
    @test seen == Set(NBig.(-2:2))

    # singleton and empty ranges
    @test rand(rng, NBig(7):NBig(7)) == NBig(7)
    @test_throws ArgumentError rand(rng, NBig(2):NBig(1))

    # wide multi-limb range: bounds hold, values differ across draws,
    # and results match BigInt round-trip ordering
    lo = NBig(big(2)^200 + 3)
    hi = NBig(big(2)^333 - 7)
    vals = [rand(rng, lo:hi) for _ in 1:50]
    @test all(v -> lo <= v <= hi, vals)
    @test length(unique(vals)) > 1

    # range spanning a power-of-two limb boundary exercises the top-limb mask
    for _ in 1:100
        v = rand(rng, NBig(0):NBig(big(2)^128))
        @test NBig(0) <= v <= NBig(big(2)^128)
    end

    # StepRange and Vector of NBig go through Random's AbstractArray
    # fallback, which bottoms out in the UnitRange{NBig} sampler
    sr = NBig(1):NBig(2):NBig(9)
    @test all(in(sr), rand(rng, sr, 100))
    pool = NBig.([3, 5, 8])
    @test all(in(pool), rand(rng, pool, 100))

    # seeded determinism
    a = rand(Xoshiro(1), NBig(1):NBig(big(10)^50), 10)
    b = rand(Xoshiro(1), NBig(1):NBig(big(10)^50), 10)
    @test a == b
    @test eltype(a) == NBig
end
