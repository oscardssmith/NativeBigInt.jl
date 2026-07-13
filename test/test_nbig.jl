using NativeBigInt: nlimbs
@testset "NBig construction/conversion" begin
    @test iszero(NBig(0)) && nlimbs(NBig(0)) == 0
    @test NBig(1) == NBig(1) && NBig(1) != NBig(2) && NBig(-1) != NBig(1)
    @test NBig(typemin(Int64)) == NBig(big(typemin(Int64)))   # abs overflow case
    @test NBig(UInt64(0xdeadbeef)) == NBig(0xdeadbeef)
    for v in (big(0), big(1), -big(1), big(2)^64, big(2)^64 - 1, -big(2)^193 + 17,
              big(typemin(Int128)), big(rand(UInt128)))
        @test BigInt(NBig(v)) == v
    end
    @test NBig(3) < NBig(4) && NBig(-4) < NBig(-3) && NBig(-1) < NBig(1)
    @test NBig(big(2)^100) > NBig(big(2)^99)
    @test -NBig(5) == NBig(-5) && abs(NBig(-5)) == NBig(5)
    @test sign(NBig(-7)) == -1 && signbit(NBig(-7)) && !signbit(NBig(7))
    @test Int64(NBig(42)) === 42 && Int64(NBig(-42)) === -42
    @test_throws InexactError Int64(NBig(big(2)^80))
    @test UInt64(NBig(7)) === UInt64(7)
    @test_throws InexactError UInt64(NBig(-7))
    @test Int128(NBig(big(typemin(Int128)))) == typemin(Int128)
    @test isodd(NBig(3)) && iseven(NBig(-4)) && iseven(NBig(0))
    @test zero(NBig) == NBig(0) && one(NBig) == NBig(1)
    @test promote_type(NBig, Int) == NBig
end

@testset "NBig arithmetic edges" begin
    @test NBig(2) + NBig(3) == NBig(5)
    @test NBig(0) + NBig(-7) == NBig(-7)
    @test NBig(5) - NBig(5) == NBig(0)
    @test NBig(3) * NBig(0) == NBig(0)
    @test NBig(-3) * NBig(4) == NBig(-12)
    @test divrem(NBig(7), NBig(2)) == (NBig(3), NBig(1))
    @test divrem(NBig(-7), NBig(2)) == (NBig(-3), NBig(-1))   # truncated, like Base
    @test divrem(NBig(3), NBig(10)) == (NBig(0), NBig(3))     # |a| < |b|
    @test mod(NBig(-7), NBig(2)) == NBig(1)
    @test_throws DivideError divrem(NBig(1), NBig(0))
    @test_throws DivideError div(NBig(0), NBig(0))
    @test NBig(2) + 3 == NBig(5) && 3 * NBig(2) == NBig(6)    # promotion
end

@testset "NBig shifts and bit ops" begin
    @test NBig(1) << 64 == NBig(big(2)^64)
    @test NBig(5) << 0 == NBig(5) && NBig(5) >> 0 == NBig(5)
    @test NBig(-1) >> 100 == NBig(-1)          # arithmetic shift floors
    @test NBig(-8) >> 2 == NBig(-2)
    @test NBig(big(2)^200) >> 200 == NBig(1)
    @test NBig(5) >> 700 == NBig(0)
    @test NBig(3) << -1 == NBig(1)             # negative count flips direction
    @test NBig(-1) & NBig(-1) == NBig(-1)
    @test NBig(-1) & NBig(0) == NBig(0)
    @test NBig(-2) | NBig(1) == NBig(-1)
    @test ~NBig(0) == NBig(-1) && ~NBig(-1) == NBig(0)
    @test trailing_zeros(NBig(big(2)^100)) == 100
    @test count_ones(NBig(big(2)^100 - 1)) == 100
end

@testset "NBig pow" begin
    @test NBig(0)^0 == NBig(1)
    @test NBig(0)^5 == NBig(0)
    @test NBig(-2)^3 == NBig(-8) && NBig(-2)^4 == NBig(16)
    @test NBig(2)^100 == NBig(big(2)^100)
    # negative exponents must match BigInt behavior exactly (value or exception)
    function outcome(f)
        try
            return (:val, f())
        catch err
            return (:err, typeof(err))
        end
    end
    for (x, e) in ((1, -5), (-1, -3), (-1, -4), (2, -1), (-2, -1), (0, -1))
        want = outcome(() -> big(x)^e)
        got = outcome(() -> BigInt(NBig(x)^e))
        @test got == want
    end
end

@testset "NBig strings/hash/float" begin
    @test string(NBig(0)) == "0"
    @test string(NBig(-255), base = 16) == "-ff"
    @test string(NBig(5), pad = 3) == "005"
    @test repr(NBig(-12345)) == "-12345"
    @test parse(NBig, "0") == NBig(0)
    @test parse(NBig, "+42") == NBig(42)
    @test parse(NBig, "-42") == NBig(-42)
    @test parse(NBig, "ff"; base = 16) == NBig(255)
    @test_throws ArgumentError parse(NBig, "12x3")
    @test_throws ArgumentError parse(NBig, "")
    @test hash(NBig(42)) == hash(42)
    @test isequal(NBig(-7), NBig(-7)) && hash(NBig(-7)) == hash(-7)
    @test Float64(NBig(0)) === 0.0
    @test Float64(NBig(-3)) === -3.0
end

@testset "large base conversion (divide-and-conquer)" begin
    rng = MersenneTwister(0xba5e)
    # Sizes spanning the STR_DC_THRESHOLD crossover so both the O(n^2) and the
    # divide-and-conquer paths get exercised against Base.BigInt.
    for bits in (500, 2000, 5000, 20000, 100000)
        for base in (10, 7, 36, 2, 16, 8)  # non-pow2 (D&C) and pow2 (bit paths)
            a = rand(rng, big(0):(big(2)^bits - 1))
            a *= rand(rng, (-1, 1))
            na = NBig(a)
            s = string(na; base)
            @test s == string(a; base)
            @test parse(NBig, s; base) == na
        end
    end
    # exact-power boundaries: values like base^m - 1 and base^m stress the
    # zero-chunk / carry edges of the recursion.
    for base in (10, 7, 36), m in (100, 1000, 3000)
        for a in (big(base)^m - 1, big(base)^m, big(base)^m + 1)
            na = NBig(a)
            @test string(na; base) == string(a; base)
            @test parse(NBig, string(na; base); base) == na
        end
    end
    @test string(NBig(0); base = 10) == "0"
    @test parse(NBig, "0"^5000; base = 10) == NBig(0)
end
