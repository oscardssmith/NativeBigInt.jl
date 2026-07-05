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
