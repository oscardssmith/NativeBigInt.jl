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
