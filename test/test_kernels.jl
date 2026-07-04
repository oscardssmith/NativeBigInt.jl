# Kernel tests
using NativeBigInt: Limb, DLimb, add_n!, sub_n!, add!, sub!, add_into!, cmp_limbs
using Random: MersenneTwister

mem(v::Vector{UInt64}) = (m = Memory{UInt64}(undef, length(v)); copyto!(m, v); m)
# reference: limbs -> BigInt
toref(m, off, n) = (x = big(0); for i in n:-1:1; x = (x << 64) | m[off+i]; end; x)

@testset "add/sub kernels" begin
    rng = MersenneTwister(42)
    for n in (1, 2, 3, 7, 30), trial in 1:50
        a = mem(rand(rng, UInt64, n)); b = mem(rand(rng, UInt64, n))
        r = Memory{UInt64}(undef, n)
        c = add_n!(r, 0, a, 0, b, 0, n)
        @test toref(r, 0, n) + (big(c) << (64n)) == toref(a, 0, n) + toref(b, 0, n)
        brw = sub_n!(r, 0, a, 0, b, 0, n)
        @test toref(r, 0, n) - (big(brw) << (64n)) == toref(a, 0, n) - toref(b, 0, n)
    end
    # carry chain: all-ones + 1
    n = 5
    a = mem(fill(typemax(UInt64), n)); b = mem([one(UInt64); zeros(UInt64, n-1)])
    r = Memory{UInt64}(undef, n)
    @test add_n!(r, 0, a, 0, b, 0, n) == 1 && all(iszero, r)
    # mixed length add!/sub!
    la, lb = 6, 2
    a = mem(fill(typemax(UInt64), la)); b = mem(rand(rng, UInt64, lb))
    r = Memory{UInt64}(undef, la)
    c = add!(r, 0, a, 0, la, b, 0, lb)
    @test toref(r, 0, la) + (big(c) << (64la)) == toref(a, 0, la) + toref(b, 0, lb)
    brw = sub!(r, 0, a, 0, la, b, 0, lb)
    @test toref(r, 0, la) == toref(a, 0, la) - toref(b, 0, lb) && brw == 0
    # add_into!
    r2 = mem([typemax(UInt64), typemax(UInt64), UInt64(0)])
    add_into!(r2, 0, 3, mem([UInt64(1)]), 0, 1)
    @test toref(r2, 0, 3) == big(1) << 128
    # aliasing: r === a
    a = mem(rand(rng, UInt64, 4)); aref = toref(a, 0, 4); b = mem(rand(rng, UInt64, 4))
    c = add_n!(a, 0, a, 0, b, 0, 4)
    @test toref(a, 0, 4) + (big(c) << 256) == aref + toref(b, 0, 4)
    # cmp_limbs
    @test cmp_limbs(mem([UInt64(1), UInt64(2)]), 0, 2, mem([UInt64(5)]), 0, 1) == 1
    @test cmp_limbs(mem([UInt64(3)]), 0, 1, mem([UInt64(3)]), 0, 1) == 0
    @test cmp_limbs(mem([UInt64(9), UInt64(1)]), 0, 2, mem([UInt64(0), UInt64(2)]), 0, 2) == -1
end
