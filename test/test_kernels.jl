# Kernel tests
using NativeBigInt: Limb, DLimb, add_n!, sub_n!, add!, sub!, add_into!, cmp_limbs, mul_1!, addmul_1!, submul_1!, mul_basecase!
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

@testset "mul kernels" begin
    rng = MersenneTwister(1)
    for n in (1, 2, 5, 20), trial in 1:50
        a = mem(rand(rng, UInt64, n)); b = rand(rng, UInt64)
        r = Memory{UInt64}(undef, n)
        c = mul_1!(r, 0, a, 0, n, b)
        @test toref(r, 0, n) + (big(c) << (64n)) == toref(a, 0, n) * b
        r0 = toref(r, 0, n)
        c2 = addmul_1!(r, 0, a, 0, n, b)
        @test toref(r, 0, n) + (big(c2) << (64n)) == r0 + toref(a, 0, n) * b
        r1 = toref(r, 0, n)
        c3 = submul_1!(r, 0, a, 0, n, b)
        @test toref(r, 0, n) - (big(c3) << (64n)) == r1 - toref(a, 0, n) * b
    end
    for (m, n) in ((1,1), (3,2), (7,7), (13,4)), trial in 1:30
        a = mem(rand(rng, UInt64, m)); b = mem(rand(rng, UInt64, n))
        r = Memory{UInt64}(undef, m + n)
        mul_basecase!(r, 0, a, 0, m, b, 0, n)
        @test toref(r, 0, m + n) == toref(a, 0, m) * toref(b, 0, n)
    end
    # max-value stress: (B^m - 1)(B^n - 1)
    a = mem(fill(typemax(UInt64), 4)); b = mem(fill(typemax(UInt64), 3))
    r = Memory{UInt64}(undef, 7)
    mul_basecase!(r, 0, a, 0, 4, b, 0, 3)
    @test toref(r, 0, 7) == toref(a, 0, 4) * toref(b, 0, 3)
end
