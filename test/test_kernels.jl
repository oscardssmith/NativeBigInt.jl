# Kernel tests
using NativeBigInt: Limb, DLimb, add_n!, sub_n!, add!, sub!, add_into!, cmp_limbs, mul_1!, addmul_1!, submul_1!, mul_basecase!, lshift!, rshift!, divrem_1!
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

@testset "add_n! SIMD cold path" begin
    # deterministic cases hitting the s == typemax guard inside 8-limb SIMD blocks
    for n in (8, 16, 24, 30)
        # full ripple: (B^n - 1) + 1 chains a carry through every lane
        a = mem(fill(typemax(UInt64), n)); b = mem([one(UInt64); zeros(UInt64, n - 1)])
        r = Memory{UInt64}(undef, n)
        @test add_n!(r, 0, a, 0, b, 0, n) == 1 && all(iszero, r)
        # typemax-sum lane with no incoming carry: cold path fires, result unchanged
        a2 = mem(fill(UInt64(1), n))
        b2v = fill(UInt64(2), n); b2v[3] = typemax(UInt64) - 1; b2v[min(10, n)] = typemax(UInt64) - 1
        b2 = mem(b2v)
        r2 = Memory{UInt64}(undef, n)
        c2 = add_n!(r2, 0, a2, 0, b2, 0, n)
        @test toref(r2, 0, n) + (big(c2) << (64n)) == toref(a2, 0, n) + toref(b2, 0, n)
        # carry generated in lane 1 chains through typemax lanes mid-block
        av = fill(typemax(UInt64), n)
        bv = zeros(UInt64, n); bv[1] = 1; bv[n] = 5
        a3 = mem(av); b3 = mem(bv)
        r3 = Memory{UInt64}(undef, n)
        c3 = add_n!(r3, 0, a3, 0, b3, 0, n)
        @test toref(r3, 0, n) + (big(c3) << (64n)) == toref(a3, 0, n) + toref(b3, 0, n)
        # carry entering a block whose lane 0 sums to typemax (block-boundary chain)
        a4v = fill(UInt64(7), n); a4v[8] = typemax(UInt64)
        b4v = fill(UInt64(9), n); b4v[8] = 1
        if n > 8
            a4v[9] = typemax(UInt64); b4v[9] = 0
        end
        a4 = mem(a4v); b4 = mem(b4v)
        r4 = Memory{UInt64}(undef, n)
        c4 = add_n!(r4, 0, a4, 0, b4, 0, n)
        @test toref(r4, 0, n) + (big(c4) << (64n)) == toref(a4, 0, n) + toref(b4, 0, n)
    end
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

@testset "addmul_2! direct" begin
    rng = MersenneTwister(99)
    addmul_2! = NativeBigInt.addmul_2!
    check2(m, av, b0, b1) = begin
        a = mem(av)
        rv = rand(rng, UInt64, m + 2)
        r = mem(copy(rv))
        addmul_2!(r, 0, a, 0, m, b0, b1)
        want = toref(mem(rv), 0, m) + toref(a, 0, m) * (big(b0) + (big(b1) << 64))
        @test toref(r, 0, m + 2) == want
    end
    # widths covering scalar-only, one vector block, blocks + tail
    for m in (1, 2, 7, 8, 9, 16, 23, 24, 31), trial in 1:20
        check2(m, rand(rng, UInt64, m), rand(rng, UInt64), rand(rng, UInt64))
    end
    # cold path: all-ones a with all-ones b limbs maximizes lane sums/carries
    for m in (8, 16, 24, 31)
        check2(m, fill(typemax(UInt64), m), typemax(UInt64), typemax(UInt64))
        # sparse typemax lanes mid-block
        av = fill(UInt64(1), m); av[3] = typemax(UInt64); av[min(10, m)] = typemax(UInt64) - 1
        check2(m, av, typemax(UInt64), UInt64(2))
    end
end

@testset "addmul_2! carry saturation" begin
    # typemax-dense operands drive the addmul_2! column carry to exactly 2^64
    # (hi0 = 2^64-2 plus two overflow bits), which a limb-sized carry drops
    rng = MersenneTwister(1234)
    for trial in 1:300
        m = rand(rng, 8:23); n = rand(rng, 2:40)
        av = rand(rng, UInt64, m); bv = rand(rng, UInt64, n)
        for i in 1:m; rand(rng) < 0.5 && (av[i] = typemax(UInt64) - rand(rng, 0:1)); end
        for i in 1:n; rand(rng) < 0.5 && (bv[i] = typemax(UInt64) - rand(rng, 0:1)); end
        a = mem(av); b = mem(bv)
        r = Memory{UInt64}(undef, m + n)
        mul_basecase!(r, 0, a, 0, m, b, 0, n)
        @test toref(r, 0, m + n) == toref(a, 0, m) * toref(b, 0, n)
    end
end

@testset "shift/div1 kernels" begin
    rng = MersenneTwister(7)
    for n in (1, 3, 9), cnt in (1, 13, 63), trial in 1:20
        a = mem(rand(rng, UInt64, n)); aref = toref(a, 0, n)
        r = Memory{UInt64}(undef, n)
        hi = lshift!(r, 0, a, 0, n, cnt)
        @test (big(hi) << (64n)) | toref(r, 0, n) == aref << cnt
        lo = rshift!(r, 0, a, 0, n, cnt)
        @test toref(r, 0, n) == aref >> cnt
        @test big(lo) >> (64 - cnt) == aref & ((big(1) << cnt) - 1)
        # in-place
        b = mem(collect(a)); lshift!(b, 0, b, 0, n, cnt)
        @test toref(b, 0, n) == (aref << cnt) & ((big(1) << (64n)) - 1)
        d = rand(rng, UInt64) | 1
        q = Memory{UInt64}(undef, n)
        rem = divrem_1!(q, 0, a, 0, n, d)
        @test toref(q, 0, n) == aref ÷ d && rem == aref % d
    end
end
