# Algorithm tests
using NativeBigInt: Limb, add_carry!, cmp_padded, abs_diff!, kar_scratch_len, KARATSUBA_THRESHOLD
using Random: MersenneTwister

amem(v::Vector{UInt64}) = (m = Memory{UInt64}(undef, length(v)); copyto!(m, v); m)
atoref(m, off, n) = (x = big(0); for i in n:-1:1; x = (x << 64) | m[off+i]; end; x)
afrombig(x::BigInt, n) = (v = zeros(UInt64, n); for i in 1:n; v[i] = UInt64(x & typemax(UInt64)); x >>= 64; end; amem(v))

@testset "kar helpers" begin
    # cmp_padded: value comparison with unnormalized (zero-padded) operands
    @test cmp_padded(amem([UInt64(5), UInt64(0)]), 0, 2, amem([UInt64(5)]), 0, 1) == 0
    @test cmp_padded(amem([UInt64(4), UInt64(1)]), 0, 2, amem([UInt64(9)]), 0, 1) == 1
    @test cmp_padded(amem([UInt64(4), UInt64(0)]), 0, 2, amem([UInt64(9)]), 0, 1) == -1
    @test cmp_padded(amem([UInt64(9), UInt64(2)]), 0, 2, amem([UInt64(9), UInt64(2)]), 0, 2) == 0

    # abs_diff!: x = [lo (2 limbs) | hi (1 limb)]
    d = Memory{UInt64}(undef, 2)
    # lo = 7, hi = 9 -> |7-9| = 2, negative
    @test abs_diff!(d, 0, amem([UInt64(7), UInt64(0), UInt64(9)]), 0, 2, 1) == true
    @test atoref(d, 0, 2) == 2
    # lo = B+9, hi = 7 -> lo-hi = B+2, positive
    @test abs_diff!(d, 0, amem([UInt64(9), UInt64(1), UInt64(7)]), 0, 2, 1) == false
    @test atoref(d, 0, 2) == (big(1) << 64) + 2
    # equal halves -> zero, non-negative
    @test abs_diff!(d, 0, amem([UInt64(3), UInt64(4), UInt64(3), UInt64(4)]), 0, 2, 2) == false
    @test atoref(d, 0, 2) == 0

    # add_carry!: ripple through a typemax limb
    r = amem([typemax(UInt64), UInt64(0)])
    add_carry!(r, 0, 2, 1, UInt64(1))
    @test atoref(r, 0, 2) == big(1) << 64
    # c == 0 is a no-op
    r2 = amem([UInt64(7)])
    add_carry!(r2, 0, 1, 1, UInt64(0))
    @test r2[1] == 7

    # kar_scratch_len
    @test kar_scratch_len(KARATSUBA_THRESHOLD - 1) == 0
    h2 = (KARATSUBA_THRESHOLD + 1) >> 1
    @test kar_scratch_len(KARATSUBA_THRESHOLD) == 4h2 + kar_scratch_len(h2)
    @test kar_scratch_len(4 * KARATSUBA_THRESHOLD) > kar_scratch_len(2 * KARATSUBA_THRESHOLD)
end

using NativeBigInt: kar_mul!

@testset "kar_mul! balanced" begin
    rng = MersenneTwister(123)
    T = KARATSUBA_THRESHOLD
    check(n, a, b) = begin
        r = Memory{UInt64}(undef, 2n)
        scratch = Memory{UInt64}(undef, kar_scratch_len(n))
        kar_mul!(r, 0, a, 0, b, 0, n, scratch, 0)
        @test atoref(r, 0, 2n) == atoref(a, 0, n) * atoref(b, 0, n)
    end
    # sizes spanning the threshold, odd/even splits, two+ recursion levels
    for n in (T - 1, T, T + 1, 2T, 2T + 1, 4T + 3), trial in 1:10
        check(n, amem(rand(rng, UInt64, n)), amem(rand(rng, UInt64, n)))
    end
    # adversarial: all-ones limbs (maximum carry chaining)
    for n in (T + 1, 2T + 1)
        a = amem(fill(typemax(UInt64), n))
        check(n, a, a)
    end
    # 2^k patterns: single high bit times single high bit minus one
    for n in (T + 1, 2T)
        a = afrombig(big(1) << (64n - 1), n)
        b = afrombig((big(1) << (64n - 1)) - 1, n)
        check(n, a, b)
    end
    # zero middle difference: a_lo == a_hi exactly
    n = 2 * (T + 1)
    half = rand(rng, UInt64, T + 1)
    check(n, amem([half; half]), amem(rand(rng, UInt64, n)))
end

using NativeBigInt: mul!

@testset "mul! general" begin
    rng = MersenneTwister(7)
    T = KARATSUBA_THRESHOLD
    checkmul(m, n, a, b) = begin
        r = Memory{UInt64}(undef, m + n)
        mul!(r, 0, a, 0, m, b, 0, n)
        @test atoref(r, 0, m + n) == atoref(a, 0, m) * atoref(b, 0, n)
    end
    # unbalanced shapes: basecase small-n, exact multiples, ragged tails,
    # tail chunk itself above/below threshold
    for (m, n) in ((100, 3), (64, 33), (2T, T), (2T + 5, T), (3T + 2, T + 1),
                   (200, T), (2T + T ÷ 2, T)), trial in 1:5
        checkmul(m, n, amem(rand(rng, UInt64, m)), amem(rand(rng, UInt64, n)))
    end
    # all-ones stress across the chunk boundaries
    m, n = 3T + 2, T + 1
    checkmul(m, n, amem(fill(typemax(UInt64), m)), amem(fill(typemax(UInt64), n)))
    # m == n delegates to balanced path
    for nn in (T, 2T + 1)
        checkmul(nn, nn, amem(rand(rng, UInt64, nn)), amem(rand(rng, UInt64, nn)))
    end
    # differential sweep vs BigInt over random sizes
    for trial in 1:60
        m = rand(rng, 1:3T); n = rand(rng, 1:m)
        checkmul(m, n, amem(rand(rng, UInt64, m)), amem(rand(rng, UInt64, n)))
    end
end
