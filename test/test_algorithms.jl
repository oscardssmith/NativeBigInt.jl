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
