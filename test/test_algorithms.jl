# Algorithm tests
using NativeBigInt: Limb, add_carry!, cmp_padded, abs_diff!, kar_scratch_len, MUL_KARATSUBA_THRESHOLD, divrem!,
    divrem_dc!, invert_pi1, DC_DIV_THRESHOLD
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
    @test kar_scratch_len(MUL_KARATSUBA_THRESHOLD - 1) == 0
    h2 = (MUL_KARATSUBA_THRESHOLD + 1) >> 1
    @test kar_scratch_len(MUL_KARATSUBA_THRESHOLD) == 4h2 + kar_scratch_len(h2)
    @test kar_scratch_len(4 * MUL_KARATSUBA_THRESHOLD) > kar_scratch_len(2 * MUL_KARATSUBA_THRESHOLD)
end

using NativeBigInt: mul_kar!

@testset "mul_kar! balanced" begin
    rng = MersenneTwister(123)
    T = MUL_KARATSUBA_THRESHOLD
    check(n, a, b) = begin
        r = Memory{UInt64}(undef, 2n)
        scratch = Memory{UInt64}(undef, kar_scratch_len(n))
        mul_kar!(r, 0, a, 0, b, 0, n, scratch, 0)
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
    T = MUL_KARATSUBA_THRESHOLD
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

using NativeBigInt: MUL_FPNTT_THRESHOLD, SQR_FPNTT_THRESHOLD, sqr!

@testset "mul!/sqr! across the Karatsuba → NTT crossover" begin
    rng = MersenneTwister(41)
    T = MUL_FPNTT_THRESHOLD
    # balanced and unbalanced end-to-end sizes straddling the dispatch switch
    for (m, n) in ((T - 1, T - 1), (T, T), (T + 1, T + 1), (3T, T ÷ 2),
                   (2T + T ÷ 2, T + 1)),
        trial in 1:3
        a = amem(rand(rng, UInt64, m)); b = amem(rand(rng, UInt64, n))
        r = Memory{UInt64}(undef, m + n)
        mul!(r, 0, a, 0, m, b, 0, n)
        @test atoref(r, 0, m + n) == atoref(a, 0, m) * atoref(b, 0, n)
    end
    for n in (SQR_FPNTT_THRESHOLD - 1, SQR_FPNTT_THRESHOLD,
              2SQR_FPNTT_THRESHOLD + 1), trial in 1:3
        a = amem(rand(rng, UInt64, n))
        r = Memory{UInt64}(undef, 2n)
        sqr!(r, 0, a, 0, n)
        @test atoref(r, 0, 2n) == atoref(a, 0, n)^2
    end
end

@testset "divrem! multi-limb" begin
    rng = MersenneTwister(23)

    function checkdiv(a::Memory{UInt64}, n, d::Memory{UInt64}, m)
        aref = atoref(a, 0, n); dref = atoref(d, 0, m)
        q = Memory{UInt64}(undef, n - m + 1)
        r = Memory{UInt64}(undef, m)
        divrem!(q, 0, r, 0, a, 0, n, d, 0, m)
        @test atoref(q, 0, n - m + 1) == aref ÷ dref
        @test atoref(r, 0, m) == aref % dref
    end

    # random sweep over sizes, normalized and unnormalized divisors
    for trial in 1:200
        n = rand(rng, 1:40); m = rand(rng, 1:n)
        a = amem(rand(rng, UInt64, n))
        d = amem(rand(rng, UInt64, m))
        d[m] == 0 && (d[m] = UInt64(1))
        rand(rng) < 0.5 && (d[m] |= UInt64(1) << 63)   # normalized divisor path
        checkdiv(a, n, d, m)
    end

    # adversarial: all-ones numerator, minimal/maximal normalized divisors
    for (n, m) in ((5, 2), (8, 3), (12, 7), (3, 3), (9, 2))
        ones_a = amem(fill(typemax(UInt64), n))
        for dv in (vcat(zeros(UInt64, m - 1), UInt64(1) << 63),       # d = B^m / 2
                   fill(typemax(UInt64), m),                          # d = B^m - 1
                   vcat(fill(typemax(UInt64), m - 1), UInt64(1) << 63))
            checkdiv(ones_a, n, amem(copy(dv)), m)
        end
    end

    # qhat == B-1 special case: numerator top limbs replicate the divisor's
    for trial in 1:50
        m = rand(rng, 2:6); n = m + rand(rng, 1:4)
        dref = atoref(amem(rand(rng, UInt64, m)), 0, m) | (big(1) << (64m - 1))
        # a = d * (B^k - 1) + small ⟹ quotient limbs of typemax
        k = n - m
        aref = dref * ((big(1) << (64k)) - 1) + rand(rng, big(0):dref-1)
        checkdiv(afrombig(aref, n), n, afrombig(dref, m), m)
    end

    # add-back stress: fat low divisor limbs make qhat overshoot as likely as possible
    for trial in 1:300
        m = rand(rng, 3:8); n = m + rand(rng, 1:6)
        dv = fill(typemax(UInt64), m); dv[m] = UInt64(1) << 63
        av = fill(typemax(UInt64), n)
        for i in 1:n
            rand(rng) < 0.3 && (av[i] = rand(rng, UInt64))
        end
        checkdiv(amem(av), n, amem(dv), m)
    end

    # exact multiples and quotient == 0
    for trial in 1:50
        m = rand(rng, 2:8); n = m + rand(rng, 0:6)
        dref = atoref(amem(rand(rng, UInt64, m)), 0, m)
        dref == 0 && continue
        dref |= big(1) << (64 * (m - 1))          # keep m limbs
        qref = rand(rng, big(0):(big(1) << (64 * (n - m))) - 1)
        aref = qref * dref
        aref < big(1) << (64n) || continue
        checkdiv(afrombig(aref, n), n, afrombig(dref, m), m)
        checkdiv(afrombig(dref - 1, m), m, afrombig(dref, m), m)   # a < d ⟹ q = 0
    end
end

@testset "divrem_dc!" begin
    rng = MersenneTwister(37)

    # Direct divrem_dc! check with a forced-low threshold to exercise deep
    # recursion. dref must be normalized (top bit of limb m set); numerator is
    # nn limbs with the top limb possibly nonzero (qh convention as divrem_bc!).
    function checkdc(aref::BigInt, dref::BigInt, nn::Int, m::Int, thr::Int)
        u = afrombig(aref, nn)
        d = afrombig(dref, m)
        v = invert_pi1(d[m], d[m-1])
        qn = nn - m
        q = Memory{UInt64}(undef, qn)
        qh = divrem_dc!(q, 0, u, 0, nn, d, 0, m, v, thr)
        qref, rref = divrem(aref, dref)
        @test (big(qh) << (64qn)) + atoref(q, 0, qn) == qref
        @test atoref(u, 0, m) == rref
    end

    randnorm(m) = atoref(amem(rand(rng, UInt64, m)), 0, m) | (big(1) << (64m - 1))

    # balanced (qn == m): random sweep, deep recursion via thr = 4
    for trial in 1:100
        m = rand(rng, 4:40)
        dref = randnorm(m)
        aref = rand(rng, big(0):(big(1) << (64 * 2m)) - 1)
        checkdc(aref, dref, 2m, m, 4)
    end

    # qn < m: truncate-and-correct path (needs qn >= thr to take the dc branch)
    for trial in 1:100
        m = rand(rng, 9:40)
        qn = rand(rng, 4:m-1)
        dref = randnorm(m)
        aref = rand(rng, big(0):(big(1) << (64 * (m + qn))) - 1)
        checkdc(aref, dref, m + qn, m, 4)
    end

    # qn > m: outer block loop, all leading-block sizes s = 1..m
    for trial in 1:100
        m = rand(rng, 4:16)
        qn = m + rand(rng, 1:3m)
        dref = randnorm(m)
        aref = rand(rng, big(0):(big(1) << (64 * (m + qn))) - 1)
        checkdc(aref, dref, m + qn, m, 4)
    end

    # qh = 1: numerator's top m limbs >= d
    for trial in 1:50
        m = rand(rng, 4:20)
        qn = rand(rng, 4:2m)
        dref = randnorm(m)
        aref = (dref << (64qn)) + rand(rng, big(0):(big(1) << (64qn)) - 1)
        checkdc(aref, dref, m + qn, m, 4)
    end

    # add-back stress: all-ones divisor tails and numerators drive the
    # block-correction (q -= 1, r += d) loops as hard as possible
    for trial in 1:200
        m = rand(rng, 4:24)
        qn = rand(rng, 4:2m)
        dv = fill(typemax(UInt64), m); dv[m] = UInt64(1) << 63
        rand(rng) < 0.5 && (dv[m] = typemax(UInt64))
        av = fill(typemax(UInt64), m + qn)
        for i in 1:m+qn
            rand(rng) < 0.25 && (av[i] = rand(rng, UInt64))
        end
        checkdc(atoref(amem(av), 0, m + qn), atoref(amem(dv), 0, m), m + qn, m, 4)
    end

    # exact multiples and tiny remainders
    for trial in 1:50
        m = rand(rng, 4:16)
        qn = rand(rng, 4:2m)
        dref = randnorm(m)
        qref = rand(rng, big(0):(big(1) << (64qn)) - 1)
        checkdc(qref * dref + rand(rng, big(0):big(1)), dref, m + qn, m, 4)
    end

    # production threshold: divrem! dispatches to dc above DC_DIV_THRESHOLD;
    # cross-check against BigInt at sizes straddling and well above it
    function checkdiv(a::Memory{UInt64}, n, d::Memory{UInt64}, m)
        aref = atoref(a, 0, n); dref = atoref(d, 0, m)
        q = Memory{UInt64}(undef, n - m + 1)
        r = Memory{UInt64}(undef, m)
        divrem!(q, 0, r, 0, a, 0, n, d, 0, m)
        @test atoref(q, 0, n - m + 1) == aref ÷ dref
        @test atoref(r, 0, m) == aref % dref
    end
    thr = DC_DIV_THRESHOLD
    for (n, m) in ((2thr, thr), (2thr + 1, thr + 1), (4thr, 2thr), (6thr, thr + 3),
                   (3thr, 2thr), (8thr, 3thr))
        a = amem(rand(rng, UInt64, n))
        d = amem(rand(rng, UInt64, m))
        d[m] == 0 && (d[m] = UInt64(1))
        rand(rng) < 0.5 && (d[m] |= UInt64(1) << 63)   # unnormalized divisor path too
        checkdiv(a, n, d, m)
    end
end

using NativeBigInt: barrett_setup, barrett_reduce!, powermod_limbs,
    BARRETT_THRESHOLD, BARRETT_EVEN_THRESHOLD

@testset "barrett_reduce!" begin
    rng = MersenneTwister(0xba44e77)

    function checkbar(Tref::BigInt, mref::BigInt, k::Int)
        mbuf = afrombig(mref, k)
        mu, lmu, scratch = barrett_setup(mbuf, 0, k)
        @test atoref(mu, 0, lmu) == (big(1) << (128k)) ÷ mref
        r = Memory{UInt64}(undef, k)
        barrett_reduce!(r, 0, afrombig(Tref, 2k), 0, mbuf, 0, k, mu, lmu, scratch, 0)
        @test atoref(r, 0, k) == mod(Tref, mref)
    end

    # random sweep: any T < β^2k, m with uniform top limb
    for trial in 1:200
        k = rand(rng, 1:40)
        mref = rand(rng, big(1) << (64k - 64):(big(1) << 64k) - 1)
        mref <= 1 && (mref = big(2))
        checkbar(rand(rng, big(0):(big(1) << 128k) - 1), mref, k)
    end

    # real usage shape (T = x·y with x, y < m) plus targeted 0/1/2-correction
    # values: T = q·m + r with r near 0 and near m
    for trial in 1:100
        k = rand(rng, 1:30)
        mref = rand(rng, big(1) << (64k - 64):(big(1) << 64k) - 1)
        mref <= 1 && (mref = big(3))
        x = rand(rng, big(0):mref-1)
        y = rand(rng, big(0):mref-1)
        checkbar(x * y, mref, k)
        q = (big(1) << 128k - 1) ÷ mref - 1
        checkbar(q * mref, mref, k)                      # r = 0
        checkbar(q * mref + rand(rng, big(0):mref-1), mref, k)
        checkbar(q * mref + mref - 1, mref, k)           # r = m - 1
    end

    # edges: m with minimal top limb, m = β^(k-1) (μ takes k+2 limbs),
    # m = β^k - 1, T near β^2k, T < m, T = 0
    for k in (1, 2, 3, 7, 20)
        small_top = (big(1) << (64k - 64)) | rand(rng, big(0):(big(1) << (64k - 64)) - 1)
        for mref in (small_top, big(1) << (64k - 64), (big(1) << 64k) - 1)
            mref <= 1 && continue
            checkbar((big(1) << 128k) - 1, mref, k)
            checkbar(big(0), mref, k)
            checkbar(mref - 1, mref, k)
            checkbar(mref + 1, mref, k)
            checkbar((mref - 1)^2, mref, k)
        end
    end
end

@testset "powermod_limbs Barrett path" begin
    rng = MersenneTwister(0xb42)
    # force the Barrett branch at small k (cheap) and cross-check vs BigInt,
    # odd and even moduli, exponents crossing the window-size breakpoints
    for trial in 1:60
        k = rand(rng, 1:25)
        mref = rand(rng, big(2):(big(1) << 64k) - 1)
        mref |= big(1) << (64k - 64)                    # keep k limbs
        trial % 2 == 0 && iseven(mref) && (mref += 1)   # both parities
        bref = rand(rng, big(1):mref-1)
        e = rand(rng, big(1):big(2)^rand(rng, (5, 30, 100)))
        lb = max(cld(ndigits(bref, base=2), 64), 1)
        r = powermod_limbs(afrombig(bref, lb), lb, e, afrombig(mref, k), k, true)
        @test atoref(r, 0, k) == powermod(bref, e, mref)
    end
    # production dispatch at the per-parity thresholds, small exponent
    for (k, low) in ((BARRETT_THRESHOLD, 1), (BARRETT_EVEN_THRESHOLD, 0))
        mref = rand(rng, big(1) << (64k - 1):(big(1) << 64k) - 1)
        mref = (mref & ~big(1)) | low
        bref = rand(rng, big(1):mref-1)
        r = powermod_limbs(afrombig(bref, k), k, big(65537), afrombig(mref, k), k)
        @test atoref(r, 0, k) == powermod(bref, big(65537), mref)
    end
end

using NativeBigInt: HgcdMatrix, hgcd_matrix_cap, hgcd!, gcd!, gcdext!, normlen

@testset "hgcd" begin
    rng = MersenneTwister(0x59cd)

    # hgcd! contract, BigInt-verified: (A; B) == M * (a'; b') exactly,
    # det(M) == +1, both outputs > s = n÷2 + 1 limbs, and nn == 0 leaves M
    # the identity. Tiny thresholds force deep recursion on small inputs.
    mval(M) = (atoref(M.m00, 0, M.n), atoref(M.m01, 0, M.n),
               atoref(M.m10, 0, M.n), atoref(M.m11, 0, M.n))
    for trial in 1:600
        n = rand(rng, 3:40)
        thr = rand(rng, (4, 8, 1000))
        a0 = rand(rng, big(1):big(2)^(64n) - 1)
        b0 = rand(rng, big(1):big(2)^(64n) - 1)
        trial % 5 == 0 && (b0 = rand(rng, big(1):big(2)^(64 * max(n ÷ 2, 1)) - 1))
        trial % 7 == 0 && (b0 = max(a0 - rand(rng, big(1):big(2)^32), big(1)))
        n = max(cld(max(ndigits(a0, base=2), ndigits(b0, base=2)), 64), 1)
        a = afrombig(a0, n + 1)
        b = afrombig(b0, n + 1)
        M = HgcdMatrix(hgcd_matrix_cap(n))
        ra, rb, rn = hgcd!(a, b, n, M, thr)
        m00, m01, m10, m11 = mval(M)
        if rn == 0
            @test (m00, m01, m10, m11) == (1, 0, 0, 1)
        else
            s = n ÷ 2 + 1
            an = normlen(ra, 0, rn)
            bn = normlen(rb, 0, rn)
            av = atoref(ra, 0, an)
            bv = atoref(rb, 0, bn)
            @test m00 * av + m01 * bv == a0
            @test m10 * av + m11 * bv == b0
            @test m00 * m11 - m01 * m10 == 1
            @test an > s && bn > s
        end
    end

    # DC driver differential vs BigInt with tiny thresholds (deep recursion),
    # including planted common factors, Fibonacci pairs (all-quotient-1
    # chains), and quotient spikes (subdiv + q-1 guard).
    for trial in 1:300
        la, lb = rand(rng, 1:50), rand(rng, 1:50)
        a0 = rand(rng, big(1):big(2)^(64la) - 1)
        b0 = rand(rng, big(1):big(2)^(64lb) - 1)
        if isodd(trial)
            g = rand(rng, big(1):big(2)^(64 * rand(rng, 1:8)))
            a0 *= g; b0 *= g
        end
        if trial % 11 == 0
            x, y = big(1), big(1)
            while ndigits(y, base=2) < 64la
                x, y = y, x + y
            end
            a0, b0 = y, x
        end
        trial % 13 == 0 &&
            (b0 = a0 * rand(rng, big(2)^200:big(2)^220) + rand(rng, big(1):a0))
        cap = max(cld(ndigits(a0, base=2), 64), cld(ndigits(b0, base=2), 64)) + 1
        lu = cld(ndigits(a0, base=2), 64)
        lv = cld(ndigits(b0, base=2), 64)
        g1, lg = gcd!(afrombig(a0, cap), lu, afrombig(b0, cap), lv;
                      dc_thr=10, hgcd_thr=6)
        @test atoref(g1, 0, lg) == gcd(a0, b0)
        g2, lg, tm, lt, tpos = gcdext!(afrombig(a0, cap + 3), lu,
                                       afrombig(b0, cap + 3), lv;
                                       dc_thr=10, hgcd_thr=6)
        gv = atoref(g2, 0, lg)
        tv = atoref(tm, 0, lt) * (tpos ? 1 : -1)
        @test gv == gcd(a0, b0)
        @test mod(gv - tv * b0, a0) == 0        # Bézout: s*a + t*b == g
        @test abs(tv) <= max(a0, b0) ÷ gv || gv == b0
    end

    # production thresholds: sizes straddling GCDEXT_DC/GCD_DC and well above
    for n in (280, 320, 500, 800)
        a0 = rand(rng, big(1):big(2)^(64n) - 1)
        b0 = rand(rng, big(1):big(2)^(64n) - 1)
        g = rand(rng, big(1):big(2)^320)
        a0 *= g; b0 *= g
        cap = max(cld(ndigits(a0, base=2), 64), cld(ndigits(b0, base=2), 64)) + 1
        lu = cld(ndigits(a0, base=2), 64)
        lv = cld(ndigits(b0, base=2), 64)
        g1, lg = gcd!(afrombig(a0, cap), lu, afrombig(b0, cap), lv)
        @test atoref(g1, 0, lg) == gcd(a0, b0)
        g2, lg, tm, lt, tpos = gcdext!(afrombig(a0, cap + 3), lu,
                                       afrombig(b0, cap + 3), lv)
        gv = atoref(g2, 0, lg)
        tv = atoref(tm, 0, lt) * (tpos ? 1 : -1)
        @test gv == gcd(a0, b0)
        @test mod(gv - tv * b0, a0) == 0
    end
end
