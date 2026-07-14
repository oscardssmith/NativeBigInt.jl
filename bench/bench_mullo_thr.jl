# Local-only: mullo!/sqrlo! threshold sweep — short product vs the full
# product it replaces (mul!/sqr! at the same operand shape, low k kept), and
# the raw truncated basecase vs the dispatched path (tunes
# MULLO_BASECASE_THRESHOLD / SQRLO_BASECASE_THRESHOLD; if the basecase column
# beats the dispatch column at a size, the threshold belongs above it).
# Balanced la = lb = k is the Mulders-worst shape; la = k is the sqrt-chain
# shape for sqrlo.
using NativeBigInt, BenchmarkTools, Random
using NativeBigInt: Limb, mul!, sqr!, mullo!, sqrlo!, mullo_basecase!,
    sqrlo_basecase!, mullo_scratch_len, sqrlo_scratch_len

rng = MersenneTwister(1)
println("k     mullo/mul   mullo_bc/mul   sqrlo/sqr   sqrlo_bc/sqr")
for k in (16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 256, 320,
          384, 448, 512, 640, 768)
    a = Memory{Limb}(undef, k); rand!(rng, a)
    b = Memory{Limb}(undef, k); rand!(rng, b)
    r = Memory{Limb}(undef, 2k + 2)
    ms = Memory{Limb}(undef, max(1, mullo_scratch_len(k)))
    ss = Memory{Limb}(undef, max(1, sqrlo_scratch_len(k)))
    fm = @belapsed mul!($r, 0, $a, 0, $k, $b, 0, $k)
    fs = @belapsed sqr!($r, 0, $a, 0, $k)
    lo = @belapsed mullo!($r, 0, $a, 0, $k, $b, 0, $k, $k, $ms, 0)
    slo = @belapsed sqrlo!($r, 0, $a, 0, $k, $k, $ss, 0)
    # raw basecases (precondition la + lb > k + 2 holds at la = lb = k > 2)
    lob = k <= 320 ? (@belapsed mullo_basecase!($r, 0, $a, 0, $k, $b, 0, $k, $k)) : NaN
    slob = k <= 320 ? (@belapsed sqrlo_basecase!($r, 0, $a, 0, $k, $k)) : NaN
    println(rpad("k=$k", 6), rpad(round(lo / fm, digits=2), 12),
            rpad(round(lob / fm, digits=2), 15),
            rpad(round(slo / fs, digits=2), 12), round(slob / fs, digits=2))
    flush(stdout)
end
