# Local-only: sweep for BARRETT_THRESHOLD (odd m, vs Montgomery redc!) and
# BARRETT_EVEN_THRESHOLD (even m, vs divrem!): powermod_limbs with Barrett
# forced on vs off, fixed 512-bit exponent so the mul/reduce count per k is
# constant. The barrett flag on powermod_limbs forces the path. Run this
# alone — a concurrent bench process corrupts the timings. Last tuned
# 2026-07: odd crossover ~68, even ~240 (where Barrett's products hit the
# fp NTT; below that DC division holds its own).
using NativeBigInt, BenchmarkTools, Random
using NativeBigInt: Limb, powermod_limbs

rng = MersenneTwister(1)
e = rand(rng, big(2)^511:big(2)^512-1)

println("k      odd-cur    odd-bar    ratio  |  even-cur   even-bar   ratio")
for k in (4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 80, 96, 128, 192, 256)
    m = Memory{Limb}(undef, k); rand!(rng, m)
    m[k] |= Limb(1) << 62                 # top limb well away from 0
    b = Memory{Limb}(undef, k); rand!(rng, b)
    b[k] = m[k] >> 1                      # guarantees b < m
    row = String[]
    for par in (1, 0)
        m[1] = (m[1] & ~Limb(1)) | Limb(par)
        cur = @belapsed powermod_limbs($b, $k, $e, $m, $k, false) seconds = 2
        bar = @belapsed powermod_limbs($b, $k, $e, $m, $k, true) seconds = 2
        push!(row, string(rpad(round(cur * 1e6, digits=1), 11),
                          rpad(round(bar * 1e6, digits=1), 11),
                          rpad(round(bar / cur, digits=2), 5)))
    end
    println(rpad("k=$k", 7), row[1], "|  ", row[2], "   (µs)")
    flush(stdout)
end
