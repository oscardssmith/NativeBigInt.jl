# Local-only: KARATSUBA_THRESHOLD sweep, all candidates in one process.
using NativeBigInt, BenchmarkTools, Random
using NativeBigInt: Limb, mul!

gmpn_mul(r, a, m, b, n) = ccall((:__gmpn_mul, :libgmp), Limb,
    (Ptr{Limb}, Ptr{Limb}, Clong, Ptr{Limb}, Clong), r, a, m, b, n)

const THRS = (25, 29, 33, 37, 41, 49, 57, 65)
rng = MersenneTwister(1)
println("n    ", join(("T=$t" for t in THRS), "   "), "   (ratio vs gmp)")
for n in (32, 48, 64, 80, 96, 128, 192, 256)
    a = Memory{Limb}(undef, n); rand!(rng, a)
    b = Memory{Limb}(undef, n); rand!(rng, b)
    r = Memory{Limb}(undef, 2n)
    g = @belapsed gmpn_mul($r, $a, $n, $b, $n)
    ratios = [round((@belapsed mul!($r, 0, $a, 0, $n, $b, 0, $n, $t)) / g, digits=2) for t in THRS]
    println("n=$n  ", join(ratios, "  "))
    flush(stdout)
end
