# Local-only: MUL_TOOM3_THRESHOLD / SQR_TOOM3_THRESHOLD sweep, all candidates
# in one process. A huge thr column is pure Karatsuba for that size, so the
# ratio vs it shows what Toom-3 buys. Ratios are vs GMP's mpn layer.
using NativeBigInt, BenchmarkTools, Random
using NativeBigInt: Limb, mul_toom3!, mul_kar!, mul_scratch_len, kar_scratch_len,
                    sqr_toom3!, sqr_kar!, sqr_scratch_len, SQR_KARATSUBA_THRESHOLD

gmpn_mul(r, a, m, b, n) = ccall((:__gmpn_mul, :libgmp), Limb,
    (Ptr{Limb}, Ptr{Limb}, Clong, Ptr{Limb}, Clong), r, a, m, b, n)
gmpn_sqr(r, a, n) = ccall((:__gmpn_sqr, :libgmp), Cvoid,
    (Ptr{Limb}, Ptr{Limb}, Clong), r, a, n)

const THRS = (60, 80, 100, 120, 150, 200)
const SIZES = (64, 96, 128, 160, 200, 256, 320, 448, 640, 896)
rng = MersenneTwister(1)

println("mul: n    ", join(("T=$t" for t in THRS), "   "), "   kar   (ratio vs gmp)")
for n in SIZES
    a = Memory{Limb}(undef, n); rand!(rng, a)
    b = Memory{Limb}(undef, n); rand!(rng, b)
    r = Memory{Limb}(undef, 2n)
    scratch = Memory{Limb}(undef, max(mul_scratch_len(n, minimum(THRS)), kar_scratch_len(n)))
    g = @belapsed gmpn_mul($r, $a, $n, $b, $n)
    ratios = [t <= n ? round((@belapsed mul_toom3!($r, 0, $a, 0, $b, 0, $n, $scratch, 0, $t)) / g, digits=2) :
              missing for t in THRS]
    kar = round((@belapsed mul_kar!($r, 0, $a, 0, $b, 0, $n, $scratch, 0)) / g, digits=2)
    println("n=$n  ", join(ratios, "  "), "  kar=", kar)
    flush(stdout)
end

println("\nsqr: n    ", join(("T=$t" for t in THRS), "   "), "   kar   (ratio vs gmp)")
for n in SIZES
    a = Memory{Limb}(undef, n); rand!(rng, a)
    r = Memory{Limb}(undef, 2n)
    scratch = Memory{Limb}(undef, max(sqr_scratch_len(n, minimum(THRS)),
                                      kar_scratch_len(n, SQR_KARATSUBA_THRESHOLD)))
    g = @belapsed gmpn_sqr($r, $a, $n)
    ratios = [t <= n ? round((@belapsed sqr_toom3!($r, 0, $a, 0, $n, $scratch, 0, $t)) / g, digits=2) :
              missing for t in THRS]
    kar = round((@belapsed sqr_kar!($r, 0, $a, 0, $n, $scratch, 0)) / g, digits=2)
    println("n=$n  ", join(ratios, "  "), "  kar=", kar)
    flush(stdout)
end
