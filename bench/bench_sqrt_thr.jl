# Tune SQRT_DIVAPPR_THRESHOLD: the top-level lq (quotient limbs) above which
# root-only sqrt uses divappr! + guard-limb certificate instead of exact
# divrem!. Sweeps the threshold per bit size against the exact path ("off")
# and GMP. Requires SQRT_DIVAPPR_THRESHOLD to be a mutable typed global while
# tuning. Run alone — a concurrent bench process corrupts the timings.
using BenchmarkTools, NativeBigInt, Random

const BITS = (2048, 4096, 8192, 16384, 32768, 65536)
const THRS = (4, 8, 16, 24, 32, 48, 64, 96, 128)

rng = MersenneTwister(0xbe9c)
println("| bits | limbs | lq | gmp(µs) | off | ", join(("thr=$t" for t in THRS), " | "), " |")
println("|---", "|---" ^ (4 + 1 + length(THRS)), "|")
for bits in BITS
    ab = rand(rng, big(2)^(bits-1):big(2)^bits-1)
    an = NBig(ab)
    n = cld(bits, 64); lq = (n + 1) >> 2
    tg = @belapsed isqrt($ab)
    NativeBigInt.SQRT_DIVAPPR_THRESHOLD = typemax(Int) >> 1
    toff = @belapsed isqrt($an)
    row = ["$bits", "$n", "$lq", string(round(tg * 1e6, digits=1)),
           string(round(toff / tg, digits=2))]
    for t in THRS
        NativeBigInt.SQRT_DIVAPPR_THRESHOLD = t
        push!(row, string(round((@belapsed isqrt($an)) / tg, digits=2)))
    end
    println("| ", join(row, " | "), " |")
    flush(stdout)
end
