# Local high-level benchmark (results feed the README table): NBig vs
# Base.BigInt end-to-end ops, 128 bits to 32k bits. Prints a markdown table
# of time ratios (NBig / BigInt, lower is better). Run alone — a concurrent
# bench process corrupts the timings. Notes: divrem is a 2n/n shape; gcd/gcdx
# operands share a planted n/2-bit factor; powermod uses a 512-bit-capped
# exponent so large sizes stay benchable; string/parse are decimal.
using BenchmarkTools, NativeBigInt, Random

const BITSIZES = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
const OPS = ("+", "-", "*", "divrem", "gcd", "gcdx", "isqrt",
             "powermod odd", "powermod even", "string", "parse")

rng = MersenneTwister(0xbe9c)
randbits(b) = rand(rng, big(2)^(b-1):big(2)^b-1)

results = Dict(op => Float64[] for op in OPS)
ratio!(op, tn, tb) = push!(results[op], tn / tb)

for bits in BITSIZES
    ab, bb = randbits(bits), randbits(bits)
    db = randbits(bits ÷ 2)                     # divisor: 2n/n divrem
    g = randbits(bits ÷ 2)                      # planted gcd factor
    gab, gbb = ab * g, bb * g
    modd = ab | 1
    meven = ab & ~big(1)
    base = randbits(bits - 1)
    e = randbits(min(bits, 512))
    s = string(ab)
    an, bn, dn = NBig(ab), NBig(bb), NBig(db)
    gan, gbn = NBig(gab), NBig(gbb)
    modn, meven_n, basen, en = NBig(modd), NBig(meven), NBig(base), NBig(e)

    ratio!("+", (@belapsed $an + $bn), (@belapsed $ab + $bb))
    ratio!("-", (@belapsed $an - $bn), (@belapsed $ab - $bb))
    ratio!("*", (@belapsed $an * $bn), (@belapsed $ab * $bb))
    ratio!("divrem", (@belapsed divrem($an, $dn)), (@belapsed divrem($ab, $db)))
    ratio!("gcd", (@belapsed gcd($gan, $gbn)), (@belapsed gcd($gab, $gbb)))
    ratio!("gcdx", (@belapsed gcdx($gan, $gbn)), (@belapsed gcdx($gab, $gbb)))
    ratio!("isqrt", (@belapsed isqrt($an)), (@belapsed isqrt($ab)))
    ratio!("powermod odd", (@belapsed powermod($basen, $en, $modn)),
           (@belapsed powermod($base, $e, $modd)))
    ratio!("powermod even", (@belapsed powermod($basen, $en, $meven_n)),
           (@belapsed powermod($base, $e, $meven)))
    ratio!("string", (@belapsed string($an)), (@belapsed string($ab)))
    ratio!("parse", (@belapsed parse(NBig, $s)), (@belapsed parse(BigInt, $s)))

    println("done bits=$bits"); flush(stdout)
end

println("\nratio (NativeBigInt / BigInt), lower is better\n")
hdr(b) = b >= 1024 ? "$(b ÷ 1024)k" : "$b"
println("| op | ", join((hdr(b) for b in BITSIZES), " | "), " |")
println("|---|", "---|"^length(BITSIZES))
for op in OPS
    parts = split(op)
    lbl = length(parts) == 1 ? "`$op`" : "`$(parts[1])` ($(parts[2]))"
    println("| ", lbl, " | ",
            join((round(r, digits=2) for r in results[op]), " | "), " |")
end
