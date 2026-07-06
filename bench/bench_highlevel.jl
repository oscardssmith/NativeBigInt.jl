# Local high-level benchmark (never committed): NBig vs Base.BigInt end-to-end ops.
using BenchmarkTools, NativeBigInt, Random

const BITSIZES = (128, 256, 512, 1024, 2048, 4096)
const OPS = ("+", "-", "*", "divrem")

results = Dict(op => Float64[] for op in OPS)

for bits in BITSIZES
    ab = rand(1:(big(2)^bits - 1))
    bb = rand(1:(big(2)^(bits ÷ 2) - 1)) + 1  # smaller divisor, avoids trivial divrem
    an, bn = NBig(ab), NBig(bb)

    t_add_n = @belapsed $an + $an
    t_add_b = @belapsed $ab + $ab
    push!(results["+"], t_add_n / t_add_b)

    t_sub_n = @belapsed $an - $bn
    t_sub_b = @belapsed $ab - $bb
    push!(results["-"], t_sub_n / t_sub_b)

    t_mul_n = @belapsed $an * $an
    t_mul_b = @belapsed $ab * $ab
    push!(results["*"], t_mul_n / t_mul_b)

    t_div_n = @belapsed divrem($an, $bn)
    t_div_b = @belapsed divrem($ab, $bb)
    push!(results["divrem"], t_div_n / t_div_b)

    println("done bits=$bits"); flush(stdout)
end

println("\nratio (NativeBigInt / BigInt), lower is better")
println(rpad("op", 10), join(rpad("$(b)b", 8) for b in BITSIZES))
for op in OPS
    println(rpad(op, 10), join(rpad(round(r, digits=2), 8) for r in results[op]))
end
