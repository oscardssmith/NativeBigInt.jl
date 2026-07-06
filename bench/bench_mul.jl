# Local-only: mul! vs __gmpn_mul. Run: julia --startup-file=no --project=. bench/bench_mul.jl
using NativeBigInt, BenchmarkTools, Random
using NativeBigInt: Limb, mul!

function gmpn_mul!(r::Memory{Limb}, a::Memory{Limb}, m::Int, b::Memory{Limb}, n::Int)
    ccall((:__gmpn_mul, :libgmp), Limb,
          (Ptr{Limb}, Ptr{Limb}, Clong, Ptr{Limb}, Clong), r, a, m, b, n)
    return r
end

rng = MersenneTwister(1)
for n in (8, 16, 25, 32, 48, 64, 96, 128, 192, 256)
    a = Memory{Limb}(undef, n); rand!(rng, a)
    b = Memory{Limb}(undef, n); rand!(rng, b)
    r = Memory{Limb}(undef, 2n)
    t_n = @belapsed mul!($r, 0, $a, 0, $n, $b, 0, $n)
    t_g = @belapsed gmpn_mul!($r, $a, $n, $b, $n)
    println("n=$n  nbig=$(round(t_n*1e9, digits=1))ns  gmp=$(round(t_g*1e9, digits=1))ns  ratio=$(round(t_n/t_g, digits=2))")
    flush(stdout)
end
