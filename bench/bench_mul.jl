# Local-only: mul! vs __gmpn_mul. Run: julia --startup-file=no --project=. bench/bench_mul.jl
# Above the NTT thresholds mul! dispatches to mul_ntt!; the kara/toom columns
# call mul_kar!/mul_toom3! directly to time the paths the NTT replaced, for
# tuning MUL_NTT_MIN/MUL_NTT_THRESHOLD.
using NativeBigInt, BenchmarkTools, Random
using NativeBigInt: Limb, mul!, mul_kar!, mul_toom3!, kar_scratch_len,
                    mul_scratch_len, MUL_NTT_THRESHOLD

# large sizes are µs–ms scale; a 0.25 s budget per measurement is plenty
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.25

function gmpn_mul!(r::Memory{Limb}, a::Memory{Limb}, m::Int, b::Memory{Limb}, n::Int)
    ccall((:__gmpn_mul, :libgmp), Limb,
          (Ptr{Limb}, Ptr{Limb}, Clong, Ptr{Limb}, Clong), r, a, m, b, n)
    return r
end

rng = MersenneTwister(1)
for n in (8, 16, 25, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
          1536, 2048)
    a = Memory{Limb}(undef, n); rand!(rng, a)
    b = Memory{Limb}(undef, n); rand!(rng, b)
    r = Memory{Limb}(undef, 2n)
    t_n = @belapsed mul!($r, 0, $a, 0, $n, $b, 0, $n)
    t_g = @belapsed gmpn_mul!($r, $a, $n, $b, $n)
    line = "n=$n limbs ($(64n) bits)  nbig=$(round(t_n*1e9, digits=1))ns  gmp=$(round(t_g*1e9, digits=1))ns  ratio=$(round(t_n/t_g, digits=2))"
    if n >= MUL_NTT_THRESHOLD
        scratch = Memory{Limb}(undef, max(kar_scratch_len(n), mul_scratch_len(n)))
        t_k = @belapsed mul_kar!($r, 0, $a, 0, $b, 0, $n, $scratch, 0)
        t_t = @belapsed mul_toom3!($r, 0, $a, 0, $b, 0, $n, $scratch, 0)
        line *= "  kara=$(round(t_k*1e9, digits=1))ns  toom=$(round(t_t*1e9, digits=1))ns  ntt/toom=$(round(t_n/t_t, digits=2))"
    end
    println(line)
    flush(stdout)
end
