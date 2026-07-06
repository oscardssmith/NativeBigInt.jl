# gcd / isqrt / powermod: NBig vs Base.BigInt (GMP) across the target regime.
# Local-only; run with: julia --startup-file=no --project=. bench/bench_numtheory.jl
using NativeBigInt, BenchmarkTools, Random, Printf

rng = MersenneTwister(0xbe9c)
randbits(b) = rand(rng, big(2)^(b-1):big(2)^b-1)

println("op        bits      BigInt      NBig      ratio (NBig/BigInt)")
for bits in (128, 256, 512, 1024, 2048, 4096)
    a, b = randbits(bits), randbits(bits)
    g = randbits(bits ÷ 2)
    ag, bg = a * g, b * g
    na, nb = NBig(ag), NBig(bg)
    t1 = @belapsed gcd($ag, $bg)
    t2 = @belapsed gcd($na, $nb)
    @printf("gcd     %5d  %9.3g s %9.3g s   %5.2f\n", bits, t1, t2, t2 / t1)
end
for bits in (128, 256, 512, 1024, 2048, 4096)
    x = randbits(2bits)
    nx = NBig(x)
    t1 = @belapsed isqrt($x)
    t2 = @belapsed isqrt($nx)
    @printf("isqrt   %5d  %9.3g s %9.3g s   %5.2f\n", 2bits, t1, t2, t2 / t1)
end
for bits in (128, 256, 512, 1024, 2048, 4096)
    m = randbits(bits) | 1   # odd modulus (Miller–Rabin shape)
    a = randbits(bits - 1)
    e = randbits(bits)
    nm, na, ne = NBig(m), NBig(a), NBig(e)
    t1 = @belapsed powermod($a, $e, $m)
    t2 = @belapsed powermod($na, $ne, $nm)
    @printf("powmod  %5d  %9.3g s %9.3g s   %5.2f\n", bits, t1, t2, t2 / t1)
end
