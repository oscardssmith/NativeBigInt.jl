# Primes.jl-level benchmark: NBig vs Base.BigInt as the integer type driving
# isprime / nextprime / factor. Needs the dev'd Primes with the generic
# isprime(::Integer) path (otherwise NBig would be converted to BigInt and
# this would benchmark GMP against itself).
#
# Run: julia --startup-file=no --project=bench bench/bench_primes.jl

using BenchmarkTools, NativeBigInt, Primes, Random, Printf

# deterministic operands; the same numeric value feeds both types
rng = Xoshiro(42)
randprime(bits) = nextprime(rand(rng, big(2)^(bits - 1):big(2)^bits))

function row(op, bits, f, x::BigInt)
    tB = @belapsed $f($x) seconds = 1
    xN = NBig(x)
    tN = @belapsed $f($xN) seconds = 1
    @printf("| %-18s | %4d | %10.3f | %10.3f | %5.2fx |\n",
            op, bits, 1e3 * tN, 1e3 * tB, tN / tB)
end

println("| op                 | bits | NBig (ms)  | BigInt (ms)| NBig/BigInt |")
println("|--------------------|------|------------|------------|-------------|")
for bits in (128, 256, 512, 1024, 2048)
    p = randprime(bits)
    # composite with no small factors: fails in the first Miller-Rabin round
    c = randprime(bits ÷ 2) * randprime(bits - bits ÷ 2)
    # semiprime with one ~20-bit factor: Pollard rho finds it quickly
    s = randprime(20) * randprime(bits - 20)
    start = rand(rng, big(2)^(bits - 1):big(2)^bits)

    row("isprime(prime)", bits, isprime, p)
    row("isprime(composite)", bits, isprime, c)
    row("nextprime", bits, nextprime, start)
    row("factor(semiprime)", bits, factor, s)
end
