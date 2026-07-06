# Local-only: row kernels vs GMP counterparts, to locate the basecase gap.
using NativeBigInt, BenchmarkTools, Random
using NativeBigInt: Limb, mul_1!, addmul_1!, addmul_2!

gmpn_mul_1(r, a, n, b) = ccall((:__gmpn_mul_1, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Clong, Limb), r, a, n, b)
gmpn_addmul_1(r, a, n, b) = ccall((:__gmpn_addmul_1, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Clong, Limb), r, a, n, b)
gmpn_addmul_2(r, a, n, bp) = ccall((:__gmpn_addmul_2, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Clong, Ptr{Limb}), r, a, n, bp)
gmpn_mul_basecase(r, a, m, b, n) = ccall((:__gmpn_mul_basecase, :libgmp), Cvoid, (Ptr{Limb}, Ptr{Limb}, Clong, Ptr{Limb}, Clong), r, a, m, b, n)

rng = MersenneTwister(2)
for m in (8, 16, 24, 32)
    a = Memory{Limb}(undef, m); rand!(rng, a)
    r = Memory{Limb}(undef, m + 2); rand!(rng, r)
    bp = Memory{Limb}(undef, 2); rand!(rng, bp)
    b = bp[1]
    t = @belapsed mul_1!($r, 0, $a, 0, $m, $b);      g = @belapsed gmpn_mul_1($r, $a, $m, $b)
    println("m=$m  mul_1!     $(round(t*1e9, digits=1))ns vs $(round(g*1e9, digits=1))ns  ratio=$(round(t/g, digits=2))")
    t = @belapsed addmul_1!($r, 0, $a, 0, $m, $b);   g = @belapsed gmpn_addmul_1($r, $a, $m, $b)
    println("m=$m  addmul_1!  $(round(t*1e9, digits=1))ns vs $(round(g*1e9, digits=1))ns  ratio=$(round(t/g, digits=2))")
    t = @belapsed addmul_2!($r, 0, $a, 0, $m, $bp[1], $bp[2]); g = @belapsed gmpn_addmul_2($r, $a, $m, $bp)
    println("m=$m  addmul_2!  $(round(t*1e9, digits=1))ns vs $(round(g*1e9, digits=1))ns  ratio=$(round(t/g, digits=2))")
    flush(stdout)
end

# whole basecase at the sizes Karatsuba bottoms out at
for n in (16, 24, 32)
    a = Memory{Limb}(undef, n); rand!(rng, a)
    b = Memory{Limb}(undef, n); rand!(rng, b)
    r = Memory{Limb}(undef, 2n)
    t = @belapsed NativeBigInt.mul_basecase!($r, 0, $a, 0, $n, $b, 0, $n)
    g = @belapsed gmpn_mul_basecase($r, $a, $n, $b, $n)
    println("n=$n  mul_basecase!  $(round(t*1e9, digits=1))ns vs $(round(g*1e9, digits=1))ns  ratio=$(round(t/g, digits=2))")
    flush(stdout)
end
