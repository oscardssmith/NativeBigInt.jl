# Local kernel micro-benchmarks (never committed): NativeBigInt kernels vs GMP mpn.
using BenchmarkTools, NativeBigInt
using NativeBigInt: Limb, add_n!, sub_n!, mul_1!, addmul_1!, submul_1!,
                    mul_basecase!, lshift!, rshift!

# non-inlined wrappers so there is a standalone specialization to time
@noinline add_n_wrap!(r, a, b, n)    = add_n!(r, 0, a, 0, b, 0, n)
@noinline sub_n_wrap!(r, a, b, n)    = sub_n!(r, 0, a, 0, b, 0, n)
@noinline mul_1_wrap!(r, a, n, x)    = mul_1!(r, 0, a, 0, n, x)
@noinline addmul_1_wrap!(r, a, n, x) = addmul_1!(r, 0, a, 0, n, x)
@noinline submul_1_wrap!(r, a, n, x) = submul_1!(r, 0, a, 0, n, x)
@noinline basecase_wrap!(r, a, b, n) = mul_basecase!(r, 0, a, 0, n, b, 0, n)
@noinline lshift_wrap!(r, a, n, c)   = lshift!(r, 0, a, 0, n, c)
@noinline rshift_wrap!(r, a, n, c)   = rshift!(r, 0, a, 0, n, c)

g_add!(r, a, b, n)  = ccall((:__gmpn_add_n, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Ptr{Limb}, Clong), r, a, b, n)
g_sub!(r, a, b, n)  = ccall((:__gmpn_sub_n, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Ptr{Limb}, Clong), r, a, b, n)
g_mul1!(r, a, n, x) = ccall((:__gmpn_mul_1, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Clong, Limb), r, a, n, x)
g_am1!(r, a, n, x)  = ccall((:__gmpn_addmul_1, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Clong, Limb), r, a, n, x)
g_sm1!(r, a, n, x)  = ccall((:__gmpn_submul_1, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Clong, Limb), r, a, n, x)
g_mbc!(r, a, m, b, n) = ccall((:__gmpn_mul_basecase, :libgmp), Cvoid, (Ptr{Limb}, Ptr{Limb}, Clong, Ptr{Limb}, Clong), r, a, m, b, n)
g_lsh!(r, a, n, c)  = ccall((:__gmpn_lshift, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Clong, Cuint), r, a, n, c)
g_rsh!(r, a, n, c)  = ccall((:__gmpn_rshift, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Clong, Cuint), r, a, n, c)

const NS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
const NAMES = ("add_n", "sub_n", "mul_1", "addmul_1", "submul_1", "basecase", "lshift", "rshift")
results = Dict(name => Float64[] for name in NAMES)

for n in NS
    a = Memory{Limb}(rand(Limb, n)); b = Memory{Limb}(rand(Limb, n))
    r = Memory{Limb}(rand(Limb, 2n + 1))
    x = rand(Limb); cnt = 13

    push!(results["add_n"],    (@belapsed add_n_wrap!($r, $a, $b, $n))    / (@belapsed g_add!($r, $a, $b, $n)))
    push!(results["sub_n"],    (@belapsed sub_n_wrap!($r, $a, $b, $n))    / (@belapsed g_sub!($r, $a, $b, $n)))
    push!(results["mul_1"],    (@belapsed mul_1_wrap!($r, $a, $n, $x))    / (@belapsed g_mul1!($r, $a, $n, $x)))
    push!(results["addmul_1"], (@belapsed addmul_1_wrap!($r, $a, $n, $x)) / (@belapsed g_am1!($r, $a, $n, $x)))
    push!(results["submul_1"], (@belapsed submul_1_wrap!($r, $a, $n, $x)) / (@belapsed g_sm1!($r, $a, $n, $x)))
    push!(results["basecase"], (@belapsed basecase_wrap!($r, $a, $b, $n)) / (@belapsed g_mbc!($r, $a, $n, $b, $n)))
    push!(results["lshift"],   (@belapsed lshift_wrap!($r, $a, $n, $cnt)) / (@belapsed g_lsh!($r, $a, $n, $cnt)))
    push!(results["rshift"],   (@belapsed rshift_wrap!($r, $a, $n, $cnt)) / (@belapsed g_rsh!($r, $a, $n, $cnt)))
    println("done n=$n"); flush(stdout)
end

println("\nratio (NativeBigInt / GMP), lower is better")
println(rpad("kernel", 10), join(rpad("n=$n", 7) for n in NS))
for name in NAMES
    println(rpad(name, 10), join(rpad(round(r, digits=2), 7) for r in results[name]))
end
