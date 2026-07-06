# Dump native assembly for the hot kernels (local only).
using NativeBigInt
using NativeBigInt: Limb, add_n!, sub_n!, mul_1!, addmul_1!, submul_1!, mul_basecase!
using InteractiveUtils

@noinline add_n_wrap!(r, a, b, n)    = add_n!(r, 0, a, 0, b, 0, n)
@noinline sub_n_wrap!(r, a, b, n)    = sub_n!(r, 0, a, 0, b, 0, n)
@noinline mul_1_wrap!(r, a, n, x)    = mul_1!(r, 0, a, 0, n, x)
@noinline addmul_1_wrap!(r, a, n, x) = addmul_1!(r, 0, a, 0, n, x)
@noinline submul_1_wrap!(r, a, n, x) = submul_1!(r, 0, a, 0, n, x)
@noinline basecase_wrap!(r, a, b, n) = mul_basecase!(r, 0, a, 0, n, b, 0, n)

M = Memory{Limb}
for (name, f, tt) in (
        ("add_n", add_n_wrap!, Tuple{M,M,M,Int}),
        ("sub_n", sub_n_wrap!, Tuple{M,M,M,Int}),
        ("mul_1", mul_1_wrap!, Tuple{M,M,Int,Limb}),
        ("addmul_1", addmul_1_wrap!, Tuple{M,M,Int,Limb}),
        ("submul_1", submul_1_wrap!, Tuple{M,M,Int,Limb}),
        ("basecase", basecase_wrap!, Tuple{M,M,M,Int}))
    println("========== $name ==========")
    code_native(stdout, f, tt; syntax=:intel, debuginfo=:none)
    println()
end
