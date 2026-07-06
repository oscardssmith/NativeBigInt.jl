# Compare divrem! / divrem_1! against GMP's mpn_tdiv_qr / mpn_divrem_1.
# Local-only bench script — never commit.
using BenchmarkTools, Random
using NativeBigInt
using NativeBigInt: Limb, divrem!, divrem_1!

const libgmp = Base.GMP.libgmp

function gmp_tdiv_qr!(q::Vector{Limb}, r::Vector{Limb}, a::Vector{Limb}, d::Vector{Limb})
    ccall((:__gmpn_tdiv_qr, libgmp), Cvoid,
          (Ptr{Limb}, Ptr{Limb}, Clong, Ptr{Limb}, Clong, Ptr{Limb}, Clong),
          q, r, 0, a, length(a), d, length(d))
end

function gmp_divrem_1!(q::Vector{Limb}, a::Vector{Limb}, d::Limb)
    ccall((:__gmpn_divrem_1, libgmp), Limb,
          (Ptr{Limb}, Clong, Ptr{Limb}, Clong, Limb),
          q, 0, a, length(a), d)
end

rng = MersenneTwister(42)

println("=== divrem_1! vs mpn_divrem_1 (n limbs / 1) ===")
for n in (2, 4, 8, 16, 32, 64, 128)
    a = rand(rng, Limb, n)
    d = rand(rng, Limb) | 1
    am = Memory{Limb}(a)
    qm = Memory{Limb}(undef, n)
    qv = Vector{Limb}(undef, n)
    t_n = @belapsed divrem_1!($qm, 0, $am, 0, $n, $d)
    t_g = @belapsed gmp_divrem_1!($qv, $a, $d)
    println("n=$(lpad(n,3)):  nbig $(round(t_n*1e9, digits=1)) ns   gmp $(round(t_g*1e9, digits=1)) ns   ratio $(round(t_n/t_g, digits=2))")
end

println("\n=== divrem! vs mpn_tdiv_qr ===")
for (n, m) in ((4, 2), (8, 4), (16, 8), (32, 16), (64, 32), (128, 64),
               (16, 2), (64, 4), (128, 8), (64, 60), (128, 120))
    a = rand(rng, Limb, n)
    d = rand(rng, Limb, m)
    d[end] |= Limb(1) << 63  # normalized divisor (also bench unnormalized below)
    am, dm = Memory{Limb}(a), Memory{Limb}(d)
    qm = Memory{Limb}(undef, n - m + 1)
    rm = Memory{Limb}(undef, m)
    qv = Vector{Limb}(undef, n - m + 1)
    rv = Vector{Limb}(undef, m)
    t_n = @belapsed divrem!($qm, 0, $rm, 0, $am, 0, $n, $dm, 0, $m)
    t_g = @belapsed gmp_tdiv_qr!($qv, $rv, $a, $d)
    println("n=$(lpad(n,3)) m=$(lpad(m,3)) (norm):  nbig $(round(t_n*1e9, digits=1)) ns   gmp $(round(t_g*1e9, digits=1)) ns   ratio $(round(t_n/t_g, digits=2))")
end

println("\n=== divrem! vs mpn_tdiv_qr, unnormalized divisor ===")
for (n, m) in ((16, 8), (64, 32), (128, 64))
    a = rand(rng, Limb, n)
    d = rand(rng, Limb, m)
    d[end] = (d[end] >> 17) | 1  # top bit clear → shift path
    am, dm = Memory{Limb}(a), Memory{Limb}(d)
    qm = Memory{Limb}(undef, n - m + 1)
    rm = Memory{Limb}(undef, m)
    qv = Vector{Limb}(undef, n - m + 1)
    rv = Vector{Limb}(undef, m)
    t_n = @belapsed divrem!($qm, 0, $rm, 0, $am, 0, $n, $dm, 0, $m)
    t_g = @belapsed gmp_tdiv_qr!($qv, $rv, $a, $d)
    println("n=$(lpad(n,3)) m=$(lpad(m,3)) (unnm):  nbig $(round(t_n*1e9, digits=1)) ns   gmp $(round(t_g*1e9, digits=1)) ns   ratio $(round(t_n/t_g, digits=2))")
end
