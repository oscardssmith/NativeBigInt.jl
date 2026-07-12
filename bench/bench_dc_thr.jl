# Local-only: DC_DIV_THRESHOLD sweep on balanced 2m/m divisions, all
# candidates in one process, plus schoolbook (bc) and GMP mpn_tdiv_qr
# reference columns. divrem_dc!/divrem_bc! destroy the numerator, so each
# sample re-copies it (setup + evals=1); sizes here are large enough that the
# copy is noise.
using NativeBigInt, BenchmarkTools, Random
using NativeBigInt: Limb, divrem_dc!, divrem_bc!, invert_pi1

gmpn_tdiv_qr(q, r, a, nn, d, m) = ccall((:__gmpn_tdiv_qr, :libgmp), Cvoid,
    (Ptr{Limb}, Ptr{Limb}, Clong, Ptr{Limb}, Clong, Ptr{Limb}, Clong),
    q, r, 0, a, nn, d, m)

const THRS = (30, 40, 50, 65, 80, 110)
rng = MersenneTwister(1)
println("m      bc    ", join(("T=$t" for t in THRS), "   "), "   (time / gmp mpn_tdiv_qr)")
for m in (32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048)
    nn = 2m
    d = Memory{Limb}(undef, m); rand!(rng, d)
    d[m] |= Limb(1) << 63                      # normalized divisor
    u = Memory{Limb}(undef, nn); rand!(rng, u)
    v = invert_pi1(d[m], d[m-1])
    q = Memory{Limb}(undef, nn - m + 1)
    r = Memory{Limb}(undef, m)
    g = @belapsed gmpn_tdiv_qr($q, $r, $u, $nn, $d, $m)
    bc = (@belapsed divrem_bc!($q, 0, uu, 0, $nn, $d, 0, $m, $v) setup=(uu=copy($u)) evals=1) / g
    ratios = [round((@belapsed divrem_dc!($q, 0, uu, 0, $nn, $d, 0, $m, $v, $t) setup=(uu=copy($u)) evals=1) / g, digits=2)
              for t in THRS]
    println(rpad("m=$m", 7), rpad(round(bc, digits=2), 6), join(ratios, "  "))
    flush(stdout)
end
