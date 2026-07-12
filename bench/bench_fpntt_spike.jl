# fpntt spike (never committed): is the fp-Shoup butterfly enough faster than
# the Goldilocks butterfly to pay the single-prime 1.37x / two-prime 1.13x
# point-count penalty?  Times the raw mulmod primitive and a full radix-4
# stage pass in both domains, L1-resident.
using BenchmarkTools, SIMD, NativeBigInt
using NativeBigInt: V8, gf_mulv, gf_addv, gf_subv, gf_mul_iv, dft4_fwd, GF_P

const VF8 = SIMD.Vec{8,Float64}

const P = 1125625028935681.0            # 2^50 - 2^38 + 1
const PN = -P
const PI_ = UInt64(1125625028935681)
const MAGIC = 6755399441055744.0        # 1.5*2^52: (x + M) - M == round(x), |x| < 2^51

# r ≡ w*x (mod p), inputs |x| < 4p, w in [0,p); output in (-p, p).
# q = round(x * wpinv) runs in parallel with h = x*w (no dependency).
# Plain ops for the magic round (Julia never contracts bare *,+), explicit
# fma elsewhere.
@inline function fp_mulmod(x::VF8, w::VF8, wpinv::VF8)
    h = x * w
    l = fma(x, w, -h)
    q = (x * wpinv + MAGIC) - MAGIC
    return fma(q, PN, h) + l
end
@inline function fp_mulmod(x::Float64, w::Float64, wpinv::Float64)
    h = x * w
    l = fma(x, w, -h)
    q = (x * wpinv + MAGIC) - MAGIC
    return fma(q, PN, h) + l
end

# --- correctness gate: fp_mulmod vs UInt128 reference, balanced inputs ------
canon(r::Float64) = (v = r < 0 ? r + P : r; UInt64(v) % PI_)
let ok = true
    for _ in 1:100_000
        w = rand(UInt64) % PI_
        x = rand(UInt64) % PI_
        xf = rand(Bool) ? Float64(x) : Float64(x) - P   # both residue classes
        r = fp_mulmod(xf, Float64(w), Float64(w) / P)
        abs(r) < P || (ok = false; @show w x r; break)
        ref = UInt64(UInt128(w) * (UInt128(xf < 0 ? x + (UInt64(1) << 50) - (UInt64(1) << 38) + 1 - PI_ : x)) % PI_)
        ref = UInt64(UInt128(w) * x % PI_)   # same residue either way
        canon(r) == ref || (ok = false; @show w x r ref; break)
    end
    ok || error("fp_mulmod correctness check FAILED")
    println("fp_mulmod correctness: 100k random cases OK")
end

# --- kernels -----------------------------------------------------------------
# raw mulmod streams: x[i] = mulmod(x[i], w[i])
@noinline function fp_mul_stream!(x::Vector{Float64}, w::Vector{Float64},
                                  wpinv::Vector{Float64})
    @inbounds for i in 1:8:length(x)-7
        v = fp_mulmod(vload(VF8, x, i), vload(VF8, w, i), vload(VF8, wpinv, i))
        vstore(v, x, i)
    end
end
@noinline function gf_mul_stream!(x::Vector{UInt64}, w::Vector{UInt64})
    @inbounds for i in 1:8:length(x)-7
        vstore(gf_mulv(vload(V8, x, i), vload(V8, w, i)), x, i)
    end
end

# one radix-4 DIF stage over N elements (q = N/4), the ntt_fwd_pow2 hot shape.
# fp: adds unreduced (inputs (-p,p), sums < 4p < 2^52), i-rotation and the
# three twiddles are full mulmods.
@noinline function fp_stage!(x::Vector{Float64}, q::Int,
                             w1, w1p, w2, w2p, w3, w3p, wi::Float64, wip::Float64)
    vwi, vwip = VF8(wi), VF8(wip)
    @inbounds for j in 0:8:q-8
        i0 = j + 1
        a = vload(VF8, x, i0)
        b = vload(VF8, x, i0 + q)
        c = vload(VF8, x, i0 + 2q)
        d = vload(VF8, x, i0 + 3q)
        apc = a + c; amc = a - c
        bpd = b + d
        ibmd = fp_mulmod(b - d, vwi, vwip)
        y0 = apc + bpd
        y1 = amc + ibmd
        y2 = apc - bpd
        y3 = amc - ibmd
        vstore(y0, x, i0)
        vstore(fp_mulmod(y1, vload(VF8, w1, i0), vload(VF8, w1p, i0)), x, i0 + q)
        vstore(fp_mulmod(y2, vload(VF8, w2, i0), vload(VF8, w2p, i0)), x, i0 + 2q)
        vstore(fp_mulmod(y3, vload(VF8, w3, i0), vload(VF8, w3p, i0)), x, i0 + 3q)
    end
end
@noinline function gf_stage!(x::Vector{UInt64}, q::Int, w1, w2, w3)
    @inbounds for j in 0:8:q-8
        i0 = j + 1
        a = vload(V8, x, i0)
        b = vload(V8, x, i0 + q)
        c = vload(V8, x, i0 + 2q)
        d = vload(V8, x, i0 + 3q)
        y0, y1, y2, y3 = dft4_fwd(a, b, c, d, gf_addv, gf_subv, gf_mul_iv)
        vstore(y0, x, i0)
        vstore(gf_mulv(y1, vload(V8, w1, i0)), x, i0 + q)
        vstore(gf_mulv(y2, vload(V8, w2, i0)), x, i0 + 2q)
        vstore(gf_mulv(y3, vload(V8, w3, i0)), x, i0 + 3q)
    end
end

# --- timing ------------------------------------------------------------------
const N = 4096
const Q = N >> 2

randres() = rand(UInt64, N) .% PI_
gfx0 = rand(UInt64, N) .% GF_P
gw = [rand(UInt64, Q) .% GF_P for _ in 1:3]
fx0 = Float64.(randres()) .- (P / 2)                 # balanced-ish inputs
fw  = [Float64.(randres()) for _ in 1:3]
fwp = [w ./ P for w in fw]
fwm = Float64.(rand(UInt64, N) .% PI_)
fwmp = fwm ./ P
gwm = rand(UInt64, N) .% GF_P
wi = Float64(rand(UInt64) % PI_); wip = wi / P

gfx = copy(gfx0); fx = copy(fx0)
gmw = copy(gwm)

t_fmul = @belapsed fp_mul_stream!($fx, $fwm, $fwmp)
t_gmul = @belapsed gf_mul_stream!($gfx, $gwm)
t_fst = @belapsed fp_stage!(x, $Q, $(fw[1]), $(fwp[1]), $(fw[2]), $(fwp[2]),
                            $(fw[3]), $(fwp[3]), $wi, $wip) setup=(x = copy($fx0)) evals=1
t_gst = @belapsed gf_stage!(x, $Q, $(gw[1]), $(gw[2]), $(gw[3])) setup=(x = copy($gfx0)) evals=1

nspe(t) = 1e9 * t / N
println()
println("N = $N elements, L1-resident, ns/element:")
println(rpad("  mulmod stream  fp:", 24), round(nspe(t_fmul), digits=3),
        "   gf: ", round(nspe(t_gmul), digits=3),
        "   ratio fp/gf: ", round(t_fmul / t_gmul, digits=3))
println(rpad("  radix-4 stage  fp:", 24), round(nspe(t_fst), digits=3),
        "   gf: ", round(nspe(t_gst), digits=3),
        "   ratio fp/gf: ", round(t_fst / t_gst, digits=3))
println()
r = t_fst / t_gst
println("point-count-adjusted stage cost (lower than 1.0 = fpntt wins):")
println("  single-prime (1.37x points): ", round(1.37r, digits=3))
println("  two-prime    (1.13x points): ", round(1.13r, digits=3))
