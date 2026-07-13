# Threshold sweep for the subquadratic (HGCD) gcd/gcdx path (never committed
# results; run standalone). Compares pure Lehmer (dc_thr = typemax) against
# the DC driver at various GCD_DC/GCDEXT_DC/HGCD thresholds, and against
# GMP's mpn_gcd for the success bar.
using BenchmarkTools, NativeBigInt, Random
using NativeBigInt: Limb, gcd!, gcdext!

function randmag(rng, n)
    m = Memory{Limb}(rand(rng, Limb, n))
    m[n] |= Limb(1) << 63
    return m
end

# mpn_gcd clobbers both inputs and requires v odd-ish normalization; use the
# documented contract: up (un limbs) >= vp (vn limbs), vp odd, returns gn.
g_gcd!(gp, up, un, vp, vn) =
    ccall((:__gmpn_gcd, :libgmp), Clong,
          (Ptr{Limb}, Ptr{Limb}, Clong, Ptr{Limb}, Clong), gp, up, un, vp, vn)

function bench_one(rng, n; dc, hg)
    a = randmag(rng, n); b = randmag(rng, n)
    b[1] |= one(Limb)
    return @belapsed (u = copy($a); v = copy($b);
                      gcd!(u, $n, v, $n; dc_thr=$dc, hgcd_thr=$hg)) evals=1
end

function bench_ext(rng, n; dc, hg)
    a = randmag(rng, n); b = randmag(rng, n)
    return @belapsed (u = Memory{Limb}(undef, $n + 3); v = Memory{Limb}(undef, $n + 3);
                      copyto!(u, 1, $a, 1, $n); copyto!(v, 1, $b, 1, $n);
                      gcdext!(u, $n, v, $n; dc_thr=$dc, hgcd_thr=$hg)) evals=1
end

function bench_gmp(rng, n)
    a = randmag(rng, n); b = randmag(rng, n)
    b[1] |= one(Limb)
    g = Memory{Limb}(undef, n)
    return @belapsed (u = copy($a); v = copy($b); g_gcd!($g, u, $n, v, $n)) evals=1
end

rng = MersenneTwister(1234)
const BIG = typemax(Int) ÷ 2

println("=== gcd: Lehmer vs DC(dc,hgcd) vs GMP ===")
for n in (100, 150, 200, 300, 400, 600, 900, 1400, 2000, 3000)
    leh = bench_one(rng, n; dc=BIG, hg=BIG)
    gmp = bench_gmp(rng, n)
    print("n=$(lpad(n,5)):  lehmer=$(round(leh*1e6, digits=1))us  gmp=$(round(gmp*1e6, digits=1))us ")
    for (dc, hg) in ((150, 100), (250, 100), (400, 130), (600, 130))
        dc > n && continue
        t = bench_one(rng, n; dc=dc, hg=hg)
        print(" dc$dc/hg$hg=$(round(t*1e6, digits=1))us")
    end
    println()
end

println("=== gcdext: Lehmer vs DC ===")
for n in (100, 150, 200, 300, 400, 600, 900, 1400, 2000)
    leh = bench_ext(rng, n; dc=BIG, hg=BIG)
    print("n=$(lpad(n,5)):  lehmer=$(round(leh*1e6, digits=1))us ")
    for (dc, hg) in ((120, 100), (200, 100), (300, 130), (500, 130))
        dc > n && continue
        t = bench_ext(rng, n; dc=dc, hg=hg)
        print(" dc$dc/hg$hg=$(round(t*1e6, digits=1))us")
    end
    println()
end
