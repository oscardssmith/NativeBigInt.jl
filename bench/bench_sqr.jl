# Squaring: sqr! vs __gmpn_sqr, vs our own mul path, plus the crossovers for
# SQR_KARATSUBA_THRESHOLD (basecase vs Karatsuba) and SQR_FPNTT_THRESHOLD
# (Karatsuba vs NTT) tuning.
using NativeBigInt, BenchmarkTools
using NativeBigInt: Limb, sqr!, sqr_basecase!, sqr_kar!, sqr_fpntt!,
                    kar_scratch_len, sqr_scratch_len, mul!

g_sqr!(r, a, n) = ccall((:__gmpn_sqr, :libgmp), Cvoid,
                        (Ptr{Limb}, Ptr{Limb}, Clong), r, a, n)

println("n | sqr! | gmpn_sqr | ratio | mul!(a,a) | sqr/mul")
for n in (2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 96, 128, 192, 256)
    a = Memory{Limb}(rand(Limb, n))
    r = Memory{Limb}(undef, 2n)
    ts = @belapsed sqr!($r, 0, $a, 0, $n)
    tg = @belapsed g_sqr!($r, $a, $n)
    tm = @belapsed mul!($r, 0, $a, 0, $n, $a, 0, $n)
    println(rpad(n, 4), lpad(round(ts*1e9, digits=1), 9), " ns",
            lpad(round(tg*1e9, digits=1), 9), " ns",
            lpad(round(ts/tg, digits=2), 7),
            lpad(round(tm*1e9, digits=1), 9), " ns",
            lpad(round(ts/tm, digits=2), 7))
end

println("\ncrossover: basecase vs one-level Karatsuba")
for n in (24, 32, 40, 48, 56, 64, 80)
    a = Memory{Limb}(rand(Limb, n))
    r = Memory{Limb}(undef, 2n)
    s = Memory{Limb}(undef, kar_scratch_len(n, n))  # force one kar level
    tb = @belapsed sqr_basecase!($r, 0, $a, 0, $n)
    tk = @belapsed sqr_kar!($r, 0, $a, 0, $n, $s, 0, $n)
    println(rpad(n, 4), lpad(round(tb*1e9, digits=1), 9), " ns bc",
            lpad(round(tk*1e9, digits=1), 9), " ns kar",
            lpad(round(tb/tk, digits=2), 7))
end

println("\ncrossover: Karatsuba vs fp NTT")
for n in (256, 320, 384, 448, 512, 640, 768, 1024)
    a = Memory{Limb}(rand(Limb, n))
    r = Memory{Limb}(undef, 2n)
    s = Memory{Limb}(undef, sqr_scratch_len(n))
    tk = @belapsed sqr_kar!($r, 0, $a, 0, $n, $s, 0)
    tn = @belapsed sqr_fpntt!($r, 0, $a, 0, $n)
    println(rpad(n, 6), lpad(round(tk*1e6, digits=1), 8), " us kar",
            lpad(round(tn*1e6, digits=1), 8), " us ntt",
            lpad(round(tn/tk, digits=2), 7))
end
