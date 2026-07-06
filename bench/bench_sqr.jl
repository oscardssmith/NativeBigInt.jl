# Squaring: sqr! vs __gmpn_sqr, vs our own mul path, and the
# basecase/Karatsuba crossover for SQR_KARATSUBA_THRESHOLD tuning.
using NativeBigInt, BenchmarkTools
using NativeBigInt: Limb, sqr!, sqr_basecase!, kar_sqr!, kar_scratch_len, mul!

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
    tk = @belapsed kar_sqr!($r, 0, $a, 0, $n, $s, 0, $n)
    println(rpad(n, 4), lpad(round(tb*1e9, digits=1), 9), " ns bc",
            lpad(round(tk*1e9, digits=1), 9), " ns kar",
            lpad(round(tb/tk, digits=2), 7))
end
