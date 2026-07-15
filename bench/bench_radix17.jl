# Tie points for the 17-family odd multipliers in ntt_len (NTT_LEN_GATES).
# For a target T just above the largest old size below it, each new
# multiplier competes with the old family's next admissible length:
#   17·2^k  vs 20·2^k  (5·2^{k+2})
#   51·2^k  vs 60·2^k  (15·2^{k+2})
#   85·2^k  vs 96·2^k  (3·2^{k+5})
#   255·2^k vs 256·2^k (2^{k+8})
# Cost proxy is one mul's transform work at length N: 2 forward + 1 inverse
# + 1 pointwise, summed over both primes.  Run serially.
#
# Measured 2026-07-14 (AVX-512 desktop) — first k where new/old < 1:
#   17·2^15 = 557056   (0.94–0.99 above; 1.03–1.37 below)
#   51·2^14 = 835584   (0.97–0.98 above)
#   85·2^13 = 696320   (0.98–0.99 above, one noisy 1.02 at 2^15)
#   255·2^14 = 4177920 (0.95, then 0.80 at 255·2^15 vs 2^23)
# ntt_len deliberately uses a uniform k >= 14 floor instead of these per-m
# tie points: it costs <= ~5% in a few narrow bands (15·2^10..2^13 denied at
# 0.95–0.99, 17·2^14 admitted at ~1.03) in exchange for the simpler rule.

using NativeBigInt, Random
const NB = NativeBigInt

function mulcost(N::Int; reps=5)
    best = Inf
    for F in (NB.FP_CTX1, NB.FP_CTX2)
        NB.fp_ntt_plan(N, F)
    end
    for _ in 1:reps
        t = 0.0
        for F in (NB.FP_CTX1, NB.FP_CTX2)
            plan = NB.fp_ntt_plan(N, F)
            p = NB.fp_prime(F)
            xa = Float64.(rand(0:p-1, N))
            xb = Float64.(rand(0:p-1, N))
            NB.fp_ntt_fwd!(copy(xa), plan)   # warm
            t += @elapsed begin
                NB.fp_ntt_fwd!(xa, plan)
                NB.fp_ntt_fwd!(xb, plan)
                NB.fp_ntt_pointwise!(xa, xb, F)
                NB.fp_ntt_rev!(xa, plan)
            end
        end
        best = min(best, t)
    end
    return best
end

Random.seed!(1)
println("k | new N (m)      | old N (m)      | new (ms) | old (ms) | new/old")
for (mnew, mold, kold) in ((17, 5, 2), (51, 15, 2), (85, 3, 5), (255, 1, 8)),
    k in 6:14

    Nnew = mnew << k
    Nold = mold << (k + kold)
    Nnew > 1 << 23 && continue
    tn = mulcost(Nnew)
    to = mulcost(Nold)
    println("$k | $(lpad(Nnew, 8)) ($mnew) | $(lpad(Nold, 8)) ($mold) | ",
            "$(lpad(round(tn * 1e3, digits=3), 8)) | $(lpad(round(to * 1e3, digits=3), 8)) | ",
            round(tn / to, digits=3))
end
