# Barrett reduction (HAC 14.42), used by powermod_limbs above
# BARRETT_THRESHOLD: both per-product reductions ride mul! (so Karatsuba/NTT),
# with the reciprocal μ = ⌊β²ᵏ/m⌋ computed once per modulus, and any parity of
# m works. Below the threshold the Montgomery/divrem! paths stay — Barrett
# pays ~2 products per reduction vs redc!'s single addmul sweep. Future
# optimizations (out of scope): Mulders short products for the two multiplies,
# Newton/invertappr reciprocal.

# powermod_limbs switches its reduction to Barrett at these modulus sizes
# (limbs), tuned by bench/bench_barrett_thr.jl. The baselines differ by
# parity, so the crossovers do too: Montgomery redc! (odd m) is schoolbook
# and falls behind by ~68 limbs, while divrem! (even m) rides DC division
# and holds out until Barrett's products reach the fp NTT (~240 limbs).
const BARRETT_THRESHOLD = 68
const BARRETT_EVEN_THRESHOLD = 240

# Scratch limbs barrett_reduce! needs for a k-limb modulus: the q1·μ product
# (≤ 2k+3 limbs, region reused for r) followed by the q3·m product (≤ 2k+1).
barrett_scratch_len(k::Int) = 4k + 4

# μ = ⌊β²ᵏ/m⌋ into mu[1..k+2] via one divrem! (m has k limbs, m[k] ≠ 0);
# returns μ's normalized length: k+1, or k+2 iff m = β^(k-1) exactly.
function barrett_mu!(mu::Memory{Limb}, m::Memory{Limb}, mo::Int, k::Int)
    num = Memory{Limb}(undef, 2k + 1)
    fill!(num, zero(Limb))
    @inbounds num[2k+1] = one(Limb)
    rem = Memory{Limb}(undef, k)
    divrem!(mu, 0, rem, 0, num, 0, 2k + 1, m, mo, k)
    return normlen(mu, 0, k + 2)
end

# Once-per-modulus setup: (μ, its length, reduce scratch) for the k-limb m.
function barrett_setup(m::Memory{Limb}, mo::Int, k::Int)
    mu = Memory{Limb}(undef, k + 2)
    lmu = barrett_mu!(mu, m, mo, k)
    return mu, lmu, Memory{Limb}(undef, barrett_scratch_len(k))
end

# r[1..k] = T mod m (unnormalized), T = t[to+1..to+2k] zero-padded, T < β²ᵏ.
# m has k limbs with m[k] ≠ 0; mu/lmu from barrett_mu!; scratch has
# barrett_scratch_len(k) limbs at so. r must not alias t, m, mu, or scratch;
# t is read-only. Steps: q1 = ⌊T/β^(k-1)⌋ (a slice), q3 = ⌊q1·μ/β^(k+1)⌋,
# r = (T − q3·m) mod β^(k+1), then q ≤ q3 + 2 gives at most two corrections.
function barrett_reduce!(r::Memory{Limb}, ro::Int, t::Memory{Limb}, to::Int,
                         m::Memory{Limb}, mo::Int, k::Int,
                         mu::Memory{Limb}, lmu::Int,
                         scratch::Memory{Limb}, so::Int)
    kp1 = k + 1
    q2 = so           # q1·μ product; its region is reused for r afterwards
    r2 = so + 2k + 3  # q3·m product
    l1 = normlen(t, to + k - 1, kp1)
    lq3 = 0
    if l1 > 0
        if l1 >= lmu
            mul!(scratch, q2, t, to + k - 1, l1, mu, 0, lmu)
        else
            mul!(scratch, q2, mu, 0, lmu, t, to + k - 1, l1)
        end
        lq3 = normlen(scratch, q2 + kp1, l1 + lmu - kp1)
    end
    if lq3 == 0
        # q3 = 0 forces ⌊T/m⌋ ≤ 2, so T < 3m < β^(k+1): r starts as T itself
        copyto!(scratch, so + 1, t, to + 1, kp1)
    else
        # true r < 3m < β^(k+1), so the k+1-limb borrow is discardable
        if k >= lq3
            mul!(scratch, r2, m, mo, k, scratch, q2 + kp1, lq3)
        else
            mul!(scratch, r2, scratch, q2 + kp1, lq3, m, mo, k)
        end
        sub_n!(scratch, so, t, to, scratch, r2, kp1)
    end
    lr = normlen(scratch, so, kp1)
    while cmp_limbs(scratch, so, lr, m, mo, k) >= 0
        sub!(scratch, so, scratch, so, lr, m, mo, k)
        lr = normlen(scratch, so, lr)
    end
    copyto!(r, ro + 1, scratch, so + 1, lr)
    fill!(view(r, ro+lr+1:ro+k), zero(Limb))
    return nothing
end
