module NativeBigIntRandomExt

# Uniform sampling from NBig unit ranges, mirroring Random's SamplerBigInt.
# Without this, Random's AbstractArray fallback recurses forever (length of
# an NBig range is itself an NBig). Ranges/arrays of NBig (StepRange, Vector)
# reach this through that same fallback, exactly as they do for BigInt.
# BigInt's rand!(rng, x, sp) has no analogue here: NBig is immutable.

using NativeBigInt
using NativeBigInt: Limb, nlimbs, nbig_from_limbs, cmp_limbs
using Random
using Random: Sampler, Repetition

struct SamplerNBig{SP<:Sampler{Limb}} <: Random.Sampler{NBig}
    a::NBig        # first
    m::NBig        # range length - 1
    nlimbs::Int    # number of limbs in generated magnitudes (z ∈ [0, m])
    highsp::SP     # sampler for the highest limb of z
end

function SamplerNBig(::Type{RNG}, r::AbstractUnitRange{NBig},
                     N::Repetition=Val(Inf)) where {RNG<:AbstractRNG}
    isempty(r) && throw(ArgumentError("collection must be non-empty"))
    m = last(r) - first(r)
    n = nlimbs(m)
    hm = n == 0 ? Limb(0) : m.limbs[n]
    highsp = Sampler(RNG, Limb(0):hm, N)
    return SamplerNBig(first(r), m, n, highsp)
end

Random.Sampler(::Type{RNG}, r::AbstractUnitRange{NBig}, N::Repetition) where {RNG<:AbstractRNG} =
    SamplerNBig(RNG, r, N)

function Random.rand(rng::AbstractRNG, sp::SamplerNBig)
    n = sp.nlimbs
    n == 0 && return sp.a
    hm = sp.m.limbs[n]
    # we randomize z ∈ [0, m] with rejection sampling:
    # 1. the low nlimbs-1 limbs of z are uniformly randomized
    # 2. the high limb of z is sampled from 0:hm where hm is the high limb of m
    # We repeat 1. and 2. until z <= m
    t = Memory{Limb}(undef, n)
    while true
        for i in 1:n-1
            t[i] = rand(rng, Limb)
        end
        hx = t[n] = rand(rng, sp.highsp)
        hx < hm && break # avoid the full comparison most of the time
        cmp_limbs(t, 0, n, sp.m.limbs, 0, n) <= 0 && break
    end
    return sp.a + nbig_from_limbs(1, t, n)
end

end
