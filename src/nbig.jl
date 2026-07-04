# NBig type definition and interface (mpz layer)

struct NBig <: Signed
    signlen::Int           # sign(x) * limb count; 0 ⟺ x == 0
    limbs::Memory{Limb}    # little-endian, normalized (top limb ≠ 0), may be over-allocated
end

const EMPTY_LIMBS = Memory{Limb}(undef, 0)

@inline nlimbs(x::NBig) = abs(x.signlen)
@inline Base.sign(x::NBig) = sign(x.signlen)
@inline Base.signbit(x::NBig) = x.signlen < 0
@inline Base.iszero(x::NBig) = x.signlen == 0

function nbig_from_limbs(sgn::Int, limbs::Memory{Limb}, n::Int)
    while n > 0 && limbs[n] == 0
        n -= 1
    end
    n == 0 && return NBig(0, EMPTY_LIMBS)
    return NBig(sgn * n, limbs)
end

function NBig(x::Integer)
    x == 0 && return NBig(0, EMPTY_LIMBS)
    sgn = x < 0 ? -1 : 1
    u = unsigned(sgn < 0 ? -widen(x) : widen(x))
    n = 0
    v = u
    while v != 0
        n += 1
        v >>= 64
    end
    limbs = Memory{Limb}(undef, n)
    v = u
    for i in 1:n
        limbs[i] = v % Limb
        v >>= 64
    end
    return NBig(sgn * n, limbs)
end
NBig(x::Bool) = NBig(Int(x))
NBig(x::NBig) = x

function NBig(x::BigInt)
    x == 0 && return NBig(0, EMPTY_LIMBS)
    sgn = x < 0 ? -1 : 1
    v = abs(x)
    n = 0
    while v != 0
        n += 1
        v >>= 64
    end
    limbs = Memory{Limb}(undef, n)
    v = abs(x)
    for i in 1:n
        limbs[i] = v % Limb
        v >>= 64
    end
    return NBig(sgn * n, limbs)
end

function Base.BigInt(x::NBig)
    n = nlimbs(x)
    n == 0 && return big(0)
    r = big(0)
    for i in n:-1:1
        r = (r << 64) | big(x.limbs[i])
    end
    return signbit(x) ? -r : r
end

function Base.:(==)(a::NBig, b::NBig)
    a.signlen != b.signlen && return false
    n = nlimbs(a)
    for i in 1:n
        a.limbs[i] != b.limbs[i] && return false
    end
    return true
end

function Base.cmp(a::NBig, b::NBig)
    sa, sb = sign(a.signlen), sign(b.signlen)
    sa != sb && return sa < sb ? -1 : 1
    c = cmp_limbs(a.limbs, 0, nlimbs(a), b.limbs, 0, nlimbs(b))
    return sa < 0 ? -c : c
end
Base.isless(a::NBig, b::NBig) = cmp(a, b) < 0
Base.:<(a::NBig, b::NBig) = cmp(a, b) < 0

Base.:-(x::NBig) = iszero(x) ? x : NBig(-x.signlen, x.limbs)
Base.abs(x::NBig) = iszero(x) || !signbit(x) ? x : NBig(-x.signlen, x.limbs)

Base.isodd(x::NBig) = nlimbs(x) > 0 && isodd(x.limbs[1])
Base.iseven(x::NBig) = !isodd(x)

Base.zero(::Type{NBig}) = NBig(0, EMPTY_LIMBS)
Base.one(::Type{NBig}) = NBig(1)

Base.promote_rule(::Type{NBig}, ::Type{<:Integer}) = NBig

function to_uint(::Type{T}, x::NBig) where {T<:Unsigned}
    signbit(x) && throw(InexactError(:UInt64, T, x))
    n = nlimbs(x)
    n == 0 && return zero(T)
    bits = sizeof(T) * 8
    if n * 64 > bits
        for i in (bits ÷ 64 + 1):n
            x.limbs[i] != 0 && throw(InexactError(:UInt64, T, x))
        end
    end
    r = zero(T)
    for i in min(n, cld(bits, 64)):-1:1
        r = (r << 64) | (T(x.limbs[i]))
    end
    return r
end

function Base.UInt64(x::NBig)
    signbit(x) && throw(InexactError(:UInt64, UInt64, x))
    n = nlimbs(x)
    n == 0 && return UInt64(0)
    n > 1 && throw(InexactError(:UInt64, UInt64, x))
    return x.limbs[1]
end

function Base.Int64(x::NBig)
    n = nlimbs(x)
    n == 0 && return Int64(0)
    n > 1 && throw(InexactError(:Int64, Int64, x))
    v = x.limbs[1]
    if signbit(x)
        v > UInt64(typemax(Int64)) + 1 && throw(InexactError(:Int64, Int64, x))
        v == UInt64(typemax(Int64)) + 1 && return typemin(Int64)
        return -Int64(v)
    else
        v > UInt64(typemax(Int64)) && throw(InexactError(:Int64, Int64, x))
        return Int64(v)
    end
end

function Base.Int128(x::NBig)
    n = nlimbs(x)
    n == 0 && return Int128(0)
    n > 2 && throw(InexactError(:Int128, Int128, x))
    v = UInt128(x.limbs[1])
    n == 2 && (v |= UInt128(x.limbs[2]) << 64)
    if signbit(x)
        v > UInt128(typemax(Int128)) + 1 && throw(InexactError(:Int128, Int128, x))
        v == UInt128(typemax(Int128)) + 1 && return typemin(Int128)
        return -Int128(v)
    else
        v > UInt128(typemax(Int128)) && throw(InexactError(:Int128, Int128, x))
        return Int128(v)
    end
end
