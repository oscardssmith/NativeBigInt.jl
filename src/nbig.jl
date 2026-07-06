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
    n = normlen(limbs, 0, n)
    n == 0 && return NBig(0, EMPTY_LIMBS)
    return NBig(sgn * n, limbs)
end

# Build an NBig with the given sign from an unsigned magnitude `mag` supporting
# `>>= 64` and `% Limb` (a Julia unsigned, or a nonnegative BigInt).
function nbig_from_magnitude(sgn::Int, mag)
    n = 0
    v = mag
    while v != 0
        n += 1
        v >>= 64
    end
    limbs = Memory{Limb}(undef, n)
    v = mag
    for i in 1:n
        limbs[i] = v % Limb
        v >>= 64
    end
    return NBig(sgn * n, limbs)
end

function NBig(x::Integer)
    x == 0 && return NBig(0, EMPTY_LIMBS)
    sgn = x < 0 ? -1 : 1
    return nbig_from_magnitude(sgn, unsigned(sgn < 0 ? -widen(x) : widen(x)))
end
NBig(x::Bool) = NBig(Int(x))
NBig(x::NBig) = x

function NBig(x::BigInt)
    x == 0 && return NBig(0, EMPTY_LIMBS)
    return nbig_from_magnitude(x < 0 ? -1 : 1, abs(x))
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

# Magnitude add/sub; the result sign is decided here from operand signs and
# magnitude comparison, then magnitudes combine via the kernels.
function Base.:+(a::NBig, b::NBig)
    iszero(a) && return b
    iszero(b) && return a
    la, lb = nlimbs(a), nlimbs(b)
    if sign(a) == sign(b)
        if la < lb
            a, b = b, a
            la, lb = lb, la
        end
        r = Memory{Limb}(undef, la + 1)
        c = add!(r, 0, a.limbs, 0, la, b.limbs, 0, lb)
        @inbounds r[la+1] = c
        return nbig_from_limbs(sign(a), r, la + 1)
    else
        c = cmp_limbs(a.limbs, 0, la, b.limbs, 0, lb)
        c == 0 && return NBig(0, EMPTY_LIMBS)
        if c < 0
            a, b = b, a
            la, lb = lb, la
        end
        r = Memory{Limb}(undef, la)
        sub!(r, 0, a.limbs, 0, la, b.limbs, 0, lb)
        return nbig_from_limbs(sign(a), r, la)
    end
end
Base.:-(a::NBig, b::NBig) = a + (-b)

function Base.:*(a::NBig, b::NBig)
    (iszero(a) || iszero(b)) && return NBig(0, EMPTY_LIMBS)
    la, lb = nlimbs(a), nlimbs(b)
    if a.limbs === b.limbs
        # same magnitude (covers x*x and x^2 via power_by_squaring): square it
        r = Memory{Limb}(undef, 2la)
        sqr!(r, 0, a.limbs, 0, la)
        return nbig_from_limbs(sign(a) * sign(b), r, 2la)
    end
    if la < lb
        a, b = b, a
        la, lb = lb, la
    end
    r = Memory{Limb}(undef, la + lb)
    mul!(r, 0, a.limbs, 0, la, b.limbs, 0, lb)
    return nbig_from_limbs(sign(a) * sign(b), r, la + lb)
end

# Truncated division (Base semantics): rem takes the sign of a.
function Base.divrem(a::NBig, b::NBig)
    iszero(b) && throw(DivideError())
    la, lb = nlimbs(a), nlimbs(b)
    cmp_limbs(a.limbs, 0, la, b.limbs, 0, lb) < 0 && return (NBig(0, EMPTY_LIMBS), a)
    q = Memory{Limb}(undef, la - lb + 1)
    r = Memory{Limb}(undef, lb)
    divrem!(q, 0, r, 0, a.limbs, 0, la, b.limbs, 0, lb)
    return (nbig_from_limbs(sign(a) * sign(b), q, la - lb + 1),
            nbig_from_limbs(sign(a), r, lb))
end
Base.div(a::NBig, b::NBig) = divrem(a, b)[1]
Base.rem(a::NBig, b::NBig) = divrem(a, b)[2]

function Base.mod(a::NBig, b::NBig)
    r = rem(a, b)
    return (iszero(r) || sign(r) == sign(b)) ? r : r + b
end
function Base.fld(a::NBig, b::NBig)
    q, r = divrem(a, b)
    return (!iszero(r) && sign(r) != sign(b)) ? q - one(NBig) : q
end
function Base.cld(a::NBig, b::NBig)
    q, r = divrem(a, b)
    return (!iszero(r) && sign(r) == sign(b)) ? q + one(NBig) : q
end

# Base semantics: result = mod(a^n, m), so the sign follows m.
function Base.powermod(a::NBig, n::NBig, m::NBig)
    iszero(m) && throw(DivideError())
    signbit(n) && throw(DomainError(n, "powermod: negative exponent (invmod not implemented)"))
    mm = abs(m)
    k = nlimbs(mm)
    k == 1 && (@inbounds mm.limbs[1]) == 1 && return NBig(0, EMPTY_LIMBS)
    b = mod(a, mm)
    r0 = if iszero(n)
        one(NBig)
    elseif iszero(b)
        NBig(0, EMPTY_LIMBS)
    else
        r = powermod_limbs(b.limbs, nlimbs(b), n.limbs, nlimbs(n), mm.limbs, k)
        nbig_from_limbs(1, r, k)
    end
    return signbit(m) && !iszero(r0) ? r0 + m : r0
end
Base.powermod(a::NBig, n::Integer, m::NBig) = powermod(a, NBig(n), m)

function Base.gcd(a::NBig, b::NBig)
    iszero(a) && return abs(b)
    iszero(b) && return abs(a)
    la, lb = nlimbs(a), nlimbs(b)
    cap = max(la, lb) + 1
    u = Memory{Limb}(undef, cap)
    v = Memory{Limb}(undef, cap)
    copyto!(u, 1, a.limbs, 1, la)
    copyto!(v, 1, b.limbs, 1, lb)
    mem, n = gcd!(u, la, v, lb)
    return nbig_from_limbs(1, mem, n)
end

# isqrt: normalize by an even bit shift so sqrtrem!'s precondition
# (top limb >= 2^62) holds, then shift the root back down by half.
function Base.isqrt(x::NBig)
    signbit(x) && throw(DomainError(x, "isqrt requires a nonnegative argument"))
    iszero(x) && return x
    n = nlimbs(x)
    if n <= 2
        return NBig(isqrt(extract_window(x.limbs, n, 0)) % Limb)
    end
    bits = 64n - leading_zeros(@inbounds x.limbs[n])
    e = (64n - bits) & ~1
    # sqrtrem! needs an even limb count; odd n gets a zero low limb (a β
    # multiply — a further even shift), undone below via the root shift.
    pad = isodd(n) ? 1 : 0
    nn = n + pad
    a = Memory{Limb}(undef, nn)
    pad == 1 && (@inbounds a[1] = 0)
    if e == 0
        copyto!(a, pad + 1, x.limbs, 1, n)
    else
        lshift!(a, pad, x.limbs, 0, n, e)
    end
    h = nn >> 1
    s = Memory{Limb}(undef, h)
    scratch = Memory{Limb}(undef, 5h + 8)
    sqrtrem!(s, 0, a, 0, nn, scratch)
    sh = (e >> 1) + 32pad
    sh > 0 && rshift!(s, 0, s, 0, h, sh)
    return nbig_from_limbs(1, s, h)
end

function Base.:<<(x::NBig, c::UInt)
    (iszero(x) || c == 0) && return x
    lx = nlimbs(x)
    lw, cnt = Int(c >> 6), Int(c & 63)
    n = lx + lw + 1
    r = Memory{Limb}(undef, n)
    @inbounds for i in 1:lw
        r[i] = 0
    end
    @inbounds r[n] = lshift!(r, lw, x.limbs, 0, lx, cnt)
    return nbig_from_limbs(sign(x), r, n)
end

function Base.:>>(x::NBig, c::UInt)
    (iszero(x) || c == 0) && return x
    lx = nlimbs(x)
    lw, cnt = Int(c >> 6), Int(c & 63)
    if lw >= lx
        # magnitude fully shifted out: 0, or -1 by flooring
        return signbit(x) ? NBig(-1) : NBig(0, EMPTY_LIMBS)
    end
    n = lx - lw
    r = Memory{Limb}(undef, n)
    lost = rshift!(r, 0, x.limbs, lw, n, cnt) != 0
    if signbit(x) && !lost
        @inbounds for i in 1:lw
            x.limbs[i] != 0 && (lost = true; break)
        end
    end
    q = nbig_from_limbs(sign(x), r, n)
    # arithmetic shift floors: negative with dropped bits rounds away from zero
    return (signbit(x) && lost) ? q - one(NBig) : q
end
Base.:>>>(x::NBig, c::UInt) = x >> c

Base.:~(x::NBig) = -(x + one(NBig))

# In-place two's-complement negation of t[1..n]: t <- ~t + 1.
@inline function negate_twos!(t::Memory{Limb}, n::Int)
    c = Limb(1)
    @inbounds for i in 1:n
        t[i], c = add_limb_c(~t[i], c, zero(Limb))
    end
    return t
end

# Two's-complement image of x in t[1..n]; requires n > nlimbs(x) so the sign
# extends into at least one full limb.
function twos_complement!(t::Memory{Limb}, x::NBig, n::Int)
    lx = nlimbs(x)
    @inbounds for i in 1:n
        t[i] = i <= lx ? x.limbs[i] : zero(Limb)
    end
    signbit(x) && negate_twos!(t, n)
    return t
end

# Interpret t[1..n] as two's complement (sign limb is all-0 or all-1).
function from_twos_complement!(t::Memory{Limb}, n::Int)
    t[n] >> 63 == 0 && return nbig_from_limbs(1, t, n)
    negate_twos!(t, n)
    return nbig_from_limbs(-1, t, n)
end

for (op, f) in ((:&, :&), (:|, :|), (:xor, :xor))
    @eval function Base.$op(a::NBig, b::NBig)
        n = max(nlimbs(a), nlimbs(b)) + 1
        ta = twos_complement!(Memory{Limb}(undef, n), a, n)
        tb = twos_complement!(Memory{Limb}(undef, n), b, n)
        @inbounds for i in 1:n
            ta[i] = $f(ta[i], tb[i])
        end
        return from_twos_complement!(ta, n)
    end
end

function Base.trailing_zeros(x::NBig)
    iszero(x) && throw(DomainError(x, "trailing_zeros of zero is undefined"))
    i = 1
    @inbounds while x.limbs[i] == 0
        i += 1
    end
    return 64 * (i - 1) + trailing_zeros(@inbounds x.limbs[i])
end

function Base.count_ones(x::NBig)
    signbit(x) && throw(DomainError(x, "count_ones of a negative NBig is undefined"))
    c = 0
    @inbounds for i in 1:nlimbs(x)
        c += count_ones(x.limbs[i])
    end
    return c
end

const DIGIT_CHARS = codeunits("0123456789abcdefghijklmnopqrstuvwxyz")

function Base.string(x::NBig; base::Integer = 10, pad::Integer = 1)
    2 <= base <= 36 || throw(ArgumentError("base must be in 2:36, got $base"))
    n = nlimbs(x)
    bb, k = big_base(Int(base))
    scratch = Memory{Limb}(undef, n)
    copyto!(scratch, 1, x.limbs, 1, n)
    chunks = radix_chunks!(scratch, n, bb)
    # digit count of the top chunk (chunks may be empty for zero)
    ndig = (length(chunks) - 1) * k
    top = isempty(chunks) ? zero(Limb) : chunks[end]
    while top > 0
        ndig += 1
        top = div(top, base % Limb)
    end
    width = max(ndig, Int(pad), 1)
    neg = signbit(x)
    buf = Base.StringMemory(width + neg)
    j = width + neg
    for (ci, c) in enumerate(chunks)
        lim = ci == length(chunks) ? ndig - (length(chunks) - 1) * k : k
        for _ in 1:lim
            c, d = divrem(c, base % Limb)
            @inbounds buf[j] = DIGIT_CHARS[d+1]
            j -= 1
        end
    end
    @inbounds while j > (neg ? 1 : 0)
        buf[j] = UInt8('0')
        j -= 1
    end
    neg && (@inbounds buf[1] = UInt8('-'))
    return Base.unsafe_takestring(buf)
end

Base.show(io::IO, x::NBig) = print(io, string(x))

function Base.tryparse(::Type{NBig}, s::AbstractString; base::Integer = 10)
    2 <= base <= 36 || throw(ArgumentError("base must be in 2:36, got $base"))
    cs = lstrip(isspace, rstrip(isspace, s))
    isempty(cs) && return nothing
    neg = false
    c1 = cs[1]
    if c1 == '-' || c1 == '+'
        neg = c1 == '-'
        cs = SubString(cs, nextind(cs, 1))
        isempty(cs) && return nothing
    end
    bb, k = big_base(Int(base))
    bigb = NBig(bb)
    x = NBig(0, EMPTY_LIMBS)
    chunk = zero(Limb)
    nd = 0
    for c in cs
        d = '0' <= c <= '9' ? c - '0' :
            'a' <= c <= 'z' ? c - 'a' + 10 :
            'A' <= c <= 'Z' ? c - 'A' + 10 : 99
        d < base || return nothing
        chunk = chunk * (base % Limb) + (d % Limb)
        nd += 1
        if nd == k
            x = x * bigb + NBig(chunk)
            chunk = zero(Limb)
            nd = 0
        end
    end
    if nd > 0
        x = x * NBig(Limb(base)^nd) + NBig(chunk)
    end
    return neg ? -x : x
end

function Base.parse(::Type{NBig}, s::AbstractString; base::Integer = 10)
    r = tryparse(NBig, s; base)
    r === nothing && throw(ArgumentError("invalid base-$base digit string: $(repr(s))"))
    return r
end

# Delegating keeps isequal/hash consistent with Int and BigInt; a native
# limb-walking hash is a post-v1 optimization.
Base.hash(x::NBig, h::UInt) = hash(BigInt(x), h)

Base.Float64(x::NBig) = Float64(BigInt(x))
Base.AbstractFloat(x::NBig) = Float64(x)

Base.zero(::Type{NBig}) = NBig(0, EMPTY_LIMBS)
Base.one(::Type{NBig}) = NBig(1)

Base.promote_rule(::Type{NBig}, ::Type{<:Integer}) = NBig

function to_uint(::Type{T}, x::NBig) where {T<:Unsigned}
    signbit(x) && throw(InexactError(nameof(T), T, x))
    n = nlimbs(x)
    n == 0 && return zero(T)
    bits = sizeof(T) * 8
    if n * 64 > bits
        for i in (bits ÷ 64 + 1):n
            x.limbs[i] != 0 && throw(InexactError(nameof(T), T, x))
        end
    end
    r = zero(T)
    for i in min(n, cld(bits, 64)):-1:1
        r = (r << 64) | (T(x.limbs[i]))
    end
    return r
end

# Signed conversion: read the magnitude as unsigned U (the same-width unsigned
# of S), then apply the sign, allowing exactly typemin(S) = -(typemax + 1).
function to_sint(::Type{S}, x::NBig) where {S<:Signed}
    U = unsigned(S)
    v = to_uint(U, abs(x))
    if signbit(x)
        v > unsigned(typemax(S)) + one(U) && throw(InexactError(nameof(S), S, x))
        v == unsigned(typemax(S)) + one(U) && return typemin(S)
        return -(v % S)
    else
        v > unsigned(typemax(S)) && throw(InexactError(nameof(S), S, x))
        return v % S
    end
end

Base.UInt64(x::NBig) = to_uint(UInt64, x)
Base.Int64(x::NBig) = to_sint(Int64, x)
Base.Int128(x::NBig) = to_sint(Int128, x)
