# Kernel functions for arbitrary-precision arithmetic

@inline function add_n!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, b::Memory{Limb}, bo::Int, n::Int)
    c = zero(Limb)
    @inbounds for i in 1:n
        s1, o1 = Base.add_with_overflow(a[ao+i], b[bo+i])
        s2, o2 = Base.add_with_overflow(s1, c)
        r[ro+i] = s2
        c = Limb(o1 | o2)
    end
    return c
end

@inline function sub_n!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, b::Memory{Limb}, bo::Int, n::Int)
    brw = zero(Limb)
    @inbounds for i in 1:n
        d1, o1 = Base.sub_with_overflow(a[ao+i], b[bo+i])
        d2, o2 = Base.sub_with_overflow(d1, brw)
        r[ro+i] = d2
        brw = Limb(o1 | o2)
    end
    return brw
end

@inline function add!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, la::Int, b::Memory{Limb}, bo::Int, lb::Int)
    c = add_n!(r, ro, a, ao, b, bo, lb)
    @inbounds for i in lb+1:la
        s, o = Base.add_with_overflow(a[ao+i], c)
        r[ro+i] = s
        c = Limb(o)
    end
    return c
end

@inline function sub!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, la::Int, b::Memory{Limb}, bo::Int, lb::Int)
    brw = sub_n!(r, ro, a, ao, b, bo, lb)
    @inbounds for i in lb+1:la
        d, o = Base.sub_with_overflow(a[ao+i], brw)
        r[ro+i] = d
        brw = Limb(o)
    end
    return brw
end

@inline function add_into!(r::Memory{Limb}, ro::Int, rlen::Int, b::Memory{Limb}, bo::Int, lb::Int)
    c = add_n!(r, ro, r, ro, b, bo, lb)
    i = lb + 1
    @inbounds while c != 0
        @assert i <= rlen "add_into! carry out of range"
        s, o = Base.add_with_overflow(r[ro+i], c)
        r[ro+i] = s
        c = Limb(o)
        i += 1
    end
    return nothing
end

@inline function cmp_limbs(a::Memory{Limb}, ao::Int, la::Int, b::Memory{Limb}, bo::Int, lb::Int)
    la != lb && return la < lb ? -1 : 1
    @inbounds for i in la:-1:1
        a[ao+i] != b[bo+i] && return a[ao+i] < b[bo+i] ? -1 : 1
    end
    return 0
end

@inline function mul_1!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int, b::Limb)
    c = zero(Limb)
    @inbounds for i in 1:n
        p = widemul(a[ao+i], b) + c
        r[ro+i] = p % Limb
        c = (p >> 64) % Limb
    end
    return c
end

@inline function addmul_1!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int, b::Limb)
    c = zero(Limb)
    @inbounds for i in 1:n
        p = widemul(a[ao+i], b) + r[ro+i] + c
        r[ro+i] = p % Limb
        c = (p >> 64) % Limb
    end
    return c
end

@inline function submul_1!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int, b::Limb)
    c = zero(Limb)
    @inbounds for i in 1:n
        p = widemul(a[ao+i], b) + c
        lo = p % Limb
        d, o = Base.sub_with_overflow(r[ro+i], lo)
        r[ro+i] = d
        c = ((p >> 64) % Limb) + Limb(o)
    end
    return c
end

function mul_basecase!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, m::Int, b::Memory{Limb}, bo::Int, n::Int)
    @inbounds r[ro+m+1] = mul_1!(r, ro, a, ao, m, b[bo+1])
    @inbounds for j in 2:n
        r[ro+m+j] = addmul_1!(r, ro+j-1, a, ao, m, b[bo+j])
    end
    return nothing
end

@inline function lshift!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int, cnt::Int)
    ret = @inbounds a[ao+n] >> (64 - cnt)
    @inbounds for i in n:-1:2
        r[ro+i] = (a[ao+i] << cnt) | (a[ao+i-1] >> (64 - cnt))
    end
    @inbounds r[ro+1] = a[ao+1] << cnt
    return ret
end

@inline function rshift!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int, cnt::Int)
    ret = @inbounds a[ao+1] << (64 - cnt)
    @inbounds for i in 1:n-1
        r[ro+i] = (a[ao+i] >> cnt) | (a[ao+i+1] << (64 - cnt))
    end
    @inbounds r[ro+n] = a[ao+n] >> cnt
    return ret
end

@inline function divrem_1!(q::Memory{Limb}, qo::Int, a::Memory{Limb}, ao::Int, n::Int, d::Limb)
    rem = zero(Limb)
    @inbounds for i in n:-1:1
        num = (DLimb(rem) << 64) | a[ao+i]
        q[qo+i] = (num ÷ d) % Limb
        rem = (num % d) % Limb
    end
    return rem
end
