# Kernel functions for arbitrary-precision arithmetic

@inline function add_n!(r, ro, a, ao, b, bo, n)
    c = zero(Limb)
    @inbounds for i in 1:n
        s1, o1 = Base.add_with_overflow(a[ao+i], b[bo+i])
        s2, o2 = Base.add_with_overflow(s1, c)
        r[ro+i] = s2
        c = Limb(o1 | o2)
    end
    return c
end

@inline function sub_n!(r, ro, a, ao, b, bo, n)
    brw = zero(Limb)
    @inbounds for i in 1:n
        d1, o1 = Base.sub_with_overflow(a[ao+i], b[bo+i])
        d2, o2 = Base.sub_with_overflow(d1, brw)
        r[ro+i] = d2
        brw = Limb(o1 | o2)
    end
    return brw
end

@inline function add!(r, ro, a, ao, la, b, bo, lb)
    c = add_n!(r, ro, a, ao, b, bo, lb)
    @inbounds for i in lb+1:la
        s, o = Base.add_with_overflow(a[ao+i], c)
        r[ro+i] = s
        c = Limb(o)
    end
    return c
end

@inline function sub!(r, ro, a, ao, la, b, bo, lb)
    brw = sub_n!(r, ro, a, ao, b, bo, lb)
    @inbounds for i in lb+1:la
        d, o = Base.sub_with_overflow(a[ao+i], brw)
        r[ro+i] = d
        brw = Limb(o)
    end
    return brw
end

@inline function add_into!(r, ro, rlen, b, bo, lb)
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

@inline function cmp_limbs(a, ao, la, b, bo, lb)
    la != lb && return la < lb ? -1 : 1
    @inbounds for i in la:-1:1
        a[ao+i] != b[bo+i] && return a[ao+i] < b[bo+i] ? -1 : 1
    end
    return 0
end

@inline function mul_1!(r, ro, a, ao, n, b::Limb)
    c = zero(Limb)
    @inbounds for i in 1:n
        p = widemul(a[ao+i], b) + c
        r[ro+i] = p % Limb
        c = (p >> 64) % Limb
    end
    return c
end

@inline function addmul_1!(r, ro, a, ao, n, b::Limb)
    c = zero(Limb)
    @inbounds for i in 1:n
        p = widemul(a[ao+i], b) + r[ro+i] + c
        r[ro+i] = p % Limb
        c = (p >> 64) % Limb
    end
    return c
end

@inline function submul_1!(r, ro, a, ao, n, b::Limb)
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

function mul_basecase!(r, ro, a, ao, m, b, bo, n)
    @inbounds r[ro+m+1] = mul_1!(r, ro, a, ao, m, b[bo+1])
    @inbounds for j in 2:n
        r[ro+m+j] = addmul_1!(r, ro+j-1, a, ao, m, b[bo+j])
    end
    return nothing
end
