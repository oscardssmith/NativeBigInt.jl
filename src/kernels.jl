# Kernel functions for arbitrary-precision arithmetic

@inline function add_limb_c(a::Limb, b::Limb, c::Limb)
    s1, o1 = Base.add_with_overflow(a, b)
    s2, o2 = Base.add_with_overflow(s1, c)
    return s2, Limb(o1 | o2)
end

@inline function sub_limb_b(a::Limb, b::Limb, brw::Limb)
    d1, o1 = Base.sub_with_overflow(a, b)
    d2, o2 = Base.sub_with_overflow(d1, brw)
    return d2, Limb(o1 | o2)
end

const V8 = SIMD.Vec{8,Limb}

# SIMD fast path: within an 8-limb block, unless some limb sum equals typemax
# (probability ~2^-64 per limb), an incoming carry bit cannot chain, so the
# carry-into vector is just the generate vector (s < a) shifted one lane with
# the block carry-in inserted at lane 0. Blocks where a carry could chain take
# the cold scalar branch. r must alias a/b exactly or not at all.
@inline function add_n!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, b::Memory{Limb}, bo::Int, n::Int)
    c = zero(Limb)
    i = 1
    @inbounds while i + 7 <= n
        va = SIMD.vload(V8, a, ao + i)
        vb = SIMD.vload(V8, b, bo + i)
        vs = va + vb
        if any(vs == V8(typemax(Limb)))
            Base.Cartesian.@nexprs 8 k -> ((r[ro+i+k-1], c) = add_limb_c(a[ao+i+k-1], b[bo+i+k-1], c))
        else
            g = SIMD.vifelse(vs < va, V8(one(Limb)), V8(zero(Limb)))
            cv = SIMD.shufflevector(g, V8(c), Val((8, 0, 1, 2, 3, 4, 5, 6)))
            SIMD.vstore(vs + cv, r, ro + i)
            c = g[8]
        end
        i += 8
    end
    @inbounds while i <= n
        (r[ro+i], c) = add_limb_c(a[ao+i], b[bo+i], c)
        i += 1
    end
    return c
end

# LLVM emits sbb chains within a straight-line block but re-materializes the
# borrow at every loop back-edge; wide manual unrolls amortize that fixup.
# (No SIMD cold-path analog here: for subtraction "propagate" is d == 0, i.e.
# equal limbs, which is common in real inputs, not 2^-64-rare.)
@inline function sub_n!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, b::Memory{Limb}, bo::Int, n::Int)
    brw = zero(Limb)
    i = 1
    @inbounds while i + 15 <= n
        Base.Cartesian.@nexprs 16 k -> ((r[ro+i+k-1], brw) = sub_limb_b(a[ao+i+k-1], b[bo+i+k-1], brw))
        i += 16
    end
    @inbounds while i + 7 <= n
        Base.Cartesian.@nexprs 8 k -> ((r[ro+i+k-1], brw) = sub_limb_b(a[ao+i+k-1], b[bo+i+k-1], brw))
        i += 8
    end
    @inbounds while i + 3 <= n
        Base.Cartesian.@nexprs 4 k -> ((r[ro+i+k-1], brw) = sub_limb_b(a[ao+i+k-1], b[bo+i+k-1], brw))
        i += 4
    end
    @inbounds while i <= n
        (r[ro+i], brw) = sub_limb_b(a[ao+i], b[bo+i], brw)
        i += 1
    end
    return brw
end

# Once the carry/borrow dies (expected after ~1 limb), the rest of the tail is
# a plain copy; keeping it out of the carry loop lets LLVM vectorize it.
@inline function copy_tail!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, from::Int, to::Int)
    (r === a && ro == ao) && return nothing
    @inbounds for i in from:to
        r[ro+i] = a[ao+i]
    end
    return nothing
end

@inline function add!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, la::Int, b::Memory{Limb}, bo::Int, lb::Int)
    c = add_n!(r, ro, a, ao, b, bo, lb)
    i = lb + 1
    @inbounds while c != 0 && i <= la
        s, o = Base.add_with_overflow(a[ao+i], c)
        r[ro+i] = s
        c = Limb(o)
        i += 1
    end
    copy_tail!(r, ro, a, ao, i, la)
    return c
end

@inline function sub!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, la::Int, b::Memory{Limb}, bo::Int, lb::Int)
    brw = sub_n!(r, ro, a, ao, b, bo, lb)
    i = lb + 1
    @inbounds while brw != 0 && i <= la
        d, o = Base.sub_with_overflow(a[ao+i], brw)
        r[ro+i] = d
        brw = Limb(o)
        i += 1
    end
    copy_tail!(r, ro, a, ao, i, la)
    return brw
end

# ripple-add the carry c into r starting at index i
@inline function add_carry!(r::Memory{Limb}, ro::Int, rlen::Int, i::Int, c::Limb)
    @inbounds while c != 0
        @assert i <= rlen "add_carry! carry out of range"
        s, o = Base.add_with_overflow(r[ro+i], c)
        r[ro+i] = s
        c = Limb(o)
        i += 1
    end
    return nothing
end

@inline function add_into!(r::Memory{Limb}, ro::Int, rlen::Int, b::Memory{Limb}, bo::Int, lb::Int)
    c = add_n!(r, ro, r, ro, b, bo, lb)
    add_carry!(r, ro, rlen, lb + 1, c)
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
