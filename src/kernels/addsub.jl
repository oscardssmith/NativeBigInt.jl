# Addition/subtraction kernels and limb-length/comparison helpers (mpn layer).

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
# a plain memcpy. The alias guard skips the common in-place case (r === a) where
# source and dest coincide; otherwise the ranges are disjoint (partial overlap
# is unsupported), so copyto! is safe.
@inline function copy_tail!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, from::Int, to::Int)
    (r === a && ro == ao) && return nothing
    from <= to && copyto!(r, ro + from, a, ao + from, to - from + 1)
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
# r[1..n] = a[1..n] - b, returns the borrow out (0 or 1)
@inline function sub_1!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int, b::Limb)
    brw = b
    i = 1
    @inbounds while brw != 0 && i <= n
        d, o = Base.sub_with_overflow(a[ao+i], brw)
        r[ro+i] = d
        brw = Limb(o)
        i += 1
    end
    r === a && ro == ao || copy_tail!(r, ro, a, ao, i, n)
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

# Bit length of the n-limb magnitude at a[ao+1..ao+n], n >= 1; requires a
# normalized (nonzero) top limb.
@inline magnitude_bits(a, ao::Int, n::Int) = 64 * (n - 1) + Base.top_set_bit(@inbounds a[ao+n])

# length of x[1..n] with zero top limbs stripped
@inline function normlen(x::Memory{Limb}, xo::Int, n::Int)
    @inbounds while n > 0 && x[xo+n] == 0
        n -= 1
    end
    return n
end

@inline function cmp_limbs(a::Memory{Limb}, ao::Int, la::Int, b::Memory{Limb}, bo::Int, lb::Int)
    la != lb && return la < lb ? -1 : 1
    @inbounds for i in la:-1:1
        a[ao+i] != b[bo+i] && return a[ao+i] < b[bo+i] ? -1 : 1
    end
    return 0
end
