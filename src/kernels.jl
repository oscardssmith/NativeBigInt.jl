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

const LO32 = 0x00000000ffffffff

# per-lane 64x64->128 via 32-bit halves; each lane `*` has both operands
# < 2^32 so LLVM lowers it to vpmuludq.
@inline function mul128_v8(va::V8, b0::Limb, b1::Limb)
    a0 = va & LO32
    a1 = va >> 32
    t = a0 * b0
    u = a1 * b0 + (t >> 32)
    v = a0 * b1 + (u & LO32)
    lo = (t & LO32) | (v << 32)
    hi = a1 * b1 + (u >> 32) + (v >> 32)
    return lo, hi
end

# SIMD fast path (same cold-propagate idea as add_n!): per 8-limb block compute
# product lo/hi vectors, sum r + lo + (hi shifted one lane); the carry-in
# (0..2) can only chain when some lane sum >= typemax-1, which is ~2^-63-rare
# per lane and falls back to a scalar block. Tail/small n use the scalar chain.
@inline function addmul_1!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int, b::Limb)
    b0 = b & LO32; b1 = b >> 32
    c = zero(Limb)    # resolved carry into the next limb (0..2)
    hin = zero(Limb)  # product hi pending from the previous limb
    i = 1
    @inbounds while i + 7 <= n
        va = SIMD.vload(V8, a, ao + i)
        vr = SIMD.vload(V8, r, ro + i)
        lo, hi = mul128_v8(va, b0, b1)
        ph = SIMD.shufflevector(hi, V8(hin), Val((8, 0, 1, 2, 3, 4, 5, 6)))
        s1 = vr + lo
        g1 = SIMD.vifelse(s1 < vr, V8(one(Limb)), V8(zero(Limb)))
        s2 = s1 + ph
        g2 = SIMD.vifelse(s2 < s1, V8(one(Limb)), V8(zero(Limb)))
        if any(s2 >= V8(typemax(Limb) - one(Limb)))
            Base.Cartesian.@nexprs 8 k -> begin
                p = widemul(a[ao+i+k-1], b) + UInt128(r[ro+i+k-1]) + UInt128(c) + UInt128(hin)
                r[ro+i+k-1] = p % Limb
                c = zero(Limb)
                hin = (p >> 64) % Limb
            end
        else
            g = g1 + g2
            cv = SIMD.shufflevector(g, V8(c), Val((8, 0, 1, 2, 3, 4, 5, 6)))
            SIMD.vstore(s2 + cv, r, ro + i)
            c = g[8]
            hin = hi[8]
        end
        i += 8
    end
    @inbounds while i <= n
        p = widemul(a[ao+i], b) + UInt128(r[ro+i]) + UInt128(c) + UInt128(hin)
        r[ro+i] = p % Limb
        c = zero(Limb)
        hin = (p >> 64) % Limb
        i += 1
    end
    return hin + c
end

# Borrow analog of addmul_1!'s SIMD path: r -= a*b with borrow generates; the
# borrow-in (0..2) can only chain when some lane difference <= 1.
@inline function submul_1!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int, b::Limb)
    b0 = b & LO32; b1 = b >> 32
    brw = zero(Limb)  # resolved borrow from the next limb (0..2)
    hin = zero(Limb)  # product hi pending from the previous limb
    i = 1
    @inbounds while i + 7 <= n
        va = SIMD.vload(V8, a, ao + i)
        vr = SIMD.vload(V8, r, ro + i)
        lo, hi = mul128_v8(va, b0, b1)
        ph = SIMD.shufflevector(hi, V8(hin), Val((8, 0, 1, 2, 3, 4, 5, 6)))
        d1 = vr - lo
        g1 = SIMD.vifelse(vr < lo, V8(one(Limb)), V8(zero(Limb)))
        d2 = d1 - ph
        g2 = SIMD.vifelse(d1 < ph, V8(one(Limb)), V8(zero(Limb)))
        if any(d2 <= V8(one(Limb)))
            Base.Cartesian.@nexprs 8 k -> begin
                p = widemul(a[ao+i+k-1], b) + UInt128(brw) + UInt128(hin)
                d, o = Base.sub_with_overflow(r[ro+i+k-1], p % Limb)
                r[ro+i+k-1] = d
                brw = Limb(o)
                hin = (p >> 64) % Limb
            end
        else
            g = g1 + g2
            bv = SIMD.shufflevector(g, V8(brw), Val((8, 0, 1, 2, 3, 4, 5, 6)))
            SIMD.vstore(d2 - bv, r, ro + i)
            brw = g[8]
            hin = hi[8]
        end
        i += 8
    end
    @inbounds while i <= n
        p = widemul(a[ao+i], b) + UInt128(brw) + UInt128(hin)
        d, o = Base.sub_with_overflow(r[ro+i], p % Limb)
        r[ro+i] = d
        brw = Limb(o)
        hin = (p >> 64) % Limb
        i += 1
    end
    return hin + brw
end

# Two rows of b per pass over a: the two mulx chains are independent and
# interleave in the OOO window, and r is traversed half as often. The column
# carry hi0 + o1 + o2 + o3 can be exactly 2^64 (doesn't fit a limb), so o1/o2
# ride into the next iteration's 128-bit p0 accumulation (cb, weight-correct
# and headroom-safe) and c0 = hi0 + o3 <= typemax stays representable.
@inline function addmul_2!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, m::Int, b0::Limb, b1::Limb)
    c0 = zero(Limb); c1 = zero(Limb); prev_lo1 = zero(Limb); cb = zero(Limb)
    @inbounds for i in 1:m
        p0 = widemul(a[ao+i], b0) + cb
        lo0 = p0 % Limb; hi0 = (p0 >> 64) % Limb
        p1 = widemul(a[ao+i], b1) + c1
        lo1 = p1 % Limb; c1 = (p1 >> 64) % Limb
        t, o1 = Base.add_with_overflow(r[ro+i], lo0)
        t, o2 = Base.add_with_overflow(t, prev_lo1)
        t, o3 = Base.add_with_overflow(t, c0)
        r[ro+i] = t
        cb = Limb(o1) + Limb(o2)
        c0 = hi0 + Limb(o3)
        prev_lo1 = lo1
    end
    @inbounds begin
        t, oa = Base.add_with_overflow(prev_lo1, c0)
        t, ob = Base.add_with_overflow(t, cb)
        r[ro+m+1] = t
        r[ro+m+2] = c1 + Limb(oa) + Limb(ob)
    end
    return nothing
end

# Row-width crossover between addmul_2! rows and SIMD addmul_1! rows.
# 0 = SIMD addmul_1! everywhere: two SIMD rows now beat one scalar addmul_2!.
const MUL_BC_SIMD_THRESHOLD = 0

function mul_basecase!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, m::Int, b::Memory{Limb}, bo::Int, n::Int)
    @inbounds r[ro+m+1] = mul_1!(r, ro, a, ao, m, b[bo+1])
    if m < MUL_BC_SIMD_THRESHOLD
        j = 2
        @inbounds while j + 1 <= n
            addmul_2!(r, ro+j-1, a, ao, m, b[bo+j], b[bo+j+1])
            j += 2
        end
        @inbounds if j <= n
            r[ro+m+j] = addmul_1!(r, ro+j-1, a, ao, m, b[bo+j])
        end
    else
        @inbounds for j in 2:n
            r[ro+m+j] = addmul_1!(r, ro+j-1, a, ao, m, b[bo+j])
        end
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
