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

# One scalar lane of submul_2!: r[j] -= lo(a*b0) + pending state, returning
# the updated (brw, qin, hin2, hin1). Product stream identical to
# addmul_2_lane; the subtraction's two's-complement hi limb gives the 0..3
# borrow directly.
@inline function submul_2_lane(rj::Limb, aj::Limb, b0::Limb, b1::Limb, brw::Limb, qin::Limb, hin2::Limb, hin1::Limb)
    p0 = widemul(aj, b0)
    p1 = widemul(aj, b1)
    t = DLimb(p0 % Limb) + qin + hin2 + brw
    dif = DLimb(rj) - t
    q, o = Base.add_with_overflow((p0 >> 64) % Limb, p1 % Limb)
    return dif % Limb, -((dif >> 64) % Limb), q, hin1, ((p1 >> 64) % Limb) + Limb(o)
end

# r[1..n] -= a[1..n] * (b0 + b1·β) (mod β^n); returns (co1, co0): the two-limb
# value of weight β^n still to subtract (co ≤ β²-1 since ⌊a·b/β^n⌋ ≤ β²-2 and
# the final borrow adds at most 1). Same product-stream layout and SIMD
# cold-propagate scheme as addmul_2!, with a 0..3 borrow chain in place of the
# carry chain: the resolved borrow-in can only chain when some lane difference
# is <= 2, which falls back to a scalar block.
@inline function submul_2!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int, b0::Limb, b1::Limb)
    b00 = b0 & LO32; b01 = b0 >> 32
    b10 = b1 & LO32; b11 = b1 >> 32
    brw = zero(Limb)   # resolved borrow into the next lane (0..3)
    qin = zero(Limb)   # q = hi0+lo1 pending from the previous lane
    hin2 = zero(Limb)  # h = hi1+overflow(q) pending from two lanes back
    hin1 = zero(Limb)  # h pending from the previous lane
    z = zero(Limb)
    i = 1
    @inbounds while i + 7 <= n
        va = SIMD.vload(V8, a, ao + i)
        vr = SIMD.vload(V8, r, ro + i)
        lo0, hi0 = mul128_v8(va, b00, b01)
        lo1, hi1 = mul128_v8(va, b10, b11)
        q = hi0 + lo1
        h = hi1 + SIMD.vifelse(q < hi0, V8(one(Limb)), V8(zero(Limb)))
        pq = SIMD.shufflevector(q, V8(qin), Val((8, 0, 1, 2, 3, 4, 5, 6)))
        ph = SIMD.shufflevector(h, V8((hin2, hin1, z, z, z, z, z, z)),
                                Val((8, 9, 0, 1, 2, 3, 4, 5)))
        s1 = vr - lo0
        g1 = SIMD.vifelse(vr < lo0, V8(one(Limb)), V8(zero(Limb)))
        s2 = s1 - pq
        g2 = SIMD.vifelse(s1 < pq, V8(one(Limb)), V8(zero(Limb)))
        s3 = s2 - ph
        g3 = SIMD.vifelse(s2 < ph, V8(one(Limb)), V8(zero(Limb)))
        if any(s3 <= V8(Limb(2)))
            Base.Cartesian.@nexprs 8 k -> ((r[ro+i+k-1], brw, qin, hin2, hin1) =
                submul_2_lane(r[ro+i+k-1], a[ao+i+k-1], b0, b1, brw, qin, hin2, hin1))
        else
            g = g1 + g2 + g3
            bv = SIMD.shufflevector(g, V8(brw), Val((8, 0, 1, 2, 3, 4, 5, 6)))
            SIMD.vstore(s3 - bv, r, ro + i)
            brw = g[8]
            qin = q[8]
            hin2 = h[7]
            hin1 = h[8]
        end
        i += 8
    end
    @inbounds while i <= n
        (r[ro+i], brw, qin, hin2, hin1) =
            submul_2_lane(r[ro+i], a[ao+i], b0, b1, brw, qin, hin2, hin1)
        i += 1
    end
    co = (DLimb(hin1) << 64) + qin + hin2 + brw
    return (co >> 64) % Limb, co % Limb
end

# One scalar lane of addmul_2!: r[j] += lo(a*b0) + carry-in state, returning
# the updated (c, qin, hin2, hin1). q merges hi(a*b0) + lo(a*b1) (both weight
# +1); its overflow bit has weight +2 and folds into the hi(a*b1) stream,
# which is <= typemax-1 so the fold cannot overflow.
@inline function addmul_2_lane(rj::Limb, aj::Limb, b0::Limb, b1::Limb, c::Limb, qin::Limb, hin2::Limb, hin1::Limb)
    p0 = widemul(aj, b0)
    p1 = widemul(aj, b1)
    t = UInt128(rj) + (p0 % Limb) + qin + hin2 + c
    q, o = Base.add_with_overflow((p0 >> 64) % Limb, p1 % Limb)
    return t % Limb, (t >> 64) % Limb, q, hin1, ((p1 >> 64) % Limb) + Limb(o)
end

# Two rows of b per pass over a, SIMD (same cold-propagate idea as addmul_1!):
# per lane r += lo0 + q<<64 + h<<128 with q = hi0+lo1, h = hi1+overflow(q), so
# the resolved carry-in is 0..3 and can only chain when some lane sum
# >= typemax-2 (~2^-62-rare), which falls back to a scalar block. Versus two
# addmul_1! passes this halves the r loads/stores and loop overhead.
# Writes (not adds) the two overflow limbs r[m+1], r[m+2].
@inline function addmul_2!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, m::Int, b0::Limb, b1::Limb)
    b00 = b0 & LO32; b01 = b0 >> 32
    b10 = b1 & LO32; b11 = b1 >> 32
    c = zero(Limb)     # resolved carry into the next lane (0..3)
    qin = zero(Limb)   # q = hi0+lo1 pending from the previous lane
    hin2 = zero(Limb)  # h = hi1+carry(q) pending from two lanes back
    hin1 = zero(Limb)  # h pending from the previous lane
    z = zero(Limb)
    i = 1
    @inbounds while i + 7 <= m
        va = SIMD.vload(V8, a, ao + i)
        vr = SIMD.vload(V8, r, ro + i)
        lo0, hi0 = mul128_v8(va, b00, b01)
        lo1, hi1 = mul128_v8(va, b10, b11)
        q = hi0 + lo1
        h = hi1 + SIMD.vifelse(q < hi0, V8(one(Limb)), V8(zero(Limb)))
        pq = SIMD.shufflevector(q, V8(qin), Val((8, 0, 1, 2, 3, 4, 5, 6)))
        ph = SIMD.shufflevector(h, V8((hin2, hin1, z, z, z, z, z, z)),
                                Val((8, 9, 0, 1, 2, 3, 4, 5)))
        s1 = vr + lo0
        g1 = SIMD.vifelse(s1 < vr, V8(one(Limb)), V8(zero(Limb)))
        s2 = s1 + pq
        g2 = SIMD.vifelse(s2 < s1, V8(one(Limb)), V8(zero(Limb)))
        s3 = s2 + ph
        g3 = SIMD.vifelse(s3 < s2, V8(one(Limb)), V8(zero(Limb)))
        if any(s3 >= V8(typemax(Limb) - Limb(2)))
            Base.Cartesian.@nexprs 8 k -> ((r[ro+i+k-1], c, qin, hin2, hin1) =
                addmul_2_lane(r[ro+i+k-1], a[ao+i+k-1], b0, b1, c, qin, hin2, hin1))
        else
            g = g1 + g2 + g3
            cv = SIMD.shufflevector(g, V8(c), Val((8, 0, 1, 2, 3, 4, 5, 6)))
            SIMD.vstore(s3 + cv, r, ro + i)
            c = g[8]
            qin = q[8]
            hin2 = h[7]
            hin1 = h[8]
        end
        i += 8
    end
    @inbounds while i <= m
        (r[ro+i], c, qin, hin2, hin1) =
            addmul_2_lane(r[ro+i], a[ao+i], b0, b1, c, qin, hin2, hin1)
        i += 1
    end
    @inbounds begin
        t = UInt128(qin) + hin2 + c
        r[ro+m+1] = t % Limb
        # the full product fits m+2 limbs, so this top add cannot overflow
        r[ro+m+2] = hin1 + (t >> 64) % Limb
    end
    return nothing
end

# Non-adding addmul_2!: r[1..m+2] = a[1..m] * (b0 + b1*B), r is not read.
# Same stream layout minus the r addend, so carry-in is 0..2 and the cold
# guard is typemax-1; scalar lanes reuse addmul_2_lane with rj = 0.
@inline function mul_2!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, m::Int, b0::Limb, b1::Limb)
    b00 = b0 & LO32; b01 = b0 >> 32
    b10 = b1 & LO32; b11 = b1 >> 32
    c = zero(Limb); qin = zero(Limb); hin2 = zero(Limb); hin1 = zero(Limb)
    z = zero(Limb)
    i = 1
    @inbounds while i + 7 <= m
        va = SIMD.vload(V8, a, ao + i)
        lo0, hi0 = mul128_v8(va, b00, b01)
        lo1, hi1 = mul128_v8(va, b10, b11)
        q = hi0 + lo1
        h = hi1 + SIMD.vifelse(q < hi0, V8(one(Limb)), V8(zero(Limb)))
        pq = SIMD.shufflevector(q, V8(qin), Val((8, 0, 1, 2, 3, 4, 5, 6)))
        ph = SIMD.shufflevector(h, V8((hin2, hin1, z, z, z, z, z, z)),
                                Val((8, 9, 0, 1, 2, 3, 4, 5)))
        s1 = lo0 + pq
        g1 = SIMD.vifelse(s1 < lo0, V8(one(Limb)), V8(zero(Limb)))
        s2 = s1 + ph
        g2 = SIMD.vifelse(s2 < s1, V8(one(Limb)), V8(zero(Limb)))
        if any(s2 >= V8(typemax(Limb) - Limb(1)))
            Base.Cartesian.@nexprs 8 k -> ((r[ro+i+k-1], c, qin, hin2, hin1) =
                addmul_2_lane(zero(Limb), a[ao+i+k-1], b0, b1, c, qin, hin2, hin1))
        else
            g = g1 + g2
            cv = SIMD.shufflevector(g, V8(c), Val((8, 0, 1, 2, 3, 4, 5, 6)))
            SIMD.vstore(s2 + cv, r, ro + i)
            c = g[8]
            qin = q[8]
            hin2 = h[7]
            hin1 = h[8]
        end
        i += 8
    end
    @inbounds while i <= m
        (r[ro+i], c, qin, hin2, hin1) =
            addmul_2_lane(zero(Limb), a[ao+i], b0, b1, c, qin, hin2, hin1)
        i += 1
    end
    @inbounds begin
        t = UInt128(qin) + hin2 + c
        r[ro+m+1] = t % Limb
        r[ro+m+2] = hin1 + (t >> 64) % Limb
    end
    return nothing
end

# addmul_2! beats two addmul_1! rows at every width, so always pair rows;
# mul_2! covers the first pair without a separate mul_1! pass.
function mul_basecase!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, m::Int, b::Memory{Limb}, bo::Int, n::Int)
    if n == 1
        @inbounds r[ro+m+1] = mul_1!(r, ro, a, ao, m, b[bo+1])
        return nothing
    end
    @inbounds mul_2!(r, ro, a, ao, m, b[bo+1], b[bo+2])
    j = 3
    @inbounds while j + 1 <= n
        addmul_2!(r, ro+j-1, a, ao, m, b[bo+j], b[bo+j+1])
        j += 2
    end
    @inbounds if j <= n
        r[ro+m+j] = addmul_1!(r, ro+j-1, a, ao, m, b[bo+j])
    end
    return nothing
end

# r[1..2n] = a[1..n]^2; r must not alias a. Half the multiplies of
# mul_basecase!: build the off-diagonal triangle sum_{i<j} a_i*a_j once with
# paired mul_2!/addmul_2! rows (rows j,j+1 over a[j+2..n] share one pass; the
# split-off a[j]*a[j+1] term is added separately), then one fused pass doubles
# the triangle and adds the diagonal squares a_i^2.
function sqr_basecase!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int)
    @inbounds begin
        if n == 1
            p = widemul(a[ao+1], a[ao+1])
            r[ro+1] = p % Limb
            r[ro+2] = (p >> 64) % Limb
            return nothing
        end
        # Off-diagonal triangle into r[2..2n-1]: a[j]*a[j+k] lands at limb
        # index 2j+k-1, so rows j,j+1 over the shared range a[j+2..n] form one
        # (a[j], a[j+1]) two-row pass based at r[2j+1].
        if n == 2
            p = widemul(a[ao+1], a[ao+2])
            r[ro+2] = p % Limb
            r[ro+3] = (p >> 64) % Limb
        else
            mul_2!(r, ro + 2, a, ao + 2, n - 2, a[ao+1], a[ao+2])
            p = widemul(a[ao+1], a[ao+2])
            r[ro+2] = p % Limb
            t = UInt128(r[ro+3]) + (p >> 64) % Limb
            r[ro+3] = t % Limb
            c = (t >> 64) % Limb
            k = 4
            while c != zero(Limb)
                t = UInt128(r[ro+k]) + c
                r[ro+k] = t % Limb
                c = (t >> 64) % Limb
                k += 1
            end
            j = 3
            while j <= n - 2
                addmul_2!(r, ro + 2j, a, ao + j + 1, n - j - 1, a[ao+j], a[ao+j+1])
                p = widemul(a[ao+j], a[ao+j+1])
                t = UInt128(r[ro+2j]) + p % Limb
                r[ro+2j] = t % Limb
                # hi(p) <= 2^64-2, so hi + carry cannot overflow a limb
                c = ((t >> 64) % Limb) + (p >> 64) % Limb
                k = 2j + 1
                while c != zero(Limb)
                    t = UInt128(r[ro+k]) + c
                    r[ro+k] = t % Limb
                    c = (t >> 64) % Limb
                    k += 1
                end
                j += 2
            end
            if j == n - 1
                # unpaired last row: the single product a[n-1]*a[n]
                p = widemul(a[ao+j], a[ao+j+1])
                t = UInt128(r[ro+2n-2]) + p % Limb
                r[ro+2n-2] = t % Limb
                r[ro+2n-1] = ((t >> 64) % Limb) + (p >> 64) % Limb
            end
        end
        # Fused double-and-add-diagonal: slot i covers r[2i-1..2i]; the bit
        # shifted out of a slot and the additive carry both have weight
        # 2^128 = bit 0 of the next slot.
        r[ro+1] = zero(Limb)
        r[ro+2n] = zero(Limb)
        bit = zero(Limb)
        cc = zero(Limb)
        for i in 1:n
            u_lo = r[ro+2i-1]
            u_hi = r[ro+2i]
            d_lo = (u_lo << 1) | bit
            d_hi = (u_hi << 1) | (u_lo >> 63)
            bit = u_hi >> 63
            p = widemul(a[ao+i], a[ao+i])
            t = UInt128(d_lo) + (p % Limb) + cc
            t2 = (t >> 64) + ((p >> 64) % Limb) + d_hi
            r[ro+2i-1] = t % Limb
            r[ro+2i] = t2 % Limb
            cc = (t2 >> 64) % Limb
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

@inline sterm(c::Int64, x::Limb) =
    c >= 0 ? Int128(widemul(Limb(c), x)) : -Int128(widemul(Limb(-c), x))

# Fused Lehmer matrix apply: r1 = A*U + B*V, r2 = C*U + D*V in one pass over
# the operands via exact two's-complement limb accumulation (signed Int128
# carries; |carry| stays < 2^63 for |cofactor| < 2^62). Valid cofactor
# matrices give nonnegative results, so the final carries are the top limbs.
# Writes n+1 limbs each, n = max(lu, lv); returns n+1.
function lehmer_apply!(r1::Memory{Limb}, r1o::Int, r2::Memory{Limb}, r2o::Int,
                       u::Memory{Limb}, lu::Int, v::Memory{Limb}, lv::Int,
                       A::Int64, B::Int64, C::Int64, D::Int64)
    n = max(lu, lv)
    c1 = Int128(0)
    c2 = Int128(0)
    @inbounds for i in 1:n
        ui = i <= lu ? u[i] : zero(Limb)
        vi = i <= lv ? v[i] : zero(Limb)
        a1 = c1 + sterm(A, ui) + sterm(B, vi)
        r1[r1o+i] = a1 % Limb
        c1 = a1 >> 64
        a2 = c2 + sterm(C, ui) + sterm(D, vi)
        r2[r2o+i] = a2 % Limb
        c2 = a2 >> 64
    end
    @inbounds r1[r1o+n+1] = c1 % Limb
    @inbounds r2[r2o+n+1] = c2 % Limb
    return n + 1
end

# -m0^(-1) mod β for odd m0: Hensel/Newton doubling from the 3-bit-correct
# seed x = m0 (m0*m0 ≡ 1 mod 8); five doublings give ≥ 64 correct bits.
@inline function mont_ninv(m0::Limb)
    x = m0
    for _ in 1:5
        x *= Limb(2) - m0 * x
    end
    return -x
end

# Montgomery reduction: t[1..2k+1] holds T < m·β^k with t[2k+1] = 0 on entry;
# writes r[1..k] = T·β^(-k) mod m (< m). ninv = -m^(-1) mod β. t is destroyed.
function redc!(r::Memory{Limb}, ro::Int, t::Memory{Limb}, to::Int,
               m::Memory{Limb}, mo::Int, k::Int, ninv::Limb)
    @inbounds for i in 1:k
        q = t[to+i] * ninv
        c = addmul_1!(t, to + i - 1, m, mo, k, q)
        add_carry!(t, to, 2k + 1, i + k, c)
    end
    if (@inbounds t[to+2k+1]) != 0 || cmp_limbs(t, to + k, k, m, mo, k) >= 0
        sub_n!(r, ro, t, to + k, m, mo, k)
    else
        @inbounds for i in 1:k
            r[ro+i] = t[to+k+i]
        end
    end
    return nothing
end

# Möller–Granlund reciprocal: v = ⌊(β²-1)/d⌋ - β for normalized d (top bit set).
# One hardware 128/64 divide at setup; every per-limb divide becomes 2 muls.
@inline function invert_limb(d::Limb)
    return ((typemax(DLimb) - (DLimb(d) << 64)) ÷ d) % Limb
end

# Rare second-fixup bodies. @noinline keeps them as real (predicted-not-taken)
# branches: inlined, LLVM if-converts them to cmovs on the loop-carried
# remainder chain, costing ~3 cycles per limb in the divrem_1!/divrem_2! loops.
@noinline div_fixup(q1::Limb, r::Limb, d::Limb) = (q1 + one(Limb), r - d)
@noinline div_fixup(q1::Limb, r::DLimb, dd::DLimb) = (q1 + one(Limb), r - dd)

# Divide (u1:u0) by normalized d given v = invert_limb(d); requires u1 < d.
# The first fixup fires ~50% of the time, so it is masked (branch-free) to keep
# the loop-carried remainder chain free of mispredicts; the second is rare and branchy.
@inline function div_2by1(u1::Limb, u0::Limb, d::Limb, v::Limb)
    p = DLimb(v) * u1 + ((DLimb(u1) << 64) | u0)
    q1 = (p >> 64) % Limb + one(Limb)
    q0 = p % Limb
    r = u0 - q1 * d
    mask = -Limb(r > q0)
    q1 += mask
    r += d & mask
    if r >= d
        q1, r = div_fixup(q1, r, d)
    end
    return q1, r
end

# 3/2 reciprocal (Möller–Granlund Alg. 6): v = ⌊(β³-1)/(d1·β+d0)⌋ - β,
# for normalized d1; refines invert_limb(d1) by two conditional corrections.
@inline function invert_pi1(d1::Limb, d0::Limb)
    v = invert_limb(d1)
    p = d1 * v
    p += d0
    if p < d0
        v -= one(Limb)
        if p >= d1
            v -= one(Limb)
            p -= d1
        end
        p -= d1
    end
    t = DLimb(v) * d0
    t1 = (t >> 64) % Limb
    t0 = t % Limb
    p += t1
    if p < t1
        v -= one(Limb)
        if (p > d1) | ((p == d1) & (t0 >= d0))
            v -= one(Limb)
        end
    end
    return v
end

# Divide ⟨u2,u1,u0⟩ by normalized ⟨d1,d0⟩ given v = invert_pi1(d1, d0);
# requires ⟨u2,u1⟩ < ⟨d1,d0⟩. Returns (q, r1, r0) with ⟨r1,r0⟩ the remainder.
# Möller–Granlund Alg. 2: candidate off by at most 1 either way, same fixup
# structure as div_2by1 (first masked, second rare and branchy).
@inline function div_3by2(u2::Limb, u1::Limb, u0::Limb, d1::Limb, d0::Limb, v::Limb)
    dd = (DLimb(d1) << 64) | d0
    q = DLimb(v) * u2 + ((DLimb(u2) << 64) | u1)
    q1 = (q >> 64) % Limb
    q0 = q % Limb
    # r = ⟨u1,u0⟩ - q1*⟨d1,d0⟩ - dd (mod β²), regrouped so the dd subtraction
    # runs off the q1 critical path and both q1 muls merge into low128(q1*dd)
    w = ((DLimb(u1) << 64) | u0) - dd
    r = w - (DLimb(q1) * d0 + (DLimb(q1 * d1) << 64))
    q1 += one(Limb)
    # first fixup fires ~50%: masked to avoid mispredicts on the remainder chain
    mask = -Limb((r >> 64) % Limb >= q0)
    q1 += mask
    r += dd & ((DLimb(mask) << 64) | mask)
    if r >= dd
        q1, r = div_fixup(q1, r, dd)
    end
    return q1, (r >> 64) % Limb, r % Limb
end

# Schoolbook (Knuth Algorithm D) quotient/remainder, two quotient limbs per
# pass (radix β²). u (nn limbs) is destroyed: the m-limb remainder is left in
# u[1..m]. d must be normalized (top bit of d[m] set), m ≥ 2, v = invert_pi1 of
# its top two limbs. Writes q[1..nn-m]; returns the extra top quotient bit qh.
#
# Each super-row: a 4/2 division of the top window (two chained div_3by2)
# yields the exact quotient ⟨qhi,qlo⟩ of the top four limbs by ⟨d1,d0⟩ — a
# single-super-digit Knuth estimate, at most 2 too large for normalized d
# (TAOCP 4.3.1 Thm A/B) — then one submul_2! sweep subtracts both rows at
# once, halving the passes over u versus limb-at-a-time. The top two window
# limbs ⟨n1, n0⟩ live in registers across rows (their memory slots are stale).
function divrem_bc!(q::Memory{Limb}, qo::Int, u::Memory{Limb}, uo::Int, nn::Int,
                    d::Memory{Limb}, do_::Int, m::Int, v::Limb)
    qn = nn - m
    qh = zero(Limb)
    if cmp_limbs(u, uo+qn, m, d, do_, m) >= 0
        qh = one(Limb)
        sub_n!(u, uo+qn, u, uo+qn, d, do_, m)
    end
    d1 = @inbounds d[do_+m]
    d0 = @inbounds d[do_+m-1]
    dd = (DLimb(d1) << 64) | d0
    n1 = @inbounds u[uo+nn]
    n0 = @inbounds u[uo+nn-1]
    j = qn
    @inbounds while j >= 2
        if n1 == d1 && n0 == d0
            # ⟨n1, n0⟩ == ⟨d1, d0⟩ violates the div_3by2 precondition:
            # qhat = β-1 exactly, and the window bound W < β·d guarantees no
            # borrow past the top limb. One scalar row, then re-pair.
            qhat = typemax(Limb)
            u[uo+j+m-1] = n0   # submul needs the stale slot refreshed
            submul_1!(u, uo+j-1, d, do_, m, qhat)
            n1 = u[uo+j+m-1]
            n0 = u[uo+j+m-2]
            q[qo+j] = qhat
            j -= 1
            continue
        end
        qhi, t1, t0 = div_3by2(n1, n0, u[uo+j+m-2], d1, d0, v)
        qlo, r1, r0 = div_3by2(t1, t0, u[uo+j+m-3], d1, d0, v)
        co1, co0 = m > 2 ? submul_2!(u, uo+j-2, d, do_, m-2, qlo, qhi) :
                           (zero(Limb), zero(Limb))
        rr = (DLimb(r1) << 64) | r0
        co = (DLimb(co1) << 64) | co0
        brw = rr < co
        rr -= co
        qq = (DLimb(qhi) << 64) | qlo
        while brw   # rare: estimate 1 or 2 too large, add the divisor back
            qq -= one(DLimb)
            c = m > 2 ? add_n!(u, uo+j-2, u, uo+j-2, d, do_, m-2) : zero(Limb)
            s, o1 = Base.add_with_overflow(rr, dd)
            s, o2 = Base.add_with_overflow(s, DLimb(c))
            brw = !(o1 | o2)   # 128-bit overflow cancels the borrow
            rr = s
        end
        q[qo+j] = (qq >> 64) % Limb
        q[qo+j-1] = qq % Limb
        n1 = (rr >> 64) % Limb
        n0 = rr % Limb
        j -= 2
    end
    @inbounds if j == 1   # leftover scalar row (3/2 qhat, error ≤ 1)
        if n1 == d1 && n0 == d0
            qhat = typemax(Limb)
            u[uo+m] = n0
            submul_1!(u, uo, d, do_, m, qhat)
            n1 = u[uo+m]
            n0 = u[uo+m-1]
        else
            qhat, r1, r0 = div_3by2(n1, n0, u[uo+m-1], d1, d0, v)
            cy = m > 2 ? submul_1!(u, uo, d, do_, m-2, qhat) : zero(Limb)
            cy1 = Limb(r0 < cy)
            r0 -= cy
            cy2 = r1 < cy1
            r1 -= cy1
            if cy2   # rare: qhat one too large, add the divisor back
                qhat -= one(Limb)
                c = m > 2 ? add_n!(u, uo, u, uo, d, do_, m-2) : zero(Limb)
                s = ((DLimb(r1) << 64) | r0) + dd + c
                r1 = (s >> 64) % Limb   # 128-bit overflow cancels the borrow
                r0 = s % Limb
            end
            n1 = r1
            n0 = r0
        end
        q[qo+1] = qhat
    end
    @inbounds u[uo+m] = n1
    @inbounds u[uo+m-1] = n0
    return qh
end

# a[1..n] ÷ ⟨d1,d0⟩ (d1 ≠ 0, n ≥ 2): writes n-1 quotient limbs (top may be
# zero), returns the remainder (r1, r0). The two-limb remainder window stays in
# registers and unnormalized divisors are handled by shifting the numerator on
# the fly, so no scratch or numerator copy is needed (cf. mpn_divrem_2).
# q may alias a at the same offset.
function divrem_2!(q::Memory{Limb}, qo::Int, a::Memory{Limb}, ao::Int, n::Int, d1::Limb, d0::Limb)
    l = leading_zeros(d1)
    if l == 0
        v = invert_pi1(d1, d0)
        dd = (DLimb(d1) << 64) | d0
        w = @inbounds (DLimb(a[ao+n]) << 64) | a[ao+n-1]
        qh = w >= dd
        qh && (w -= dd)
        @inbounds q[qo+n-1] = qh
        r1 = (w >> 64) % Limb
        r0 = w % Limb
        @inbounds for j in n-2:-1:1
            qhat, r1, r0 = div_3by2(r1, r0, a[ao+j], d1, d0, v)
            q[qo+j] = qhat
        end
        return r1, r0
    end
    # shifted numerator has n+1 limbs; its top spill a[n] >> (64-l) < 2^l ≤ dn1
    # seeds the window, so exactly n-1 quotient limbs come out of the loop
    dn1 = (d1 << l) | (d0 >> (64 - l))
    dn0 = d0 << l
    v = invert_pi1(dn1, dn0)
    r1 = @inbounds a[ao+n] >> (64 - l)
    r0 = @inbounds (a[ao+n] << l) | (a[ao+n-1] >> (64 - l))
    @inbounds for j in n-1:-1:2
        u = (a[ao+j] << l) | (a[ao+j-1] >> (64 - l))
        qhat, r1, r0 = div_3by2(r1, r0, u, dn1, dn0, v)
        q[qo+j] = qhat
    end
    qhat, r1, r0 = div_3by2(r1, r0, @inbounds(a[ao+1]) << l, dn1, dn0, v)
    @inbounds q[qo+1] = qhat
    return r1 >> l, (r0 >> l) | (r1 << (64 - l))
end

# Two dividend limbs per iteration with a lazy (unreduced, two-limb) running
# remainder R < β²: per pair, R ← r1·B3 + r0·B2 + ⟨u1,u0⟩ folded mod β²
# (B2 = β² mod d, B3 = β³ mod d), so the loop-carried chain is two independent
# muls plus adds — no 2/1 division inside the loop. Quotient mass accrues off
# the chain via K2 = ⌊β²/d⌋ = β+v, K3 = ⌊β³/d⌋ = β·K2 + q3 into a sliding
# two-limb pending window ⟨p1,p0⟩; the finalized limbs written each iteration
# cannot receive later carries (all future pieces land strictly below), so no
# ripple into written quotient limbs is needed. Requires d normalized,
# d ≠ 2^63 (K2 = 2β wouldn't fit the k2lo limb), n ≥ 4.
function divrem_1_pi2!(q::Memory{Limb}, qo::Int, a::Memory{Limb}, ao::Int, n::Int, d::Limb, v::Limb)
    # setup: no hardware divide — B2 = β² - K2·d via wrapping negation
    b2 = (-(widemul(d, v) + (DLimb(d) << 64))) % Limb
    q3, b3 = div_2by1(b2, zero(Limb), d, v)     # ⟨B2,0⟩ = q3·d + B3
    k3l = q3                                    # K3 - β² = ⟨v, q3⟩
    r1 = @inbounds a[ao+n]
    r0 = @inbounds a[ao+n-1]
    p1 = zero(Limb)
    p0 = zero(Limb)
    i = n - 3
    @inbounds while i >= 1
        u1 = a[ao+i+1]
        u0 = a[ao+i]
        # remainder chain: W = r1·B3 + r0·B2 + ⟨u1,u0⟩ (≤ 130 bits), then fold
        # the c ∈ {0,1,2} overflows of weight β² back in as c·B2
        m3 = widemul(r1, b3)
        m2 = widemul(r0, b2)
        w, o1 = Base.add_with_overflow(m3, m2)
        w, o2 = Base.add_with_overflow(w, (DLimb(u1) << 64) | u0)
        c = Limb(o1) + Limb(o2)
        # c·B2 as two masked adds keeps a mul off the loop-carried chain
        w, o3 = Base.add_with_overflow(w, DLimb(b2 & (-Limb(o1))) + (b2 & (-Limb(o2))))
        if o3                                    # rare second fold; w < 3β now
            w += b2
            c += one(Limb)
        end
        # quotient piece Qp = r1·K3 + (r0+c)·K2 < 2β³ at weight β^(i-1) with
        # K3 = β² + ⟨k3h,k3l⟩, K2 = β + v. f = r0+c may wrap: the missing β·K2
        # contributes fc·v at weight β¹ and fc at weight β². By weight:
        #   L0: lo(r1·k3l), lo(f·v)
        #   L1: hi(r1·k3l), hi(f·v), lo(r1·k3h), f, fc·v
        #   L2: hi(r1·k3h), r1, fc, p0   (+ carries)
        #   L3: p1                        (+ carries; no carry-out: S is final)
        f = r0 + c
        fc = Limb(f < r0)
        ml = widemul(r1, k3l)
        mh = widemul(r1, v)                      # k3h == v
        A, cA = Base.add_with_overflow(ml, widemul(f, v))
        A, cA2 = Base.add_with_overflow(A, DLimb(v & (-fc)) << 64)
        s0 = A % Limb
        B = DLimb((A >> 64) % Limb) + (mh % Limb) + f
        s1 = B % Limb
        C = DLimb((mh >> 64) % Limb) + r1 + fc + p0 +
            ((B >> 64) % Limb) + Limb(cA) + Limb(cA2)
        s2 = C % Limb
        s3 = p1 + ((C >> 64) % Limb)   # no wrap: finalized mass < β²
        q[qo+i+3] = s3
        q[qo+i+2] = s2
        p1 = s1
        p0 = s0
        r1 = (w >> 64) % Limb
        r0 = w % Limb
        i -= 2
    end
    if i == 0   # one leftover limb: V = R·β + u[1], quotient at weight β⁰
        ul = @inbounds a[ao+1]
        m2 = widemul(r1, b2)
        w, o1 = Base.add_with_overflow(m2, (DLimb(r0) << 64) | ul)
        c = Limb(o1)
        w, o2 = Base.add_with_overflow(w, DLimb(c) * b2)
        if o2
            w += b2
            c += one(Limb)
        end
        x1 = (w >> 64) % Limb
        x0 = w % Limb
        b = x1 >= d
        b && (x1 -= d)
        q2, rem = div_2by1(x1, x0, d, v)
        # Qt = (r1+c)·K2 + b·β + q2; pending sits at (q[3], q[2])
        f = r1 + c
        fc = Limb(f < r1)
        A, cA = Base.add_with_overflow(widemul(f, v), DLimb(q2))
        A, cA2 = Base.add_with_overflow(A, DLimb(v & (-fc)) << 64)
        B = DLimb((A >> 64) % Limb) + f + Limb(b) + p0
        C = DLimb((B >> 64) % Limb) + fc + p1 + Limb(cA) + Limb(cA2)
        @inbounds q[qo+1] = A % Limb
        @inbounds q[qo+2] = B % Limb
        @inbounds q[qo+3] = C % Limb   # no wrap: finalized mass < β
        return rem
    end
    # no leftover: reduce R = ⟨r1,r0⟩, quotient Qt = b·β + q2 into ⟨p1,p0⟩
    b = r1 >= d
    b && (r1 -= d)
    q2, rem = div_2by1(r1, r0, d, v)
    s = DLimb(p0) + q2
    @inbounds q[qo+1] = s % Limb
    @inbounds q[qo+2] = p1 + ((s >> 64) % Limb) + Limb(b)
    return rem
end

function divrem_1!(q::Memory{Limb}, qo::Int, a::Memory{Limb}, ao::Int, n::Int, d::Limb)
    l = leading_zeros(d)
    dn = d << l
    v = invert_limb(dn)
    if l == 0
        # below ~10 limbs the pi2 setup (one div_2by1 + muls) outweighs the
        # faster loop; benchmark-tuned crossover
        if n >= 10 && d != (one(Limb) << 63)
            return divrem_1_pi2!(q, qo, a, ao, n, d, v)
        end
        rem = zero(Limb)
        @inbounds for i in n:-1:1
            q[qo+i], rem = div_2by1(rem, a[ao+i], d, v)
        end
        return rem
    end
    # Divide the left-shifted numerator by dn; quotient limbs are unchanged
    # and the true remainder is rem >> l. The spill a[n] >> (64-l) < 2^l ≤ dn
    # seeds the running remainder, so no extra quotient limb is needed.
    rem = @inbounds a[ao+n] >> (64 - l)
    @inbounds for i in n:-1:2
        u = (a[ao+i] << l) | (a[ao+i-1] >> (64 - l))
        q[qo+i], rem = div_2by1(rem, u, dn, v)
    end
    @inbounds q[qo+1], rem = div_2by1(rem, a[ao+1] << l, dn, v)
    return rem >> l
end
