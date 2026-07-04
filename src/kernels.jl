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

# Möller–Granlund reciprocal: v = ⌊(β²-1)/d⌋ - β for normalized d (top bit set).
# One hardware 128/64 divide at setup; every per-limb divide becomes 2 muls.
@inline function invert_limb(d::Limb)
    return ((typemax(DLimb) - (DLimb(d) << 64)) ÷ d) % Limb
end

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
        q1 += one(Limb)
        r -= d
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
    r1 = u1 - q1 * d1
    r = ((DLimb(r1) << 64) | u0) - DLimb(q1) * d0 - dd
    q1 += one(Limb)
    if (r >> 64) % Limb >= q0
        q1 -= one(Limb)
        r += dd
    end
    if r >= dd
        q1 += one(Limb)
        r -= dd
    end
    return q1, (r >> 64) % Limb, r % Limb
end

# Schoolbook (Knuth Algorithm D) quotient/remainder with 3/2 qhat estimation.
# u (nn limbs) is destroyed: the m-limb remainder is left in u[1..m].
# d must be normalized (top bit of d[m] set), m ≥ 2, v = invert_pi1 of its top
# two limbs. Writes q[1..nn-m]; returns the extra top quotient bit qh ∈ {0,1}.
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
    n1 = @inbounds u[uo+nn]   # top window limb lives in a register; its memory slot is stale
    @inbounds for j in qn:-1:1
        if n1 == d1 && u[uo+j+m-1] == d0
            # ⟨n1, u1⟩ == ⟨d1, d0⟩: qhat = β-1 exactly, and the window bound
            # W < β·d guarantees no borrow past the top limb.
            qhat = typemax(Limb)
            submul_1!(u, uo+j-1, d, do_, m, qhat)
            n1 = u[uo+j+m-1]
        else
            qhat, r1, r0 = div_3by2(n1, u[uo+j+m-1], u[uo+j+m-2], d1, d0, v)
            cy = m > 2 ? submul_1!(u, uo+j-1, d, do_, m-2, qhat) : zero(Limb)
            cy1 = Limb(r0 < cy)
            r0 -= cy
            cy2 = r1 < cy1
            r1 -= cy1
            if cy2   # rare: qhat one too large, add the divisor back
                qhat -= one(Limb)
                c = m > 2 ? add_n!(u, uo+j-1, u, uo+j-1, d, do_, m-2) : zero(Limb)
                s = ((DLimb(r1) << 64) | r0) + ((DLimb(d1) << 64) | d0) + c
                r1 = (s >> 64) % Limb   # 128-bit overflow cancels the borrow
                r0 = s % Limb
            end
            u[uo+j+m-2] = r0
            n1 = r1
        end
        q[qo+j] = qhat
    end
    @inbounds u[uo+m] = n1
    return qh
end

function divrem_1!(q::Memory{Limb}, qo::Int, a::Memory{Limb}, ao::Int, n::Int, d::Limb)
    l = leading_zeros(d)
    dn = d << l
    v = invert_limb(dn)
    if l == 0
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
