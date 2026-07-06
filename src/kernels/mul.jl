# Multiplication and squaring kernels (mpn layer).

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

# Straight-line squares for n <= 4: cross products column-summed into
# triangle limbs T, doubled via shifts, then the diagonal squares added in one
# UInt128 chain. Skips the triangle/double-pass structure entirely.
@inline function sqr_2!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int)
    @inbounds begin
        a1 = a[ao+1]; a2 = a[ao+2]
        p11 = widemul(a1, a1); p12 = widemul(a1, a2); p22 = widemul(a2, a2)
        T1 = p12 % Limb; T2 = (p12 >> 64) % Limb
        r[ro+1] = p11 % Limb
        s = (p11 >> 64) + (T1 << 1)
        r[ro+2] = s % Limb
        s = (s >> 64) + (p22 % Limb) + ((T2 << 1) | (T1 >> 63))
        r[ro+3] = s % Limb
        r[ro+4] = ((s >> 64) % Limb) + ((p22 >> 64) % Limb) + (T2 >> 63)
    end
    return nothing
end

@inline function sqr_3!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int)
    @inbounds begin
        a1 = a[ao+1]; a2 = a[ao+2]; a3 = a[ao+3]
        p11 = widemul(a1, a1); p22 = widemul(a2, a2); p33 = widemul(a3, a3)
        p12 = widemul(a1, a2); p13 = widemul(a1, a3); p23 = widemul(a2, a3)
        T1 = p12 % Limb
        u = (p12 >> 64) + (p13 % Limb)
        T2 = u % Limb
        u = (u >> 64) + ((p13 >> 64) % Limb) + (p23 % Limb)
        T3 = u % Limb
        T4 = ((u >> 64) % Limb) + ((p23 >> 64) % Limb)  # cannot overflow
        r[ro+1] = p11 % Limb
        s = (p11 >> 64) + (T1 << 1)
        r[ro+2] = s % Limb
        s = (s >> 64) + (p22 % Limb) + ((T2 << 1) | (T1 >> 63))
        r[ro+3] = s % Limb
        s = (s >> 64) + ((p22 >> 64) % Limb) + ((T3 << 1) | (T2 >> 63))
        r[ro+4] = s % Limb
        s = (s >> 64) + (p33 % Limb) + ((T4 << 1) | (T3 >> 63))
        r[ro+5] = s % Limb
        r[ro+6] = ((s >> 64) % Limb) + ((p33 >> 64) % Limb) + (T4 >> 63)
    end
    return nothing
end

@inline function sqr_4!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int)
    @inbounds begin
        a1 = a[ao+1]; a2 = a[ao+2]; a3 = a[ao+3]; a4 = a[ao+4]
        p12 = widemul(a1, a2); p13 = widemul(a1, a3); p14 = widemul(a1, a4)
        p23 = widemul(a2, a3); p24 = widemul(a2, a4); p34 = widemul(a3, a4)
        T1 = p12 % Limb
        u = (p12 >> 64) + (p13 % Limb)
        T2 = u % Limb
        u = (u >> 64) + ((p13 >> 64) % Limb) + (p14 % Limb) + (p23 % Limb)
        T3 = u % Limb
        u = (u >> 64) + ((p14 >> 64) % Limb) + ((p23 >> 64) % Limb) + (p24 % Limb)
        T4 = u % Limb
        u = (u >> 64) + ((p24 >> 64) % Limb) + (p34 % Limb)
        T5 = u % Limb
        T6 = ((u >> 64) % Limb) + ((p34 >> 64) % Limb)  # cannot overflow
        p11 = widemul(a1, a1); p22 = widemul(a2, a2)
        p33 = widemul(a3, a3); p44 = widemul(a4, a4)
        r[ro+1] = p11 % Limb
        s = (p11 >> 64) + (T1 << 1)
        r[ro+2] = s % Limb
        s = (s >> 64) + (p22 % Limb) + ((T2 << 1) | (T1 >> 63))
        r[ro+3] = s % Limb
        s = (s >> 64) + ((p22 >> 64) % Limb) + ((T3 << 1) | (T2 >> 63))
        r[ro+4] = s % Limb
        s = (s >> 64) + (p33 % Limb) + ((T4 << 1) | (T3 >> 63))
        r[ro+5] = s % Limb
        s = (s >> 64) + ((p33 >> 64) % Limb) + ((T5 << 1) | (T4 >> 63))
        r[ro+6] = s % Limb
        s = (s >> 64) + (p44 % Limb) + ((T6 << 1) | (T5 >> 63))
        r[ro+7] = s % Limb
        r[ro+8] = ((s >> 64) % Limb) + ((p44 >> 64) % Limb) + (T6 >> 63)
    end
    return nothing
end

# r[1..2n] = a[1..n]^2; r must not alias a. Half the multiplies of
# mul_basecase!: build the off-diagonal triangle sum_{i<j} a_i*a_j once with
# paired mul_2!/addmul_2! rows (rows j,j+1 over a[j+2..n] share one pass; the
# split-off a[j]*a[j+1] term is added separately), then one fused pass doubles
# the triangle and adds the diagonal squares a_i^2. n <= 4 dispatches to the
# straight-line sqr_2!/sqr_3!/sqr_4! (1.7-2.8x faster at those sizes).
function sqr_basecase!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int)
    @inbounds begin
        if n <= 2
            if n == 1
                p = widemul(a[ao+1], a[ao+1])
                r[ro+1] = p % Limb
                r[ro+2] = (p >> 64) % Limb
            else
                sqr_2!(r, ro, a, ao)
            end
            return nothing
        elseif n == 3
            return sqr_3!(r, ro, a, ao)
        elseif n == 4
            return sqr_4!(r, ro, a, ao)
        end
        # Off-diagonal triangle into r[2..2n-1]: a[j]*a[j+k] lands at limb
        # index 2j+k-1, so rows j,j+1 over the shared range a[j+2..n] form one
        # (a[j], a[j+1]) two-row pass based at r[2j+1].
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
