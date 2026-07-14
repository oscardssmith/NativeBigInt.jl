# Multi-limb algorithms built on the kernels: Karatsuba sqrt, powermod, and
# radix conversion. (Multiplication lives in mul.jl, division in div.jl,
# Montgomery reduction in montgomery.jl, the Lehmer gcds in gcd.jl.)

# Karatsuba square root (Zimmermann, INRIA RR-3805): s[1..h] = isqrt(a[1..n]),
# h = (n+1)>>1. Requires n even (or n <= 2) and a[ao+n] >= 2^62 (caller
# normalizes by an even bit shift plus a zero low limb for odd lengths); an
# odd-length high part would leave S' half-normalized, 2S' < β^hh, and the
# quotient step unbounded.
# On return a[ao+1..ao+h] holds the low limbs of the remainder a - s^2; the
# return value is its top limb (0 or 1). scratch needs sqrt_scratch_len(h)
# limbs at sco; recursion levels share it (a level touches it only after its
# child returns).
# With needrem = false only the root is guaranteed (a is still destroyed and
# the return value is meaningless): above SQRT_DIVAPPR_THRESHOLD the top-level
# division runs as divappr! (no remainder computed) and a guard-limb
# certificate settles sign(R), R = A - S², outright — U is reconstructed with
# one mul only in the ambiguous band (~2^-57 of inputs, plus perfect squares).
# Below the threshold the exact division's remainder feeds cheap positivity
# checks, and the final remainder phase — a quarter-size Q² square plus an
# h-limb subtract — runs only when those can't fire. The recursion always
# needs the child's remainder, so only the top level skips.
# num + quotient slot + division scratch; the divisor is read from the root
# buffer and the remainder/Q² overlay the division scratch, so ≈ 3.5h + 8.
sqrt_scratch_len(h::Int) = 4h - (h >> 1) + 8

# Build the level's division numerator at scratch[num+1..]: a ×β guard limb,
# then N = (c1, R', A1) halved in place — (Q, U) = divrem(N, 2S') is run as
# ⌊(N>>1)/S'⌋ with U = 2·(N>>1 mod S') + ε, so the divisor is the normalized
# root itself (no 2S' buffer, whose carry limb would force a 63-bit
# renormalization of both operands at every level). The guard limb holds
# ε·2^63 after the halving, so the buffer is N·β/2 exactly and the divappr
# guard quotient keeps its meaning. Returns (numlen, ε); numlen ≤ h after
# the halving strips c1's bit into the limb below.
@inline function sqrt_build_num!(scratch::Memory{Limb}, num::Int, a::Memory{Limb},
                                 ao::Int, lq::Int, h::Int, hh::Int, c1::Int)
    @inbounds scratch[num+1] = zero(Limb)
    copyto!(scratch, num + 2, a, ao + lq + 1, h)
    numlen = h
    if c1 != 0
        numlen = h + 1
        @inbounds scratch[num+1+numlen] = c1 % Limb
    end
    ε = @inbounds scratch[num+2] & one(Limb)
    rshift!(scratch, num, scratch, num, numlen + 1, 1)
    @inbounds while numlen > hh && scratch[num+1+numlen] == 0
        numlen -= 1
    end
    return numlen, ε
end

function sqrtrem!(s::Memory{Limb}, so::Int, a::Memory{Limb}, ao::Int, n::Int,
                  scratch::Memory{Limb}, sco::Int=0, needrem::Bool=true)
    if n <= 2
        v = n == 1 ? UInt128(@inbounds a[ao+1]) :
            (UInt128(@inbounds a[ao+2]) << 64) | (@inbounds a[ao+1])
        rt = isqrt(v)
        rm = v - rt * rt
        @inbounds s[so+1] = rt % Limb
        @inbounds a[ao+1] = rm % Limb
        return Int((rm >> 64) % Limb)
    end
    h = (n + 1) >> 1
    lq = h >> 1       # low-root limbs (the quotient Q)
    hh = h - lq       # high-root limbs (S')
    nh = n - 2lq      # limbs of the high part
    # a = Ahi*β^2lq + A1*β^lq + A0; recurse: Ahi = S'^2 + R', R' <= 2S'.
    c1 = sqrtrem!(s, so + lq, a, ao + 2lq, nh, scratch, sco)
    num = sco                    # h+2 limbs: ×β guard, then (c1, R', A1)/2
    qq = sco + h + 2             # lq+3 limbs: quotient Q (divappr: ĝ below Q)
    dv = sco + h + lq + 5        # ≤ h+3+2hh limbs: divrem!/divappr! scratch
                                 # (the divisor S' is normalized, so its copy
                                 # slot inside the contract is never touched);
                                 # doubles as U's slot — divrem! leaves the
                                 # remainder at its scratch base, so passing
                                 # (r, ro) = (scratch, dv) makes the final
                                 # copy a no-op — and as Q²/Q·S' space once
                                 # the division is over
    numlen, ε = sqrt_build_num!(scratch, num, a, ao, lq, h, hh, c1)
    rhi = 0
    appr = 0
    if !needrem && lq >= SQRT_DIVAPPR_THRESHOLD
        appr, rhi = sqrt_appr_top!(s, so, a, ao, h, lq, hh, ε, c1,
                                   scratch, num, numlen, qq, dv)
        appr == 1 && return 0    # certificate settled the root
        # the engines destroyed the numerator; rebuild it for the exact retry
        appr == 0 && (numlen, ε = sqrt_build_num!(scratch, num, a, ao, lq, h, hh, c1))
    end
    if appr == 0
        # Pre-zero the quotient slot so untouched high limbs read as zero.
        fill!(view(scratch, qq+1:qq+lq+2), zero(Limb))
        qlen = numlen - hh + 1
        if hh >= 2
            # The numerator buffer is disposable and S' normalized, so run
            # the division engines on it directly — no defensive copy, no
            # entry dispatch, remainder left in place. The appended zero top
            # limb pins the extra quotient bit to zero (Q < β^qlen).
            nn = numlen + 1
            @inbounds scratch[num+1+nn] = zero(Limb)
            v = @inbounds invert_pi1(s[so+h], s[so+h-1])
            if hh >= DC_DIV_THRESHOLD && nn - hh >= DC_DIV_THRESHOLD
                divrem_dc!(scratch, qq, scratch, num + 1, nn, s, so + lq, hh, v,
                           DC_DIV_THRESHOLD, scratch, dv)
            else
                divrem_bc!(scratch, qq, scratch, num + 1, nn, s, so + lq, hh, v)
            end
            uc = lshift!(a, ao + lq, scratch, num + 1, hh, 1)  # U = 2U₁+ε < 2S'
        else
            divrem!(scratch, qq, scratch, dv, scratch, num + 1, numlen,
                    s, so + lq, hh, scratch, dv)
            uc = lshift!(a, ao + lq, scratch, dv, hh, 1)
        end
        @inbounds a[ao+lq+1] |= ε
        rhi = Int(uc)
        # Q <= β^lq; if Q = β^lq exactly, clamp to β^lq - 1 and put 2S' back in U
        # (still >= the true root; the correction loop repairs the remainder).
        toobig = any(!iszero, view(scratch, qq+lq+1:qq+qlen))
        if toobig
            fill!(view(scratch, qq+1:qq+lq), typemax(Limb))
            rhi += Int(add_n!(a, ao + lq, a, ao + lq, s, so + lq, hh))
            rhi += Int(add_n!(a, ao + lq, a, ao + lq, s, so + lq, hh))
        end
        copyto!(s, so + 1, scratch, qq + 1, lq)
    end
    if !needrem
        # S never undershoots (the correction loop below only decrements),
        # so S is exact iff R = V - Q² ≥ 0 with V = rhi·β^h + U·β^lq + A0.
        # Provably nonnegative when V ≥ β^2lq > Q²: rhi ≠ 0, U ≥ β^lq, or
        # V's top two limbs clear (q1+1)² ≥ Q²/β^(2lq-2). Otherwise fall
        # through and settle it exactly.
        rhi != 0 && return 0
        normlen(a, ao + 2lq, h - 2lq) != 0 && return 0
        q1 = @inbounds s[so+lq]
        if q1 != typemax(Limb)
            v = (UInt128(@inbounds a[ao+2lq]) << 64) | (@inbounds a[ao+2lq-1])
            widemul(q1 + one(Limb), q1 + one(Limb)) <= v && return 0
        end
    end
    # R = U*β^lq + A0 - Q^2, tracked as (rhi, a[ao+1..ao+h]) with rhi signed
    sqr!(scratch, dv, s, so, lq)
    rhi -= Int(sub!(a, ao, a, ao, h, scratch, dv, 2lq))
    while rhi < 0
        # (S+1)^2 overshoots: R += 2S - 1, S -= 1
        tc = lshift!(scratch, num, s, so, h, 1)
        sub_1!(scratch, num, scratch, num, h, one(Limb))
        c = add_n!(a, ao, a, ao, scratch, num, h)
        rhi += Int(c) + Int(tc)
        sub_1!(s, so, s, so, h, one(Limb))
    end
    return rhi
end

# Root-only square root: sqrtrem! with the top-level remainder phase elided;
# same contract, but a's contents on return are unspecified.
function sqrt!(s::Memory{Limb}, so::Int, a::Memory{Limb}, ao::Int, n::Int,
               scratch::Memory{Limb}, sco::Int=0)
    sqrtrem!(s, so, a, ao, n, scratch, sco, false)
    return nothing
end

# Root-only top level switches from exact divrem! (whose remainder doubles as
# the nonnegativity certificate) to divappr! + guard-limb certificate once the
# quotient is long enough for the skipped remainder work to beat the rare
# fallback mul (bench/bench_sqrt_thr.jl: flat 4-48, gains from lq ≈ 16 up —
# 4k bits — and no measurable win below; the division there is schoolbook,
# where divappr_bc!'s triangle is the whole saving).
const SQRT_DIVAPPR_THRESHOLD = 16

# a·2^δ ≥ b, exactly, for UInt128 mantissas with a signed binary exponent gap.
@inline function ge_scaled(a::UInt128, b::UInt128, δ::Int)
    if δ >= 0
        δ >= 128 && return a != 0 || b == 0
        mask = (UInt128(1) << δ) - 1
        return a >= (b >> δ) + ((b & mask) != 0 ? UInt128(1) : UInt128(0))
    end
    sδ = -δ
    sδ >= 128 && return b == 0
    b > (typemax(UInt128) >> sδ) && return false
    return a >= (b << sδ)
end

# Certificate for the divappr sqrt path: decide sign(R), R = U·β^lq + A0 - Q²,
# without U. With Q exact and g the guard limb, floor(U·β/D) ∈ [g-E, g], so
# U ∈ [(g-E)·D/β, (g+1)·D/β); D = 2S' and Q are bracketed by their top 64
# bits (db·2^ed ≤ D < (db+1)·2^ed, likewise qb/eq) — S' is top-bit
# normalized, so db is its top limb exactly and ed carries the doubling.
# Returns 1 for R ≥ 0 certain (S is the root), -1 for R < 0 certain (root is
# S-1: R > -β^2lq ≥ -(2S-1) since 2S ≥ β^hh·β^lq, so a single decrement
# always lands), 0 for the ambiguous band around R = 0 (~2^-57 of inputs,
# plus perfect squares).
function sqrt_root_cert(s::Memory{Limb}, so::Int, lq::Int, hh::Int, g::Limb)
    E = DIVAPPR_ERR
    db = @inbounds s[so+lq+hh]
    ed = 64 * (hh - 1) + 1
    i = lq
    @inbounds while i > 0 && s[so+i] == zero(Limb)
        i -= 1
    end
    i == 0 && return 1                    # Q = 0: R = U·β^lq + A0 ≥ 0
    q1 = @inbounds s[so+i]
    lzq = leading_zeros(q1)
    qb = lzq == 0 ? q1 :
         (q1 << lzq) | (i > 1 ? (@inbounds(s[so+i-1]) >> (64 - lzq)) : zero(Limb))
    eq = 64 * (i - 1) - lzq
    eA = ed - 64 + 64lq
    # R ≥ 0 ⟸ (g-E)·db·2^eA ≥ (qb+1)²·2^2eq > Q² (A0 ≥ 0 only helps)
    B, eB = qb == typemax(Limb) ? (UInt128(1), 2eq + 128) :
                                  (widemul(qb + one(Limb), qb + one(Limb)), 2eq)
    ge_scaled(widemul(g - E, db), B, eA - eB) && return 1
    # R < 0 ⟸ (g+1)·(db+1)·2^eA + β^lq ≤ qb²·2^2eq ≤ Q²; the A0 < β^lq slack
    # folds into one LHS ulp since β^lq ≤ 2^eA (dl ≥ 3 here). The two typemax
    # corners would overflow the mantissa product; punt them to the fallback.
    if g != typemax(Limb) && db != typemax(Limb)
        A2 = widemul(g + one(Limb), db + one(Limb))
        ge_scaled(widemul(qb, qb), A2 + UInt128(1), 2eq - eA) && return -1
    end
    return 0
end

# Top-level root-only quotient via divappr! (spec §3): N·β ÷ 2S' with one
# guard limb ĝ, run as (N·β/2) ÷ S' on the pre-halved num buffer (guard limb
# ε·2^63) so the divisor is the normalized root itself. Returns (code, rhi):
# code 1 — certificate settled the root in s (possibly decremented); code 0 —
# retry exactly (guard wrap ĝ < E leaves the integer part uncertain, or the
# quotient overflowed β^lq; ~2^-59); code 2 — Q is exact and in s but
# sign(R) is ambiguous: U = 2·(N₁ - Q·S') + ε has been reconstructed with
# one hh×lq mul into a[ao+lq+1..] (top limb in rhi) and the caller finishes
# with the shared exact remainder phase.
function sqrt_appr_top!(s::Memory{Limb}, so::Int, a::Memory{Limb}, ao::Int,
                        h::Int, lq::Int, hh::Int, ε::Limb, c1::Int,
                        scratch::Memory{Limb}, num::Int, numlen::Int,
                        qq::Int, dv::Int)
    nA = numlen + 1
    @inbounds while nA > hh && scratch[num+nA] == zero(Limb)
        nA -= 1
    end
    # N too short for a meaningful guard (needs R' = 0 and a zero A1 top —
    # e.g. a perfect-square upper half): let the exact path handle it
    @inbounds scratch[num+nA] == zero(Limb) && return 0, 0
    fill!(view(scratch, qq+1:qq+lq+3), zero(Limb))
    # run the divappr engines on the numerator buffer directly (it is
    # disposable — the rare paths that still need N rebuild it from a); the
    # appended zero top limb pins the carry-out to zero
    nn = nA + 1
    @inbounds scratch[num+nn] = zero(Limb)
    v = @inbounds invert_pi1(s[so+h], s[so+h-1])
    if hh >= DC_DIV_THRESHOLD && nn - hh >= DC_DIV_THRESHOLD
        divappr_dc!(scratch, qq, scratch, num, nn, s, so + lq, hh, v,
                    DC_DIV_THRESHOLD, scratch, dv)
    else
        divappr_bc!(scratch, qq, scratch, num, nn, s, so + lq, hh, v)
    end
    qext = nA - hh + 1               # written limbs: guard, then integer part
    @inbounds for i in qq+lq+2:qq+qext
        scratch[i] != zero(Limb) && return 0, 0
    end
    g = @inbounds scratch[qq+1]
    g < DIVAPPR_ERR && return 0, 0
    copyto!(s, so + 1, scratch, qq + 2, lq)
    verdict = sqrt_root_cert(s, so, lq, hh, g)
    if verdict != 0
        verdict < 0 && sub_1!(s, so, s, so, h, one(Limb))
        return 1, 0
    end
    numlen, ε = sqrt_build_num!(scratch, num, a, ao, lq, h, hh, c1)
    mul!(scratch, dv, s, so + lq, hh, scratch, qq + 1, lq)   # Q·S' ≤ N₁
    plen = hh + lq
    @inbounds while plen > numlen && scratch[dv+plen] == zero(Limb)
        plen -= 1
    end
    sub!(scratch, num + 1, scratch, num + 1, numlen, scratch, dv, plen)
    uc = lshift!(a, ao + lq, scratch, num + 1, hh, 1)        # U = 2U₁ + ε
    @inbounds a[ao+lq+1] |= ε
    return 2, Int(uc)
end

# O(1) exponent bit access; NBig overloads live in nbig.jl.
@inline expbit(e::Integer, i::Int) = (e >>> i) % Bool
@inline expbits(e::Integer) = Base.top_set_bit(e)

# b^e mod m on magnitudes: m has k limbs (m[k] ≠ 0, m > 1), 0 < b < m
# (lb limbs), e > 0 any Integer supporting expbit/expbits. Returns a k-limb
# Memory (unnormalized). Sliding-window exponentiation; for odd m the values
# live in Montgomery form with redc! after each mul/sqr, for even m each
# product is reduced with divrem! instead. Above the per-parity Barrett
# thresholds both give way to plain-domain Barrett reduction, which rides
# mul!'s subquadratic engines (the flag is overridable for threshold
# benchmarks/tests).
function powermod_limbs(b::Memory{Limb}, lb::Int, e::Integer,
                        m::Memory{Limb}, k::Int,
                        barrett::Bool = k >= (isodd(@inbounds m[1]) ?
                                              BARRETT_THRESHOLD : BARRETT_EVEN_THRESHOLD))
    odd = !barrett && isodd(@inbounds m[1])
    ninv = odd ? mont_ninv(@inbounds m[1]) : zero(Limb)
    mu, lmu, bscratch = barrett ? barrett_setup(m, 0, k) :
                                  (EMPTY_LIMBS, 0, EMPTY_LIMBS)
    nbits = expbits(e)
    w = nbits <= 8 ? 1 : nbits <= 24 ? 2 : nbits <= 80 ? 3 : nbits <= 240 ? 4 : 5
    tsize = 1 << (w - 1)   # table of odd powers b^1, b^3, …, b^(2^w - 1)
    acc = Memory{Limb}(undef, k)
    prod = Memory{Limb}(undef, 2k + 1)
    qbuf = Memory{Limb}(undef, k + 2)
    table = Memory{Limb}(undef, tsize * k)

    # dst[1..k] = x*y (or x^2) brought back into the working domain
    function mulred!(dst::Memory{Limb}, dsto::Int, x::Memory{Limb}, xo::Int,
                     y::Memory{Limb}, yo::Int)
        if x === y && xo == yo
            sqr!(prod, 0, x, xo, k)
        else
            mul!(prod, 0, x, xo, k, y, yo, k)
        end
        @inbounds prod[2k+1] = 0
        if barrett
            barrett_reduce!(dst, dsto, prod, 0, m, 0, k, mu, lmu, bscratch, 0)
        elseif odd
            redc!(dst, dsto, prod, 0, m, 0, k, ninv)
        else
            lt = normlen(prod, 0, 2k)
            if lt < k || cmp_limbs(prod, 0, lt, m, 0, k) < 0
                copyto!(dst, dsto + 1, prod, 1, lt)
                fill!(view(dst, dsto+lt+1:dsto+k), zero(Limb))
            else
                divrem!(qbuf, 0, dst, dsto, prod, 0, lt, m, 0, k)
            end
        end
    end

    # table slot 0 = base: b·β^k mod m in Montgomery form, else b zero-padded
    if odd
        fill!(view(prod, 1:k), zero(Limb))
        copyto!(prod, k + 1, b, 1, lb)
        divrem!(qbuf, 0, table, 0, prod, 0, k + lb, m, 0, k)
    else
        copyto!(table, 1, b, 1, lb)
        fill!(view(table, lb+1:k), zero(Limb))
    end
    if tsize > 1
        bsq = Memory{Limb}(undef, k)
        mulred!(bsq, 0, table, 0, table, 0)
        for j in 1:tsize-1
            mulred!(table, j * k, table, (j - 1) * k, bsq, 0)
        end
    end

    first = true
    i = nbits - 1
    while i >= 0
        if !expbit(e, i)
            first || mulred!(acc, 0, acc, 0, acc, 0)
            i -= 1
        else
            l = min(w, i + 1)
            while !expbit(e, i - l + 1)   # window must end on a set bit
                l -= 1
            end
            val = 0
            for j in 0:l-1
                val = (val << 1) | Int(expbit(e, i - j))
            end
            if first
                copyto!(acc, 1, table, (val >> 1) * k + 1, k)
                first = false
            else
                for _ in 1:l
                    mulred!(acc, 0, acc, 0, acc, 0)
                end
                mulred!(acc, 0, acc, 0, table, (val >> 1) * k)
            end
            i -= l
        end
    end
    if odd   # leave Montgomery form: one reduction of the bare value
        copyto!(prod, 1, acc, 1, k)
        fill!(view(prod, k+1:2k+1), zero(Limb))
        redc!(acc, 0, prod, 0, m, 0, k, ninv)
    end
    return acc
end

# Largest power of base that fits in a limb: (base^k, k).
function big_base(base::Int)
    bb = Limb(base)
    k = 1
    while true
        nb, ovf = Base.mul_with_overflow(bb, Limb(base))
        ovf && return bb, k
        bb, k = nb, k + 1
    end
end

# Little-endian base^k digit chunks of the n-limb magnitude in a (destroyed:
# a is divided down in place). Repeated divrem_1! — O(n²), fine for v1;
# divide-and-conquer is a post-v1 extension for very large n. A power-of-two
# base^k = 2^c needs no division: each chunk is a fixed-width c-bit window (O(n)).
function radix_chunks!(a::Memory{Limb}, n::Int, bb::Limb)
    ispow2(bb) && return radix_chunks_pow2(a, n, trailing_zeros(bb))
    chunks = Limb[]
    while n > 0
        push!(chunks, divrem_1!(a, 0, a, 0, n, bb))
        n = normlen(a, 0, n)
    end
    return chunks
end

# Little-endian c-bit windows of the n-limb magnitude (a[n] ≠ 0, c ≤ 63):
# identical output to repeatedly dividing by 2^c, but O(n). Does not touch a.
function radix_chunks_pow2(a::Memory{Limb}, n::Int, c::Int)
    n == 0 && return Limb[]
    bitlen = magnitude_bits(a, 0, n)
    nchunks = cld(bitlen, c)
    chunks = Vector{Limb}(undef, nchunks)
    mask = (one(Limb) << c) - one(Limb)
    @inbounds for t in 0:nchunks-1
        b = t * c
        q = b >> 6
        off = b & 63
        w = a[q+1] >> off
        if off != 0 && q+2 <= n   # window straddles a limb boundary
            w |= a[q+2] << (64 - off)
        end
        chunks[t+1] = w & mask
    end
    return chunks
end

