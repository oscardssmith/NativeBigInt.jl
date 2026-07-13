# Multi-limb division: Knuth Algorithm D quotient/remainder (divrem!) with a
# small-quotient subtraction fast path, dispatching to a divide-and-conquer
# basecase (divrem_dc!, GMP mpn_dcpi1 style) above DC_DIV_THRESHOLD. Built on
# the division kernels (divrem_1!/divrem_2!/divrem_bc!/invert_pi1), the
# shift/add/sub kernels, and mul!/sqr! from mul.jl.

# Quotient/remainder: a (n limbs) ÷ d (m limbs, d[m] ≠ 0), n ≥ m ≥ 1.
# For m ≥ 3, a[n] must be nonzero unless n == m (the small-quotient fast
# path bounds the quotient by a[n]'s bit position).
# Writes n-m+1 quotient limbs (top may be zero) and m remainder limbs
# (unnormalized). One scratch Memory holds the shifted numerator copy (n+1
# limbs) and, for unnormalized d, the shifted divisor; a is not modified.
function divrem!(q::Memory{Limb}, qo::Int, r::Memory{Limb}, ro::Int,
                 a::Memory{Limb}, ao::Int, n::Int, d::Memory{Limb}, do_::Int, m::Int)
    if m == 1
        @inbounds r[ro+1] = divrem_1!(q, qo, a, ao, n, d[do_+1])
        return nothing
    end
    if m == 2
        r1, r0 = @inbounds divrem_2!(q, qo, a, ao, n, d[do_+2], d[do_+1])
        @inbounds r[ro+1] = r0
        @inbounds r[ro+2] = r1
        return nothing
    end
    # Small-quotient fast path: bitlength(a) - bitlength(d) ≤ 2 bounds the
    # quotient below 8 (a/d < 2^(Δ+1)), so at most 7 subtraction sweeps beat
    # the scratch-alloc + normalize + invert + basecase machinery. Δ ≤ 2 also
    # forces n ≤ m+1 with a[n] < 4, so the value fits r plus one register.
    dbits = magnitude_bits(a, ao, n) - magnitude_bits(d, do_, m)
    if dbits <= 2
        copyto!(r, ro + 1, a, ao + 1, m)
        t = n > m ? (@inbounds a[ao+n]) : zero(Limb)
        c = zero(Limb)
        while t != zero(Limb) || cmp_limbs(r, ro, m, d, do_, m) >= 0
            t -= sub_n!(r, ro, r, ro, d, do_, m)
            c += one(Limb)
        end
        @inbounds q[qo+1] = c
        n > m && (@inbounds q[qo+2] = zero(Limb))
        return nothing
    end
    l = leading_zeros(@inbounds d[do_+m])
    nn = n + 1
    scratch = Memory{Limb}(undef, nn + (l > 0 ? m : 0))
    if l == 0
        copyto!(scratch, 1, a, ao + 1, n)
        @inbounds scratch[nn] = zero(Limb)
        dv, dvo = d, do_
    else
        @inbounds scratch[nn] = lshift!(scratch, 0, a, ao, n, l)
        lshift!(scratch, nn, d, do_, m, l)
        dv, dvo = scratch, nn
    end
    v = @inbounds invert_pi1(dv[dvo+m], dv[dvo+m-1])
    # qh == 0 either way: Q < β^(nn-m)
    if m >= DC_DIV_THRESHOLD && nn - m >= DC_DIV_THRESHOLD
        divrem_dc!(q, qo, scratch, 0, nn, dv, dvo, m, v)
    else
        divrem_bc!(q, qo, scratch, 0, nn, dv, dvo, m, v)
    end
    if l == 0
        copyto!(r, ro + 1, scratch, 1, m)
    else
        rshift!(r, ro, scratch, 0, m, l)
    end
    return nothing
end

# Schoolbook → divide-and-conquer division crossover, in limbs (GMP's
# DC_DIV_QR_THRESHOLD analogue; tuned by bench/bench_dc_thr.jl). divrem!
# dispatches to divrem_dc! only when both the divisor and the quotient reach
# it — a short quotient over a long divisor costs O(qn·m) either way.
# Balanced 2m/m sweep: schoolbook wins through m = 96, ties at m = 128, and
# falls behind from m = 192 (1.1-3x GMP by m = 2048 vs dc's 0.8-1.0x); as the
# recursion cutoff, 100-110 also edges out lower values at large m.
const DC_DIV_THRESHOLD = 100

# Balanced 2n/n divide-and-conquer division step (GMP mpn_dcpi1_div_qr_n).
# u[uo+1..uo+2n] ÷ d[do_+1..do_+n], d normalized, v = invert_pi1 of its top
# two limbs. Writes q[qo+1..qo+n], leaves the n-limb remainder in
# u[uo+1..uo+n], returns the extra top quotient bit qh. scratch holds the
# n-limb cross products at so; recursion levels share it (a level touches it
# only after its child returns). thr >= 4 keeps the divrem_bc! basecase at
# m >= 2.
#
# Split n = hi + lo. Dividing the top 2·hi limbs by the top hi limbs of d
# overshoots the true high quotient block by at most 2 (the ignored d_lo only
# makes the divisor larger): subtract the cross product q_hi·d_lo — plus
# qh·d_lo at β^hi for the implicit qh row — and repair each resulting borrow
# by adding d back and decrementing the block. The same step on the remaining
# n + lo live limbs yields the low block. The pi1 inverse stays valid down
# the recursion because every divisor suffix keeps d's top two limbs.
function divrem_dc_n!(q::Memory{Limb}, qo::Int, u::Memory{Limb}, uo::Int,
                      d::Memory{Limb}, do_::Int, n::Int, v::Limb,
                      scratch::Memory{Limb}, so::Int, thr::Int)
    lo = n >> 1
    hi = n - lo
    qh = hi < thr ? divrem_bc!(q, qo + lo, u, uo + 2lo, 2hi, d, do_ + lo, hi, v) :
                    divrem_dc_n!(q, qo + lo, u, uo + 2lo, d, do_ + lo, hi, v, scratch, so, thr)
    mul!(scratch, so, q, qo + lo, hi, d, do_, lo)          # q_hi × d_lo, n limbs
    cy = sub_n!(u, uo + lo, u, uo + lo, scratch, so, n)
    if qh != zero(Limb)
        cy += sub_n!(u, uo + n, u, uo + n, d, do_, lo)
    end
    while cy != zero(Limb)
        qh -= sub_1!(q, qo + lo, q, qo + lo, hi, one(Limb))
        cy -= add_n!(u, uo + lo, u, uo + lo, d, do_, n)
    end
    ql = lo < thr ? divrem_bc!(q, qo, u, uo + hi, 2lo, d, do_ + hi, lo, v) :
                    divrem_dc_n!(q, qo, u, uo + hi, d, do_ + hi, lo, v, scratch, so, thr)
    mul!(scratch, so, d, do_, hi, q, qo, lo)               # q_lo × d_lo', hi >= lo
    cy = sub_n!(u, uo, u, uo, scratch, so, n)
    if ql != zero(Limb)
        cy += sub_n!(u, uo + lo, u, uo + lo, d, do_, hi)
    end
    while cy != zero(Limb)
        sub_1!(q, qo, q, qo, lo, one(Limb))   # borrow folds into ql's correction
        cy -= add_n!(u, uo, u, uo, d, do_, n)
    end
    return qh
end

# Leading partial quotient block: u[uo+1..uo+m+s] ÷ d (full m limbs), s <= m.
# Writes q[qo+1..qo+s], leaves the m-limb remainder in u[uo+1..uo+m], returns
# qh. s == m is the balanced step; small s stays schoolbook (O(s·m), one-off);
# otherwise divide the top 2s limbs by the top s limbs of d, then subtract the
# cross product q·d_lo with the same add-back repair (GMP mpn_dcpi1_div_qr's
# qn < dn arm).
function divrem_dc_partial!(q::Memory{Limb}, qo::Int, u::Memory{Limb}, uo::Int,
                            d::Memory{Limb}, do_::Int, m::Int, s::Int, v::Limb,
                            scratch::Memory{Limb}, so::Int, thr::Int)
    s == m && return divrem_dc_n!(q, qo, u, uo, d, do_, m, v, scratch, so, thr)
    s < thr && return divrem_bc!(q, qo, u, uo, m + s, d, do_, m, v)
    qh = divrem_dc_n!(q, qo, u, uo + (m - s), d, do_ + (m - s), s, v, scratch, so, thr)
    if s >= m - s                                          # q × d_lo, m limbs
        mul!(scratch, so, q, qo, s, d, do_, m - s)
    else
        mul!(scratch, so, d, do_, m - s, q, qo, s)
    end
    cy = sub_n!(u, uo, u, uo, scratch, so, m)
    if qh != zero(Limb)
        cy += sub_n!(u, uo + s, u, uo + s, d, do_, m - s)
    end
    while cy != zero(Limb)
        qh -= sub_1!(q, qo, q, qo, s, one(Limb))
        cy -= add_n!(u, uo, u, uo, d, do_, m)
    end
    return qh
end

# Divide-and-conquer quotient/remainder with divrem_bc!'s exact contract:
# u (nn limbs, destroyed; remainder left in u[uo+1..uo+m]) ÷ normalized
# m-limb d with v = invert_pi1 of its top two limbs; writes q[1..nn-m],
# returns the extra top quotient bit qh. Requires nn > m. The quotient is
# peeled from the top: a leading partial block of s limbs (qn reduced mod m
# into [1, m]), then full balanced 2m/m blocks — each later window tops out
# with the previous remainder (< d), so only the leading block can set qh.
function divrem_dc!(q::Memory{Limb}, qo::Int, u::Memory{Limb}, uo::Int, nn::Int,
                    d::Memory{Limb}, do_::Int, m::Int, v::Limb,
                    thr::Int=DC_DIV_THRESHOLD)
    qn = nn - m
    thr = max(thr, 4)
    scratch = Memory{Limb}(undef, m)
    s = qn <= m ? qn : qn - m * ((qn - 1) ÷ m)
    off = qn - s
    qh = divrem_dc_partial!(q, qo + off, u, uo + off, d, do_, m, s, v, scratch, 0, thr)
    while off > 0
        off -= m
        divrem_dc_n!(q, qo + off, u, uo + off, d, do_, m, v, scratch, 0, thr)
    end
    return qh
end
