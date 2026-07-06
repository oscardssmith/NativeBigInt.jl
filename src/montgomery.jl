# Montgomery reduction, used by powermod_limbs for odd moduli.

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
        copyto!(r, ro + 1, t, to + k + 1, k)
    end
    return nothing
end
