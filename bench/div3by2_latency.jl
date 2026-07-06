# Serial-chain latency of div_3by2 variants: each iteration's window feeds the
# next, mimicking the divrem_bc!/divrem_2! loop-carried dependency. Local-only.
using BenchmarkTools
using NativeBigInt: Limb, DLimb, invert_pi1, div_3by2, div_fixup

# pre-rewrite version (two 128-bit subs, branchy first fixup)
@inline function div_3by2_old(u2::Limb, u1::Limb, u0::Limb, d1::Limb, d0::Limb, v::Limb)
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
        q1, r = div_fixup(q1, r, dd)
    end
    return q1, (r >> 64) % Limb, r % Limb
end

function chain(f::F, r1::Limb, r0::Limb, d1::Limb, d0::Limb, v::Limb, n::Int) where {F}
    acc = zero(Limb)
    for i in 1:n
        q, r1, r0 = f(r1, r0, acc ⊻ Limb(i), d1, d0, v)
        acc += q
    end
    return acc + r1 + r0
end

d1 = Limb(0x9234567812345678)   # normalized
d0 = Limb(0xdeadbeefcafebabe)
v = invert_pi1(d1, d0)
r1 = d1 - 1
r0 = Limb(42)
N = 1000
t_old = @belapsed chain($div_3by2_old, $r1, $r0, $d1, $d0, $v, $N)
t_new = @belapsed chain($div_3by2, $r1, $r0, $d1, $d0, $v, $N)
println("old: $(round(t_old*1e9/N, digits=2)) ns/iter")
println("new: $(round(t_new*1e9/N, digits=2)) ns/iter")
