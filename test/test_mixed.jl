# NBig ↔ BitInteger64 fast paths: methods must dispatch to NativeBigInt (not
# Base's promotion fallback), and agree bit-for-bit with BigInt.

const BITINT64 = (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64)

adversarial(::Type{T}) where {T} = T[0, 1, typemax(T), typemax(T) - 1,
                                     (T <: Signed ? T[-1, typemin(T), typemin(T) + 1] : T[])...]

@testset "mixed dispatch specificity" begin
    for T in BITINT64
        for (f, at) in ((+, (NBig, T)), (+, (T, NBig)),
                        (-, (NBig, T)), (-, (T, NBig)),
                        (*, (NBig, T)), (*, (T, NBig)),
                        (divrem, (NBig, T)), (divrem, (T, NBig)),
                        (div, (NBig, T)), (div, (T, NBig)),
                        (rem, (NBig, T)), (rem, (T, NBig)),
                        (mod, (NBig, T)), (mod, (T, NBig)),
                        (fld, (NBig, T)), (fld, (T, NBig)),
                        (cld, (NBig, T)), (cld, (T, NBig)),
                        (==, (NBig, T)), (==, (T, NBig)),
                        (<, (NBig, T)), (<, (T, NBig)),
                        (<=, (NBig, T)), (<=, (T, NBig)),
                        (isless, (NBig, T)), (isless, (T, NBig)),
                        (cmp, (NBig, T)), (cmp, (T, NBig)),
                        (gcd, (NBig, T)), (gcd, (T, NBig)),
                        (powermod, (NBig, NBig, T)))
            @test which(f, Tuple{at...}).module === NativeBigInt
        end
    end
end

@testset "mixed differential" begin
    rng = MersenneTwister(0x316d)
    for T in BITINT64
        smalls = collect(adversarial(T))
        for _ in 1:20
            push!(smalls, rand(rng, T))
        end
        for trial in 1:60
            la = rand(rng, 0:8)
            trial % 10 == 0 && (la = rand(rng, 20:60))
            a = diff_randbig(rng, la)
            na = NBig(a)
            for b in smalls
                bb = big(b)
                @test BigInt(na + b) == a + bb
                @test BigInt(b + na) == bb + a
                @test BigInt(na - b) == a - bb
                @test BigInt(b - na) == bb - a
                @test BigInt(na * b) == a * bb
                @test BigInt(b * na) == bb * a
                @test (na == b) == (a == bb)
                @test (b == na) == (bb == a)
                @test (na < b) == (a < bb)
                @test (b < na) == (bb < a)
                @test (na <= b) == (a <= bb)
                @test (b <= na) == (bb <= a)
                @test cmp(na, b) == cmp(a, bb)
                @test cmp(b, na) == cmp(bb, a)
                @test BigInt(gcd(na, b)) == gcd(a, bb)
                @test BigInt(gcd(b, na)) == gcd(bb, a)
                if iszero(b)
                    @test_throws DivideError divrem(na, b)
                    @test_throws DivideError mod(na, b)
                else
                    q, r = divrem(na, b)
                    @test BigInt(q) == div(a, bb) && BigInt(r) == rem(a, bb)
                    @test BigInt(div(na, b)) == div(a, bb)
                    @test BigInt(rem(na, b)) == rem(a, bb)
                    @test BigInt(mod(na, b)) == mod(a, bb)
                    @test BigInt(fld(na, b)) == fld(a, bb)
                    @test BigInt(cld(na, b)) == cld(a, bb)
                end
                if iszero(a)
                    @test_throws DivideError divrem(b, na)
                    @test_throws DivideError mod(b, na)
                else
                    q, r = divrem(b, na)
                    @test BigInt(q) == div(bb, a) && BigInt(r) == rem(bb, a)
                    @test BigInt(div(b, na)) == div(bb, a)
                    @test BigInt(rem(b, na)) == rem(bb, a)
                    @test BigInt(mod(b, na)) == mod(bb, a)
                    @test BigInt(fld(b, na)) == fld(bb, a)
                    @test BigInt(cld(b, na)) == cld(bb, a)
                end
            end
        end
    end
end

@testset "mixed powermod" begin
    rng = MersenneTwister(0x93d)
    for T in (Int8, Int32, Int64, UInt8, UInt64)
        for trial in 1:40
            m = rand(rng, T)
            (iszero(m) || m == typemin(T)) && (m = T(3))
            a = diff_randbig(rng, rand(rng, 0:6))
            n = abs(diff_randbig(rng, rand(rng, 1:3)))
            e = rand(rng, 0:40)
            r = powermod(NBig(a), NBig(n), m)
            @test r isa typeof(powermod(0, 0, m))
            @test big(r) == powermod(a, n, big(m))
            @test big(powermod(NBig(a), NBig(e), m)) == powermod(a, big(e), big(m))
        end
    end
    @test_throws DivideError powermod(NBig(2), NBig(5), 0)
end
