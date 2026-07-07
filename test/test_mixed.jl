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
            @test big(powermod(NBig(a), e, m)) == powermod(a, big(e), big(m))
            @test big(powermod(NBig(a), UInt16(e), m)) == powermod(a, big(e), big(m))
            if gcd(a, big(m)) == 1
                @test big(powermod(NBig(a), -e, m)) == powermod(a, -big(e), big(m))
            end
        end
    end
    @test_throws DivideError powermod(NBig(2), NBig(5), 0)
end

@testset "native conversions" begin
    BITS = (Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32, UInt64, UInt128)
    for T in BITS
        @test which(rem, Tuple{NBig, Type{T}}).module === NativeBigInt
        @test which(T, Tuple{NBig}).module === NativeBigInt
    end
    rng = MersenneTwister(0xc0417)
    for trial in 1:400
        a = trial % 3 == 0 ? big(rand(rng, -300:300)) :
                             diff_randbig(rng, rand(rng, 0:4))
        na = NBig(a)
        for T in BITS
            @test rem(na, T) === rem(a, T)
            if typemin(T) <= a <= typemax(T)
                @test T(na) === T(a)
            else
                @test_throws InexactError T(na)
            end
        end
    end
    for T in BITS
        lo, hi = big(typemin(T)), big(typemax(T))
        @test T(NBig(lo)) === typemin(T)
        @test T(NBig(hi)) === typemax(T)
        @test_throws InexactError T(NBig(hi + 1))
        @test_throws InexactError T(NBig(lo - 1))
    end
end

@testset "invmod" begin
    for at in ((NBig, NBig), (NBig, Int64), (Int64, NBig), (NBig, UInt8))
        @test which(invmod, Tuple{at...}).module === NativeBigInt
    end

    rng = MersenneTwister(0xacc5)
    for trial in 1:200
        a = diff_randbig(rng, rand(rng, 0:8))
        m = diff_randbig(rng, rand(rng, 1:8))
        iszero(m) && (m = big(7))
        na, nm = NBig(a), NBig(m)
        if gcd(a, m) == 1
            r = invmod(na, nm)
            @test r isa NBig
            @test big(r) == invmod(a, m)
        else
            @test_throws DomainError invmod(na, nm)
        end
    end
    # mixed invmod: native modulus returns typeof(m); NBig modulus returns NBig
    for T in (Int8, Int64, UInt8, UInt64), trial in 1:60
        m = rand(rng, T)
        (iszero(m) || m == typemin(T)) && (m = T(5))
        a = diff_randbig(rng, rand(rng, 0:5))
        if gcd(a, big(m)) == 1
            r = invmod(NBig(a), m)
            @test r isa T
            @test big(r) == invmod(a, big(m))
        else
            @test_throws DomainError invmod(NBig(a), m)
        end
        b = rand(rng, T)
        mm = diff_randbig(rng, rand(rng, 1:4))
        iszero(mm) && (mm = big(9))
        if gcd(big(b), mm) == 1
            @test big(invmod(b, NBig(mm))) == invmod(big(b), mm)
        else
            @test_throws DomainError invmod(b, NBig(mm))
        end
    end
    # edge cases (Base semantics: result sign follows m; |m| == 1 -> 0)
    @test_throws DomainError invmod(NBig(3), NBig(0))
    @test_throws DomainError invmod(NBig(0), NBig(5))
    @test invmod(NBig(3), NBig(1)) == 0
    @test invmod(NBig(3), NBig(-1)) == 0
    @test big(invmod(NBig(3), NBig(-5))) == invmod(big(3), big(-5))

    # negative exponents in powermod now route through invmod
    for trial in 1:60
        m = diff_randbig(rng, rand(rng, 1:4))
        (iszero(m) || abs(m) == 1) && (m = big(10)^9 + 7)
        a = diff_randbig(rng, rand(rng, 0:5))
        gcd(a, m) == 1 || continue
        n = abs(diff_randbig(rng, rand(rng, 1:2)))
        iszero(n) && (n = big(3))
        @test big(powermod(NBig(a), NBig(-n), NBig(m))) == powermod(a, -n, m)
        nm64 = rand(rng, Int64) | 1  # odd, nonzero
        if gcd(a, big(nm64)) == 1
            @test big(powermod(NBig(a), NBig(-n), nm64)) == powermod(a, -n, big(nm64))
        end
    end
end
