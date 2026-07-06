# Bit-shift kernels (mpn layer).

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
