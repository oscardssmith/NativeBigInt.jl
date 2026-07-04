module NativeBigInt

export NBig

const Limb = UInt64
const DLimb = UInt128

include("kernels.jl")
include("algorithms.jl")
include("nbig.jl")

end
