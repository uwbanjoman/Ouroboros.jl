module Entropy
using CUDA
using ..Registry

export register!

function entropy_update!(F, Δt)
    # simple GPU-friendly entropy evolution example
    @. F.C += Δt * abs(F.C) * (1f0 - abs(F.C))
end

function register!()
    Registry.register_module!(:entropy, entropy_update!)
end

end # module

#===
module Entropy

using CUDA

export update!

function update!(F, dt)
    @. F.C += dt * (abs(F.C) * (1f0 - abs(F.C)))
end

end
===#

#
