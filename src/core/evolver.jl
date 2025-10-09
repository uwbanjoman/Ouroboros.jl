
using ..Registry

export evolve_step!

"""
    evolve_step!(F::Field3D, Δt::Float32)

Calls all registered physics modules on the field F for a single timestep.
"""
function evolve_step!(F, Δt::Float32)
    for (_, update_fn) in Registry.get_modules()
        update_fn(F, Δt)   # each module updates the field
    end
end

