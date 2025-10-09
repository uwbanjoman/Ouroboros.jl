module LeapfrogGPU

using CUDA
import ..Registry: get_modules

export leapfrog_step!

"""
    leapfrog_step!(F, dt)

Advance all registered modules by one leapfrog step of size `dt`.
Each module must define `update!(F, dt)` operating on GPU arrays.
"""
function leapfrog_step!(F, dt)
    for mod in get_modules()
        if hasproperty(mod, :update!)
            mod.update!(F, dt)
        end
    end
    return F
end

end
