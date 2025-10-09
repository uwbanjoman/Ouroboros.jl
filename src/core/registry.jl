
export register_module!, get_modules

const MODULES = Dict{Symbol, Function}()

"""
    register_module!(name::Symbol, update_fn::Function)

Registers a physics module by name.
`update_fn(F::Field3D, Δt)` will be called each evolution step.
"""
function register_module!(name::Symbol, update_fn::Function)
    MODULES[name] = update_fn
end

"""
    get_modules() -> Dict{Symbol, Function}

Returns all registered modules.
"""
get_modules() = MODULES

#===
module Registry

const registered_modules = Ref(Vector{Module}())

export register_module!, get_modules

function register_module!(mod::Module)
    push!(registered_modules[], mod)
end

get_modules() = registered_modules[]

end
#===

