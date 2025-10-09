using CUDA
include("src/core/Registry.jl")
include("src/core/Evolver.jl")
include("src/physics/Entropy.jl")

using .Registry
using .Evolver
using .Entropy

# Setup field
F = Field3D(
    C = CuArray(rand(Float32, 48,48,48)),
    Q = CuArray(zeros(Float32, 48,48,48)),
    I = CuArray(zeros(Float32, 3,48,48,48)),
    ρ = CuArray(ones(Float32, 48,48,48)),
    τ = CuArray(zeros(Float32, 48,48,48)),
    extras = Dict{Symbol, Any}(),
    Δt = 0.01f0,
    Δx = 1f0,
    nx = 48, ny = 48, nz = 48
)

# Register modules
Entropy.register!()

# Evolve
for t in 1:100
    Evolver.evolve_step!(F, F.Δt)
end

#===
using .Registry, .Entropy, .LeapfrogGPU

# Register the entropy module
register_module!(Entropy)

# Suppose F is your Field3D struct on GPU
for step in 1:100
    leapfrog_step!(F, 0.01f0)
end
===#
