using CUDA
using GLMakie
using Consciousness3D

# Grid & parameters
Nx, Ny, Nz = 128, 128, 64
params = PhiParams(dt=0.002f0, dx=1.0f0, c=1.0f0, mu=0.4f0, lambda=0.01f0, gamma=0.002f0)

println("Initializing consciousness field on GPU...")
layer = init_consciousness!(Nx, Ny, Nz; p=params)

# Set Gaussian pulse at center
cx, cy, cz = div(Nx,2), div(Ny,2), div(Nz,2)
set_consciousness_pulse!(layer, cx, cy, cz; A=10f0, sigma=3.0f0)

# Evolve
steps_total = 400
save_every = 10
frames = Vector{Matrix{Float32}}()

println("Running evolution for $steps_total steps...")
for step in 1:steps_total
    update_consciousness!(layer)
    if step % save_every == 0
        φ_host = Array(layer.φ)
        # Take a central slice in z
        push!(frames, φ_host[:,:,cz])
    end
end

# Visualization
println("Creating visualization...")
nframes = length(frames)
x = 1:Nx; y = 1:Ny
scene = Scene(resolution = (600, 600))
heatmap_obj = heatmap!(scene, x, y, frames[1], colormap=:plasma)
Colorbar(scene, heatmap_obj, label="φ (coherence)")
record(scene, "phi_expand.gif", 1:nframes; framerate=20) do i
    heatmap_obj[3] = frames[i]
end

println("✅ Animation saved as phi_expand.gif")
