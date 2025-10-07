module Consciousness3D

using CUDA

export init_consciousness!, set_consciousness_pulse!, update_consciousness!, evolve_consciousness!

# Params struct
struct PhiParams
    dt::Float32
    dx::Float32
    c::Float32           # wave speed
    mu::Float32          # mass term (for V = -1/2 mu^2 phi^2 + 1/4 lambda phi^4 use negative mu^2 sign appropriately)
    lambda::Float32      # self-interaction
    gamma::Float32       # damping
    phi_clip::Float32    # safety clamp on |phi|
    threads::NTuple{3,Int32}
end

function PhiParams(; dt=0.002f0, dx=1.0f0, c=1.0f0, mu=0.5f0, lambda=0.01f0, gamma=0.01f0, phi_clip=1e3f0, threads=(8,8,4))
    PhiParams(Float32(dt), Float32(dx), Float32(c), Float32(mu), Float32(lambda), Float32(gamma), Float32(phi_clip), ntuple(Int32,3) do i threads[i] end)
end

# Core storage
struct PhiLayer3D
    φ::CuArray{Float32,3}
    π::CuArray{Float32,3}
    φ_new::CuArray{Float32,3}
    π_new::CuArray{Float32,3}
    params::PhiParams
    Nx::Int32; Ny::Int32; Nz::Int32
end

function init_consciousness!(Nx::Int, Ny::Int, Nz::Int; p=PhiParams())
    φ = CUDA.zeros(Float32, Nx, Ny, Nz)
    π = CUDA.zeros(Float32, Nx, Ny, Nz)
    φ_new = similar(φ); π_new = similar(π)
    return PhiLayer3D(φ, π, φ_new, π_new, p, Int32(Nx), Int32(Ny), Int32(Nz))
end

# small kernel to set a Gaussian pulse at center (or arbitrary pos)
function kernel_set_gauss_pulse!(φ, cx::Int32, cy::Int32, cz::Int32, A::Float32, σ2::Float32)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    j = (blockIdx().y-1)*blockDim().y + threadIdx().y
    k = (blockIdx().z-1)*blockDim().z + threadIdx().z
    if i < 1 || j < 1 || k < 1 || i > size(φ,1) || j > size(φ,2) || k > size(φ,3)
        return
    end
    dx = Float32(i - cx); dy = Float32(j - cy); dz = Float32(k - cz)
    r2 = dx*dx + dy*dy + dz*dz
    φ[i,j,k] = A * exp(-0.5f0 * r2 / σ2)
    return
end

function set_consciousness_pulse!(layer::PhiLayer3D, ix::Int, iy::Int, iz::Int; A=10f0, sigma=2.0f0)
    threads = layer.params.threads
    blocks = (cld(layer.Nx,threads[1]), cld(layer.Ny,threads[2]), cld(layer.Nz,threads[3]))
    @cuda threads=Tuple(threads) blocks=blocks kernel_set_gauss_pulse!(layer.φ, Int32(ix), Int32(iy), Int32(iz), Float32(A), Float32(sigma*sigma))
    return nothing
end

# Main update kernel: integrates phi and pi one dt step
function kernel_update_phi_3d!(
    φ, π, φ_new, π_new,
    Nx::Int32, Ny::Int32, Nz::Int32,
    dt::Float32, inv_dx2::Float32,
    c::Float32, mu::Float32, lambda::Float32, gamma::Float32, phi_clip::Float32
)
    ix = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    iz = (blockIdx().z-1)*blockDim().z + threadIdx().z

    if ix < 1 || ix > Nx || iy < 1 || iy > Ny || iz < 1 || iz > Nz
        return
    end

    # neighbor indices with simple clamping (absorbing-like)
    function clamp(i,N)
        if i < 1 return Int32(1) elseif i > N return N else return i end
    end
    ip = clamp(ix+1,Nx); im = clamp(ix-1,Nx)
    jp = clamp(iy+1,Ny); jm = clamp(iy-1,Ny)
    kp = clamp(iz+1,Nz); km = clamp(iz-1,Nz)

    φ000 = φ[ix,iy,iz]
    # Laplacian (6-point)
    φxp = φ[ip,iy,iz]; φxm = φ[im,iy,iz]
    φyp = φ[ix,jp,iz]; φym = φ[ix,jm,iz]
    φzp = φ[ix,iy,kp]; φzm = φ[ix,iy,km]
    lapφ = (φxp + φxm + φyp + φym + φzp + φzm - 6f0*φ000) * inv_dx2

    π000 = π[ix,iy,iz]

    # potential derivative V'(φ) for V = -1/2 mu^2 φ^2 + 1/4 λ φ^4  (symmetry-breaking style)
    # V'(φ) = -mu^2 * φ + λ * φ^3
    Vp = - (mu*mu) * φ000 + lambda * (φ000*φ000*φ000)

    # RHS for π (∂t π)
    rhsπ = c*c * lapφ - Vp - gamma * π000

    # explicit step
    πn = π000 + dt * rhsπ
    φn = φ000 + dt * πn

    # clamp phi safe
    if φn > phi_clip
        φn = phi_clip
    elseif φn < -phi_clip
        φn = -phi_clip
    end

    π_new[ix,iy,iz] = πn
    φ_new[ix,iy,iz] = φn

    return
end

function update_consciousness!(layer::PhiLayer3D)
    p = layer.params
    threads = p.threads
    blocks = (cld(layer.Nx,threads[1]), cld(layer.Ny,threads[2]), cld(layer.Nz,threads[3]))
    inv_dx2 = 1f0 / (p.dx * p.dx)
    @cuda threads=Tuple(threads) blocks=blocks kernel_update_phi_3d!(
        layer.φ, layer.π, layer.φ_new, layer.π_new,
        layer.Nx, layer.Ny, layer.Nz,
        p.dt, inv_dx2,
        p.c, p.mu, p.lambda, p.gamma, p.phi_clip
    )
    # swap
    tmpφ = layer.φ; layer.φ = layer.φ_new; layer.φ_new = tmpφ
    tmpπ = layer.π; layer.π = layer.π_new; layer.π_new = tmpπ
    return nothing
end

function evolve_consciousness!(layer::PhiLayer3D, steps::Int)
    for s in 1:steps
        update_consciousness!(layer)
    end
    return nothing
end

end # module
