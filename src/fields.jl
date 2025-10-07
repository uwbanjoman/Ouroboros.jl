# ------------------------------
# 3D Field Structure
# ------------------------------
struct Field3D
    C::CuArray{Float32,3}      # Consciousness
    Q::CuArray{Float32,3}      # Dual field / matter
    I::CuArray{SVector{3,Float32},3}  # Interaction / energy flow (vector field)
    ρ::CuArray{Float32,3}      # Charge / density
    τ::CuArray{Float32,3}      # Experienced time
    Δt::Float32
    Δx::Float32
    nx::Int
    ny::Int
    nz::Int
end

# ------------------------------
# Initialization
# ------------------------------
function init_field3D(nx::Int, ny::Int, nz::Int; Δt=0.01f0, Δx=1.0f0)
    C_cpu = zeros(Float32, nx, ny, nz)
    Q_cpu = zeros(Float32, nx, ny, nz)
    I_cpu = zeros(Float32, nx, ny, nz)
    ρ_cpu = ones(Float32, nx, ny, nz)
    τ_cpu = zeros(Float32, nx, ny, nz)

    # Localized pulse in the center
    C_cpu[nx ÷ 2, ny ÷ 2, nz ÷ 2] = 1f0

    # Upload to GPU
    C, Q, I, ρ, τ = CuArray.(Ref.((C_cpu, Q_cpu, I_cpu, ρ_cpu, τ_cpu)))[]

    return Field3D(C, Q, I, ρ, τ, Δt, Δx, nx, ny, nz)
end

# ------------------------------
# Leap-frog Kernel Update
# ------------------------------
function kernel_update_field!(F::Field3D)
    @cuda threads=256 blocks=ceil(Int,F.nx*F.ny*F.nz/256) update_kernel!(
        F.C, F.Q, F.I, F.ρ, F.τ, F.Δt, F.Δx, F.nx, F.ny, F.nz)
end

@cuda function update_kernel!(
        C, Q, I, ρ, τ, Δt, Δx, nx, ny, nz)

    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = nx*ny*nz
    if idx > total return end

    # Compute 3D indices
    k = fld(idx-1, nx*ny) + 1
    j = fld(idx-1 - (k-1)*nx*ny, nx) + 1
    i = idx - (k-1)*nx*ny - (j-1)*nx

    # 3D neighbors with periodic boundary
    ip = i < nx ? i+1 : 1
    im = i > 1 ? i-1 : nx
    jp = j < ny ? j+1 : 1
    jm = j > 1 ? j-1 : ny
    kp = k < nz ? k+1 : 1
    km = k > 1 ? k-1 : nz

    # Simple Laplacian / diffusion approximation for C
    lap_C = (C[ip,j,k] + C[im,j,k] + C[i,jp,k] + C[i,jm,k] +
             C[i,j,kp] + C[i,j,km] - 6f0*C[i,j,k]) / (Δx^2)

    # Update rules (conceptual)
    C[i,j,k] += Δt * (lap_C - Q[i,j,k] + ρ[i,j,k])
    Q[i,j,k] += Δt * (C[i,j,k] - Q[i,j,k]*0.1f0)
    ρ[i,j,k] += Δt * (-lap_C*0.01f0)
    τ[i,j,k] += Δt

    # Interaction field update (gradient approximation)
    I[i,j,k] = SVector(
        (C[ip,j,k] - C[im,j,k])/(2f0*Δx),
        (C[i,jp,k] - C[i,jm,k])/(2f0*Δx),
        (C[i,j,kp] - C[i,j,km])/(2f0*Δx)
    )
end
