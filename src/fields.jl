# ------------------------------
# 3D Field Structure
# ------------------------------
struct Field3D
    C::CuArray{Float32,3}      # Consciousness
    Q::CuArray{Float32,3}      # Dual field / matter
    I::CuArray{SVector{3,Float32},3}  # Interaction / energy flow (vector field)
    ρ::CuArray{Float32,3}      # Charge / density
    τ::CuArray{Float32,3}      # Experienced time
    extras::Dict{Symbol,CuArray{Float32,3}} # domain specific extra fields
    Δt::Float32
    Δx::Float32
    nx::Int
    ny::Int
    nz::Int
end

# ------------------------------
# Initialization
# ------------------------------
function init_field3D(nx::Int, ny::Int, nz::Int;
                      Δt::Float32=0.01f0, Δx::Float32=1.0f0,
                      extras::Dict{Symbol, CuArray{Float32,3}}=Dict{Symbol,CuArray{Float32,3}}())
    # CPU arrays
    C_cpu = zeros(Float32, nx, ny, nz)
    Q_cpu = zeros(Float32, nx, ny, nz)
    I_cpu = fill(SVector{3,Float32}(0f0,0f0,0f0), nx, ny, nz)
    ρ_cpu = ones(Float32, nx, ny, nz)
    τ_cpu = zeros(Float32, nx, ny, nz)

    # Pulse in the center
    C_cpu[nx ÷ 2, ny ÷ 2, nz ÷ 2] = 1f0

    # Upload to GPU
    C = CuArray(C_cpu)
    Q = CuArray(Q_cpu)
    I = CuArray(I_cpu)
    ρ = CuArray(ρ_cpu)
    τ = CuArray(τ_cpu)
    
    # Upload extras
    extras_gpu = Dict{Symbol, CuArray{Float32,3}}()
    for (k,v) in extras
        extras_gpu[k] = CuArray(v)
    end

    return Field3D(C, Q, I, ρ, τ, extras_gpu, Δt, Δx, nx, ny, nz)
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

# below here is adjusted code
# added later, might not be in the right file.
# this combines with below leapfrog3D!() function
function kernel_leapfrog!(
    C::CuDeviceArray{Float32,3},
    Q::CuDeviceArray{Float32,3},
    I::CuDeviceArray{Float32,4},  # vector field: (3,nx,ny,nz)
    ρ::CuDeviceArray{Float32,3},
    τ::CuDeviceArray{Float32,3},
    Δt::Float32
)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    j = (blockIdx().y-1)*blockDim().y + threadIdx().y
    k = (blockIdx().z-1)*blockDim().z + threadIdx().z

    nx, ny, nz = size(C)

    if i <= nx && j <= ny && k <= nz
        @inbounds begin
            eps = Float32(1e-6)

            # Update scalar fields
            C[i,j,k] += Δt * (Q[i,j,k] - sum(I[:,i,j,k])) / (ρ[i,j,k] + eps)
            Q[i,j,k] += Δt * (C[i,j,k] - τ[i,j,k])

            # Update vector field component-wise
            for d in 1:3
                I[d,i,j,k] = 0.99f0 * I[d,i,j,k] + 0.01f0 * C[i,j,k]
            end
        end
    end
    return
end

function kernel_leapfrog_gpu!(
    C::CuDeviceArray{Float32,3},
    Q::CuDeviceArray{Float32,3},
    I::CuDeviceArray{Float32,4},
    ρ::CuDeviceArray{Float32,3},
    τ::CuDeviceArray{Float32,3},
    Δt::Float32
)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    j = (blockIdx().y-1)*blockDim().y + threadIdx().y
    k = (blockIdx().z-1)*blockDim().z + threadIdx().z

    nx, ny, nz = size(C)

    if i <= nx && j <= ny && k <= nz
        @inbounds begin
            eps = 1e-6f0

            # sum(I[:,i,j,k]) → handmatig uitgeschreven omdat sum niet GPU-compatibel is
            I_sum = I[1,i,j,k] + I[2,i,j,k] + I[3,i,j,k]

            C[i,j,k] += Δt * (Q[i,j,k] - I_sum) / (ρ[i,j,k] + eps)
            Q[i,j,k] += Δt * (C[i,j,k] - τ[i,j,k])
            I[1,i,j,k] = 0.99f0 * I[1,i,j,k] + 0.01f0 * C[i,j,k]
            I[2,i,j,k] = 0.99f0 * I[2,i,j,k] + 0.01f0 * C[i,j,k]
            I[3,i,j,k] = 0.99f0 * I[3,i,j,k] + 0.01f0 * C[i,j,k]
        end
    end
    return
end

# this combines with above kernel_leapfrog! function
function leapfrog3D!(F::Field3D)
    threads = (8,8,8)
    blocks = (
        cld(F.nx, threads[1]),
        cld(F.ny, threads[2]),
        cld(F.nz, threads[3])
    )
    @cuda threads=threads blocks=blocks kernel_leapfrog_gpu!(
        F.C, F.Q, F.I, F.ρ, F.τ, F.Δt
    )
    return nothing
end
