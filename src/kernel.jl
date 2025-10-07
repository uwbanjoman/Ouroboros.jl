using CUDA

# ----------------------------------------------
# Kernelfunctie: 3D leap-frog met ρ en τ
# ----------------------------------------------
function leapfrog3D_kernel!(C, Q, I, ρ, τ, Δt, Δx, nx, ny, nz)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    iz = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if 2 ≤ ix ≤ nx-1 && 2 ≤ iy ≤ ny-1 && 2 ≤ iz ≤ nz-1
        # Lokale effectieve tijdvertraging
        Δt_eff = Δt / (1f0 + τ[ix, iy, iz])

        # Discrete Laplace-operator
        lapC = (
            C[ix+1, iy, iz] + C[ix-1, iy, iz] +
            C[ix, iy+1, iz] + C[ix, iy-1, iz] +
            C[ix, iy, iz+1] + C[ix, iy, iz-1] -
            6f0 * C[ix, iy, iz]
        ) / (Δx^2)

        # Leap-frog updates
        Q[ix, iy, iz] += (Δt_eff / ρ[ix, iy, iz]) * lapC
        C[ix, iy, iz] += Δt_eff * Q[ix, iy, iz]
        I[ix, iy, iz] = C[ix, iy, iz] * Q[ix, iy, iz]
    end

    return
end

# ----------------------------------------------
# Host wrapper
# ----------------------------------------------
function leapfrog3D!(C, Q, I, ρ, τ; Δt=0.01f0, Δx=1.0f0)
    nx, ny, nz = size(C)
    threads = (8, 8, 8)
    blocks  = (
        cld(nx, threads[1]),
        cld(ny, threads[2]),
        cld(nz, threads[3])
    )

    @cuda threads=threads blocks=blocks leapfrog3D_kernel!(C, Q, I, ρ, τ, Δt, Δx, nx, ny, nz)
    return nothing
end

# ----------------------------------------------
# Testinit (alle velden)
# ----------------------------------------------
function init_field3D(nx, ny, nz)
    C = CUDA.rand(Float32, nx, ny, nz)
    Q = CUDA.zeros(Float32, nx, ny, nz)
    I = CUDA.zeros(Float32, nx, ny, nz)
    ρ = CUDA.fill(1.0f0, nx, ny, nz)     # uniforme dichtheid
    τ = CUDA.fill(0.0f0, nx, ny, nz)     # geen tijdvertraging
    return C, Q, I, ρ, τ
end

# ----------------------------------------------
# Demo
# ----------------------------------------------
C, Q, I, ρ, τ = init_field3D(64, 64, 64)
leapfrog3D!(C, Q, I, ρ, τ)
