# ===============================================
# File: src/OuroboroHiggs.jl
# Description: Multi-Higgs entropic Ouroboro field model
# Author: (Jan Bouwman)
# ===============================================

module OuroboroHiggs

using CUDA
using WriteVTK
using Statistics

const F32 = Float32

# -------------------------------------------------------
# === Utility helpers ===
# -------------------------------------------------------

@inline function get_idx(i,j,k,nx,ny,nz)
    return (i-1) + (j-1)*nx + (k-1)*nx*ny + 1
end

# -------------------------------------------------------
# === Core GPU kernel ===
# -------------------------------------------------------
function kernel_ouro_multihiggs!(
    C::CuDeviceArray{F32,3},
    Q::CuDeviceArray{F32,3},
    I::CuDeviceArray{F32,4},
    ρ::CuDeviceArray{F32,3},
    τ::CuDeviceArray{F32,3},
    higgs::CuDeviceArray{F32,4},
    entropy::CuDeviceArray{F32,3},
    Δt::F32, Δx::F32,
    nx::Int, ny::Int, nz::Int, nh::Int,
    κ0::F32, κ1::F32, g::F32, β::F32, λ::F32, v::F32
)
    idx = (blockIdx().x-1) * blockDim().x + threadIdx().x
    N = nx * ny * nz
    if idx < 1 || idx > N
        return
    end

    i = (idx-1) % nx + 1
    j = ((idx-1) ÷ nx) % ny + 1
    k = (idx-1) ÷ (nx*ny) + 1

    @inbounds begin
        Cijk = C[i,j,k]
        Qijk = Q[i,j,k]
        rho = ρ[i,j,k]
        tau = τ[i,j,k]
    end

    eps = 1e-6f0

    # Higgs mean
    phi_sum = 0f0
    @inbounds for h in 1:nh
        phi_sum += higgs[h,i,j,k]
    end
    phi_mean = phi_sum / nh
    m_eff2 = g * g * (phi_mean * phi_mean)

    # Laplace (6-point)
    lap = 0f0
    @inbounds begin
        if i < nx; lap += C[i+1,j,k] - Cijk; end
        if i > 1;  lap += C[i-1,j,k] - Cijk; end
        if j < ny; lap += C[i,j+1,k] - Cijk; end
        if j > 1;  lap += C[i,j-1,k] - Cijk; end
        if k < nz; lap += C[i,j,k+1] - Cijk; end
        if k > 1;  lap += C[i,j,k-1] - Cijk; end
    end
    lap /= (Δx * Δx)

    D = κ0 + κ1 * min(abs(Cijk), 10f0)

    I1 = I[1,i,j,k]
    I2 = I[2,i,j,k]
    I3 = I[3,i,j,k]
    I_sum = I1 + I2 + I3

    Δt_eff = Δt / (1f0 + 0.5f0 * tau)
    Qnew = Qijk + Δt * (lap - m_eff2 * Cijk - β * phi_mean)
    Qnew = Qnew / (1f0 + 0.1f0 * tau)

    Cnew = Cijk + Δt_eff * (Qnew - I_sum) / (rho + eps)
    Cnew += Δt * D * lap

    γ = 0.01f0 * g
    @inbounds for h in 1:nh
        φ = higgs[h,i,j,k]
        dφ = -λ * (φ*φ - v*v) * φ + γ * Cijk
        higgs[h,i,j,k] = φ + 0.5f0 * Δt * dφ
    end

    Ccap = 1e2f0
    if !(abs(Cnew) <= Ccap)
        overflow = Cnew - sign(Cnew)*Ccap
        Cnew = sign(Cnew) * Ccap
        entropy[i,j,k] += 0.5f0 * abs(overflow)
    end

    C[i,j,k] = Cnew
    Q[i,j,k] = Qnew

    return
end


# -------------------------------------------------------
# === Wrapper launch ===
# -------------------------------------------------------
function launch_leapfrog_multihiggs!(
    C::CuArray{F32,3}, Q::CuArray{F32,3}, I::CuArray{F32,4},
    ρ::CuArray{F32,3}, τ::CuArray{F32,3}, higgs::CuArray{F32,4},
    entropy::CuArray{F32,3},
    Δt::F32, Δx::F32,
    κ0::F32, κ1::F32, g::F32, β::F32, λ::F32, v::F32
)
    nx, ny, nz = size(C)
    nh = size(higgs, 1)
    N = nx * ny * nz
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks kernel_ouro_multihiggs!(
        C,Q,I,ρ,τ,higgs,entropy,Δt,Δx,nx,ny,nz,nh,κ0,κ1,g,β,λ,v
    )
    return
end


# -------------------------------------------------------
# === Demo runner ===
# -------------------------------------------------------
function run_demo(; nx=48, ny=48, nz=48, nh=2, n_steps=200)
    Δx = 1f0
    Δt = 0.01f0

    C_cpu = zeros(F32, nx, ny, nz)
    Q_cpu = zeros(F32, nx, ny, nz)
    I_cpu = zeros(F32, 3, nx, ny, nz)
    ρ_cpu = ones(F32, nx, ny, nz)
    τ_cpu = ones(F32, nx, ny, nz)
    entropy_cpu = zeros(F32, nx, ny, nz)
    higgs_cpu = zeros(F32, nh, nx, ny, nz)

    cx,cy,cz = (nx+1)/2, (ny+1)/2, (nz+1)/2
    for k in 1:nz, j in 1:ny, i in 1:nx
        r2 = (i-cx)^2 + (j-cy)^2 + (k-cz)^2
        C_cpu[i,j,k] = exp(-r2 / (2f0 * (min(nx,ny,nz)/8)^2))
        for h in 1:nh
            higgs_cpu[h,i,j,k] = 1f0 + 0.01f0 * (rand(Float32)-0.5f0)
        end
    end

    C = CuArray(C_cpu)
    Q = CuArray(Q_cpu)
    I = CuArray(I_cpu)
    ρ = CuArray(ρ_cpu)
    τ = CuArray(τ_cpu)
    entropy = CuArray(entropy_cpu)
    higgs = CuArray(higgs_cpu)

    κ0, κ1 = 0.001f0, 0.01f0
    g, β, λ, v = 1.0f0, 0.01f0, 0.5f0, 1.0f0

    max_abs_C = zeros(Float32, n_steps)
    E_tot = zeros(Float32, n_steps)

    for t in 1:n_steps
        launch_leapfrog_multihiggs!(C,Q,I,ρ,τ,higgs,entropy,Δt,Δx,κ0,κ1,g,β,λ,v)
        CUDA.synchronize()

        C_cpu .= Array(C)
        Q_cpu .= Array(Q)
        hig_cpu = Array(higgs)
        max_abs_C[t] = maximum(abs.(C_cpu))
        E_tot[t] = sum(abs.(C_cpu)) + sum(abs.(Q_cpu))

        if t % 50 == 0 || t == n_steps
            fn = "ouroboro_C_step$(t).vti"
            vtk = vtk_grid(fn, (nx,ny,nz))
            vtk_point_data(vtk, "C", Array(C))
            vtk_point_data(vtk, "Q", Array(Q))
            vtk_point_data(vtk, "entropy", Array(entropy))
            mean_higgs = mean(hig_cpu; dims=1) |> dropdims
            vtk_point_data(vtk, "higgs_mean", Array(mean_higgs))
            close(vtk)
            @info "Wrote VTK snapshot $fn"
        end
    end

    return max_abs_C, E_tot
end

end # module
