# detailed_balance.jl
#
# KMS / detailed-balance thermometer — ED-validation side. The active Component 4.
#
# Principle. For ANY thermal state the total-magnetization spectral function
# S(ω) = ∫dt e^{iωt} ⟨δM(t) δM(0)⟩,  M = Σ_i Mloc_i,  obeys the exact KMS relation
#     S(ω) / S(−ω) = e^{βω}.
# So β = slope of ln[S(ω)/S(−ω)] vs ω, and the line being straight through the
# origin is the thermalization certificate (a non-thermal state breaks it).
#
# This file builds S(ω) EXACTLY from an ED spectrum (no time discretization):
#     S(ω) = Σ_{n,m} p_n |M_{nm}|² δ(ω − (E_m − E_n)),   p_n = Gibbs weights,
# binned onto a symmetric ω-grid. The TEBD driver (two_time_correlator.jl) will
# instead measure C(t) and FT it, then reuse `beta_from_spectral` below.
#
# INV-2: ED is the ruler — KMS is an identity in a Gibbs state, so this validates
#   the machinery against a known β; it does not test thermalization.
# INV-3: Gibbs weights via ed_thermal's log-sum-exp.

include("ED.jl")    # build_hamiltonian_ed, n_sector_indices, site_op, _Nloc, boltzmann_weights, t_hop
include("kms_extract.jl")   # linfit, beta_from_spectral, window_weights, beta_from_Ct (no ED/ITensors deps)
using LinearAlgebra

const DB_SEED = 42      # == H_SEED (the TEBD disorder seed)
const DB_W    = 0.2

"""
    perturbed_spectrum(L, N, α; seed=DB_SEED, W=DB_W) -> (vals, vecs, idx)

Full N-sector spectrum of H_pert(α) (hopping, V·MM, Ω·FlipAB, q·Na/Nb, tilt
α·Σ_j j·Nloc_j), disorder drawn at `seed`/`W` (the TEBD convention).
"""
function perturbed_spectrum(L, N, α; seed = DB_SEED, W = DB_W)
    Random.seed!(seed)
    Ω = W .* randn(L); V = W .* randn(L - 1); q = W .* randn(L, 2)
    H = build_hamiltonian_ed(L, t_hop, Ω, V, q)
    if α != 0
        for j in 1:L
            H = H + (α * j) * site_op(_Nloc, j, L)
        end
    end
    idx = n_sector_indices(L, N)
    Hn  = Symmetric(Matrix(H[idx, idx]))
    vals, vecs = eigen(Hn)
    return vals, vecs, idx
end

"""
    total_mag_diagonal(idx, L) -> Mdiag

Diagonal of M = Σ_i Mloc_i in the sector occupation basis (Mloc: A→+1, B→−1,
vac→0). M is diagonal in this basis, so a vector suffices.
"""
function total_mag_diagonal(idx, L)
    Mdiag = zeros(Float64, length(idx))
    for a in eachindex(idx)
        kk = idx[a] - 1
        s = 0
        for j in 1:L
            d = div(kk, 3^(L - j)) % 3
            if d == 1
                s += 1
            elseif d == 2
                s -= 1
            end
        end
        Mdiag[a] = s
    end
    return Mdiag
end

"""
    spectral_function(vals, vecs, Mdiag, p; nbins=161, ωmax=nothing) -> (ω_centers, S)

S(ω) = Σ_{n,m} p_n |M_{nm}|² δ(ω−(E_m−E_n)) binned onto a SYMMETRIC ω-grid (so
bin i mirrors bin nbins+1−i at −ω). `p` is the diagonal weight vector in the
energy basis (Gibbs weights for a thermal state; a one-hot or |⟨E_n|ψ⟩|² for a
non-thermal state). `nbins` is forced odd so a bin is centered at ω=0.
"""
function spectral_function(vals, vecs, Mdiag, p; nbins = 161, ωmax = nothing)
    nbins = isodd(nbins) ? nbins : nbins + 1
    Mmat = vecs' * (Mdiag .* vecs)                 # M_{nm}
    D = length(vals)
    if ωmax === nothing
        ωmax = (vals[end] - vals[1]) * 1.0000001
    end
    Δω = 2ωmax / nbins
    S = zeros(Float64, nbins)
    @inbounds for n in 1:D
        pn = p[n]
        pn == 0.0 && continue
        En = vals[n]
        for m in 1:D
            ω = vals[m] - En
            (ω <= -ωmax || ω >= ωmax) && continue
            b = floor(Int, (ω + ωmax) / Δω) + 1
            if b < 1
                b = 1
            elseif b > nbins
                b = nbins
            end
            S[b] += pn * abs2(Mmat[n, m])
        end
    end
    centers = [(-ωmax + (i - 0.5) * Δω) for i in 1:nbins]
    return centers, S ./ Δω
end

"""
    exact_Ct(vals, vecs, Mdiag, β, tgrid) -> C::Vector{ComplexF64}

Exact connected two-time correlator C(t) = ⟨δM(t) δM(0)⟩_β from the spectrum, on
one-sided `tgrid` (t ≥ 0). C(t) = Σ_{n,m} p_n |δM_{nm}|² e^{i(E_n−E_m)t}, evaluated
as uᵀ(B conj(u)) with u_n = e^{iE_n t}, B_{nm} = p_n|δM_{nm}|² (O(D²) per t).
C(0) = Var(M). The ED reference for the truncation gate.
"""
function exact_Ct(vals, vecs, Mdiag, β, tgrid)
    p = boltzmann_weights(vals, β)
    M = vecs' * (Mdiag .* vecs)
    Mexp = sum(p .* diag(M))
    δM = M - Mexp * I
    B = p .* abs2.(δM)                       # p broadcasts over rows (n index)
    C = Vector{ComplexF64}(undef, length(tgrid))
    for it in eachindex(tgrid)
        u = cis.(vals .* tgrid[it])
        C[it] = transpose(u) * (B * conj(u))
    end
    return C
end

"""
    kms_beta(vals, vecs, Mdiag, β; kwargs...) -> (β_est, intercept, R², nfit)

Convenience for the ED validation: build the Gibbs spectral function at known β
and read β back off the KMS slope. (Connected correlator: the n=m / ⟨M⟩² piece
sits at ω=0 and is excluded by the ω>0 fit window, so no explicit subtraction is
needed for the slope.)
"""
function kms_beta(vals, vecs, Mdiag, β; nbins = 161, kwargs...)
    p = boltzmann_weights(vals, β)
    centers, S = spectral_function(vals, vecs, Mdiag, p; nbins = nbins)
    return beta_from_spectral(centers, S; kwargs...)
end
