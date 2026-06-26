# ed_thermal.jl
#
# Exact Gibbs-state thermal observables for the perturbed model from its
# diagonalized N-sector spectrum (vals, vecs). Reusable: depends only on the
# spectrum and sector-basis operators — included by ED.jl and by the component-4
# thermometer calibration.
#
# INV-3: every Boltzmann weight is formed in LOG-SUM-EXP (subtract max log-weight
# before exponentiating). The perturbed spectrum spans tens of units, so a bare
# exp(-β·E) overflows on cold / negative-β runs. There is no bare exp(-β·E)
# anywhere in this file.
#
# INV-2: ED is the ruler, not the experiment. These routines return EXACT thermal
# observables and the exact E0↔β map; they do NOT test thermalization, and in
# particular this file contains NO fluctuation-dissipation "verification" (in a
# Gibbs state χ = β·C is an algebraic identity, so checking it learns nothing).

using LinearAlgebra

"""
    boltzmann_weights(vals, β) -> w

Normalized Gibbs weights w_n = e^{-β E_n}/Z, computed in log-sum-exp:
`logw = -β·vals; logw .-= maximum(logw); w = exp.(logw); w ./= sum(w)`.
The max-shift guarantees the largest exponent is 0, so no overflow regardless of
β or the spectral width; underflowed tails simply round to 0 harmlessly.
"""
function boltzmann_weights(vals::AbstractVector{<:Real}, β::Real)
    logw = -β .* vals
    logw .-= maximum(logw)
    w = exp.(logw)
    w ./= sum(w)
    return w
end

"""
    thermal_expect_diag(vals, vecs, Odiag, β) -> ⟨O⟩_β

Thermal expectation of an observable DIAGONAL in the sector occupation basis
(e.g. n_i, given its diagonal `Odiag`). Eigenvectors are real, so
⟨E_n|O|E_n⟩ = Σ_k |vecs[k,n]|² Odiag[k].
"""
function thermal_expect_diag(vals, vecs, Odiag, β)
    w = boltzmann_weights(vals, β)
    return sum(w[n] * dot(vecs[:, n] .^ 2, Odiag) for n in eachindex(vals))
end

"""
    thermal_expect_op(vals, vecs, Op_sector, β) -> ⟨O⟩_β

Thermal expectation of a general (possibly off-diagonal) operator given in the
sector basis (`Op_sector`, dense or sparse). Builds M = vecs' (Op_sector vecs)
and returns Σ_n w_n M_nn.
"""
function thermal_expect_op(vals, vecs, Op_sector, β)
    w = boltzmann_weights(vals, β)
    M = vecs' * (Op_sector * vecs)
    return sum(w[n] * real(M[n, n]) for n in eachindex(vals))
end

"""
    beta_from_E0_ed(vals, E0; bracket=(-2.0, 2.0)) -> β

Exact E0 → β map for the perturbed model: bisect ⟨H⟩_β = E0. Since
d⟨H⟩/dβ = −Var_β(H) ≤ 0, ⟨H⟩_β is monotone decreasing in β and the root is
unique. The bracket straddles zero and is expanded outward (×2) if E0 lies
outside the initial energy window — β<0 is physical (bounded spectrum).
"""
function beta_from_E0_ed(vals, E0; bracket = (-2.0, 2.0))
    energy(β) = sum(boltzmann_weights(vals, β) .* vals)
    βlo, βhi = bracket
    flo = energy(βlo) - E0
    fhi = energy(βhi) - E0
    k = 0
    while flo * fhi > 0 && k < 100
        βlo *= 2; βhi *= 2
        flo = energy(βlo) - E0
        fhi = energy(βhi) - E0
        k += 1
    end
    @assert flo * fhi <= 0 "could not bracket β for E0=$E0 (outside [E_min, E_max]?)"
    βm = 0.5 * (βlo + βhi)
    for _ in 1:200
        βm = 0.5 * (βlo + βhi)
        fm = energy(βm) - E0
        if abs(fm) < 1e-12 || (βhi - βlo) < 1e-14
            return βm
        end
        if (fm > 0) == (flo > 0)
            βlo, flo = βm, fm
        else
            βhi, fhi = βm, fm
        end
    end
    return βm
end
