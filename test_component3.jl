# test_component3.jl
#
# Component 3 acceptance gates — exact perturbed Gibbs benchmark via ED.
# Run:  julia --project=. test_component3.jl       (stdlib only; --project optional)
#
# Builds the L=10 N-sector perturbed spectrum (same disorder convention as ED.jl,
# seed recorded below) and checks the ed_thermal.jl Gibbs routines:
#   C3.1 boltzmann_weights is log-sum-exp: finite & normalized at β=-1.0
#   C3.2 exact thermal Σ_i ⟨n_i⟩ = N at β ∈ {-1.0, 0.0, +1.0}
#   C3.3 beta_from_E0_ed is monotone and round-trips ⟨H⟩_{β*} ≈ E0 to 1e-9
#   C3.4 β→0 ⟨n_i⟩ → N/L uniformly (infinite-T flat profile)
# No FDT verification (INV-2): in a Gibbs state χ=βC is an identity, not a test.

include("ED.jl")   # guarded: provides builders, ed_thermal routines, consts L,N,seed,W,t_hop
using Printf, LinearAlgebra, Random

function main()
    @printf("Component 3 — exact ED Gibbs benchmark  (L=%d N=%d seed=%d W=%.2f)\n\n", L, N, seed, W)

    # ── build & diagonalize the N-sector (records the fixed disorder seed) ──
    Random.seed!(seed)
    Ω = W .* randn(L); V = W .* randn(L - 1); q = W .* randn(L, 2)
    H   = build_hamiltonian_ed(L, t_hop, Ω, V, q)
    idx = n_sector_indices(L, N)
    Hn  = Symmetric(Matrix(H[idx, idx]))
    vals, vecs = eigen(Hn)
    nocc = [sector_occupations(idx, i, L) for i in 1:L]   # diagonal of n_i in sector basis
    @printf("sector dim = %d, spectrum ∈ [%.3f, %.3f]\n\n", length(vals), vals[1], vals[end])

    # ── C3.1 log-sum-exp: finite & normalized at β=-1.0 ──
    w = boltzmann_weights(vals, -1.0)
    ok31 = all(isfinite, w) && isapprox(sum(w), 1.0; atol = 1e-12)
    @printf("C3.1  β=-1.0: all weights finite=%s, Σw=%.15f  (max e^{-βE} bare would be %.1e)\n",
            all(isfinite, w), sum(w), exp(-(-1.0) * vals[end]))

    # ── C3.2 Σ_i ⟨n_i⟩ = N ──
    ok32 = true
    for β in (-1.0, 0.0, 1.0)
        s = sum(thermal_expect_diag(vals, vecs, nocc[i], β) for i in 1:L)
        ok32 &= abs(s - N) < 1e-9
        @printf("C3.2  β=%+.1f: Σ_i⟨n_i⟩ = %.12f  (N=%d, dev %.1e)\n", β, s, N, abs(s - N))
    end

    # ── C3.3 monotone ⟨H⟩(β) + round-trip ──
    energy(β) = sum(boltzmann_weights(vals, β) .* vals)
    βgrid = -2.0:0.5:2.0
    Egrid = energy.(βgrid)
    monotone = all(diff(Egrid) .< 0)
    ok33 = monotone
    for E0 in (energy(-0.5), energy(0.2), energy(0.8))
        βstar = beta_from_E0_ed(vals, E0)
        dev = abs(energy(βstar) - E0)
        ok33 &= dev < 1e-9
        @printf("C3.3  E0=%+.4f → β*=%+.6f, ⟨H⟩_{β*}-E0 = %.1e\n", E0, βstar, dev)
    end
    @printf("C3.3  ⟨H⟩(β) monotone decreasing on [-2,2]: %s\n", monotone)

    # ── C3.4 β→0 flat profile → N/L ──
    ni0 = [thermal_expect_diag(vals, vecs, nocc[i], 0.0) for i in 1:L]
    ok34 = maximum(abs.(ni0 .- N / L)) < 1e-9
    @printf("C3.4  β=0 ⟨n_i⟩ = [%s]\n", join((@sprintf("%.6f", x) for x in ni0), ", "))
    @printf("C3.4  N/L = %.6f, max|⟨n_i⟩-N/L| = %.2e\n", N / L, maximum(abs.(ni0 .- N / L)))

    println("\n", "="^60)
    tbl(name, ok) = @printf("%-46s %s\n", name, ok ? "PASS" : "FAIL")
    tbl("C3.1 log-sum-exp finite & normalized @β=-1", ok31)
    tbl("C3.2 Σ_i⟨n_i⟩ = N (1e-9)",                   ok32)
    tbl("C3.3 β(E0) monotone + round-trip (1e-9)",     ok33)
    tbl("C3.4 β→0 flat ⟨n_i⟩ = N/L (1e-9)",           ok34)
    println("="^60)
    allpass = ok31 && ok32 && ok33 && ok34
    println(allpass ? "ALL PASS" : "SOME FAILED")
    exit(allpass ? 0 : 1)
end

main()
