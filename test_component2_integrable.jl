# test_component2_integrable.jl
#
# Integrable-side acceptance checks for Component 2. Run:
#   julia test_component2_integrable.jl [params_file]
#
# Covers the SETTLED gates:
#   - C^n exact fixed-N sum rule        Σ_j C^n_ij = 0           (canonical constraint)
#   - C^n correctness via Kubo identity ∂n_i/∂h_j = −β C^n_ij    (vs finite difference)
#   - C2.2 integrable C^m is diagonal,  C^m_ij = δ_ij ⟨n_i⟩
# and runs C2.1 with route2 defined LITERALLY as the plain bulk-window sum
# Σ_{j∈bulk} C^n_ij (per sign-off). This does NOT close to 1% against route1
# (κ/β): the fixed-N constraint forces Σ_j C^n_ij = 0, so the plain window sum
# cannot reach the finite-N value route1 = 8/33. C2.1 is therefore reported FAIL
# and the gate stands red pending a decision on the cross-check definition.
# (The canonical C^n itself is proven correct by the sum-rule and Kubo gates
# above; the FDT first-moment Σ_k k·∂_i C^n_ik = 0.2426 does match route1, but
# is intentionally NOT used here.)
#
# The perturbed-side C2.3 (connected_corr on the ITensors product IC) lives in
# the ITensors world and is checked separately.

include("wannier_stark_integrable.jl")
using Printf, LinearAlgebra

function n_profile_fixedbeta(L, α, β, N; hfield = ())
    h = Matrix(single_particle_matrix(L, T_HOP, α))
    for (j, hj) in hfield; h[j, j] += hj; end
    F = eigen(Hermitian(h))
    return (F.vectors .^ 2) * orbital_occupations(F.values, N, β)
end

function main()
    params = isempty(ARGS) ?
        joinpath(@__DIR__, "params/params_L12_seed42_ps188_sweepmin.csv") : ARGS[1]
    occ = read_occupancy(params); L = length(occ); N = count(occ); O = [i for i in 1:L if occ[i]]
    c = L ÷ 2                                  # center site index (1-based), n≈2/3
    println("Component 2 (integrable) — L=$L N=$N center-site=$c\n")

    ok_sr = true; ok_kubo = true; ok_cm = true; ok_c21 = true
    bulk = 2:(L - 1)                            # "bulk window": drop the two edge sites
    for α in (0.10, 0.25)
        β  = solve_beta(L, T_HOP, α, N, α * sum(O))
        Cn = connected_corr_n_integrable(L, T_HOP, α, β, N)
        Cm = connected_corr_m_integrable(L, T_HOP, α, β, N)
        nt = integrable_thermal_at_beta(L, T_HOP, α, β, N).n_tot

        # C^n fixed-N sum rule
        sr = maximum(abs.(vec(sum(Cn, dims = 2))))
        ok_sr &= sr < 1e-9

        # Kubo: ∂n_i/∂h_j = −β C^n_ij   (finite difference at j=c)
        dh = 1e-5
        fd = (n_profile_fixedbeta(L, α, β, N; hfield = ((c, dh),)) .-
              n_profile_fixedbeta(L, α, β, N; hfield = ((c, -dh),))) ./ (2dh)
        kubo = maximum(abs.(fd .- (-β .* Cn[:, c])))
        ok_kubo &= kubo < 1e-4

        # C2.2: C^m diagonal and = ⟨n_i⟩
        offdiag = maximum(abs.(Cm .- Diagonal(diag(Cm))))
        diagerr = maximum(abs.(diag(Cm) .- nt))
        ok_cm &= offdiag < 1e-10 && diagerr < 1e-12

        # C2.1 cross-check — route2 = LITERAL plain bulk-window sum Σ_{j∈bulk} C^n_cj
        route1 = -(1 / (α * β)) * ((nt[c + 1] - nt[c - 1]) / 2)   # profile κ/β
        route2 = sum(Cn[c, bulk])                                  # plain window sum
        reldiff = abs(route2 - route1) / abs(route1)
        ok_c21 &= reldiff < 0.01

        @printf("α=%.2f β=%+.5e\n", α, β)
        @printf("  [C^n] sum rule max|Σ_j C^n_ij| = %.2e   Kubo max|fd-(-βC^n)| = %.2e\n", sr, kubo)
        @printf("  [C^m] max|offdiag| = %.2e   max|C^m_ii-⟨n_i⟩| = %.2e\n", offdiag, diagerr)
        @printf("  [C2.1] route1 κ/β = %.5f | route2 plain-window Σ_{j∈bulk} C^n = %.5f | reldiff = %.1f%%\n\n",
                route1, route2, 100 * reldiff)
    end

    println("="^64)
    @printf("%-50s %s\n", "C^n fixed-N sum rule (<1e-9)",         ok_sr   ? "PASS" : "FAIL")
    @printf("%-50s %s\n", "C^n Kubo identity ∂n/∂h=-βC^n (<1e-4)", ok_kubo ? "PASS" : "FAIL")
    @printf("%-50s %s\n", "C2.2 C^m diagonal = ⟨n_i⟩",            ok_cm   ? "PASS" : "FAIL")
    @printf("%-50s %s\n", "C2.1 κ/β vs plain-window Σ C^n (<1%)", ok_c21  ? "PASS" : "FAIL")
    println("="^64)
    println("C2.1 is expected FAIL (plain window cannot reach 8/33 at fixed N); gate is RED.")
    exit((ok_sr && ok_kubo && ok_cm && ok_c21) ? 0 : 1)
end

main()
