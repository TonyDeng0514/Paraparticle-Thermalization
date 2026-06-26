# test_component4C.jl
#
# Component 4C gate (ED validation of the KMS detailed-balance thermometer).
# Run:  julia --project=. test_component4C.jl     (stdlib only; --project optional)
#
# C4C.1  On an L=9/N=6 ED Gibbs state at known β, the S(ω)/S(−ω) slope returns β
#        to ~2%, with ln[S/S(−ω)] straight (R² > 0.99).
# C4C.2  On non-thermal states (a single eigenstate; a random in-sector
#        superposition), the ln-ratio is NOT a single straight line through the
#        origin — proving the certificate can fail when it should.
#
# INV-2: this validates the machinery against a known β; it is not a thermalization
# claim. The thermalization test lives in the TEBD driver (two_time_correlator.jl).

include("detailed_balance.jl")
using Printf, LinearAlgebra, Random

function main()
    L, N, α = 9, 6, 0.0
    @printf("Component 4C — KMS thermometer ED validation (L=%d N=%d α=%.2f, seed=%d)\n\n",
            L, N, α, DB_SEED)
    vals, vecs, idx = perturbed_spectrum(L, N, α)
    Mdiag = total_mag_diagonal(idx, L)
    @printf("sector dim = %d, spectrum ∈ [%.3f, %.3f]\n\n", length(vals), vals[1], vals[end])

    # ── C4C.1 recover known β ──
    println("C4C.1  recover known β from KMS slope:")
    βtest = (-1.0, -0.5, 0.2, 0.5, 1.0)
    ok1 = true
    for β in βtest
        β_est, intercept, r2, nfit = kms_beta(vals, vecs, Mdiag, β)
        reldev = abs(β_est - β) / abs(β)
        pass = reldev < 0.02 && r2 > 0.99
        ok1 &= pass
        @printf("  β=%+.2f → β_est=%+.4f  reldev=%.2f%%  R²=%.5f  (nfit=%d)  %s\n",
                β, β_est, 100reldev, r2, nfit, pass ? "PASS" : "FAIL")
    end

    # ── C4C.2 discrimination: non-thermal states must break KMS ──
    println("\nC4C.2  non-thermal states break the KMS line (must NOT look thermal):")
    is_thermal_looking(r2, intercept, nfit) = (nfit >= 3 && r2 > 0.99 && abs(intercept) < 0.1)

    # (a) single mid-spectrum eigenstate
    k = length(vals) ÷ 2
    p_eig = zeros(Float64, length(vals)); p_eig[k] = 1.0
    c1, S1 = spectral_function(vals, vecs, Mdiag, p_eig)
    s1, i1, r1, n1 = beta_from_spectral(c1, S1)
    therm1 = is_thermal_looking(r1, i1, n1)

    # (b) random in-sector superposition (diagonal weights |⟨E_n|ψ⟩|²)
    Random.seed!(7)
    ψ = randn(length(idx)); ψ ./= norm(ψ)
    cE = vecs' * ψ
    p_rand = abs2.(cE); p_rand ./= sum(p_rand)
    c2, S2 = spectral_function(vals, vecs, Mdiag, p_rand)
    s2, i2, r2b, n2 = beta_from_spectral(c2, S2)
    therm2 = is_thermal_looking(r2b, i2, n2)

    @printf("  single eigenstate     : slope=%+.3f intercept=%+.3f R²=%.4f nfit=%d → thermal-looking=%s\n",
            s1, i1, r1, n1, therm1)
    @printf("  random superposition  : slope=%+.3f intercept=%+.3f R²=%.4f nfit=%d → thermal-looking=%s\n",
            s2, i2, r2b, n2, therm2)
    ok2 = !therm1 && !therm2

    println("\n" * "="^60)
    @printf("%-44s %s\n", "C4C.1 KMS recovers known β (2%, R²>0.99)", ok1 ? "PASS" : "FAIL")
    @printf("%-44s %s\n", "C4C.2 non-thermal states rejected",       ok2 ? "PASS" : "FAIL")
    println("="^60)
    println((ok1 && ok2) ? "ALL PASS — KMS thermometer validated; proceed to TEBD driver." :
                           "SOME FAILED — stop and inspect.")
    exit((ok1 && ok2) ? 0 : 1)
end

main()
