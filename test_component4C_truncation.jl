# test_component4C_truncation.jl
#
# Gate C4C.2.5 — finite-T_max (windowing/truncation) effect on the KMS thermometer.
# Sets the cold reach and the cluster T_max budget before the TEBD driver is built.
# Run on the committed Julia (L=9 ED fits in memory):
#   julia --project=. test_component4C_truncation.jl
#
# Builds the EXACT connected C(t) from the L=9 spectrum, samples on 0:Δt:T_max
# (Δt=0.1, Nyquist for bandwidth ~11), and feeds it through `beta_from_Ct`, which
# reuses the C4C.1-validated `beta_from_spectral` fitter — so any difference from
# C4C.1 is purely the windowing/truncation being measured.

include("detailed_balance.jl")
using Printf, LinearAlgebra

function main()
    L, N, α = 9, 6, 0.0
    vals, vecs, idx = perturbed_spectrum(L, N, α)
    Mdiag = total_mag_diagonal(idx, L)
    bw = vals[end] - vals[1]
    Δt = 0.1
    ωm = bw * 1.05
    @printf("Component 4C.2.5 truncation gate (L=%d N=%d, bandwidth=%.2f, Δt=%.2f)\n\n", L, N, bw, Δt)

    Tmaxs = (10.0, 20.0, 40.0, 80.0)
    βs    = (-1.0, -0.5, 0.5, 1.0, 2.0)

    # ── C4C.2.5a: T_max × β sweep, Blackman-Harris ──
    println("C4C.2.5a  β_rec / R² / intercept vs T_max  [Blackman-Harris]:")
    minTmax = Dict{Float64,Any}()
    for β in βs
        @printf("  β_true=%+.2f\n", β)
        rec = nothing
        for Tmax in Tmaxs
            tgrid = collect(0:Δt:Tmax)
            C = exact_Ct(vals, vecs, Mdiag, β, tgrid)
            r = beta_from_Ct(tgrid, C; window = :blackmanharris, ωmax_grid = ωm)
            reldev = abs(r.β - β) / abs(β)
            @printf("    T_max=%4.0f : β_rec=%+.4f  reldev=%5.2f%%  R²=%.5f  intercept=%+.4f\n",
                    Tmax, r.β, 100reldev, r.r2, r.intercept)
            if rec === nothing && reldev < 0.02 && abs(r.intercept) < 0.05
                rec = Tmax
            end
        end
        minTmax[β] = rec
    end
    println("\n  Minimum T_max to recover β to ~2% (|intercept|<0.05):")
    for β in βs
        v = minTmax[β]
        @printf("    β=%+.2f : T_max %s\n", β, v === nothing ? "> 80 (not reached)" : "≥ $(Int(v))")
    end

    # ── C4C.2.5b: Blackman-Harris vs Hann, fit-cutoff sweep (β=1, T_max=40) ──
    println("\nC4C.2.5b  BH vs Hann (β=1, T_max=40): β_rec vs fit cutoff ωfrac")
    tg = collect(0:Δt:40.0)
    C1 = exact_Ct(vals, vecs, Mdiag, 1.0, tg)
    for ωf in (0.3, 0.5, 0.7, 0.9)
        rb = beta_from_Ct(tg, C1; window = :blackmanharris, ωmax_grid = ωm, ωfrac = ωf)
        rh = beta_from_Ct(tg, C1; window = :hann,          ωmax_grid = ωm, ωfrac = ωf)
        @printf("    ωfrac=%.1f : BH β=%+.4f (R²=%.4f) | Hann β=%+.4f (R²=%.4f)\n",
                ωf, rb.β, rb.r2, rh.β, rh.r2)
    end
    println("    (low-side-lobe BH should hold β≈1 to larger ωfrac; Hann drifts earlier)")

    # ── C4C.2.5c: intercept scaling ∝ β²/T_max² (falsifiable) ──
    println("\nC4C.2.5c  |intercept|·T_max² vs T_max (β=1, expect ≈ const ⇒ 1/T_max²):")
    for Tmax in Tmaxs
        tgrid = collect(0:Δt:Tmax)
        C = exact_Ct(vals, vecs, Mdiag, 1.0, tgrid)
        r = beta_from_Ct(tgrid, C; window = :blackmanharris, ωmax_grid = ωm)
        @printf("    T_max=%4.0f : |intercept|=%.4e   |intercept|·T_max²=%.4f\n",
                Tmax, abs(r.intercept), abs(r.intercept) * Tmax^2)
    end
    println("    (≈ const ⇒ 1/T_max² confirmed. If it does NOT scale this way, STOP & flag.)")

    println("\nHALT after C4C.2.5 for sign-off.")
end

main()
