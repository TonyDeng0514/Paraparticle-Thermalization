# test_component1.jl
#
# Acceptance tests C1.1–C1.3 for Component 1 (integrable thermal state at given β).
# Run:  julia test_component1.jl [params_file]
# Prints an explicit PASS/FAIL table with the actual numbers, then exits nonzero
# if anything failed. The whole body lives in main() so loop accumulators are
# function-local (top-level `for` loops have soft scope and would shadow them).

include("wannier_stark_integrable.jl")
using Printf

# Pre-refactor solve_beta (original code path), kept here only to verify C1.3:
# bisect ⟨H⟩_β = E0 on the spectrum directly.
function solve_beta_old(ε::AbstractVector{Float64}, N::Int, E0::Float64)
    f(β) = sum(ε .* orbital_occupations(ε, N, β)) - E0
    βlo, βhi = -1.0, 1.0
    flo, fhi = f(βlo), f(βhi); k = 0
    while flo * fhi > 0 && k < 100
        βlo *= 2; βhi *= 2; flo, fhi = f(βlo), f(βhi); k += 1
    end
    for _ in 1:200
        βm = 0.5 * (βlo + βhi); fm = f(βm)
        (abs(fm) < 1e-13 || (βhi - βlo) < 1e-14) && return βm
        if (fm > 0) == (flo > 0); βlo, flo = βm, fm else βhi, fhi = βm, fm end
    end
    return 0.5 * (βlo + βhi)
end

function main()
    params = isempty(ARGS) ?
        joinpath(@__DIR__, "params/params_L12_seed42_ps188_sweepmin.csv") : ARGS[1]

    results = Tuple{String,Bool,String}[]   # (name, pass, detail)
    add!(name, pass, detail) = push!(results, (name, pass, detail))

    occ = read_occupancy(params)
    L   = length(occ); N = count(occ)
    O   = [i for i in 1:L if occ[i]]
    println("Component 1 acceptance tests")
    println("  params = $(basename(params));  L=$L  N=$N  occupied=$O  ΣO=$(sum(O))\n")

    # ── C1.1  dynamic range of x_a at β = −1.0 (colder-than-scan stress test) ──
    gmin, gmax = Inf, -Inf
    for α in α_LIST
        ε = eigen(single_particle_matrix(L, T_HOP, α)).values
        x = shifted_fugacities(ε, -1.0)
        gmin = min(gmin, minimum(x)); gmax = max(gmax, maximum(x))
    end
    # "within ~30 orders of magnitude of 1": both |log10| well under 30 ⇒ no Inf/0.
    ok11 = isfinite(gmin) && isfinite(gmax) && gmin > 0 &&
           abs(log10(gmax)) < 30 && abs(log10(gmin)) < 30
    add!("C1.1 x_a dynamic range @β=-1.0", ok11,
         @sprintf("min=%.4e (log10=%.3f)  max=%.4e (log10=%.3f)  span=%.2f decades",
                  gmin, log10(gmin), gmax, log10(gmax), log10(gmax) - log10(gmin)))

    # ── C1.2  sum rule + occupation bounds at several β ───────────────────────
    ok12 = true; det12 = String[]
    for β in (-1.0, -0.05, 0.0, 0.5)
        bad_sum, bad_bounds = false, false
        worst_lo, worst_hi = 0.0, 1.0
        for α in α_LIST
            st = integrable_thermal_at_beta(L, T_HOP, α, β, N)
            abs(sum(st.n_tot) - N) > 1e-9 && (bad_sum = true)
            worst_lo = min(worst_lo, minimum(st.n_orb))
            worst_hi = max(worst_hi, maximum(st.n_orb))
            (minimum(st.n_orb) < -1e-12 || maximum(st.n_orb) > 1 + 1e-12) && (bad_bounds = true)
        end
        ok = !bad_sum && !bad_bounds; ok12 &= ok
        push!(det12, @sprintf("β=%+.2f: Σn=N ✓=%s  n_orb∈[%.3e, %.6f]", β,
                              bad_sum ? "NO" : "yes", worst_lo, worst_hi))
    end
    add!("C1.2 Σn=N(1e-9) & n_orb∈[0,1]", ok12, join(det12, "  |  "))

    # ── C1.3  round-trip: solve_beta matches pre-refactor + E(β*)≈E0 ──────────
    ok13 = true; det13 = String[]
    for α in α_LIST
        E0 = α * sum(O)
        βnew = solve_beta(L, T_HOP, α, N, E0)
        ε    = eigen(single_particle_matrix(L, T_HOP, α)).values
        βold = solve_beta_old(ε, N, E0)
        Eback = integrable_thermal_at_beta(L, T_HOP, α, βnew, N).E
        dβ = abs(βnew - βold); dE = abs(Eback - E0)
        ok = dβ < 1e-10 && dE < 1e-9; ok13 &= ok
        push!(det13, @sprintf("α=%.2f: β=%.6e |Δβ|=%.1e |E(β*)-E0|=%.1e", α, βnew, dβ, dE))
    end
    add!("C1.3 β round-trip(1e-10) & E≈E0(1e-9)", ok13, join(det13, "\n      "))

    # ── PASS/FAIL table ───────────────────────────────────────────────────────
    println("="^78)
    @printf("%-42s  %s\n", "TEST", "RESULT")
    println("-"^78)
    allpass = true
    for (name, pass, detail) in results
        allpass &= pass
        @printf("%-42s  %s\n", name, pass ? "PASS" : "FAIL")
        println("      ", detail)
    end
    println("="^78)
    println(allpass ? "ALL PASS" : "SOME FAILED")
    return allpass
end

exit(main() ? 0 : 1)
