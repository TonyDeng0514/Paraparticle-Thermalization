# two_time_correlator.jl
#
# KMS detailed-balance thermometer — TEBD driver (L=12, cluster). Field-free
# (INV-5): the two-time correlator is measured under the UNPERTURBED H_pert(α).
# Per-state: one product state → thermalize → measure C(t) → read β off the KMS
# slope. Run it over many states (4E set) to sweep warm→cold.
#
# Protocol (field-free):
#   1. |ψ_th⟩ = product state evolved under H_pert(α) to T_therm.
#   2. ⟨M⟩ = ⟨ψ_th|M|ψ_th⟩,  |φ⟩ = M|ψ_th⟩,  M = Σ_i Mloc_i.
#   3. Evolve |ψ_th⟩ and |φ⟩ forward under the SAME H_pert(α) over 0:dt:T_corr
#      (no renormalization in the correlator phase — truncation-induced norm loss
#      of |φ(t)⟩ is the trustworthiness signal).
#   4. C(t) = ⟨ψ(t)|M|φ(t)⟩ − ⟨M⟩².
#   5. β / R² / intercept from `beta_from_Ct` (the C4C.1-validated extractor).
#
# Reports against the locked 2.5a budget: β=1 needs T_corr=40, β=2 needs T_corr=80,
# β≈2 is the resolution edge. The make-or-break read is C4C.3b — the largest T_corr
# at χ=1024 before |φ(t)⟩'s bond dimension saturates — which sets the cold limit.
#
# Usage:  julia two_time_correlator.jl <params_file> [alpha]
#   Hamiltonian disorder fixed at H_SEED. alpha defaults to 0.0.

include("hilbert.jl")
include("gates.jl")
include("observable.jl")        # connected_corr (for the C4C.3a equal-time cross-check)
include("product_states.jl")    # superposition_product_state, read_params, params_label, H_SEED
include("kms_extract.jl")       # beta_from_Ct (no ED/ITensors deps)

function run_two_time()
    L = 12; t_hop = 1.0
    params_file = ARGS[1]
    α = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.0
    occ, θ, φ_ang = read_params(params_file, L)
    N = count(occ); label = params_label(params_file, L)

    dt      = 0.05
    T_therm = 40.0                 # thermalization time (state is locally thermal by here)
    T_corr  = 80.0                 # correlator ladder — push to the cold-stretch target
    n_therm = round(Int, T_therm / dt)
    n_corr  = round(Int, T_corr / dt)
    cutoff  = 1e-12
    maxdim  = 1024
    sat_normloss = 1e-3            # |φ| norm-loss flag = truncation onset (C4C.3b)

    sites = siteinds("Tri", L; conserve_qns = true)
    Random.seed!(H_SEED)
    Ω = 0.2 .* randn(L); V = 0.2 .* randn(L - 1); q = 0.2 .* randn(L, 2)
    gates = tebd_gates(sites, dt, t_hop, Ω, V, q; α = α)

    println("Two-time KMS correlator  (H_SEED=$H_SEED, L=$L, N=$N, α=$α)")
    println("  IC = $params_file ;  T_therm=$T_therm  T_corr=$T_corr  χmax=$maxdim")

    # ── 1. thermalize (normalized, as usual) ──
    ψ, _ = superposition_product_state(sites, occ, θ, φ_ang)
    for _ in 1:n_therm
        ψ = apply(gates, ψ; cutoff = cutoff, maxdim = maxdim)
        normalize!(ψ)
    end

    # ── 2. ⟨M⟩, total-M MPO, |φ⟩ = M|ψ_th⟩ ──
    osM = OpSum(); for i in 1:L; osM += "Mloc", i; end
    M_mpo = MPO(osM, sites)
    Mexp  = real(inner(ψ', M_mpo, ψ))                 # ⟨M⟩
    Cm_eq = sum(real.(connected_corr(ψ, "Mloc")))     # Var(M) equal-time = Σ_ij C^m_ij (C4C.3a ref)
    φ = apply(M_mpo, ψ; cutoff = cutoff, maxdim = maxdim)
    φ0norm = norm(φ)

    # ── 3–4. correlator phase: evolve both, no renormalization ──
    times = collect(0:n_corr) .* dt
    Ct = Vector{ComplexF64}(undef, n_corr + 1)
    χψ = zeros(Int, n_corr + 1); χφ = zeros(Int, n_corr + 1)
    φnorm = zeros(Float64, n_corr + 1)
    Ct[1]   = inner(ψ', M_mpo, φ) - Mexp^2            # C(0) = Var(M)
    χψ[1]   = maxlinkdim(ψ); χφ[1] = maxlinkdim(φ); φnorm[1] = norm(φ)
    sat_step = 0
    for step in 1:n_corr
        ψ = apply(gates, ψ; cutoff = cutoff, maxdim = maxdim)
        φ = apply(gates, φ; cutoff = cutoff, maxdim = maxdim)
        Ct[step+1] = inner(ψ', M_mpo, φ) - Mexp^2
        χψ[step+1] = maxlinkdim(ψ); χφ[step+1] = maxlinkdim(φ); φnorm[step+1] = norm(φ)
        if sat_step == 0 && (χφ[step+1] >= maxdim && 1 - φnorm[step+1] / φ0norm > sat_normloss)
            sat_step = step + 1
        end
    end
    T_trust = sat_step == 0 ? T_corr : times[sat_step]

    # ── C4C.3a equal-time cross-check ──
    C0 = real(Ct[1])
    a_dev = abs(C0 - Cm_eq) / abs(Cm_eq)

    # ── 5 / C4C.3c: β from the (trustworthy) correlator window ──
    n_use = sat_step == 0 ? n_corr + 1 : sat_step
    tcut = times[1:n_use]; Ccut = Ct[1:n_use]
    βres = try
        beta_from_Ct(tcut, Ccut)
    catch err
        @warn "beta_from_Ct failed (likely C(t) too short / truncated): $err"
        (β = NaN, r2 = NaN, intercept = NaN, nfit = 0, S = Float64[], ωgrid = Float64[])
    end

    # ── C4C.3d: moment-ratio diagnostic ωbar = ∫ωS/∫S (cold-end cross-check) ──
    ωbar = NaN
    if !isempty(βres.S)
        sS = sum(βres.S); ωbar = sS == 0 ? NaN : sum(βres.ωgrid .* βres.S) / sS
    end

    # ── save C(t) + bond-dim diagnostics ──
    outdir = joinpath(@__DIR__, "results/two_time"); mkpath(outdir)
    outfile = joinpath(outdir, "Ct_L$(L)_chi$(maxdim)_alpha$(α)_seed$(H_SEED)_$(label).csv")
    open(outfile, "w") do io
        println(io, "# params_file = $(params_file)")
        println(io, "# H_SEED=$H_SEED L=$L N=$N alpha=$α  <M>=$Mexp")
        println(io, "# C4C.3a  C(0)=$C0  Cm_eq(Var M)=$Cm_eq  reldev=$a_dev")
        println(io, "# C4C.3b  T_trust(χ_φ sat & normloss>$sat_normloss)=$T_trust  (sat_step=$sat_step)")
        println(io, "# C4C.3c  beta=$(βres.β)  R2=$(βres.r2)  intercept=$(βres.intercept)  nfit=$(βres.nfit)")
        println(io, "# C4C.3d  omega_bar=$ωbar")
        println(io, "time,Re_C,Im_C,chi_psi,chi_phi,phi_normloss")
        for k in 1:(n_corr+1)
            @printf(io, "%.4f,%.10e,%.10e,%d,%d,%.3e\n",
                    times[k], real(Ct[k]), imag(Ct[k]), χψ[k], χφ[k], 1 - φnorm[k]/φ0norm)
        end
    end

    println("\n── C4C.3 reads ──")
    @printf("  3a  C(0)=%.5f  vs  Σ_ij C^m_ij=%.5f   reldev=%.2f%%  %s\n",
            C0, Cm_eq, 100a_dev, a_dev < 0.01 ? "PASS" : "CHECK")
    @printf("  3b  max trustworthy T_corr at χ=%d : %.1f   (χ_φ: %d→%d, |φ| norm-loss→%.1e)\n",
            maxdim, T_trust, χφ[1], maximum(χφ), 1 - φnorm[end]/φ0norm)
    @printf("      BUDGET: β=1 needs 40, β=2 needs 80 (edge). T_trust=%.1f ⇒ cold limit %s\n",
            T_trust, T_trust >= 80 ? "|β|≈2" : (T_trust >= 40 ? "|β|≈1" : "|β|<1"))
    @printf("  3c  β=%.4f  R²=%.5f  intercept=%.4f  (nfit=%d)\n",
            βres.β, βres.r2, βres.intercept, βres.nfit)
    @printf("  3d  ω̄ (moment) = %.4f\n", ωbar)
    println("Wrote $(outfile)")
    println("\nNOTE: β/R² use the trustworthy window t ≤ T_trust. If T_trust < 40 the")
    println("state is colder than χ=1024 can resolve — report it as the honest cold limit.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_two_time()
end
