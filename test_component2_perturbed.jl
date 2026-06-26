# test_component2_perturbed.jl
#
# C2.3 — perturbed-side connected_corr sanity on the product-state IC.
# Run (from the repo root, with the project env so ITensors is available):
#
#   julia --project=. test_component2_perturbed.jl [params_file]
#
# Defaults to params/params_L12_seed42_ps188_sweepmin.csv.
#
# Checks, on the bond-dim-1 initial product state (t=0, no TEBD needed):
#   (a) C^n = connected_corr(psi,"Nloc") and C^m = connected_corr(psi,"Mloc")
#       are REAL and SYMMETRIC;
#   (b) C^n_ii ≈ n_i(1−n_i). Each site is a Nloc-eigenstate (0 for |vac>, 1 for the
#       a/b superposition), so n_i ∈ {0,1} and the RHS is 0 — the product state has
#       no charge fluctuation. (C^m_ii = sin²2θ_i ≠ 0 in general; only C^n is gated.)
#
# It then evolves the SAME IC under TEBD to a short T_end = 5.0 (entangling the
# state) and re-checks real/symmetric. The diagonal identity C^n_ii = n_i(1−n_i)
# is IC-only (it relies on n_i ∈ {0,1}) and is NOT expected to hold once the state
# is correlated, so it is not re-checked post-TEBD.

include("hilbert.jl")          # SiteType"Tri", ops, states  (loads ITensorMPS/ITensors)
include("gates.jl")            # tebd_gates
include("observable.jl")       # connected_corr
include("product_states.jl")   # superposition_product_state, read_params (defines H_SEED)

# real/symmetric diagnostics for the two connected correlators on a state
function corr_diagnostics(psi)
    Cn = connected_corr(psi, "Nloc")
    Cm = connected_corr(psi, "Mloc")
    realness = max(maximum(abs.(imag.(Cn))), maximum(abs.(imag.(Cm))))
    symmetry = max(maximum(abs.(Cn .- transpose(Cn))), maximum(abs.(Cm .- transpose(Cm))))
    return Cn, realness, symmetry
end

function main()
    params = isempty(ARGS) ?
        joinpath(@__DIR__, "params/params_L12_seed42_ps188_sweepmin.csv") : ARGS[1]
    L = 12; t_hop = 1.0
    occ, θ, φ = read_params(params, L)
    sites = siteinds("Tri", L; conserve_qns = true)
    psi0, _ = superposition_product_state(sites, occ, θ, φ)

    # ── Part 1: product IC ───────────────────────────────────────────────────
    Cn0, real0, sym0 = corr_diagnostics(psi0)
    n = expect(psi0, "Nloc")
    diag_err = maximum(abs.(real.(diag(Cn0)) .- n .* (1 .- n)))
    ic_real = real0 < 1e-12; ic_sym = sym0 < 1e-12; ic_diag = diag_err < 1e-10

    println("C2.3 perturbed connected_corr  (params = $(basename(params)))\n")
    println("[Part 1] product IC (t=0)")
    @printf("  real (max|imag C|)             = %.2e   %s\n", real0, ic_real ? "PASS" : "FAIL")
    @printf("  symmetric (max|C - Cᵀ|)        = %.2e   %s\n", sym0,  ic_sym  ? "PASS" : "FAIL")
    @printf("  C^n_ii ≈ n_i(1-n_i) (max dev)  = %.2e   %s\n", diag_err, ic_diag ? "PASS" : "FAIL")

    # ── Part 2: short TEBD-evolved (entangled) state ─────────────────────────
    dt = 0.05; T_end = 5.0; Nsteps = round(Int, T_end / dt)
    α = 0.0; cutoff = 1e-12; maxdim = 1024
    Random.seed!(H_SEED)                       # same fixed disorder realization
    Ω = 0.2 .* randn(L); V = 0.2 .* randn(L - 1); q = 0.2 .* randn(L, 2)
    gates = tebd_gates(sites, dt, t_hop, Ω, V, q; α = α)

    psi = psi0
    for _ in 1:Nsteps
        psi = apply(gates, psi; cutoff = cutoff, maxdim = maxdim)
        normalize!(psi)
    end
    _, realT, symT = corr_diagnostics(psi)
    t_real = realT < 1e-10; t_sym = symT < 1e-10   # looser: accumulated Trotter/truncation

    println("\n[Part 2] after TEBD to T_end=$T_end (H_SEED=$H_SEED, α=$α, χmax=$(maxlinkdim(psi)))")
    @printf("  real (max|imag C|)             = %.2e   %s\n", realT, t_real ? "PASS" : "FAIL")
    @printf("  symmetric (max|C - Cᵀ|)        = %.2e   %s\n", symT,  t_sym  ? "PASS" : "FAIL")

    allpass = ic_real && ic_sym && ic_diag && t_real && t_sym
    println("\n", allpass ? "C2.3 PASS" : "C2.3 FAIL")
    exit(allpass ? 0 : 1)
end

main()
