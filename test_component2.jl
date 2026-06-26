# test_component2.jl
#
# Consolidated Component 2 acceptance gates. Run with the project env (the
# perturbed C2.3 part needs ITensors):
#
#   julia --project=. test_component2.jl [params_file]
#
# Integrable side (standalone linear algebra):
#   - C^n exact fixed-N sum rule        Σ_j C^n_ij = 0        (number conservation)
#   - C^n correctness via Kubo identity ∂n_i/∂h_j = −β C^n_ij (vs finite difference)
#   - C2.1 charge-compressibility cross-check, FIRST-MOMENT form:
#        route1 = κ_i/β = −(1/αβ)·d⟨n_i⟩/di            (profile)
#        route2 = d/di[Σ_k k·C^n_ik]                   (correlator, Kubo first moment)
#     route1 ≈ route2 to <1%. DISCRIMINATION: the same first moment from the naive
#     grand-canonical Wick form must MISS route1 by >5% (so the gate still catches a
#     wrong fixed-N correction). [The plain window sum Σ_j C^n_ij cannot be used:
#     the fixed-N constraint pins it to zero; the compressibility is the first
#     moment of the correlator, not its sum. The 0.222→0.242 gap is the finite-N
#     factor L/(L−1), i.e. the constraint, not the −|ρ_ij|² exchange term.]
#   - C2.2 integrable C^m is diagonal,  C^m_ij = δ_ij ⟨n_i⟩
#
# Perturbed side (ITensors):
#   - C2.3 connected_corr is REAL and SYMMETRIC on the product IC and on a short
#     TEBD-evolved (entangled) state; C^n_ii ≈ n_i(1−n_i) on the IC (n_i ∈ {0,1}).

include("wannier_stark_integrable.jl")   # integrable correlators (standalone)
include("hilbert.jl")                    # SiteType"Tri" (loads ITensorMPS/ITensors)
include("gates.jl")                      # tebd_gates
include("observable.jl")                 # connected_corr
include("product_states.jl")             # superposition_product_state, read_params, H_SEED
using Printf, LinearAlgebra

# grand-canonical Wick correlator, used ONLY for the C2.1 discrimination assertion
function gc_corr_n(ε, ψ, N, β)
    n_orb = orbital_occupations(ε, N, β)
    nt = (ψ .^ 2) * n_orb
    ρ  = ψ * Diagonal(n_orb) * ψ'
    C  = -(ρ .^ 2)
    for i in eachindex(nt); C[i, i] += nt[i]; end
    return C
end

first_moment_gradient(C, c, L) = sum((1:L) .* ((C[c + 1, :] .- C[c - 1, :]) ./ 2))

function n_profile_fixedbeta(L, α, β, N; hfield = ())
    h = Matrix(single_particle_matrix(L, T_HOP, α))
    for (j, hj) in hfield; h[j, j] += hj; end
    F = eigen(Hermitian(h))
    return (F.vectors .^ 2) * orbital_occupations(F.values, N, β)
end

function main()
    params = isempty(ARGS) ?
        joinpath(@__DIR__, "params/params_L12_seed42_ps188_sweepmin.csv") : ARGS[1]
    L = 12; t_hop = 1.0
    occ, θ, φ = read_params(params, L)
    N = count(occ); O = [i for i in 1:L if occ[i]]; c = L ÷ 2

    println("Component 2 acceptance — L=$L N=$N (params $(basename(params)))\n")
    ok_sr = true; ok_kubo = true; ok_c21 = true; ok_disc = true; ok_cm = true

    # ── Integrable gates ─────────────────────────────────────────────────────
    for α in (0.10, 0.25)
        β  = solve_beta(L, t_hop, α, N, α * sum(O))
        Cn = connected_corr_n_integrable(L, t_hop, α, β, N)
        Cm = connected_corr_m_integrable(L, t_hop, α, β, N)
        F  = eigen(single_particle_matrix(L, t_hop, α)); ε = F.values; ψ = F.vectors
        nt = (ψ .^ 2) * orbital_occupations(ε, N, β)

        sr = maximum(abs.(vec(sum(Cn, dims = 2)))); ok_sr &= sr < 1e-9
        dh = 1e-5
        fd = (n_profile_fixedbeta(L, α, β, N; hfield = ((c, dh),)) .-
              n_profile_fixedbeta(L, α, β, N; hfield = ((c, -dh),))) ./ (2dh)
        kubo = maximum(abs.(fd .- (-β .* Cn[:, c]))); ok_kubo &= kubo < 1e-4

        route1 = -(1 / (α * β)) * ((nt[c + 1] - nt[c - 1]) / 2)
        route2 = first_moment_gradient(Cn, c, L)
        route2_gc = first_moment_gradient(gc_corr_n(ε, ψ, N, β), c, L)
        rel_can = abs(route2 - route1) / abs(route1)
        rel_gc  = abs(route2_gc - route1) / abs(route1)
        ok_c21  &= rel_can < 0.01
        ok_disc &= rel_gc  > 0.05

        offdiag = maximum(abs.(Cm .- Diagonal(diag(Cm)))); diagerr = maximum(abs.(diag(Cm) .- nt))
        ok_cm &= offdiag < 1e-10 && diagerr < 1e-12

        @printf("α=%.2f β=%+.5e\n", α, β)
        @printf("  [C^n] sum rule %.2e | Kubo %.2e\n", sr, kubo)
        @printf("  [C2.1] route1 κ/β=%.5f  route2 first-moment=%.5f (Δ=%.2f%%)  GC route2=%.5f (Δ=%.1f%%)\n",
                route1, route2, 100rel_can, route2_gc, 100rel_gc)
        @printf("  [C2.2] C^m offdiag %.2e | diag-⟨n⟩ %.2e\n\n", offdiag, diagerr)
    end

    # ── Perturbed C2.3 ───────────────────────────────────────────────────────
    sites = siteinds("Tri", L; conserve_qns = true)
    psi0, _ = superposition_product_state(sites, occ, θ, φ)
    diagnostics(psi) = begin
        Cn = connected_corr(psi, "Nloc"); Cm = connected_corr(psi, "Mloc")
        (Cn,
         max(maximum(abs.(imag.(Cn))), maximum(abs.(imag.(Cm)))),
         max(maximum(abs.(Cn .- transpose(Cn))), maximum(abs.(Cm .- transpose(Cm)))))
    end
    Cn0, real0, sym0 = diagnostics(psi0)
    n = expect(psi0, "Nloc")
    diag_err = maximum(abs.(real.(diag(Cn0)) .- n .* (1 .- n)))
    ic_real = real0 < 1e-12; ic_sym = sym0 < 1e-12; ic_diag = diag_err < 1e-10

    dt = 0.05; T_end = 5.0; Nsteps = round(Int, T_end / dt)
    Random.seed!(H_SEED)
    Ω = 0.2 .* randn(L); V = 0.2 .* randn(L - 1); q = 0.2 .* randn(L, 2)
    gates = tebd_gates(sites, dt, t_hop, Ω, V, q; α = 0.0)
    psi = psi0
    for _ in 1:Nsteps
        psi = apply(gates, psi; cutoff = 1e-12, maxdim = 1024); normalize!(psi)
    end
    _, realT, symT = diagnostics(psi)
    t_real = realT < 1e-10; t_sym = symT < 1e-10

    @printf("[C2.3 IC]   real %.2e | sym %.2e | C^n_ii-n(1-n) %.2e\n", real0, sym0, diag_err)
    @printf("[C2.3 TEBD] real %.2e | sym %.2e  (T_end=%.1f, χmax=%d)\n\n", realT, symT, T_end, maxlinkdim(psi))

    println("="^66)
    tbl(name, ok) = @printf("%-52s %s\n", name, ok ? "PASS" : "FAIL")
    tbl("C^n fixed-N sum rule (<1e-9)",                 ok_sr)
    tbl("C^n Kubo identity ∂n/∂h=-βC^n (<1e-4)",        ok_kubo)
    tbl("C2.1 route1 vs first-moment route2 (<1%)",     ok_c21)
    tbl("C2.1 discrimination: GC misses route1 (>5%)",  ok_disc)
    tbl("C2.2 C^m diagonal = ⟨n_i⟩",                    ok_cm)
    tbl("C2.3 perturbed real/sym (IC + TEBD), C^n_ii",  ic_real && ic_sym && ic_diag && t_real && t_sym)
    println("="^66)
    allpass = ok_sr && ok_kubo && ok_c21 && ok_disc && ok_cm &&
              ic_real && ic_sym && ic_diag && t_real && t_sym
    println(allpass ? "ALL PASS" : "SOME FAILED")
    exit(allpass ? 0 : 1)
end

main()
