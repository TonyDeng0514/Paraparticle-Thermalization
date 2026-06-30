using LinearAlgebra
using Base.Threads
BLAS.set_num_threads(Threads.nthreads())

include("hilbert.jl")
include("gates.jl")
include("observable.jl")
include("product_states.jl")

outdir  = joinpath(@__DIR__, "results/temp_probe_Cij/")
mkpath(outdir)

function thermalize(α::Float64, params_file::String; T_end::Float64 = 1000.0)
    L = 10
    t_hop = 1.0

    occ, θ_params, φ_params = read_params(params_file, L)
    N   = count(occ)
    label   = params_label(params_file, L)

    sites   = siteinds("Tri", L; conserve_qns=true)

    dt  = 0.05
    # T_end default 1000: M(t) fast relaxation is complete by ~1000 for all alpha
    Nsteps  = round(Int, T_end / dt)

    cutoff  = 1e-12
    maxdim  = 1024

    times   = collect(0:Nsteps) .* dt

    println("QN sectors active: ", hasqns(sites[1]))

    println("\n========== H_SEED = $H_SEED ==========")

    Random.seed!(H_SEED)
    Ω   = 0.2 .* randn(L)
    V   = 0.2 .* randn(L - 1)
    q   = 0.2 .* randn(L, 2)

    println("   Initial condition: $(params_file)   (N = $N)")

    psi, _ = superposition_product_state(sites, occ, θ_params, φ_params)

    n_loc   = zeros(Float64, Nsteps + 1, L)
    m_loc   = zeros(Float64, Nsteps + 1, L)
    n_loc[1, :] = expect(psi, "Nloc")
    m_loc[1, :] = expect(psi, "Mloc")

    gates = tebd_gates(sites, dt, t_hop, Ω, V, q; α=α)

    H   = build_hamiltonian_mpo(sites, t_hop, Ω, V, q; α=α)
    E0  = real(inner(psi', H, psi))
    println("Initial energy: $E0")

    nfile = joinpath(outdir,
        "n_profile_vs_t_L$(L)_chi$(maxdim)_alpha$(α)_seed$(H_SEED)_$(label).csv")
    nio = open(nfile, "w")
    println(nio, "# params_file = $(params_file)")
    println(nio, "# alpha = $α   E0 = $E0")
    println(nio, "time," * join(["n_$j" for j in 1:L], ","))
    println(nio, @sprintf("%.6f", times[1]) * "," *
                 join([@sprintf("%.10e", n_loc[1, j]) for j in 1:L], ","))
    flush(nio)

    mfile = joinpath(outdir,
        "m_profile_vs_t_L$(L)_chi$(maxdim)_alpha$(α)_seed$(H_SEED)_$(label).csv")
    mio = open(mfile, "w")
    println(mio, "# params_file = $(params_file)")
    println(mio, "# alpha = $α   E0 = $E0")
    println(mio, "time," * join(["m_$j" for j in 1:L], ","))
    println(mio, @sprintf("%.6f", times[1]) * "," *
                 join([@sprintf("%.10e", m_loc[1, j]) for j in 1:L], ","))
    flush(mio)

    nnfile = joinpath(outdir,
        "nn_corr_vs_t_L$(L)_chi$(maxdim)_alpha$(α)_seed$(H_SEED)_$(label).csv")
    nnio = open(nnfile, "w")
    println(nnio, "# params_file = $(params_file)")
    println(nnio, "# alpha = $α   E0 = $E0")
    println(nnio, "# C_ij(t) = <n_i n_j> - <n_i><n_j> (connected), row-major flatten of LxL (i outer, j inner)")
    println(nnio, "time," * join(["c_$(i)_$(j)" for i in 1:L for j in 1:L], ","))

    C0 = real.(connected_corr(psi, "Nloc"))
    println(nnio, @sprintf("%.6f", times[1]) * "," *
                join([@sprintf("%.10e", C0[i, j]) for i in 1:L for j in 1:L], ","))
    flush(nnio)

    println("\nstep     time    bond-dim    ⟨N_tot⟩")
    for step in 1:Nsteps
        psi = apply(gates, psi; cutoff=cutoff, maxdim=maxdim)
        normalize!(psi)
        n_loc[step + 1, :] = expect(psi, "Nloc")
        m_loc[step + 1, :] = expect(psi, "Mloc")
        C = real.(connected_corr(psi, "Nloc"))

        println(nio, @sprintf("%.6f", times[step + 1]) * "," *
                     join([@sprintf("%.10e", n_loc[step + 1, j]) for j in 1:L], ","))
        flush(nio)

        println(mio, @sprintf("%.6f", times[step + 1]) * "," *
                     join([@sprintf("%.10e", m_loc[step + 1, j]) for j in 1:L], ","))
        flush(mio)

        println(nnio, @sprintf("%.6f", times[step + 1]) * "," *
                     join([@sprintf("%.10e", C[i, j]) for i in 1:L for j in 1:L], ","))
        flush(nnio)

        @printf("%4d    %7.3f        %6d                %8.4f\n",
                step, step * dt, maxlinkdim(psi), sum(n_loc[step + 1, :]))
    end
    close(nio)
    close(mio)
    close(nnio)

    # hand the thermalized state to the two-time probe (psi is normalized here)
    return (; psi, sites, gates, dt, α, label, N, E0)
end

# φ = O_j |ψ⟩ for an on-site operator O at site j (local apply; preserves bond dim & QNs)
function apply_onsite(opname::String, sites, j::Int, psi::MPS)
    φ = copy(psi)
    orthogonalize!(φ, j)
    φ[j] = noprime(op(opname, sites[j]) * φ[j])
    return φ
end

function onsite_mpo(opname::String, sites, i::Int)
    os = OpSum()
    os += opname, i
    return MPO(os, sites)
end

# Unequal-time (KMS) probe from the thermalized state.  Field-free (INV-5): both ψ and
# every source state evolve under the SAME unperturbed gates.  Measures the full density
# two-time matrix raw_ij(t)=⟨ψ(t)|n_i|φ_j(t)⟩ with φ_j=n_j|ψ⟩, plus ⟨n_i(t)⟩, plus the
# global-M two-time correlator (the validated thermometer).  Connected forms are built in
# post-processing (08): G_ij(t)=raw_ij(t)-nbar_i*nbar_j ;  C^M(t)=raw^M(t)-⟨M⟩².
function two_time_correlator(th; T_corr::Float64 = 160.0, loss_tol::Float64 = 1e-2)
    sites = th.sites; psi = th.psi; gates = th.gates; dt = th.dt
    α = th.α; label = th.label; E0 = th.E0
    L = length(sites)
    cutoff = 1e-12
    maxdim = 1024
    n_corr = round(Int, T_corr / dt)
    times  = collect(0:n_corr) .* dt          # LOCAL axis, starts at 0

    # equal-time thermal references (for connected subtraction + the t=0 gate)
    nbar = expect(psi, "Nloc")
    Ceq  = real.(connected_corr(psi, "Nloc"))

    # global magnetization M = Σ_i Mloc_i  (validated KMS thermometer)
    osM = OpSum(); for i in 1:L; osM += "Mloc", i; end
    M_mpo = MPO(osM, sites)
    Mexp  = real(inner(psi', M_mpo, psi))
    Cm_eq = sum(real.(connected_corr(psi, "Mloc")))

    # single-site Nloc MPOs for the cross-state elements ⟨ψ|n_i|φ_j⟩
    Nmpo = [onsite_mpo("Nloc", sites, i) for i in 1:L]

    # source states: φ_j = n_j|ψ⟩ (j=1..L) and φ_{L+1} = M|ψ⟩
    phis = Vector{MPS}(undef, L + 1)
    for j in 1:L
        phis[j] = apply_onsite("Nloc", sites, j, psi)
    end
    phis[L + 1] = apply(M_mpo, psi; cutoff=cutoff, maxdim=maxdim)
    φ0 = norm.(phis)

    # NO normalize! on ψ or any φ past here — it would distort the correlator amplitudes.

    gcols = String[]
    for i in 1:L, j in 1:L
        push!(gcols, "Re_g_$(i)_$(j)"); push!(gcols, "Im_g_$(i)_$(j)")
    end

    gfile = joinpath(outdir,
        "Gn_unequal_L$(L)_chi$(maxdim)_alpha$(α)_seed$(H_SEED)_$(label).csv")
    gio = open(gfile, "w")
    println(gio, "# label = $(label)   alpha = $α   E0 = $E0   T_corr = $T_corr")
    println(gio, "# density two-time raw_ij(t) = <psi(t)| n_i |phi_j(t)>,  phi_j = n_j|psi_th>")
    println(gio, "# connected:  G_ij(t) = raw_ij(t) - nbar_i*nbar_j ;  G_ij(-t) = conj(G_ji(t)) before FT")
    println(gio, "# row-major i outer, j inner;  nbar = " * join([@sprintf("%.10e", nbar[i]) for i in 1:L], ","))
    println(gio, "time," * join(gcols, ",") * "," *
                 join(["n_$j" for j in 1:L], ",") * ",chi_psi,chi_phi_max,phi_normloss_max")

    # global-M two-time file, format-compatible with temp_probe.jl (so load_ct works)
    cfile = joinpath(outdir,
        "Ct_L$(L)_chi$(maxdim)_alpha$(α)_seed$(H_SEED)_$(label).csv")
    cio = open(cfile, "w")

    # --- t = 0 junction (no evolution yet) -------------------------------------------
    rawM0 = inner(psi', M_mpo, phis[L + 1])                 # ⟨M²⟩
    rawM  = Ref(rawM0)                                       # mutable cell read by write_row!
    C0    = real(rawM0) - Mexp^2
    reldev = abs(C0 - Cm_eq) / abs(Cm_eq)

    rawG = Matrix{ComplexF64}(undef, L, L)
    @threads for j in 1:L
        for i in 1:L
            rawG[i, j] = inner(psi', Nmpo[i], phis[j])      # ⟨n_i n_j⟩
        end
    end
    G0   = real.(rawG) .- nbar * nbar'
    gate = maximum(abs.(G0 .- Ceq))
    @printf("\nt=0 gates:  density max|G_ij(0)-C^eq_ij| = %.2e %s   |   M: C(0)=%.6f Var(M)=%.6f reldev=%.2f%% %s\n",
            gate, gate < 1e-8 ? "PASS" : "CHECK",
            C0, Cm_eq, 100 * reldev, reldev < 0.01 ? "PASS" : "CHECK")

    println(cio, "# label = $(label)   alpha = $α   <M> = $Mexp")
    println(cio, "# C(0) = $C0   Var(M) = $Cm_eq   reldev = $reldev")
    println(cio, "# C(t) = complex(Re_raw,Im_raw) - <M>^2 ;  C(-t)=conj(C(t)) before FT")
    println(cio, "time,Re_raw,Im_raw,chi_psi,chi_phi,phi_normloss,M_t")

    function write_row!(s)
        nt_row    = expect(psi, "Nloc")
        Mt_val    = real(inner(psi', M_mpo, psi)) / real(inner(psi, psi))
        chi_psi   = maxlinkdim(psi)
        chi_phi_n = maximum(maxlinkdim(phis[j]) for j in 1:L)
        ploss_n   = maximum(1 - norm(phis[j]) / φ0[j] for j in 1:L)
        chi_phi_M = maxlinkdim(phis[L + 1])
        ploss_M   = 1 - norm(phis[L + 1]) / φ0[L + 1]

        gvals = String[]
        for i in 1:L, j in 1:L
            push!(gvals, @sprintf("%.10e", real(rawG[i, j])))
            push!(gvals, @sprintf("%.10e", imag(rawG[i, j])))
        end
        println(gio, @sprintf("%.4f", times[s]) * "," * join(gvals, ",") * "," *
                     join([@sprintf("%.10e", nt_row[j]) for j in 1:L], ",") * "," *
                     @sprintf("%d,%d,%.6e", chi_psi, chi_phi_n, ploss_n))
        flush(gio)

        @printf(cio, "%.4f,%.10e,%.10e,%d,%d,%.6e,%.10e\n",
                times[s], real(rawM[]), imag(rawM[]), chi_psi, chi_phi_M, ploss_M, Mt_val)
        flush(cio)

        return chi_phi_n, ploss_n
    end

    cpn, pln = write_row!(1)
    @printf("step     time     chi_psi  chi_phi_max  phi_normloss_max\n")
    @printf("%4d   %7.3f   %6d   %6d       %.3e\n", 0, times[1], maxlinkdim(psi), cpn, pln)

    for step in 1:n_corr
        psi = apply(gates, psi; cutoff=cutoff, maxdim=maxdim)
        @threads for k in 1:(L + 1)
            phis[k] = apply(gates, phis[k]; cutoff=cutoff, maxdim=maxdim)
        end

        @threads for j in 1:L
            for i in 1:L
                rawG[i, j] = inner(psi', Nmpo[i], phis[j])
            end
        end
        rawM[] = inner(psi', M_mpo, phis[L + 1])

        cpn, pln = write_row!(step + 1)
        @printf("%4d   %7.3f   %6d   %6d       %.3e\n",
                step, times[step + 1], maxlinkdim(psi), cpn, pln)

        if pln > loss_tol
            @printf("\nφ norm-loss %.2e exceeds tol %.0e at t = %.1f -> stopping probe (T_trust).\n",
                    pln, loss_tol, times[step + 1])
            break
        end
    end
    close(gio)
    close(cio)
    return nothing
end

# --- CLI entry point ---
# Guarded so this file can be `include`d (to reuse thermalize/two_time_correlator
# for an in-process alpha loop) without executing a run. See CLAUDE.md §2.
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println(stderr, "Usage: julia temp_probe_Cij.jl <alpha> <params_file> [T_end] [T_corr]")
        println(stderr, "  (Hamiltonian disorder is fixed at H_SEED in product_states.jl)")
        println(stderr, "  T_end defaults to 1000.0 (thermalization), T_corr to 160.0 (probe).")
        println(stderr, "  e.g. julia temp_probe_Cij.jl 0.05 params/params_L10_seed42_bot7.csv")
        println(stderr, "       julia temp_probe_Cij.jl 0.01 params/params_L10_seed42_bot7.csv 80 80")
        exit(1)
    end

    α = parse(Float64, ARGS[1])
    params_file = ARGS[2]
    T_end  = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1000.0
    T_corr = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 160.0

    th = thermalize(α, params_file; T_end=T_end)
    two_time_correlator(th; T_corr=T_corr)
end
