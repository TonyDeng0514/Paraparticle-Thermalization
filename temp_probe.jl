using LinearAlgebra
using Base.Threads
BLAS.set_num_threads(Threads.nthreads())

include("hilbert.jl")
include("gates.jl")
include("observable.jl")
include("product_states.jl")

outdir  = joinpath(@__DIR__, "results/temp_probe/")
mkpath(outdir)

function thermalize(α::Float64, params_file::String)
    L = 10
    t_hop = 1.0

    occ, θ_params, φ_params = read_params(params_file, L)
    N   = count(occ)
    label   = params_label(params_file, L)

    sites   = siteinds("Tri", L; conserve_qns=true)

    dt  = 0.05
    T_end   = 160.0
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

    println("\nstep     time    bond-dim    ⟨N_tot⟩")
    for step in 1:Nsteps
        psi = apply(gates, psi; cutoff=cutoff, maxdim=maxdim)
        normalize!(psi)
        n_loc[step + 1, :] = expect(psi, "Nloc")
        m_loc[step + 1, :] = expect(psi, "Mloc")

        println(nio, @sprintf("%.6f", times[step + 1]) * "," *
                     join([@sprintf("%.10e", n_loc[step + 1, j]) for j in 1:L], ","))
        flush(nio)

        println(mio, @sprintf("%.6f", times[step + 1]) * "," *
                     join([@sprintf("%.10e", m_loc[step + 1, j]) for j in 1:L], ","))
        flush(mio)

        @printf("%4d    %7.3f        %6d                %8.4f\n",
                step, step * dt, maxlinkdim(psi), sum(n_loc[step + 1, :]))
    end
    close(nio)
    close(mio)

    return (; psi, sites, gates, dt, α, n_loc, m_loc, label, N, E0)
end

function two_time_correlator(th; T_corr::Float64 = 160.0)
    sites = th.sites
    psi   = th.psi
    gates = th.gates
    dt    = th.dt

    L      = length(sites)
    cutoff = 1e-12
    maxdim = 1024
    n_corr = round(Int, T_corr / dt)
    times  = collect(0:n_corr) .* dt          # LOCAL axis, starts at 0

    # total magnetization MPO, built on the SAME sites (never re-mint siteinds)
    osM = OpSum()
    for i in 1:L
        osM += "Mloc", i
    end
    M_mpo = MPO(osM, sites)

    Mexp  = real(inner(psi', M_mpo, psi))                 # ⟨M⟩  (constant subtraction)
    Cm_eq = sum(real.(connected_corr(psi, "Mloc")))       # Var(M) = Σ_ij C^m_ij, independent route

    # |φ⟩ = M|ψ⟩ — correlator phase begins. NO normalize! on ψ or φ past here.
    φ      = apply(M_mpo, psi; cutoff=cutoff, maxdim=maxdim)
    φ0norm = norm(φ)

    raw   = Vector{ComplexF64}(undef, n_corr + 1)   # ⟨ψ(t)|M|φ(t)⟩  (you form C = raw - Mexp² in the nb)
    Mt    = zeros(Float64, n_corr + 1)              # ⟨M(t)⟩ stationarity check
    χψ    = zeros(Int, n_corr + 1)
    χφ    = zeros(Int, n_corr + 1)
    φloss = zeros(Float64, n_corr + 1)

    # t = 0 junction (no evolution yet): ⟨ψ|M|φ⟩ = ⟨ψ|M·M|ψ⟩ = ⟨M²⟩ ⇒ C(0) = Var(M)
    raw[1]   = inner(psi', M_mpo, φ)
    Mt[1]    = real(inner(psi', M_mpo, psi)) / real(inner(psi, psi))
    χψ[1]    = maxlinkdim(psi)
    χφ[1]    = maxlinkdim(φ)
    φloss[1] = 0.0

    C0     = real(raw[1]) - Mexp^2
    reldev = abs(C0 - Cm_eq) / abs(Cm_eq)
    @printf("\nC(0) = %.6f   Var(M) = %.6f   reldev = %.2f%%   %s\n",
            C0, Cm_eq, 100 * reldev, reldev < 0.01 ? "PASS" : "CHECK")


    α     = th.α
    label = th.label

    cfile = joinpath(outdir,
        "Ct_L$(L)_chi$(maxdim)_alpha$(α)_seed$(H_SEED)_$(label).csv")
    cio = open(cfile, "w")
    println(cio, "# label = $(label)   alpha = $α   <M> = $Mexp")
    println(cio, "# C(0) = $C0   Var(M) = $Cm_eq   reldev = $reldev")
    println(cio, "# C(t) = complex(Re_raw,Im_raw) - <M>^2 ;  C(-t)=conj(C(t)) before FT")
    println(cio, "time,Re_raw,Im_raw,chi_psi,chi_phi,phi_normloss,M_t")
    @printf(cio, "%.4f,%.10e,%.10e,%d,%d,%.6e,%.10e\n",
            times[1], real(raw[1]), imag(raw[1]), χψ[1], χφ[1], φloss[1], Mt[1])
    flush(cio)

    println("\nstep     time      χ_ψ     χ_φ     φ_normloss      ⟨M(t)⟩")
    for step in 1:n_corr
        psi = apply(gates, psi; cutoff=cutoff, maxdim=maxdim)
        φ   = apply(gates, φ;   cutoff=cutoff, maxdim=maxdim)

        raw[step+1]   = inner(psi', M_mpo, φ)
        Mt[step+1]    = real(inner(psi', M_mpo, psi)) / real(inner(psi, psi))
        χψ[step+1]    = maxlinkdim(psi)
        χφ[step+1]    = maxlinkdim(φ)
        φloss[step+1] = 1 - norm(φ) / φ0norm

        @printf(cio, "%.4f,%.10e,%.10e,%d,%d,%.6e,%.10e\n",
                times[step+1], real(raw[step+1]), imag(raw[step+1]),
                χψ[step+1], χφ[step+1], φloss[step+1], Mt[step+1])
        flush(cio)

        @printf("%4d   %7.3f   %6d  %6d     %.3e    %8.4f\n",
                step, times[step+1], χψ[step+1], χφ[step+1], φloss[step+1], Mt[step+1])
    end
    close(cio)
    return (; times, raw, Mexp, Cm_eq, C0, Mt, χψ, χφ, φloss)
end

# --- CLI entry point ---
# Guarded so this file can be `include`d (to reuse thermalize/two_time_correlator
# for an in-process alpha loop) without executing a run. See CLAUDE.md §2.
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 2
        println(stderr, "Usage: julia temp_probe.jl <alpha> <params_file>")
        println(stderr, "  (Hamiltonian disorder is fixed at H_SEED in product_states.jl)")
        println(stderr, "  e.g. julia temp_probe.jl 0.05 params/params_L12_seed42_cool_01.csv")
        exit(1)
    end

    α = parse(Float64, ARGS[1])
    params_file = ARGS[2]

    th = thermalize(α, params_file)
    two_time_correlator(th)
end