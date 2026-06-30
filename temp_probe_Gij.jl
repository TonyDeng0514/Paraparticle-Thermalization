using LinearAlgebra
using Base.Threads
BLAS.set_num_threads(Threads.nthreads())

include("hilbert.jl")
include("gates.jl")
include("observable.jl")
include("product_states.jl")

outdir  = joinpath(@__DIR__, "results/temp_probe_Gij/")
mkpath(outdir)

function thermalize(α::Float64, params_file::String)
    L = 10
    t_hop = 1.0

    occ, θ_params, φ_params = read_params(params_file, L)
    N   = count(occ)
    label   = params_label(params_file, L)

    sites   = siteinds("Tri", L; conserve_qns=true)

    dt  = 0.05
    T_end   = 1600.0
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

    return nothing
end

# --- CLI entry point ---
# Guarded so this file can be `include`d (to reuse thermalize for an in-process
# alpha loop) without executing a run. See CLAUDE.md §2.
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 2
        println(stderr, "Usage: julia temp_probe_Gij.jl <alpha> <params_file>")
        println(stderr, "  (Hamiltonian disorder is fixed at H_SEED in product_states.jl)")
        println(stderr, "  e.g. julia temp_probe_Gij.jl 0.05 params/params_L10_seed42_bot7.csv")
        exit(1)
    end

    α = parse(Float64, ARGS[1])
    params_file = ARGS[2]

    thermalize(α, params_file)
end