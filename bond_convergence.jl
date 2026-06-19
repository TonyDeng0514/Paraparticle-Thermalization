using LinearAlgebra
using Base.Threads
BLAS.set_num_threads(Threads.nthreads())


include("hilbert.jl")
include("gates.jl")
include("observable.jl")

function run_bond_convergence(χ::Int)
    L     = 12
    N     = 8
    Na    = 4
    seed  = 1234
    t_hop = 1.0
    W     = 0.2

    Random.seed!(seed)
    Ω = W .* randn(L)
    V = W .* randn(L - 1)
    q = W .* randn(L, 2)

    sites = siteinds("Tri", L; conserve_qns=true)

    dt     = 0.05
    T_end  = 40.0
    Nsteps = round(Int, T_end / dt)

    cutoff = 1e-12

    times = collect(0:Nsteps) .* dt
    j_mid = L ÷ 2 + 1

    gates = tebd_gates(sites, dt, t_hop, Ω, V, q)
    H = build_hamiltonian_mpo(sites, t_hop, Ω, V, q)

    begin
        println("\n========== χ = $χ ==========")

        state_labels = random_config(L, N, Na, seed=seed)
        psi = MPS(sites, state_labels)
        @assert sum(expect(psi, "Nloc")) ≈ N "Initial N wrong"
        println("QN sectors active: ", hasqns(sites[1]))

        n_loc = zeros(Float64, Nsteps + 1, L)
        # m_loc = zeros(Float64, Nsteps + 1, L)
        trunc_err = zeros(Float64, Nsteps + 1)

        S_mid = zeros(Float64, Nsteps + 1)

        energy = zeros(Float64, Nsteps + 1)

        n_loc[1, :] = expect(psi, "Nloc")
        # m_loc[1, :] = expect(psi, "Mloc")

        E0 = real(inner(psi', H, psi))
        println("Initial energy: $E0")
        energy[1] = measure_energy_bonds(psi, sites, t_hop, Ω, V, q)
        @assert isapprox(energy[1], E0; rtol=1e-8) "Bond energy mismatch: bond=$(energy[1]), MPO=$(E0), diff=$(abs(energy[1]-E0))"

        println("\nstep    time     bond-dim     ⟨N_tot⟩    S_mid   energy    truncation_error")
        for step in 1:Nsteps
            psi = apply(gates, psi; cutoff=cutoff, maxdim=χ)
            trunc_err[step + 1]  = abs(1.0 - norm(psi))
            normalize!(psi)

            orthogonalize!(psi, j_mid)
            _, S, _, _ = svd(psi[j_mid], (linkinds(psi, j_mid-1)..., siteinds(psi, j_mid)...))
            SvN = 0.0
            for n in 1:dim(S, 1)
                p = S[n,n]^2
                SvN -= p * log(p)
            end
            S_mid[step + 1]  = SvN

            n_loc[step + 1, :] = expect(psi, "Nloc")
            # m_loc[step + 1, :] = expect(psi, "Mloc")

            energy[step + 1] = measure_energy_bonds(psi, sites, t_hop, Ω, V, q)

            @printf("%4d  %7.3f   %6d   %8.4f   %8.4f   %8.4f   %.2e\n",
                    step, 
                    step * dt, 
                    maxlinkdim(psi),
                    sum(n_loc[step + 1, :]),
                    S_mid[step + 1],
                    energy[step + 1],
                    trunc_err[step + 1],
                    )
        end

        # save full density profile
        outdir = joinpath(@__DIR__, "results/bond_convergence/")
        mkpath(outdir)
        outfile_profile = joinpath(outdir, "n_profile_vs_t_L$(L)_chi$(χ).csv")
        open(outfile_profile, "w") do io
            println(io, "# E0 = $E0")
            header = "time,energy,S_mid,trunc_err," * join(["n_$j" for j in 1:L], ",")
            println(io, header)
            for k in eachindex(times)
                row = @sprintf("%.6f", times[k]) *
                    "," * @sprintf("%.10e", energy[k]) *
                    "," * @sprintf("%.10e", S_mid[k]) *
                    "," * @sprintf("%.10e", k == 1 ? 0.0 : trunc_err[k]) *
                    "," * join([@sprintf("%.10e", n_loc[k, j]) for j in 1:L], ",")
                println(io, row)
            end
        end
        println("Wrote $(length(times)) rows to $(abspath(outfile_profile))")
    end
end

# --- CLI entry point ---
if length(ARGS) != 1
    println(stderr, "Usage: julia bond_convergence.jl <chi>")
    println(stderr, "  e.g. julia bond_convergence.jl 64")
    exit(1)
end
 
χ = tryparse(Int, ARGS[1])
if isnothing(χ) || χ <= 0
    println(stderr, "Error: chi must be a positive integer, got '$(ARGS[1])'")
    exit(1)
end
 
run_bond_convergence(χ)
