using LinearAlgebra
using Base.Threads
BLAS.set_num_threads(Threads.nthreads())



include("hilbert.jl")
include("gates.jl")
include("observable.jl")



function run_thermalization_time(L, N, Na)
    seed    = 1234
    t_hop   = 1.0
    dt      = 0.005
    T_end   = 10.0
    Nsteps  = round(Int, T_end / dt)

    cutoff  = 1e-12
    χ       = 1024

    # times   = collect(0:Nsteps) .* dt

    outdir  = joinpath(@__DIR__, "results/thermalization/")
    mkpath(outdir)

    println("\n========== L = $L ==========")
    sites = siteinds("Tri", L; conserve_qns=true)
    println("QN sectors active: ", hasqns(sites[1]))
    Random.seed!(seed)
    Ω = 0.2 .* randn(L)
    V = 0.2 .* randn(L - 1)
    q = 0.2 .* randn(L, 2)
    state_labels = random_config(L, N, Na; seed=seed)
    psi = MPS(sites, state_labels)
    @assert sum(expect(psi, "Nloc")) ≈ N "Initial N wrong"
    
    j_mid   = L ÷ 2 + 1

    gates   = tebd_gates(sites, dt, t_hop, Ω, V, q)
    H       = build_hamiltonian_mpo(sites, t_hop, Ω, V, q)

    n_loc_0 = expect(psi, "Nloc")
    E0      = real(inner(psi', H, psi))
    println("Initial energy: $E0")
    energy_0 = measure_energy_bonds(psi, sites, t_hop, Ω, V, q)
    @assert isapprox(energy_0, E0; rtol=1e-8) "Bond energy mismatch: bond=$(energy_0), MPO=$(E0), diff=$(abs(energy_0-E0))"

    outfile_profile = joinpath(outdir, "n_profile_vs_t_L$(L).csv")
    io = open(outfile_profile, "w")
    println(io, "# E0 = $E0")
    header = "time, energy, S_mid," * join(["n_$j" for j in 1:L], ",")
    println(io, header)
    row0 = @sprintf("%.6f", 0.0) *
        "," * @sprintf("%.10e", energy_0) *
        "," * @sprintf("%.10e", 0.0) *
        "," * join([@sprintf("%.10e", n_loc_0[j]) for j in 1:L], ",")
    println(io, row0)
    flush(io)

    println("\nstep    time     bond-dim    S_mid   energy")
    for step in 1:Nsteps
        psi = apply(gates, psi; cutoff=cutoff, maxdim=χ)
        normalize!(psi)

        orthogonalize!(psi, j_mid)
        _, S, _, _ = svd(psi[j_mid], (linkinds(psi, j_mid-1)..., siteinds(psi, j_mid)...))
        SvN = 0.0
        for n in 1:dim(S,1)
            p = S[n,n]^2
            SvN -= p * log2(p)
        end

        n_loc_t = expect(psi, "Nloc")
        energy_t = measure_energy_bonds(psi, sites, t_hop, Ω, V, q)

        row = @sprintf("%.6f", step * dt) *
            "," * @sprintf("%.10e", energy_t) *
            "," * @sprintf("%.10e", SvN) *
            "," * join([@sprintf("%.10e", n_loc_t[j]) for j in 1:L], ",")
        println(io, row)
        flush(io)

        @printf("%4d  %7.3f   %6d   %8.4f   %8.4f\n",
                step,
                step * dt,
                maxlinkdim(psi),
                SvN,
                energy_t,
                )
    end
    close(io)
end

if length(ARGS) != 3
    println(stderr, "Usage: julia thermalization_time.jl <L> <N> <Na>")
    println(stderr, "Example: julia thermalization_time.jl 10 7 3")
    exit(1)
end

L   = tryparse(Int, ARGS[1])
N   = tryparse(Int, ARGS[2])
Na  = tryparse(Int, ARGS[3])

if isnothing(L) || isnothing(N) || isnothing(Na)
    println(stderr, "Error: L, N, Na must be integers")
    exit(1)
end

if N > L || Na > N
    println(stderr, "Error: number of particles has to be less than site length")
    exit(1)
end

run_thermalization_time(L, N, Na)