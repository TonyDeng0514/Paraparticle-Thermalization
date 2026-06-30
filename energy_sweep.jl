# energy_sweep.jl
#
# Phase B. The Hamiltonian disorder is FIXED at H_SEED (set in product_states.jl).
# Take the occupancy defined by a single product-state seed and sweep the a/b flavor
# angles, recording each state's energy under the fixed Hamiltonian. Run this after
# seed_scan.jl identifies a low-energy seed: confirm the energy stays low across angle
# variations, then pick a row to save as a params file.
#
# Usage: julia energy_sweep.jl <ps_seed> <n_samples>

using LinearAlgebra
using Base.Threads
BLAS.set_num_threads(Threads.nthreads())

include("hilbert.jl")
include("gates.jl")
include("observable.jl")
include("product_states.jl")

function run_energy_sweep()
    L     = 10
    N     = 7
    Na    = 3
    t_hop = 1.0
    W     = 0.2

    ps_seed   = parse(Int, ARGS[1])
    n_samples = parse(Int, ARGS[2])

    # --- Fixed disorder Hamiltonian ---
    Random.seed!(H_SEED)
    Ω = W .* randn(L)
    V = W .* randn(L - 1)
    q = W .* randn(L, 2)

    # --- Occupancy skeleton from the product-state seed (sweep varies only angles) ---
    config = random_config(L, N, Na; seed=ps_seed)
    occ = Bool[config[j] != "Vac" for j in 1:L]
    @assert count(occ) == N "occupancy count mismatch"

    sites = siteinds("Tri", L; conserve_qns=true)

    # One MPO, reused for the DMRG reference and every energy evaluation.
    H = build_hamiltonian_mpo(sites, t_hop, Ω, V, q; α=0.0)

    # --- Reference ground-state energy of H(H_SEED) in the N-sector (yardstick only) ---
    psi0 = MPS(sites, config)
    E0_dmrg, _ = dmrg(H, psi0; nsweeps=6,
                      maxdim=[20, 40, 256], cutoff=[1e-6, 1e-8, 1e-10],
                      outputlevel=0)
    @printf("E0_dmrg (N=%d, H_SEED=%d) = %.8f\n", N, H_SEED, E0_dmrg)

    # Angle RNG, seeded by the product-state seed (reproducible).
    rng = MersenneTwister(ps_seed)

    outdir = joinpath(@__DIR__, "results/energy_sweep/")
    mkpath(outdir)
    outfile = joinpath(outdir, "sweep_L$(L)_Hseed$(H_SEED)_ps$(ps_seed).csv")

    open(outfile, "w") do io
        println(io, "# H_SEED = $H_SEED")
        println(io, "# ps_seed = $ps_seed")
        println(io, "# E0_dmrg = $E0_dmrg")
        println(io, "# occ = " * join(Int.(occ), ""))
        header = join(["occ_$j"   for j in 1:L], ",") * "," *
                 join(["theta_$j" for j in 1:L], ",") * "," *
                 join(["phi_$j"   for j in 1:L], ",") * ",energy"
        println(io, header)

        for _ in 1:n_samples
            theta = zeros(Float64, L)
            phi   = zeros(Float64, L)
            for j in 1:L
                if occ[j]
                    theta[j] = (π / 2) * rand(rng)   # a/b mixing angle in [0, π/2]
                    phi[j]   = (2π)    * rand(rng)    # relative phase in [0, 2π)
                end
            end

            psi, _ = superposition_product_state(sites, occ, theta, phi)
            energy = real(inner(psi', H, psi))

            row = join([occ[j] ? "1" : "0" for j in 1:L], ",") * "," *
                  join([@sprintf("%.10e", theta[j]) for j in 1:L], ",") * "," *
                  join([@sprintf("%.10e", phi[j])   for j in 1:L], ",") * "," *
                  @sprintf("%.10e", energy)
            println(io, row)
        end
    end

    @printf("Wrote %d samples to %s\n", n_samples, abspath(outfile))
end

# --- CLI entry point ---
if length(ARGS) != 2
    println(stderr, "Usage: julia energy_sweep.jl <ps_seed> <n_samples>")
    println(stderr, "  e.g. julia energy_sweep.jl 7 10000")
    exit(1)
end

run_energy_sweep()
