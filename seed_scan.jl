# seed_scan.jl
#
# Phase A. The Hamiltonian disorder is FIXED at H_SEED (set in product_states.jl).
# Scan a range of product-state seeds; each seed defines one product state
# (occupancy via random_config + one random a/b angle set), and we record its energy
# under the fixed Hamiltonian. Use this to find the lowest-energy seed, then explore
# that seed's angles with energy_sweep.jl.
#
# Each output row is self-contained (occ, theta, phi columns) so the winning row can be
# saved directly as a params file. The energy for seed s here equals the FIRST sample of
# `energy_sweep.jl s ...` (same occupancy and same MersenneTwister(s) angle draw).
#
# Usage: julia seed_scan.jl <ps_seed_lo> <ps_seed_hi>

using LinearAlgebra
using Base.Threads
BLAS.set_num_threads(Threads.nthreads())

include("hilbert.jl")
include("gates.jl")
include("observable.jl")
include("product_states.jl")

function run_seed_scan()
    L     = 12
    N     = 8
    Na    = 4
    t_hop = 1.0
    W     = 0.2

    ps_lo = parse(Int, ARGS[1])
    ps_hi = parse(Int, ARGS[2])

    # --- Fixed disorder Hamiltonian ---
    Random.seed!(H_SEED)
    Ω = W .* randn(L)
    V = W .* randn(L - 1)
    q = W .* randn(L, 2)

    sites = siteinds("Tri", L; conserve_qns=true)
    H = build_hamiltonian_mpo(sites, t_hop, Ω, V, q; α=0.0)

    outdir = joinpath(@__DIR__, "results/energy_sweep/")
    mkpath(outdir)
    outfile = joinpath(outdir, "seed_scan_L$(L)_Hseed$(H_SEED)_ps$(ps_lo)-$(ps_hi).csv")

    open(outfile, "w") do io
        println(io, "# H_SEED = $H_SEED")
        header = "ps_seed," *
                 join(["occ_$j"   for j in 1:L], ",") * "," *
                 join(["theta_$j" for j in 1:L], ",") * "," *
                 join(["phi_$j"   for j in 1:L], ",") * ",energy"
        println(io, header)

        @printf("%8s   %12s\n", "ps_seed", "energy")
        for ps in ps_lo:ps_hi
            config = random_config(L, N, Na; seed=ps)
            occ = Bool[config[j] != "Vac" for j in 1:L]

            rng = MersenneTwister(ps)
            theta = zeros(Float64, L)
            phi   = zeros(Float64, L)
            for j in 1:L
                if occ[j]
                    theta[j] = (π / 2) * rand(rng)
                    phi[j]   = (2π)    * rand(rng)
                end
            end

            psi, _ = superposition_product_state(sites, occ, theta, phi)
            energy = real(inner(psi', H, psi))

            row = "$ps," *
                  join([occ[j] ? "1" : "0" for j in 1:L], ",") * "," *
                  join([@sprintf("%.10e", theta[j]) for j in 1:L], ",") * "," *
                  join([@sprintf("%.10e", phi[j])   for j in 1:L], ",") * "," *
                  @sprintf("%.10e", energy)
            println(io, row)
            @printf("%8d   %12.6f\n", ps, energy)
        end
    end
    @printf("Wrote ps_seeds %d..%d to %s\n", ps_lo, ps_hi, abspath(outfile))
end

# --- CLI entry point ---
if length(ARGS) != 2
    println(stderr, "Usage: julia seed_scan.jl <ps_seed_lo> <ps_seed_hi>")
    println(stderr, "  e.g. julia seed_scan.jl 1 500")
    exit(1)
end

run_seed_scan()
