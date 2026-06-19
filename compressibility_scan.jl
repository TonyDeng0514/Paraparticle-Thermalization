include("hilbert.jl")
include("gates.jl")
include("observable.jl")
include("product_states.jl")   # provides superposition_product_state and read_params

function run_compressibility_scan()
    L     = 12
    params_file = ARGS[1]
    t_hop = 1.0

    occ, θ_params, φ_params = read_params(params_file, L)
    N     = count(occ)
    label = params_label(params_file, L)

    α_list =    [
                0.0, 
                0.05, 
                0.1, 
                0.15, 
                0.2, 
                0.25,
                ]

    sites = siteinds("Tri", L; conserve_qns=true)

    dt     = 0.05
    T_end  = 80.0
    Nsteps = round(Int, T_end / dt)

    cutoff = 1e-12
    maxdim = 1024

    times = collect(0:Nsteps) .* dt
    # j_mid = L ÷ 2 + 1

    println("QN sectors active: ", hasqns(sites[1]))

    outdir = joinpath(@__DIR__, "results/compressibility/")
    mkpath(outdir)

    println("\n========== H_SEED = $H_SEED ==========")

    Random.seed!(H_SEED)
    Ω = 0.2 .* randn(L)
    V = 0.2 .* randn(L - 1)
    q = 0.2 .* randn(L, 2)

    println("  Initial condition: $(params_file)  (N = $N)")

    for α in α_list
        println("\n========== α = $α ==========")

        # Fixed-N product state from the params file (N-assert is inside the helper).
        psi, _ = superposition_product_state(sites, occ, θ_params, φ_params)

        n_loc = zeros(Float64, Nsteps + 1, L)
        # m_loc = zeros(Float64, Nsteps + 1, L)

        # n_mid_t   = zeros(Float64, Nsteps + 1)

        n_loc[1, :] = expect(psi, "Nloc")
        # m_loc[1, :] = expect(psi, "Mloc")

        # n_mid_t[1]   = n_loc[1, j_mid]

        gates = tebd_gates(sites, dt, t_hop, Ω, V, q; α=α)

        H  = build_hamiltonian_mpo(sites, t_hop, Ω, V, q; α=α)
        E0 = real(inner(psi', H, psi))
        println("Initial energy: $E0")

        println("\nstep    time     bond-dim     ⟨N_tot⟩")
        for step in 1:Nsteps
            psi = apply(gates, psi; cutoff=cutoff, maxdim=maxdim)
            normalize!(psi)

            n_loc[step + 1, :] = expect(psi, "Nloc")
            # m_loc[step + 1, :] = expect(psi, "Mloc")

            # n_mid_t[step + 1]   = n_loc[step + 1, j_mid]

            @printf("%4d  %7.3f   %6d   %8.4f\n",
                    step, step * dt, maxlinkdim(psi),
                    sum(n_loc[step + 1, :]))
        end

        # save full density profile
        outfile_profile = joinpath(outdir, "n_profile_vs_t_L$(L)_chi$(maxdim)_alpha$(α)_seed$(H_SEED)_$(label).csv")
        open(outfile_profile, "w") do io
            println(io, "# params_file = $(params_file)")
            println(io, "# E0 = $E0")
            header = "time," * join(["n_$j" for j in 1:L], ",")
            println(io, header)
            for k in eachindex(times)
                row = @sprintf("%.6f", times[k]) *
                    "," * join([@sprintf("%.10e", n_loc[k, j]) for j in 1:L],
                    ",",
                    )
                println(io, row)
            end
        end
        println("Wrote $(length(times)) rows to $(abspath(outfile_profile))")
    end
end

# --- CLI entry point ---
if length(ARGS) != 1
    println(stderr, "Usage: julia compressibility_scan.jl <params_file>")
    println(stderr, "  (Hamiltonian disorder is fixed at H_SEED in product_states.jl)")
    println(stderr, "  e.g. julia compressibility_scan.jl params/params_L12_seed42_cool_01.csv")
    exit(1)
end

run_compressibility_scan()
