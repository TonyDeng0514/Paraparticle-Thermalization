# integrable_at_beta.jl
#
# Reverse driver for the integrable Wannier-Stark pipeline: given a β-grid,
# build the canonical thermal state at each (α, β) and write the SAME
# alpha,beta,E0,site,n_total,n_A,n_B schema as wannier_stark_integrable.jl.
#
# This is the β-in / state-out counterpart of the E0→β scan. It exists so that
# a β measured elsewhere (component 4 FDT thermometer at L=12, or component 3
# ED-exact at L=10) can be fed in to build the matched-β integrable benchmark
# for the perturbed-vs-integrable comparison (component 5).
#
# NOTE on the E0 column: here we go β → state, so there is no initial-product
# state energy being matched. The `E0` column therefore holds the integrable
# state's OWN thermal energy E = Σ_a ε_a ⟨n_a⟩_N at that (α, β). (For the
# E0→β scan in wannier_stark_integrable.jl, E0 == E by construction anyway.)
#
# Usage:
#   julia integrable_at_beta.jl <params_file> <betas>
#     <params_file>  a params CSV; only the occ_* columns are read (→ L, N).
#     <betas>        either a comma-separated list, e.g. "-1.0,0.0,0.5",
#                    or a path to a CSV that has a `beta` column (e.g. a
#                    component-4 thermometer output). Duplicate β are kept once.

include("wannier_stark_integrable.jl")

"""
    read_betas(arg) -> Vector{Float64}

`arg` is a comma-separated list of β values, or a path to a CSV with a `beta`
column. Comment (`#`) and blank lines are skipped; β values are de-duplicated
while preserving order.
"""
function read_betas(arg::AbstractString)
    raw = Float64[]
    if isfile(arg)
        lines = filter(l -> !startswith(strip(l), "#") && !isempty(strip(l)),
                       readlines(arg))
        @assert length(lines) >= 2 "β CSV $arg needs a header and >=1 data row"
        header = strip.(split(strip(lines[1]), ","))
        j = findfirst(==("beta"), header)
        @assert j !== nothing "no `beta` column in $arg"
        for k in 2:length(lines)
            push!(raw, parse(Float64, strip(split(strip(lines[k]), ",")[j])))
        end
    else
        for tok in split(arg, ",")
            push!(raw, parse(Float64, strip(tok)))
        end
    end
    seen = Set{Float64}()
    return [b for b in raw if !(b in seen) && (push!(seen, b); true)]
end

function run_at_beta()
    @assert length(ARGS) >= 2 "usage: julia integrable_at_beta.jl <params_file> <betas|csv>"
    params_file = ARGS[1]
    betas = read_betas(ARGS[2])

    occ = read_occupancy(params_file)
    L   = length(occ)
    N   = count(occ)
    label = label_from_path(params_file, L)

    println("Integrable Wannier-Stark thermal state at fixed β")
    println("  params_file = $(params_file)")
    println("  L = $L, N = $N, t = $T_HOP")
    println("  β grid      = $(betas)")
    println("  α grid      = $(α_LIST)")
    println()

    outdir = joinpath(@__DIR__, "results/ws_integrable")
    mkpath(outdir)
    outfile = joinpath(outdir, "ws_integrable_atbeta_L$(L)_N$(N)_$(label).csv")

    rows = Vector{NTuple{7, Any}}()   # (alpha, beta, E, site, n_total, n_A, n_B)

    @printf("%-6s  %10s  %10s  %8s  %10s\n", "alpha", "beta", "E", "Σn_i", "Σn_{i,A}")
    for α in α_LIST, β in betas
        st = integrable_thermal_at_beta(L, T_HOP, α, β, N)
        @printf("%-6.2f  %10.4f  %10.4f  %8.5f  %10.5f\n",
                α, β, st.E, sum(st.n_tot), sum(st.n_A))
        @assert isapprox(sum(st.n_tot), N; atol = 1e-9) "Σ⟨n_i⟩ ≠ N at (α=$α, β=$β)"
        for i in 1:L
            push!(rows, (α, β, st.E, i, st.n_tot[i], st.n_A[i], st.n_B[i]))
        end
    end

    open(outfile, "w") do io
        println(io, "# integrable Wannier-Stark canonical thermal state at fixed β")
        println(io, "# params_file = $(params_file)")
        println(io, "# L=$L N=$N t=$T_HOP label=$label")
        println(io, "# E0 column = integrable thermal energy E at (alpha,beta)")
        println(io, "# flavor split: n_A = n_B = n_total/2 (exact, A<->B symmetry)")
        println(io, "alpha,beta,E0,site,n_total,n_A,n_B")
        for (α, β, E, i, nt, na, nb) in rows
            @printf(io, "%.4f,%.10e,%.10e,%d,%.10e,%.10e,%.10e\n",
                    α, β, E, i, nt, na, nb)
        end
    end

    println()
    println("Wrote $(length(rows)) rows to $(outfile)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_at_beta()
end
