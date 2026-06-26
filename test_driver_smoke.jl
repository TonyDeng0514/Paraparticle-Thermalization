# test_driver_smoke.jl
#
# Smoke test for integrable_at_beta.jl IO plumbing (read β, pair with α, write
# schema, reparse) — the layer that C1.1–C1.3 skip. Feeds the six β values from
# the committed E0→β scan back through the driver and checks it reproduces the
# n_total column.
#
# PRECISION FLOOR (decided, accepted): the residual bottoms out at ~5e-12, which
# is the existing CSV schema's %.10e write precision (~11 sig figs on O(1)
# values), NOT an IO bug. Proof: at α=0 the answer is analytically N/L=2/3; the
# CSV stores 6.6666666667e-01 and the true value is 0.6666666666…, differing by
# 3.33e-12 by themselves. A real IO bug (wrong β, mis-paired α, swapped
# sites/columns) would show O(≥1e-3). So the tolerance below is set to the format
# floor with margin; %.10e is kept (negligible vs the ~1% κ/β science).

include("integrable_at_beta.jl")   # defines read_betas / run_at_beta (guarded: no auto-run)

const TOL = 1e-11   # %.10e write floor (~5e-12) with safety; real IO bugs are O(≥1e-3)

function parse_profile_csv(path)
    d = Dict{Tuple{Float64,Float64,Int},Float64}()
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#") || startswith(s, "alpha")) && continue
        p = split(s, ",")
        d[(parse(Float64, p[1]), parse(Float64, p[2]), parse(Int, p[4]))] = parse(Float64, p[5])
    end
    return d
end

function main()
    repo   = @__DIR__
    params = joinpath(repo, "params/params_L12_seed42_ps188_sweepmin.csv")
    ref    = joinpath(repo, "results/ws_integrable/ws_integrable_L12_N8_ps188_sweepmin.csv")

    refd  = parse_profile_csv(ref)
    pairs = sort(unique([(a, b) for (a, b, s) in keys(refd)]))   # (α,β) in α order
    betas = unique([b for (a, b) in pairs])

    betacsv = joinpath(repo, "results/ws_integrable", "smoke_betas.csv")
    open(betacsv, "w") do io
        println(io, "beta"); for b in betas; @printf(io, "%.16e\n", b); end
    end

    empty!(ARGS); push!(ARGS, params); push!(ARGS, betacsv)   # drive through real entry point
    run_at_beta()

    occ = read_occupancy(params); L = length(occ); N = count(occ)
    label = label_from_path(params, L)
    out = joinpath(repo, "results/ws_integrable",
                   "ws_integrable_atbeta_L$(L)_N$(N)_$(label).csv")
    drv = parse_profile_csv(out)

    worst = 0.0
    for (a, b, s) in keys(refd)
        @assert haskey(drv, (a, b, s)) "driver output missing (α=$a, β=$b, site=$s)"
        worst = max(worst, abs(drv[(a, b, s)] - refd[(a, b, s)]))
    end
    rm(betacsv; force = true)

    @printf("\nDriver IO smoke test: worst |Δn_total| vs reference = %.2e  (tol %.0e)\n", worst, TOL)
    pass = worst < TOL
    println(pass ? "PASS — driver reproduces the scan to the %.10e format floor" :
                   "FAIL — discrepancy exceeds the format floor; inspect IO")
    exit(pass ? 0 : 1)
end

main()
