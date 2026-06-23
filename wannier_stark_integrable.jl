# wannier_stark_integrable.jl
#
# Standalone integrable (clean Wannier-Stark) thermal pipeline. Sibling to the
# perturbed/TEBD pipeline; it shares ONLY the IO / naming conventions and does
# not import or touch any TEBD / ITensors machinery. No time evolution, no MPS,
# no bond dimension. The whole computation lives in a single L x L matrix.
#
# Model (open chain, uniform hopping t, flavor-blind linear potential):
#
#   H = -t Σ_j Σ_{σ∈{A,B}} (c†_{jσ} c_{j+1,σ} + h.c.)  +  α Σ_i i·n_i
#
# with the hardcore constraint n_{i,A}+n_{i,B} ≤ 1 and fixed total N.
#
# Because the hopping is flavor-blind on an open line, the model has exact
# spin-charge separation: the charge (total-density) sector is spinless free
# fermions, and the flavor sector is energetically flat (a β-independent
# 2^N multiplicity that cancels in every density). So the single-β canonical
# state in the total-N sector reduces to the spinless free-fermion fixed-N
# canonical problem. The A↔B symmetry of H then forces the per-site flavor
# split to be exactly 1/2 each — no N↑ assumption is made or needed.
#
# Usage:  julia wannier_stark_integrable.jl [params_file]
#   params_file defaults to params/params_L12_seed42_ps188_sweepmin.csv.
#   Only the occ_* columns are read (which sites are occupied); the a/b angles
#   are irrelevant to the integrable energy and density.

using LinearAlgebra
using Printf

const T_HOP  = 1.0
const α_LIST = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

# ── Inputs (located, not hardcoded) ──────────────────────────────────────────

"""
    read_occupancy(path) -> occ::Vector{Bool}

Read the occupancy from a params CSV by header name (occ_1 … occ_L). Tolerant
of comment lines and extra columns. L is inferred from how many occ_* columns
are present. Self-contained so this script needs no TEBD/ITensors code.
"""
function read_occupancy(path)
    lines  = filter(l -> !startswith(strip(l), "#") && !isempty(strip(l)), readlines(path))
    @assert length(lines) >= 2 "params file needs a header row and one data row"
    header = strip.(split(strip(lines[1]), ","))
    data   = strip.(split(strip(lines[2]), ","))
    @assert length(header) == length(data) "header/data column count mismatch"

    colmap = Dict(header[k] => k for k in eachindex(header))
    L = count(h -> startswith(h, "occ_"), header)
    @assert L > 0 "no occ_* columns found in $path"
    occ = falses(L)
    for i in 1:L
        occ[i] = parse(Float64, data[colmap["occ_$i"]]) != 0
    end
    return occ
end

"""
    label_from_path(path, L) -> String

Strip the conventional `params_L{L}_seed{seed}_` prefix to get a compact label.
"""
function label_from_path(path, L)
    raw = splitext(basename(path))[1]
    return replace(raw, Regex("^params_L$(L)_seed\\d+_") => "")
end

# ── Single-particle problem ──────────────────────────────────────────────────

"""
    single_particle_matrix(L, t, α) -> Hermitian L×L

h_{ij} = -t(δ_{i,j+1}+δ_{i,j-1}) + α·i·δ_{ij}, with 1-based i to match the
perturbed Hamiltonian's V_i = α·i convention (gates.jl).
"""
function single_particle_matrix(L::Int, t::Float64, α::Float64)
    h = zeros(Float64, L, L)
    for i in 1:L
        h[i, i] = α * i              # 1-based site index, matches gates.jl
    end
    for i in 1:(L - 1)
        h[i, i + 1] = -t
        h[i + 1, i] = -t
    end
    return Hermitian(h)
end

# ── Fixed-N canonical occupations (cancellation-free) ────────────────────────

"""
    esp(x, K) -> e[1:K+1]    where e[k+1] = e_k(x)

Elementary symmetric polynomials e_0..e_K via the generating polynomial
∏_a (1 + x_a y) = Σ_k e_k y^k, accumulating one variable at a time:
e_k ← e_k + x_a·e_{k-1}. With x_a > 0 this is all additions (no cancellation).
"""
function esp(x::AbstractVector{<:Real}, K::Int)
    e = zeros(Float64, K + 1)
    e[1] = 1.0
    for xa in x
        @inbounds for k in min(K, length(x)):-1:1
            e[k + 1] += xa * e[k]
        end
    end
    return e
end

"""
    orbital_occupations(ε, N, β) -> ⟨n_a⟩_N

Fixed-N canonical occupations ⟨n_a⟩_N = x_a·e_{N-1}(x∖x_a)/e_N(x), with
x_a = e^{-βε_a}. ε is internally mean-shifted (occupations are invariant under
ε → ε − c) to keep x_a ~ O(1) and avoid dynamic-range blowup. e_{N-1}(x∖x_a)
is obtained by rebuilding the product polynomial over the L−1 orbitals
excluding a — never the subtractive leave-one-out recursion.
"""
function orbital_occupations(ε::AbstractVector{Float64}, N::Int, β::Float64)
    L = length(ε)
    εshift = ε .- (sum(ε) / L)          # invariant shift; keep x_a ~ O(1)
    x = exp.(-β .* εshift)
    eN = esp(x, N)[N + 1]               # e_N over all orbitals
    n = zeros(Float64, L)
    for a in 1:L
        xs = vcat(@view(x[1:a-1]), @view(x[a+1:end]))   # x ∖ x_a
        e  = esp(xs, N - 1)                              # e_{N-1}(x ∖ x_a) = e[N]
        n[a] = x[a] * e[N] / eN
    end
    return n
end

# physical thermal energy uses the ORIGINAL (unshifted) ε
thermal_energy(ε, N, β) = sum(ε .* orbital_occupations(ε, N, β))

"""
    solve_beta(ε, N, E0) -> β

Bisection for the unique β with ⟨H⟩_β = E0. d⟨H⟩/dβ = −Var(H) ≤ 0, so ⟨H⟩ is
monotone decreasing in β and the root is unique. The bracket straddles zero and
allows β < 0 (physical here: the band is bounded, so negative temperature is
real and E0 > E_mid ⇒ β < 0).
"""
function solve_beta(ε::AbstractVector{Float64}, N::Int, E0::Float64)
    f(β) = thermal_energy(ε, N, β) - E0
    βlo, βhi = -1.0, 1.0
    flo, fhi = f(βlo), f(βhi)           # β→−∞ ⇒ E_max (f>0); β→+∞ ⇒ E_min (f<0)
    k = 0
    while flo * fhi > 0 && k < 100
        βlo *= 2; βhi *= 2
        flo, fhi = f(βlo), f(βhi)
        k += 1
    end
    @assert flo * fhi <= 0 "could not bracket β (E0 outside [E_min, E_max]?)"
    for _ in 1:200
        βm = 0.5 * (βlo + βhi)
        fm = f(βm)
        if abs(fm) < 1e-13 || (βhi - βlo) < 1e-14
            return βm
        end
        if (fm > 0) == (flo > 0)
            βlo, flo = βm, fm
        else
            βhi, fhi = βm, fm
        end
    end
    return 0.5 * (βlo + βhi)
end

# ── Driver ───────────────────────────────────────────────────────────────────

function run()
    params_file = isempty(ARGS) ?
        joinpath(@__DIR__, "params/params_L12_seed42_ps188_sweepmin.csv") : ARGS[1]

    occ = read_occupancy(params_file)
    L   = length(occ)
    N   = count(occ)
    O   = [i for i in 1:L if occ[i]]        # occupied sites, 1-based
    label = label_from_path(params_file, L)

    println("Integrable Wannier-Stark thermal pipeline")
    println("  params_file = $(params_file)")
    println("  L = $L, N = $N, t = $T_HOP")
    println("  occupied sites = $O")
    println("  Σ occupied i   = $(sum(O))   (E0 = α·$(sum(O)))")
    println()

    outdir = joinpath(@__DIR__, "results/ws_integrable")
    mkpath(outdir)
    outfile = joinpath(outdir, "ws_integrable_L$(L)_N$(N)_$(label).csv")

    rows = Vector{NTuple{7, Any}}()   # (alpha, beta, E0, site, n_total, n_A, n_B)

    @printf("%-6s  %10s  %10s  %12s  %8s  %10s\n",
            "alpha", "E0", "E_mid", "beta", "Σn_i", "Σn_{i,A}")
    for α in α_LIST
        h = single_particle_matrix(L, T_HOP, α)
        F = eigen(h)                  # F.values = ε_a, F.vectors columns = φ_a
        ε = F.values
        ψ = F.vectors                 # ψ[i,a] = φ_a(i)  (distinct name; no shadowing)

        E0    = α * sum(O)
        E_mid = (N / L) * sum(ε)      # = (N/L)·α·L(L+1)/2

        β     = solve_beta(ε, N, E0)
        n_orb = orbital_occupations(ε, N, β)
        n_tot = (ψ .^ 2) * n_orb      # ⟨n_i⟩ = Σ_a |φ_a(i)|² ⟨n_a⟩_N

        # flavor split: exact 1/2 each by A↔B symmetry of the integrable H
        n_A = 0.5 .* n_tot
        n_B = 0.5 .* n_tot

        @printf("%-6.2f  %10.4f  %10.4f  %12.6e  %8.5f  %10.5f\n",
                α, E0, E_mid, β, sum(n_tot), sum(n_A))

        for i in 1:L
            push!(rows, (α, β, E0, i, n_tot[i], n_A[i], n_B[i]))
        end

        # ── sanity checks (analytic; do NOT rely on the distrusted ws_baseline) ──
        @assert isapprox(E0, α * sum(O); atol = 1e-12)
        @assert isapprox(sum(n_tot), N; atol = 1e-9)      "Σ⟨n_i⟩ ≠ N"
        @assert isapprox(sum(n_A), N / 2; atol = 1e-9)    "Σ⟨n_{i,A}⟩ ≠ N/2"
        if α > 0
            @assert β < 0  "expected β < 0 (E0 = $E0 > E_mid = $E_mid) but got β = $β"
        end
    end

    open(outfile, "w") do io
        println(io, "# integrable Wannier-Stark canonical thermal state")
        println(io, "# params_file = $(params_file)")
        println(io, "# L=$L N=$N t=$T_HOP O=$O label=$label")
        println(io, "# flavor split: n_A = n_B = n_total/2 (exact, A<->B symmetry)")
        println(io, "alpha,beta,E0,site,n_total,n_A,n_B")
        for (α, β, E0, i, nt, na, nb) in rows
            @printf(io, "%.4f,%.10e,%.10e,%d,%.10e,%.10e,%.10e\n",
                    α, β, E0, i, nt, na, nb)
        end
    end

    println()
    println("Wrote $(length(rows)) rows to $(outfile)")
end

run()
