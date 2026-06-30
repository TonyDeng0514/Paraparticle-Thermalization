using LinearAlgebra
using Printf

function single_particle_matrix(L::Int, t::Float64, α::Float64)
    h = zeros(Float64, L, L)
    for i in 1:L
        h[i,i] = α * i
    end
    for i in 1:(L - 1)
        h[i, i + 1] = -t
        h[i + 1, i] = -t
    end
    return Hermitian(h)
end

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

shifted_fugacities(ε::AbstractVector{Float64}, β::Float64) =
    exp.(-β .* (ε .- sum(ε) / length(ε)))

function orbital_occupations(ε::AbstractVector{Float64}, N::Int, β::Float64)
    L = length(ε)
    x = shifted_fugacities(ε, β)        # invariant shift; keep x_a ~ O(1)
    eN = esp(x, N)[N + 1]               # e_N over all orbitals
    n = zeros(Float64, L)
    for a in 1:L
        xs = vcat(@view(x[1:a-1]), @view(x[a+1:end]))   # x ∖ x_a
        e  = esp(xs, N - 1)                              # e_{N-1}(x ∖ x_a) = e[N]
        n[a] = x[a] * e[N] / eN
    end
    return n
end

function integrable_thermal_at_beta(L::Int, t::Float64, α::Float64, β::Float64, N::Int)
    h = single_particle_matrix(L, t, α)
    F = eigen(h); ε = F.values; ψ = F.vectors
    n_orb = orbital_occupations(ε, N, β)
    n_tot = (ψ .^ 2) * n_orb
    E     = sum(ε .* n_orb)
    return (; ε, ψ, n_orb, n_tot, E, n_A = 0.5 .* n_tot, n_B = 0.5 .* n_tot)
end

function orbital_pair_occupations(ε::AbstractVector{Float64}, N::Int, β::Float64)
    L  = length(ε)
    x  = shifted_fugacities(ε, β)
    eN = esp(x, N)[N + 1]
    n  = orbital_occupations(ε, N, β)
    NN = zeros(Float64, L, L)
    for a in 1:L
        NN[a, a] = n[a]
        for b in (a + 1):L
            xs = x[setdiff(1:L, (a, b))]
            eNm2 = N >= 2 ? esp(xs, N - 2)[N - 1] : 0.0   # e_{N-2}(x ∖ {a,b})
            v = x[a] * x[b] * eNm2 / eN
            NN[a, b] = NN[b, a] = v
        end
    end
    return NN, n
end

function connected_corr_n_integrable(L::Int, t::Float64, α::Float64, β::Float64, N::Int)
    F = eigen(single_particle_matrix(L, t, α)); ε = F.values; ψ = F.vectors
    NN, n = orbital_pair_occupations(ε, N, β)
    Cov = NN .- n * n'
    P   = ψ .^ 2                                   # P[i,a] = φ_a(i)²
    Cn  = P * Cov * P'                             # direct piece
    W   = n * (1 .- n)' .- Cov                     # W_ab = n_a(1−n_b) − Cov_ab
    for a in 1:L; W[a, a] = 0.0; end               # exchange excludes a=b
    ex = zeros(Float64, L, L)
    for a in 1:L, b in 1:L
        W[a, b] == 0.0 && continue
        qa = @view ψ[:, a]; qb = @view ψ[:, b]
        ex .+= W[a, b] .* (qa * qa') .* (qb * qb')
    end
    return Cn .+ ex
end

function connected_corr_m_integrable(L::Int, t::Float64, α::Float64, β::Float64, N::Int)
    st = integrable_thermal_at_beta(L, t, α, β, N)
    return Matrix(Diagonal(st.n_tot))
end

function run(α, β)
    L = 10
    N = 7
    t_hop = 1.0                      # was 1 (Int) → MethodError vs t::Float64

    label = "integrable_at_alpha$(α)_beta$(β)"

    outdir = joinpath(@__DIR__, "results/ws_integrable/")
    mkpath(outdir)
    outfile = joinpath(outdir, "ws_integrable_L$(L)_N$(N)_$(label).csv")

    rows = Vector{NTuple{7, Any}}()

    st    = integrable_thermal_at_beta(L, t_hop, α, β, N)
    ε     = st.ε
    n_tot = st.n_tot
    n_A   = st.n_A
    n_B   = st.n_B
    E0    = st.E                      # energy of this canonical thermal state
    E_mid = (N / L) * sum(ε)          # β = 0 (infinite-T) reference energy

    @printf("%-6s  %10s  %10s  %12s  %8s  %10s\n",
            "alpha", "E0", "E_mid", "beta", "Σn_i", "Σn_{i,A}")
    @printf("%-6.2f  %10.4f  %10.4f  %12.6e  %8.5f  %10.5f\n",
            α, E0, E_mid, β, sum(n_tot), sum(n_A))

    for i in 1:L
        push!(rows, (α, β, E0, i, n_tot[i], n_A[i], n_B[i]))
    end

    @assert isapprox(sum(n_tot), N; atol = 1e-9) "Σ⟨n_i⟩ ≠ N"
    atol = 1e-9
    if β > 0
        @assert E0 ≤ E_mid + atol "β > 0 should give E0 ≤ E_mid; got E0=$E0, E_mid=$E_mid"
    elseif β < 0
        @assert E0 ≥ E_mid - atol "β < 0 should give E0 ≥ E_mid; got E0=$E0, E_mid=$E_mid"
    end

    open(outfile, "w") do io
        println(io, "# integrable Wannier-Stark canonical thermal state")
        println(io, "# L=$L N=$N t=$t_hop label=$label")
        println(io, "# flavor split: n_A = n_B = n_total/2 (exact, A<->B symmetry)")
        println(io, "alpha,beta,E0,site,n_total,n_A,n_B")
        for (a, b, e0, i, nt, na, nb) in rows
            @printf(io, "%.4f,%.10e,%.10e,%d,%.10e,%.10e,%.10e\n",
                    a, b, e0, i, nt, na, nb)
        end
    end

    println()
    println("Wrote $(length(rows)) rows to $(outfile)")
end

α = parse(Float64, ARGS[1])
β = parse(Float64, ARGS[2])

run(α, β)

