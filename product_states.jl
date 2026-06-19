# product_states.jl
#
# Fixed-N product initial states for the "Tri" site type.
#
# Occupancy is fixed: each occupied site carries a flavor a/b superposition (which
# stays in the N=1 block), vacancy sites stay |vac>. The total particle number is
# therefore an EXACT eigenvalue, so this works on a conserve_qns=true site set and
# keeps the MPS at bond dimension 1.
#
# Occupied site j:  cos(theta_j)|a> + sin(theta_j) e^{i phi_j} |b>
#
# Requires hilbert.jl (SiteType"Tri" and its operators/states) to be included first.

using ITensorMPS, ITensors

# Fixed Hamiltonian disorder seed for the entire product-state thermalization study.
# Keep this FIXED across all runs; vary the product state (occupancy seed / angles)
# instead. Defined here because product_states.jl is included by every run script.
const H_SEED = 42

# On-site, N-conserving unitary rotation in the (a,b) subspace.
# Basis order [vac, a, b]; matrix is M[out, in] (same convention as hilbert.jl).
# Acting on |a>: -> cos(theta)|a> + sin(theta) e^{i phi}|b>.
# The zero vac<->particle entries keep it flux-0, so it is QN-allowed.
function ITensors.op(::OpName"Rab", ::SiteType"Tri"; theta::Real, phi::Real)
    c = cos(theta)
    s = sin(theta)
    return ComplexF64[ 1.0   0.0              0.0
                       0.0   c               -s * exp(-im * phi)
                       0.0   s * exp(im * phi)  c                 ]
end

"""
    superposition_product_state(sites, occ, theta, phi) -> (psi, n_loc_init)

Build a bond-dimension-1 MPS in the fixed-N sector defined by `occ`.

- `sites`  : a conserve_qns=true "Tri" site set.
- `occ`    : length-L Bool vector; `true` = occupied (one particle), `false` = vacancy.
- `theta`  : length-L vector; per-site a/b mixing angle (vacancy entries ignored).
- `phi`    : length-L vector; per-site a/b relative phase (vacancy entries ignored).

Returns the MPS and `n_loc_init[i] = occ[i] ? 1.0 : 0.0` (the initial density profile,
which is exactly the occupancy because both a and b have Nloc = 1).
"""
function superposition_product_state(sites,
                                     occ::AbstractVector{Bool},
                                     theta::AbstractVector{<:Real},
                                     phi::AbstractVector{<:Real})
    L = length(sites)
    @assert length(occ)   == L "occ length must equal number of sites"
    @assert length(theta) == L "theta must have length L"
    @assert length(phi)   == L "phi must have length L"

    # Start from a definite-flavor product state in the chosen N-sector...
    labels = [occ[j] ? "A" : "Vac" for j in 1:L]
    psi = MPS(sites, labels)

    # ...then rotate every occupied site into its a/b superposition.
    gates = [op("Rab", sites[j]; theta=Float64(theta[j]), phi=Float64(phi[j]))
             for j in 1:L if occ[j]]
    psi = apply(gates, psi)
    normalize!(psi)

    n_loc_init = Float64[occ[j] ? 1.0 : 0.0 for j in 1:L]
    @assert isapprox(sum(expect(psi, "Nloc")), count(occ); atol=1e-8) "Initial N wrong"

    return psi, n_loc_init
end

"""
    read_params(path, L) -> (occ, theta, phi)

Read an initial-condition params file (self-contained: occ_*, theta_*, phi_* columns).
Tolerant of extra columns (e.g. an `energy` column copied verbatim from the sweep CSV)
and of comment lines beginning with '#'. Columns are matched by header name, not order.
"""
function read_params(path, L)
    occ   = falses(L)
    theta = zeros(Float64, L)
    phi   = zeros(Float64, L)

    lines = filter(l -> !startswith(strip(l), "#") && !isempty(strip(l)), readlines(path))
    @assert length(lines) >= 2 "params file needs a header row and one data row"

    header = strip.(split(strip(lines[1]), ","))
    data   = strip.(split(strip(lines[2]), ","))
    @assert length(header) == length(data) "header/data column count mismatch"

    colmap = Dict(header[k] => k for k in eachindex(header))
    for j in 1:L
        occ[j]   = parse(Float64, data[colmap["occ_$j"]]) != 0
        theta[j] = parse(Float64, data[colmap["theta_$j"]])
        phi[j]   = parse(Float64, data[colmap["phi_$j"]])
    end
    return occ, theta, phi
end

"""
    params_label(params_file, L) -> String

Compact label for output filenames: the params basename with the conventional
`params_L{L}_seed{H_SEED}_` prefix stripped (so output files don't repeat L and seed).
Falls back to the full basename if the prefix isn't present.
"""
function params_label(params_file, L)
    raw    = splitext(basename(params_file))[1]
    prefix = "params_L$(L)_seed$(H_SEED)_"
    return startswith(raw, prefix) ? raw[length(prefix)+1:end] : raw
end
