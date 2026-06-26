# ED.jl — full-spectrum exact diagonalization for ETH analysis
#
# Observable: O = n_{j1} * n_{j2}, center bond (j1=L÷2, j2=L÷2+1)
# Saves (energy, <O>) for every eigenstate to results/eth_L{L}_N{N}_seed{seed}.csv
#
# Sector dim for L=11, N=7: C(11,7)*2^7 = 42,240  (~28 GB for H + eigenvecs)

using LinearAlgebra, SparseArrays, Random, Printf

include("ed_thermal.jl")   # exact Gibbs-state thermal observables (reused by tests / component 4)

const L    = 10
const N    = 7
const Na   = 3      # flavor-A particles; unused in diagonalization, kept for parity
const seed = 1234
const W    = 0.2
const t_hop = 1.0

# --- local 3×3 operators (basis: Vac=1, A=2, B=3) ---
const _Sp_a   = sparse([2], [1], [1.0], 3, 3)
const _Sm_a   = sparse([1], [2], [1.0], 3, 3)
const _Sp_b   = sparse([3], [1], [1.0], 3, 3)
const _Sm_b   = sparse([1], [3], [1.0], 3, 3)
const _Na     = sparse([2], [2], [1.0], 3, 3)
const _Nb     = sparse([3], [3], [1.0], 3, 3)
const _Nloc   = sparse([2, 3], [2, 3], [1.0,  1.0], 3, 3)
const _Mloc   = sparse([2, 3], [2, 3], [1.0, -1.0], 3, 3)
const _FlipAB = sparse([2, 3], [3, 2], [1.0,  1.0], 3, 3)

# Embed op A at site j in an L-site chain.
# Basis ordering: |σ_1,...,σ_L⟩ → 1-based index 1 + Σ_j σ_j·3^(L-j), σ_j ∈ {0,1,2}
site_op(A, j, L) =
    kron(kron(sparse(I, 3^(j-1), 3^(j-1)), A), sparse(I, 3^(L-j), 3^(L-j)))

bond_op(A, B, j, L) =
    kron(kron(sparse(I, 3^(j-1), 3^(j-1)), kron(A, B)), sparse(I, 3^(L-j-1), 3^(L-j-1)))

function build_hamiltonian_ed(L, t_hop, Ω, V, q)
    H = spzeros(Float64, 3^L, 3^L)
    for j in 1:(L-1)
        H = H + (-t_hop) * bond_op(_Sp_a, _Sm_a, j, L)
        H = H + (-t_hop) * bond_op(_Sm_a, _Sp_a, j, L)
        H = H + (-t_hop) * bond_op(_Sp_b, _Sm_b, j, L)
        H = H + (-t_hop) * bond_op(_Sm_b, _Sp_b, j, L)
        H = H + V[j]   * bond_op(_Mloc, _Mloc, j, L)
    end
    for j in 1:L
        H = H + Ω[j]   * site_op(_FlipAB, j, L)
        H = H + q[j,1] * site_op(_Na, j, L)
        H = H + q[j,2] * site_op(_Nb, j, L)
    end
    return H
end

# 1-based global indices of all basis states in the N-particle sector
function n_sector_indices(L, N)
    filter(1:3^L) do idx
        k = idx - 1
        count(j -> div(k, 3^(L-j)) % 3 != 0, 1:L) == N
    end
end

# Binary occupation (0.0 or 1.0) at site j for each state in the sector
function sector_occupations(idx, j, L)
    map(idx) do global_idx
        k = global_idx - 1
        div(k, 3^(L-j)) % 3 != 0 ? 1.0 : 0.0
    end
end

# Run the full ETH script only when executed directly; `include("ED.jl")` then
# exposes the builders (build_hamiltonian_ed, n_sector_indices, sector_occupations)
# and ed_thermal.jl's Gibbs routines without triggering the heavy diagonalization.
if abspath(PROGRAM_FILE) == @__FILE__

# ─── disorder (matches compressibility_scan.jl convention) ──────────────────
Random.seed!(seed)
Ω = W * randn(L)
V = W * randn(L - 1)
q = W * randn(L, 2)

# ─── sector and Hamiltonian ──────────────────────────────────────────────────
idx   = n_sector_indices(L, N)
dim_N = length(idx)
@printf "L=%d  N=%d  seed=%d  W=%.2f\n" L N seed W
@printf "N-sector dimension: %d\n" dim_N
@printf "Estimated memory (H + eigenvecs): ~%.1f GB\n\n" (2 * dim_N^2 * 8 / 1e9)

@printf "Building sparse Hamiltonian...\n"; flush(stdout)
H_sp = build_hamiltonian_ed(L, t_hop, Ω, V, q)

@printf "Extracting N-sector and converting to dense...\n"; flush(stdout)
H_N = Symmetric(Matrix(H_sp[idx, idx]))

@printf "Diagonalizing (dim=%d)...\n" dim_N; flush(stdout)
vals, vecs = eigen(H_N)

# ─── ETH observable: O = n_{j1} · n_{j2} on center bond ────────────────────
# O is diagonal in the occupation basis (values 0 or 1), so no matrix needed.
# ⟨E_n|O|E_n⟩ = vecs[:,n].² ⋅ diag_O   (eigenvectors are real)
j1, j2 = L ÷ 2, L ÷ 2 + 1    # sites 5 and 6 for L=11
diag_O = sector_occupations(idx, j1, L) .* sector_occupations(idx, j2, L)

@printf "Computing ETH matrix elements...\n"; flush(stdout)
eth_vals = [dot(vecs[:, n].^2, diag_O) for n in 1:dim_N]

# ─── save ────────────────────────────────────────────────────────────────────
mkpath("results/eth")
outfile = "results/eth/eth_L$(L)_N$(N)_seed$(seed).csv"
open(outfile, "w") do f
    println(f, "# ETH: O = n_$(j1)*n_$(j2),  L=$(L)  N=$(N)  seed=$(seed)  W=$(W)  t=$(t_hop)")
    println(f, "energy,O_expval")
    for n in 1:dim_N
        @printf f "%.10f,%.10f\n" vals[n] eth_vals[n]
    end
end
@printf "Saved %d rows → %s\n" dim_N outfile

end  # if abspath(PROGRAM_FILE) == @__FILE__
