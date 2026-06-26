"""
    connected_corr(psi, opname) -> C

Connected equal-time correlator C_ij = ⟨O_i O_j⟩ − ⟨O_i⟩⟨O_j⟩ for the on-site
operator `opname` on the MPS `psi`. Use `"Nloc"` for the charge correlator C^n
and `"Mloc"` for the magnetization correlator C^m. Real and symmetric for the
Hermitian density/magnetization operators.
"""
function connected_corr(psi, opname)
    C = correlation_matrix(psi, opname, opname)
    e = expect(psi, opname)
    return C .- e * e'
end

function build_hamiltonian_mpo(sites, t_hop, Ω, V, q; α=0.0)
    L = length(sites)
    os = OpSum()

    for j in 1:(L-1)
        os += -t_hop, "Sp_a", j, "Sm_a", j+1
        os += -t_hop, "Sm_a", j, "Sp_a", j+1
        os += -t_hop, "Sp_b", j, "Sm_b", j+1
        os += -t_hop, "Sm_b", j, "Sp_b", j+1
        os += V[j], "Mloc", j, "Mloc", j+1
    end

    for j in 1:L
        os += Ω[j], "FlipAB", j
        os += q[j,1], "Na", j
        os += q[j,2], "Nb", j
        os += α * j, "Nloc", j
    end

    return MPO(os, sites)
end

function measure_energy_bonds(psi, sites, t_hop, Ω, V, q; α=0.0)
    L = length(sites)
    E = 0.0

    orthogonalize!(psi, 1)

    for j in 1:(L - 1)
        orthogonalize!(psi, j)

        Θ  = psi[j] * psi[j+1]
        h_j = bond_hamiltonian(sites[j], sites[j+1], j, L, t_hop, Ω, V, q; α=α)

        # h_j's unprimed <In> indices contract with Θ's site <Out> indices ✓
        # noprime brings primed output back to level 0 to match dag(Θ)
        hΘ = noprime(h_j * Θ)
        E  += real(scalar(dag(Θ) * hΘ))
    end

    return E
end

