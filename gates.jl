function bond_hamiltonian(s1, s2, j::Int, L::Int, t_hop, Ω, V, q; α=0.0)
    w1 = (j     == 1) ? 1.0 : 0.5
    w2 = (j + 1 == L) ? 1.0 : 0.5

    Id1 = op("Id", s1)
    Id2 = op("Id", s2)

    # H_0 :  -t Σ_c (S+_{j,c} S-_{j+1,c} + h.c.)
    h  = -t_hop * (op("Sp_a", s1) * op("Sm_a", s2) + op("Sp_a", s2) * op("Sm_a", s1))
    h += -t_hop * (op("Sp_b", s1) * op("Sm_b", s2) + op("Sp_b", s2) * op("Sm_b", s1))

    # H_V :  V_j m_j m_{j+1}      with m_j = n_{j,a} - n_{j,b}
    h += V[j] * op("Mloc", s1) * op("Mloc", s2)

    # H_Ω :  Ω_j (S+_{j,a} S-_{j,b} + h.c.)
    h += (w1 * Ω[j])     * op("FlipAB", s1) * Id2
    h += (w2 * Ω[j + 1]) * Id1 * op("FlipAB", s2)

    # H_q :  q_{j,a} n_{j,a} + q_{j,b} n_{j,b}
    h += (w1 * q[j, 1])     * op("Na", s1) * Id2
    h += (w1 * q[j, 2])     * op("Nb", s1) * Id2
    h += (w2 * q[j + 1, 1]) * Id1 * op("Na", s2)
    h += (w2 * q[j + 1, 2]) * Id1 * op("Nb", s2)

    # Optional linear potential V(r) = α r
    h += (w1 * α * j)       * op("Nloc", s1) * Id2
    h += (w2 * α * (j + 1)) * Id1 * op("Nloc", s2)

    return h
end


function tebd_gates(sites, dt, t_hop, Ω, V, q; α=0.0)
    L = length(sites)
    half_gates = ITensor[]
    for j in 1:(L - 1)
        h  = bond_hamiltonian(sites[j], sites[j + 1], j, L, t_hop, Ω, V, q; α=α)
        Gj = exp(-1im * (dt / 2) * h)
        push!(half_gates, Gj)
    end
    return [half_gates; reverse(half_gates)]
end