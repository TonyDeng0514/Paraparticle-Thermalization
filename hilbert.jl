using ITensorMPS, ITensors
using LinearAlgebra
using Random
using Printf

function ITensors.space(::SiteType"Tri"; conserve_qns=false)
    if conserve_qns
        return [QN("N", 0) => 1, QN("N", 1) => 2]
    end
    return 3
end

ITensors.op(::OpName"Id", ::SiteType"Tri") = Matrix{Float64}(I, 3, 3)

ITensors.op(::OpName"Sp_a", ::SiteType"Tri") = [0.0 0.0 0.0
                                               1.0 0.0 0.0
                                               0.0 0.0 0.0]

ITensors.op(::OpName"Sm_a", ::SiteType"Tri") = [0.0 1.0 0.0
                                               0.0 0.0 0.0
                                               0.0 0.0 0.0]

ITensors.op(::OpName"Sp_b", ::SiteType"Tri") = [0.0 0.0 0.0
                                               0.0 0.0 0.0
                                               1.0 0.0 0.0]

ITensors.op(::OpName"Sm_b", ::SiteType"Tri") = [0.0 0.0 1.0
                                               0.0 0.0 0.0
                                               0.0 0.0 0.0]

ITensors.op(::OpName"Na",   ::SiteType"Tri") = [0.0 0.0 0.0
                                                0.0 1.0 0.0
                                                0.0 0.0 0.0]

ITensors.op(::OpName"Nb",   ::SiteType"Tri") = [0.0 0.0 0.0
                                                0.0 0.0 0.0
                                                0.0 0.0 1.0]


ITensors.op(::OpName"Nloc", ::SiteType"Tri") = [0.0 0.0 0.0
                                                0.0 1.0 0.0
                                                0.0 0.0 1.0]

ITensors.op(::OpName"Mloc", ::SiteType"Tri") = [ 0.0 0.0  0.0
                                                 0.0 1.0  0.0
                                                 0.0 0.0 -1.0]


ITensors.op(::OpName"FlipAB", ::SiteType"Tri") = [0.0 0.0 0.0
                                                  0.0 0.0 1.0
                                                  0.0 1.0 0.0]

ITensors.state(::StateName"Vac", ::SiteType"Tri") = [1.0, 0.0, 0.0]   # component 1
ITensors.state(::StateName"A",   ::SiteType"Tri") = [0.0, 1.0, 0.0]   # component 2
ITensors.state(::StateName"B",   ::SiteType"Tri") = [0.0, 0.0, 1.0]   # component 3

function random_config(L, N, Na; seed=nothing)
    rng = isnothing(seed) ? MersenneTwister() : MersenneTwister(seed)

    config = fill("Vac", L)
    particle_sites = randperm(rng, L)[1:N]
    flavors = shuffle(rng, [fill("A", Na); fill("B", N - Na)])
    config[particle_sites] = flavors

    return config 
end
