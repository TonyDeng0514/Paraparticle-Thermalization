# kms_extract.jl
#
# KMS β-extraction core: everything downstream of a spectral function S(ω) or a
# one-sided correlator C(t). NO dependency on ED or ITensors, so it can be
# included by both `detailed_balance.jl` (ED validation) and `two_time_correlator.jl`
# (TEBD driver) without pulling in either stack.
#
# The single fitter `beta_from_spectral` (validated in C4C.1) is the one place the
# S(ω)→β slope is computed; `beta_from_Ct` reuses it, so any exact-vs-windowed
# difference is purely the windowing/truncation effect (C4C.2.5).

"""
    linfit(x, y) -> (slope, intercept, R²)

Ordinary least squares y = intercept + slope·x.
"""
function linfit(x, y)
    n = length(x)
    mx = sum(x) / n; my = sum(y) / n
    sxx = 0.0; sxy = 0.0
    for i in 1:n
        sxx += (x[i] - mx)^2
        sxy += (x[i] - mx) * (y[i] - my)
    end
    slope = sxy / sxx
    intercept = my - slope * mx
    sstot = 0.0; ssres = 0.0
    for i in 1:n
        ŷ = intercept + slope * x[i]
        sstot += (y[i] - my)^2
        ssres += (y[i] - ŷ)^2
    end
    r2 = sstot == 0.0 ? 0.0 : 1 - ssres / sstot
    return slope, intercept, r2
end

"""
    beta_from_spectral(centers, S; rel_floor=1e-6, ωfrac=0.6) -> (β, intercept, R², nfit)

Extract β from the KMS slope of ln[S(ω)/S(−ω)] vs ω. Uses bins with ω>0 where
both S(ω) and S(−ω) exceed `rel_floor`·max(S), and |ω| < `ωfrac`·ωmax (the band
edges are sparsely sampled). Returns the slope (=β), the intercept (≈0 for a
thermal state), R², and the number of fitted bins. Requires a SYMMETRIC ω-grid
(bin i mirrors bin n+1−i).
"""
function beta_from_spectral(centers, S; rel_floor = 1e-6, ωfrac = 0.6)
    n = length(centers)
    floor_val = rel_floor * maximum(S)
    ωmax = centers[end] + (centers[2] - centers[1]) / 2
    xs = Float64[]; ys = Float64[]
    for i in 1:n
        ω = centers[i]
        ω <= 0 && continue
        ω > ωfrac * ωmax && continue
        j = n + 1 - i                       # mirror bin at −ω (symmetric grid)
        if S[i] > floor_val && S[j] > floor_val
            push!(xs, ω)
            push!(ys, log(S[i] / S[j]))
        end
    end
    length(xs) < 3 && return (NaN, NaN, NaN, length(xs))
    slope, intercept, r2 = linfit(xs, ys)
    return slope, intercept, r2, length(xs)
end

"""
    window_weights(tfull, Tmax, window) -> w

Symmetric even window on tfull ∈ [−Tmax, Tmax]. `window ∈ {:blackmanharris (default,
low side lobes), :hann, :boxcar}`. Parametrized by τ = (t+Tmax)/(2Tmax) ∈ [0,1].
"""
function window_weights(tfull, Tmax, window::Symbol)
    τ = (tfull .+ Tmax) ./ (2Tmax)
    if window === :blackmanharris
        return @. 0.35875 - 0.48829cos(2π * τ) + 0.14128cos(4π * τ) - 0.01168cos(6π * τ)
    elseif window === :hann
        return @. 0.5 - 0.5cos(2π * τ)
    elseif window === :boxcar
        return ones(length(tfull))
    else
        error("unknown window: $window")
    end
end

"""
    beta_from_Ct(t, C; window=:blackmanharris, pad=8, ωmax_grid=nothing,
                 rel_floor=1e-6, ωfrac=0.6, Cm_atol=1e-6)
        -> (β, r2, intercept, nfit, S, ωgrid)

Shared extraction core: turn a one-sided C(t) (t = 0:Δt:Tmax, exact-from-spectrum
OR TEBD-measured) into β via the KMS slope. Steps: assert C(0) real (= C^m) →
reconstruct C(−t) = conj(C(t)) → apply a symmetric even `window` → DFT onto a
SYMMETRIC ω-grid with spacing Δω = 2π/(pad·2Tmax) (the zero-pad ×`pad`
interpolation) → assert S(ω) real → hand (ωgrid, S) to `beta_from_spectral`. The
ONLY new code vs the exact path is negative-time reconstruction, windowing, DFT.
"""
function beta_from_Ct(t, C; window::Symbol = :blackmanharris, pad::Int = 8,
                      ωmax_grid = nothing, rel_floor = 1e-6, ωfrac = 0.6, Cm_atol = 1e-6)
    Δt = t[2] - t[1]; Tmax = t[end]
    @assert abs(imag(C[1])) < max(Cm_atol, 1e-9 * abs(C[1])) "C(0) not real (=C^m): imag=$(imag(C[1]))"
    tfull = collect(-Tmax:Δt:Tmax)
    Cfull = Vector{ComplexF64}(undef, length(tfull))
    for i in eachindex(tfull)
        tt = tfull[i]
        k = round(Int, abs(tt) / Δt) + 1
        if tt >= 0
            Cfull[i] = C[k]
        else
            Cfull[i] = conj(C[k])
        end
    end
    Cw = Cfull .* window_weights(tfull, Tmax, window)
    ωmax = ωmax_grid === nothing ? π / Δt : ωmax_grid
    Δω = 2π / (pad * 2Tmax)
    Kω = floor(Int, ωmax / Δω)
    ωgrid = [k * Δω for k in -Kω:Kω]
    S = Vector{Float64}(undef, length(ωgrid))
    maximag = 0.0
    for a in eachindex(ωgrid)
        ω = ωgrid[a]
        acc = ComplexF64(0)
        for i in eachindex(tfull)
            acc += Cw[i] * cis(ω * tfull[i])
        end
        acc *= Δt
        maximag = max(maximag, abs(imag(acc)))
        S[a] = real(acc)
    end
    @assert maximag < 1e-6 * maximum(abs.(S)) + 1e-12 "S(ω) not real: max imag = $maximag"
    slope, intercept, r2, nfit = beta_from_spectral(ωgrid, S; rel_floor = rel_floor, ωfrac = ωfrac)
    return (β = slope, r2 = r2, intercept = intercept, nfit = nfit, S = S, ωgrid = ωgrid)
end
