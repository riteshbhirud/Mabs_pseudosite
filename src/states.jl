"""
    random_bmps(sites::Vector{<:ITensors.Index}, alg::Truncated; kwargs...)

Create a random bosonic MPS using the `Truncated` algorithm.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of site indices
- alg::Truncated: Algorithm specification

Returns:
- BMPS: Random bosonic MPS
"""
function random_bmps(sites::Vector{<:ITensors.Index}, alg::Truncated; linkdims = 1)
    mps = ITensorMPS.random_mps(sites; linkdims)
    return BMPS(mps, alg)
end

"""
    vacuumstate(sites::Vector{<:ITensors.Index}, alg::Truncated)

Create a vacuum state |0,0,...,0⟩ BMPS.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of site indices
- alg::Truncated: Algorithm specification

Returns:
- BMPS: Vacuum state bosonic MPS
"""
function vacuumstate(sites::Vector{<:ITensors.Index}, alg::Truncated)
    states = fill(1, length(sites))
    return BMPS(sites, alg, states)  
end

"""
    coherentstate(sites::Vector{<:ITensors.Index}, alg::Truncated, α::Number)
    coherentstate(sites::Vector{<:ITensors.Index}, alg::Truncated, αs::Vector{<:Number})

Create an approximate coherent state `BMPS` using truncated expansion.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of site indices
- alg::Truncated: Algorithm specification
- α::Union{Number, Vector{<:Number}}: Single coherent state amplitude (applied to all modes),
  or a vector of coherent state amplitudes (one per mode)

Returns:
- BMPS: Coherent state bosonic MPS (approximated by truncation)
"""
function coherentstate(sites::Vector{<:ITensors.Index}, alg::Truncated, α::Number)
    αs = fill(α, length(sites))
    return coherentstate(sites, alg, αs)
end
function coherentstate(sites::Vector{<:ITensors.Index}, alg::Truncated, αs::Vector{<:Number})
    N = length(sites)
    length(αs) == N || error("Number of amplitudes ($(length(αs))) must match number of sites ($N)")
    tensors = Vector{ITensors.ITensor}(undef, N)
    @inbounds for (i, site) in enumerate(sites)
        α = αs[i]
        max_occ = ITensors.dim(site) - 1
        coeffs = Vector{ComplexF64}(undef, max_occ+1)
        normalization = exp(-abs2(α)/2)
        @inbounds for n in 0:max_occ
            coeff = normalization * (α^n) / sqrt(_safe_factorial(n))
            coeffs[n+1] = convert(ComplexF64, coeff)
        end
        norm_factor = sqrt(sum(abs2, coeffs))
        coeffs ./= norm_factor
        if i == 1
            if N == 1
                tensor = ITensors.ITensor(ComplexF64, site)
                @inbounds for n in 0:max_occ
                    tensor[n+1] = coeffs[n+1]
                end
            else
                right_link = ITensors.Index(1, "Link,l=$i")
                tensor = ITensors.ITensor(ComplexF64, site, right_link)
                @inbounds for n in 0:max_occ
                    tensor[n+1, 1] = coeffs[n+1]
                end
            end
        elseif i == N
            left_link = ITensors.Index(1, "Link,l=$(i-1)")
            tensor = ITensors.ITensor(ComplexF64, left_link, site)
            @inbounds for n in 0:max_occ
                tensor[1, n+1] = coeffs[n+1]
            end
        else
            left_link = ITensors.Index(1, "Link,l=$(i-1)")
            right_link = ITensors.Index(1, "Link,l=$i")
            tensor = ITensors.ITensor(ComplexF64, left_link, site, right_link)
            @inbounds for n in 0:max_occ
                tensor[1, n+1, 1] = coeffs[n+1]
            end
        end
        tensors[i] = tensor
    end
    mps = ITensorMPS.MPS(tensors)
    return BMPS(mps, alg)
end
