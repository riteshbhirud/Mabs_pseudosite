"""
    random_bmps(sites::Vector{<:ITensors.Index}, alg::MabsAlg; kwargs...)

Create a random bosonic MPS using the bosonic MPS algorithm.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of site indices
- alg::MabsAlg: Algorithm specification

Returns:
- BMPS: Random bosonic MPS
"""
function random_bmps(sites::Vector{<:ITensors.Index}, alg::Truncated; linkdims = 1)
    mps = ITensorMPS.random_mps(sites; linkdims)
    return BMPS(mps, alg)
end
function random_bmps(sites::Vector{<:ITensors.Index}, alg::PseudoSite; linkdims=1)
    nqubits = _nqubits_per_mode(sites, alg)
    n_expected = alg.nmodes * nqubits
    length(sites) == n_expected || throw(ArgumentError("Sites must match algorithm"))
    
    mps = ITensorMPS.random_mps(sites; linkdims=linkdims)
    return BMPS(mps, alg)
end

"""
    vacuumstate(sites::Vector{<:ITensors.Index}, alg::MabsAlg)

Create a vacuum state |0,0,...,0⟩ BMPS.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of site indices
- alg::MabsAlg: Algorithm specification

Returns:
- BMPS: Vacuum state bosonic MPS
"""
function vacuumstate(sites::Vector{<:ITensors.Index}, alg::Truncated)
    states = fill(1, length(sites))
    return BMPS(sites, alg, states)  
end
function vacuumstate(sites::Vector{<:ITensors.Index}, alg::PseudoSite)
    nqubits = _nqubits_per_mode(sites, alg)
    n_expected = alg.nmodes * nqubits
    length(sites) == n_expected || throw(ArgumentError("Sites must match algorithm"))
    states = fill(1, length(sites))
    mps = ITensorMPS.productMPS(sites, states) 
    return BMPS(mps, alg)
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

"""
    coherentstate(sites::Vector{<:ITensors.Index}, alg::PseudoSite, α::Number; kwargs...)
    coherentstate(sites::Vector{<:ITensors.Index}, alg::PseudoSite, αs::Vector{<:Number}; kwargs...)

Create a coherent state in the PseudoSite representation.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of qubit site indices
- alg::PseudoSite: Algorithm specification
- α::Union{Number, Vector{<:Number}}: Single coherent state amplitude (applied to all modes),
  or a vector of coherent state amplitudes (one per mode)

Keyword Arguments:
- kwargs...: Passed to underlying ITensorMPS operations (e.g., `cutoff`, `maxdim`)

Returns:
- BMPS: Coherent state bosonic MPS in PseudoSite representation
"""
function coherentstate(sites::Vector{<:ITensors.Index}, alg::PseudoSite, α::Number; kwargs...)
    nqubits = _nqubits_per_mode(sites, alg)
    n_expected = alg.nmodes * nqubits
    length(sites) == n_expected || throw(ArgumentError("Sites must match algorithm"))
    αs = fill(α, alg.nmodes)
    return coherentstate(sites, alg, αs; kwargs...)
end

function coherentstate(sites::Vector{<:ITensors.Index}, alg::PseudoSite, αs::Vector{<:Number}; kwargs...)
    nqubits = _nqubits_per_mode(sites, alg)
    n_expected = alg.nmodes * nqubits
    length(sites) == n_expected || throw(ArgumentError("Sites must match algorithm"))
    length(αs) == alg.nmodes || 
        throw(ArgumentError("Number of amplitudes must match modes"))
    fock_cutoff = 2^nqubits - 1
    if fock_cutoff <= 15
        return _coherentstate_direct_sum(sites, alg, αs; kwargs...)  
    else
        return _coherentstate_via_displacement(sites, alg, αs; kwargs...)
    end
end

"""
    _coherentstate_direct_sum(
        sites::Vector{<:ITensors.Index}, 
        alg::PseudoSite,
        αs::Vector{<:Number};
        state_threshold::Real=1e-12,
        kwargs...
    )

Direct summation approach for coherent states in PseudoSite representation.
Enumerates significant Fock state contributions and sums them as product states.

This approach is efficient for small cutoffs (≤15) where the number of significant
Fock states is manageable. For larger cutoffs, use displacement operator approach.

# Keyword Arguments
- state_threshold::Real: Relative threshold for including Fock states. States with 
  coefficients smaller than `state_threshold * max_coefficient` are excluded from 
  enumeration. Default: 1e-12 (suitable for Float64 precision)
- kwargs...: Passed to `ITensorMPS.add` (e.g., `cutoff`, `maxdim`)
"""
function _coherentstate_direct_sum(
    sites::Vector{<:ITensors.Index}, 
    alg::PseudoSite,
    αs::Vector{<:Number};
    state_threshold::Real=1e-12,
    kwargs...
)
    nmodes = alg.nmodes
    nqubits = _nqubits_per_mode(sites, alg)
    max_occ = 2^nqubits - 1
    
    # |α⟩ = exp(-|α|²/2) Σₙ (αⁿ/√n!) |n⟩
    all_fock_coeffs = [_coherent_fock_coefficients_raw(α, max_occ) for α in αs]
    
    # Determine relative cutoff for significance
    max_coeff = maximum(maximum(abs.(coeffs)) for coeffs in all_fock_coeffs)
    cutoff_threshold = state_threshold * max_coeff
    
    # Storage for significant multi-mode Fock states
    fock_states_list = Vector{Vector{Int}}()
    coefficients_list = Vector{ComplexF64}()
    
    # Recursively enumerate all significant Fock state combinations
    function enumerate_states!(current_state::Vector{Int}, mode_idx::Int)
        if mode_idx > nmodes
            coeff = ComplexF64(1.0)
            for m in 1:nmodes
                coeff *= all_fock_coeffs[m][current_state[m] + 1]
            end
            
            if abs(coeff) >= cutoff_threshold
                push!(fock_states_list, copy(current_state))
                push!(coefficients_list, coeff)
            end
            return
        end
        
        for n in 0:max_occ
            if abs(all_fock_coeffs[mode_idx][n + 1]) >= cutoff_threshold
                current_state[mode_idx] = n
                enumerate_states!(current_state, mode_idx + 1)
            end
        end
    end
    
    current_state = zeros(Int, nmodes)
    enumerate_states!(current_state, 1)
    
    if isempty(fock_states_list)
        error("No significant Fock states found - try larger cutoff or displacement method")
    end
    
    # Normalize collected coefficients
    norm_factor = sqrt(sum(abs2, coefficients_list))
    coefficients_list ./= norm_factor
    
    # Pre-allocate buffers
    qubit_states = Vector{Int}(undef, nmodes * nqubits)
    qubit_state_buffer = Vector{Int}(undef, nqubits)
    result_mps = nothing
    
    for (fock_state, coeff) in zip(fock_states_list, coefficients_list)
        idx = 1
        for mode in 1:nmodes
            _fock_to_qubit!(qubit_state_buffer, fock_state[mode], nqubits)
            copyto!(qubit_states, idx, qubit_state_buffer, 1, nqubits)
            idx += nqubits
        end
        
        term_mps = ITensorMPS.productMPS(sites, qubit_states)
        term_mps[1] *= coeff
        
        if result_mps === nothing
            result_mps = term_mps
        else
            result_mps = ITensorMPS.add(result_mps, term_mps; kwargs...)
        end
    end
    
    ITensorMPS.normalize!(result_mps)
    return BMPS(result_mps, alg)
end

"""
    _coherent_fock_coefficients_raw(α::Number, max_occ::Int)

Compute RAW (unnormalized) Fock state expansion coefficients for coherent state |α⟩.
Returns: cₙ = exp(-|α|²/2) × α^n / √(n!)

These are NOT individually normalized - the normalization happens after 
summing all product state contributions.
"""
function _coherent_fock_coefficients_raw(α::Number, max_occ::Int)
    coeffs = zeros(ComplexF64, max_occ + 1)
    exp_factor = exp(-abs2(α) / 2)
    
    for n in 0:max_occ
        coeffs[n+1] = exp_factor * (α^n) / sqrt(_safe_factorial(n))
    end    
    return coeffs
end

"""
    _coherentstate_via_displacement(
        sites::Vector{<:ITensors.Index},
        alg::PseudoSite,
        αs::Vector{<:Number};
        kwargs...
    )

Displacement operator approach for coherent states.
More efficient for large Hilbert spaces.

# Keyword Arguments
- kwargs...: Passed to `ITensors.apply` (e.g., `cutoff`, `maxdim`)
"""
function _coherentstate_via_displacement(
    sites::Vector{<:ITensors.Index},
    alg::PseudoSite,
    αs::Vector{<:Number};
    kwargs...
)
    psi = vacuumstate(sites, alg)
    @inbounds for mode_idx in 1:alg.nmodes
        α = αs[mode_idx]
        if abs(α) > 1e-10  
            cluster_sites = _mode_cluster(sites, alg, mode_idx)
            D = _displace_qubit(cluster_sites, α)
            psi.mps = ITensors.apply(D, psi.mps; kwargs...)
        end
    end    
    LinearAlgebra.normalize!(psi)
    return psi
end

"""
    _coherent_fock_coefficients(α::Number, max_occ::Int)

Compute Fock state expansion coefficients for coherent state |α⟩.
Returns normalized coefficients for direct use.
"""
function _coherent_fock_coefficients(α::Number, max_occ::Int)
    coeffs = zeros(ComplexF64, max_occ + 1)
    exp_factor = exp(-abs2(α) / 2)
    for n in 0:max_occ
        coeffs[n+1] = exp_factor * (α^n) / sqrt(_safe_factorial(n))
    end
    norm_factor = sqrt(sum(abs2, coeffs))
    coeffs ./= norm_factor
    return coeffs
end