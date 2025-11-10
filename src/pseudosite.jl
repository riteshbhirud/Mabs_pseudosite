"""
    PseudoSite <: MabsAlg
    PseudoSite(nmodes::Int)

Algorithm for representing bosonic systems using a binary mapping.
Maps a bosonic Hilbert space of dimension `2^N` to `N` qubits per mode.

Fields:
- nmodes::Int: Number of bosonic modes

The qubit mapping represents occupation number states in binary:
`|n⟩ → |bₙ₋₁⟩⊗|bₙ₋₂⟩⊗...⊗|b₀⟩` where `n = Σᵢ bᵢ × 2^i`.

# Examples

```julia
alg = PseudoSite(2) # of bosonic modes
N = 4
sites = ITensorMPS.siteinds("Qubit", 2^N - 1)
psi = random_bmps(sites, alg)
```
"""
struct PseudoSite <: MabsAlg 
    nmodes::Int
end

#=
"""
    PseudoSite(sites::Vector{ITensors.Index})

Create PseudoSite algorithm from a vector qubit site indices.
The number of modes is inferred from: n_modes = length(sites) / log₂(fock_cutoff + 1)

Arguments:
- sites::Vector{ITensors.Index}: User-provided qubit site indices
- fock_cutoff::Int: Maximum occupation number (must be 2^N - 1)

Returns:
- PseudoSite: Algorithm specification with inferred n_modes

# Example

    # Create custom sites
    sites = [ITensors.Index(2, "MyQubit,n=\$i") for i in 1:6]
    alg = PseudoSite(sites, 7)  # 2 modes (6 sites / 3 qubits per mode)
"""
function PseudoSite(sites::Vector{<:ITensors.Index}, fock_cutoff::Int)
    N = log2(fock_cutoff + 1)
    isinteger(N) || throw(ArgumentError(PSEUDOSITE_ERROR))
    n_qubits_per_mode = Int(N)
    
    length(sites) % n_qubits_per_mode == 0 || 
        throw(ArgumentError(
            "Number of sites ($(length(sites))) not divisible by n_qubits_per_mode ($n_qubits_per_mode). " *
            "Expected length to be a multiple of $n_qubits_per_mode for fock_cutoff=$fock_cutoff."
        ))
    
    n_modes = div(length(sites), n_qubits_per_mode)
    
    for (i, site) in enumerate(sites)
        ITensors.dim(site) == 2 || 
            throw(ArgumentError("Site $i has dimension $(ITensors.dim(site)), expected 2 (qubit)"))
    end
    
    return PseudoSite(n_modes, fock_cutoff)
end
=#


"""
    _nqubits_per_mode(mps::ITensorMPS.MPS, alg::PseudoSite)
    _nqubits_per_mode(sites::Vector{<:ITensors.Index}, alg::PseudoSite)

Get number of qubits per mode in the pseudo-site algorithm.
"""
_nqubits_per_mode(mps::ITensorMPS.MPS, alg::PseudoSite) = Int(length(mps) / alg.nmodes)
_nqubits_per_mode(sites::Vector{<:ITensors.Index}, alg::PseudoSite) = Int(length(sites) / alg.nmodes)

#=
"""
    qubitsites(alg::PseudoSite, )

Generate qubit sites for PseudoSite algorithm.
Creates `nmodes` × n_qubits_per_mode qubit indices.

Returns:
- Vector{ITensors.Index}: Qubit sites for the system
"""
function qubitsites(alg::PseudoSite)
    n_qubits = n_qubits_per_mode(alg)
    n_total = alg.n_modes * n_qubits
    sites = Vector{ITensors.Index}(undef, n_total)
    
    idx = 1
    for mode in 1:alg.n_modes
        for qubit in 1:n_qubits
            tag = "Qubit,Mode=$mode,Bit=$qubit"
            sites[idx] = ITensors.Index(2, tag)
            idx += 1
        end
    end
    
    return sites
end
=#

Base.:(==)(alg1::PseudoSite, alg2::PseudoSite) = alg1.nmodes == alg2.nmodes

"""
    _mode_cluster(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int)

Get the qubit cluster indices for a specific bosonic mode.

Arguments:
- sites::Vector{ITensors.Index}: Qubit sites for the system
- alg::PseudoSite: Algorithm specification
- mode::Int: Mode number

Returns:
- Vector{ITensors.Index}: Qubit sites for this mode
"""
function _mode_cluster(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int)
    (mode < 1 || mode > alg.nmodes) && 
        throw(ArgumentError("Mode $mode out of range [1, $(alg.nmodes)]"))
    nqubits = _nqubits_per_mode(sites, alg)
    start_idx = (mode - 1) * nqubits + 1
    end_idx = mode * nqubits
    return sites[start_idx:end_idx]
end

"""
    _mode_indices(alg::PseudoSite, mode::Int)

Get the position indices in MPS for a specific mode's qubit cluster.

Arguments:
- alg::PseudoSite: Algorithm specification
- mode::Int: Mode number

Returns:
- UnitRange{Int}: Position indices for this mode's qubits
"""
function _mode_indices(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int)
    (mode < 1 || mode > alg.nmodes) && 
        throw(ArgumentError("Mode $mode out of range [1, $(alg.nmodes)]"))
    
    nqubits = _nqubits_per_mode(sites, alg)
    start_idx = (mode - 1) * nqubits + 1
    end_idx = mode * nqubits
    
    return start_idx:end_idx
end

"""
    _fock_to_qubit!(buffer::Vector{Int}, n::Int, nqubits::Int)

convert occupation number to binary state vector.
Writes directly into buffer to avoid allocations.

Arguments:
- buffer::Vector{Int}: Pre-allocated buffer (length nqubits)
- n::Int: Occupation number  
- nqubits::Int: Number of qubits

Returns:
- buffer (modified in-place)
"""
function _fock_to_qubit!(buffer::Vector{Int}, n::Int, nqubits::Int)
    n >= 0 || throw(ArgumentError("Occupation number must be non-negative"))
    n < 2^nqubits || throw(ArgumentError("Occupation $n exceeds max for $nqubits qubits"))
    
    @inbounds for i in 1:nqubits
        bit = (n >> (i-1)) & 1
        buffer[i] = bit + 1
    end
    return buffer
end

"""
    _qubit_to_fock(binary_state::Vector{Int})

Convert binary state vector to decimal occupation number.

Arguments:
- binary_state::Vector{Int}: Binary state `[b₀, b₁, ..., bₙ₋₁]` where `bᵢ ∈ {1,2}`

Returns:
- Int: Occupation number `n = Σᵢ (bᵢ - 1) × 2^(i-1)`
"""
function _qubit_to_fock(binary_state::Vector{Int})
    n = 0
    for (i, bit) in enumerate(binary_state)
        (bit == 1 || bit == 2) || throw(ArgumentError("Binary state values must be 1 or 2"))
        if bit == 2 
            n += 2^(i-1)
        end
    end
    return n
end