"""
    PseudoSite <: MabsAlg

Algorithm for representing bosonic systems using quantics (binary) mapping.
Maps a bosonic Hilbert space of dimension 2^N to N qubits per mode.

Fields:
- n_modes::Int: Number of bosonic modes
- fock_cutoff::Int: Maximum occupation in bosonic space (must be 2^N - 1)

The quantics mapping represents occupation number states in binary:
|n⟩ → |b_{N-1}⟩⊗|b_{N-2}⟩⊗...⊗|b_0⟩ where n = Σᵢ bᵢ × 2^i

# Constructors

## PseudoSite(n_modes::Int, fock_cutoff::Int)
Create algorithm by specifying number of modes and Fock space cutoff.
Use with `create_qubit_sites(alg)` to generate standard qubit indices.

## PseudoSite(sites::Vector{ITensors.Index}, fock_cutoff::Int)
Create algorithm from user-provided qubit sites.
Number of modes is automatically inferred from site count.

# Examples

    # Option 1: Standard usage (let Mabs create sites)
    alg = PseudoSite(2, 7)
    sites = create_qubit_sites(alg)
    psi = random_bmps(sites, alg)

    # Option 2: Custom sites (user provides sites)
    my_sites = [ITensors.Index(2, "Q,n=\$i") for i in 1:6]
    alg = PseudoSite(my_sites, 7)  # 2 modes inferred
    psi = random_bmps(my_sites, alg)
"""
struct PseudoSite <: MabsAlg 
    n_modes::Int
    fock_cutoff::Int
    
    function PseudoSite(n_modes::Int, fock_cutoff::Int)
        N = log2(fock_cutoff + 1)
        isinteger(N) || throw(ArgumentError(PSEUDOSITE_ERROR))
        
        return new(n_modes, fock_cutoff)
    end
end

"""
    PseudoSite(sites::Vector{ITensors.Index}, fock_cutoff::Int)

Create PseudoSite algorithm from user-provided qubit sites.
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

"""
    n_qubits_per_mode(alg::PseudoSite)

Get number of qubits needed per mode: log₂(fock_cutoff + 1)
"""
n_qubits_per_mode(alg::PseudoSite) = Int(log2(alg.fock_cutoff + 1))

"""
    create_qubit_sites(alg::PseudoSite)

Generate qubit sites for PseudoSite algorithm.
Creates n_modes × n_qubits_per_mode qubit indices.

Returns:
- Vector{ITensors.Index}: Qubit sites for the system
"""
function create_qubit_sites(alg::PseudoSite)
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

function Base.:(==)(alg1::PseudoSite, alg2::PseudoSite)
    return alg1.n_modes == alg2.n_modes &&
           alg1.fock_cutoff == alg2.fock_cutoff
end

"""
    get_mode_cluster(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int)

Get the qubit cluster indices for a specific bosonic mode.

Arguments:
- sites::Vector{ITensors.Index}: Qubit sites for the system
- alg::PseudoSite: Algorithm specification
- mode::Int: Mode number (1-indexed)

Returns:
- Vector{ITensors.Index}: Qubit sites for this mode
"""
function get_mode_cluster(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int)
    (mode < 1 || mode > alg.n_modes) && 
        throw(ArgumentError("Mode $mode out of range [1, $(alg.n_modes)]"))
    
    n_qubits = n_qubits_per_mode(alg)
    start_idx = (mode - 1) * n_qubits + 1
    end_idx = mode * n_qubits
    
    return sites[start_idx:end_idx]
end

"""
    get_mode_indices(alg::PseudoSite, mode::Int)

Get the position indices in MPS for a specific mode's qubit cluster.

Arguments:
- alg::PseudoSite: Algorithm specification
- mode::Int: Mode number (1-indexed)

Returns:
- UnitRange{Int}: Position indices for this mode's qubits
"""
function get_mode_indices(alg::PseudoSite, mode::Int)
    (mode < 1 || mode > alg.n_modes) && 
        throw(ArgumentError("Mode $mode out of range [1, $(alg.n_modes)]"))
    
    n_qubits = n_qubits_per_mode(alg)
    start_idx = (mode - 1) * n_qubits + 1
    end_idx = mode * n_qubits
    
    return start_idx:end_idx
end

"""
    decimal_to_binary_state(n::Int, n_qubits::Int)

Convert decimal occupation number to binary state vector.

Arguments:
- n::Int: Occupation number
- n_qubits::Int: Number of qubits

Returns:
- Vector{Int}: Binary representation [b_0, b_1, ..., b_{N-1}] where bᵢ ∈ {1,2} (ITensor convention)
"""
function decimal_to_binary_state(n::Int, n_qubits::Int)
    n >= 0 || throw(ArgumentError("Occupation number must be non-negative"))
    n < 2^n_qubits || throw(ArgumentError("Occupation $n exceeds max for $n_qubits qubits"))
    binary_state = Vector{Int}(undef, n_qubits)
    for i in 1:n_qubits
        bit = (n >> (i-1)) & 1  
        binary_state[i] = bit + 1  
    end
    return binary_state
end

"""
    binary_state_to_decimal(binary_state::Vector{Int})

Convert binary state vector to decimal occupation number.

Arguments:
- binary_state::Vector{Int}: Binary state [b_0, b_1, ..., b_{N-1}] where bᵢ ∈ {1,2}

Returns:
- Int: Occupation number n = Σᵢ (bᵢ - 1) × 2^(i-1)
"""
function binary_state_to_decimal(binary_state::Vector{Int})
    n = 0
    for (i, bit) in enumerate(binary_state)
        (bit == 1 || bit == 2) || throw(ArgumentError("Binary state values must be 1 or 2"))
        if bit == 2 
            n += 2^(i-1)
        end
    end
    return n
end