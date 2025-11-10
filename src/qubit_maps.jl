"""
    set_cluster_matrix_element!(
        op::ITensors.ITensor, 
        cluster_sites::Vector{<:ITensors.Index},
        bra_state::Vector{Int}, 
        ket_state::Vector{Int},
        value::Number
    )

Set matrix element in qubit cluster operator.

Arguments:
- op::ITensors.ITensor: Operator to modify (must have primed and unprimed cluster indices)
- cluster_sites::Vector{<:ITensors.Index}: Qubit sites (unprimed)
- bra_state::Vector{Int}: Bra state in binary (for primed indices)
- ket_state::Vector{Int}: Ket state in binary (for unprimed indices)
- value::Number: Matrix element value
"""
function set_cluster_matrix_element!(
    op::ITensors.ITensor, 
    cluster_sites::Vector{<:ITensors.Index},
    bra_state::Vector{Int}, 
    ket_state::Vector{Int},
    value::Number
)
    nqubits = length(cluster_sites)
    @assert length(bra_state) == nqubits "Bra state length mismatch"
    @assert length(ket_state) == nqubits "Ket state length mismatch"
    
    index_specs = Vector{Pair{ITensors.Index,Int}}(undef, 2 * nqubits)
    idx = 1
    @inbounds for i in 1:nqubits
        index_specs[idx] = cluster_sites[i]' => bra_state[i]
        idx += 1
        index_specs[idx] = cluster_sites[i] => ket_state[i]
        idx += 1
    end
    op[index_specs...] = convert(ComplexF64, value)
    return op
end

"""
    _matrix_to_cluster_operator(
        matrix::Matrix{ComplexF64}, 
        cluster_sites::Vector{<:ITensors.Index}
    )

Convert a matrix in Fock basis to ITensor operator in qubit cluster basis.

Arguments:
- matrix::Matrix{ComplexF64}: Operator matrix in Fock basis (dim × dim)
- cluster_sites::Vector{<:ITensors.Index}: Qubit sites for cluster

Returns:
- ITensors.ITensor: Operator with primed and unprimed cluster indices
"""
function _matrix_to_cluster_operator(
    matrix::Matrix{ComplexF64}, 
    cluster_sites::Vector{<:ITensors.Index}
)
    nqubits = length(cluster_sites)
    dim = size(matrix, 1)
    @assert size(matrix, 2) == dim "Matrix must be square"
    @assert dim == 2^nqubits "Matrix dimension must match qubit space: $dim ≠ $(2^nqubits)"
    op = ITensors.ITensor(ComplexF64, cluster_sites'..., cluster_sites...)
    qubit_m_buffer = Vector{Int}(undef, nqubits)  
    qubit_n_buffer = Vector{Int}(undef, nqubits) 
    
    @inbounds for m in 0:(dim-1)
        _fock_to_qubit!(qubit_m_buffer, m, nqubits)  
        @inbounds for n in 0:(dim-1)
            element = matrix[m+1, n+1]
            if abs(element) > 1e-15
                _fock_to_qubit!(qubit_n_buffer, n, nqubits)  
                set_cluster_matrix_element!(op, cluster_sites, qubit_m_buffer, qubit_n_buffer, element)
            end
        end
    end
    return op
end

"""
    _number_qubit(cluster_sites::Vector{<:ITensors.Index})

Create number operator in the pseudo-site representation.

Arguments:
- cluster_sites::Vector{<:ITensors.Index}: Qubit sites for one mode

Returns:
- ITensors.ITensor: Number operator n̂ = Σₙ n|n⟩⟨n|
"""
function _number_qubit(cluster_sites::Vector{<:ITensors.Index})
    nqubits = length(cluster_sites)
    max_occ = 2^nqubits - 1
    n_matrix = zeros(ComplexF64, max_occ+1, max_occ+1)
    @inbounds for n in 0:max_occ
        n_matrix[n+1, n+1] = n
    end
    return _matrix_to_cluster_operator(n_matrix, cluster_sites)
end

"""
    _create_qubit(cluster_sites::Vector{<:ITensors.Index})

Create bosonic creation operator in the pseudo-site representation.

Arguments:
- cluster_sites::Vector{<:ITensors.Index}: Qubit sites for one mode

Returns:
- ITensors.ITensor: Creation operator a† with ⟨n+1|a†|n⟩ = √(n+1)
"""
function _create_qubit(cluster_sites::Vector{<:ITensors.Index})
    nqubits = length(cluster_sites)
    max_occ = 2^nqubits - 1
    a_dag_matrix = zeros(ComplexF64, max_occ+1, max_occ+1)
    @inbounds for n in 0:max_occ-1
        a_dag_matrix[n+2, n+1] = sqrt(n + 1)
    end
    return _matrix_to_cluster_operator(a_dag_matrix, cluster_sites)
end

"""
    _destroy_qubit(cluster_sites::Vector{<:ITensors.Index})

Create bosonic annihilation operator in the pseudo-site representation.

Arguments:
- cluster_sites::Vector{<:ITensors.Index}: Qubit sites for one mode

Returns:
- ITensors.ITensor: Annihilation operator a with ⟨n-1|a|n⟩ = √n
"""
function _destroy_qubit(cluster_sites::Vector{<:ITensors.Index})
    nqubits = length(cluster_sites)
    max_occ = 2^nqubits - 1
    a_matrix = zeros(ComplexF64, max_occ+1, max_occ+1)
    @inbounds for n in 1:max_occ
        a_matrix[n, n+1] = sqrt(n)
    end
    return _matrix_to_cluster_operator(a_matrix, cluster_sites)
end

"""
    _displace_qubit(cluster_sites::Vector{<:ITensors.Index}, α::Number)

Create displacement operator in the pseudo-site representation.
D(α) = exp(α*a† - α*a)

Arguments:
- cluster_sites::Vector{<:ITensors.Index}: Qubit sites for one mode
- α::Number: Displacement amplitude (complex)

Returns:
- ITensors.ITensor: Displacement operator
"""
function _displace_qubit(cluster_sites::Vector{<:ITensors.Index}, α::Number)
    nqubits = length(cluster_sites)
    max_occ = 2^nqubits - 1
    D_matrix = zeros(ComplexF64, max_occ+1, max_occ+1)
    exp_factor = exp(-abs2(α) / 2)
    @inbounds for m in 0:max_occ
        @inbounds for n in 0:max_occ
            D_matrix[m+1, n+1] = _displace_matrix_element(m, n, α) * exp_factor
        end
    end
    return _matrix_to_cluster_operator(D_matrix, cluster_sites)
end

"""
    _displace_matrix_element(m::Int, n::Int, α::Number)

Compute matrix element ⟨m|D(α)|n⟩ without exponential prefactor.

Uses the formula:
⟨m|D(α)|n⟩ = α^(m-n)/√((m-n)! m!/n!)  if m ≥ n
           = (-α*)^(n-m)/√((n-m)! n!/m!)  if m < n
"""
function _displace_matrix_element(m::Int, n::Int, α::Number)
    if m >= n
        k = m - n
        if k == 0
            return 1.0
        end
        return (α^k) / sqrt(_safe_factorial(k)) * 
               sqrt(_safe_factorial(m) / _safe_factorial(n))
    else
        k = n - m
        return ((-conj(α))^k) / sqrt(_safe_factorial(k)) * 
               sqrt(_safe_factorial(n) / _safe_factorial(m))
    end
end

"""
    _squeeze_qubit(cluster_sites::Vector{<:ITensors.Index}, ξ::Number)

Create squeezing operator in the pseudo-site representation.
S(ξ) = exp(0.5*(ξ*a†² - ξ*a²))

Arguments:
- cluster_sites::Vector{<:ITensors.Index}: Qubit sites for one mode
- ξ::Number: Squeezing parameter (complex)

Returns:
- ITensors.ITensor: Squeezing operator
"""
function _squeeze_qubit(cluster_sites::Vector{<:ITensors.Index}, ξ::Number)
    nqubits = length(cluster_sites)
    max_occ = 2^nqubits - 1
    S_matrix = zeros(ComplexF64, max_occ+1, max_occ+1)
    r = abs(ξ)
    φ = angle(ξ)
    @inbounds for m in 0:max_occ
        @inbounds for n in 0:max_occ
            if (m + n) % 2 == 0  
                S_matrix[m+1, n+1] = _squeezing_matrix_element(m, n, r, φ)
            end
        end
    end
    
    return _matrix_to_cluster_operator(S_matrix, cluster_sites)
end

"""
    _squeezing_matrix_element(m::Int, n::Int, r::Real, φ::Real)

Compute matrix element ⟨m|S(ξ)|n⟩ for squeezing operator.

"""
function _squeezing_matrix_element(m::Int, n::Int, r::Real, φ::Real)
    k_max = min(m, n) ÷ 2
    element = 0.0 + 0.0im
    cr = cosh(r)
    tr = tanh(r)
    phase = exp(2im * φ)
    prefactor = -0.5 * tr * phase
    inv_sqrt_cr = 1.0 / sqrt(cr)
    for k in 0:k_max
        coeff = sqrt(_safe_factorial(m) * _safe_factorial(n)) / 
               (_safe_factorial(k) * _safe_factorial(m - 2k) * _safe_factorial(n - 2k))
        
        coeff *= (prefactor^k) * inv_sqrt_cr
        if m == n && k == 0
            coeff *= inv_sqrt_cr
        end
        element += coeff
    end
    return element
end

"""
    _kerr_qubit(cluster_sites::Vector{<:ITensors.Index}, χ::Real, t::Real)

Create Kerr evolution operator in the pseudo-site representation.
K(χ,t) = exp(-i*χ*t*n²)

Arguments:
- cluster_sites::Vector{<:ITensors.Index}: Qubit sites for one mode
- χ::Real: Kerr nonlinearity strength
- t::Real: Evolution time

Returns:
- ITensors.ITensor: Kerr evolution operator
"""
function _kerr_qubit(cluster_sites::Vector{<:ITensors.Index}, χ::Real, t::Real)
    nqubits = length(cluster_sites)
    max_occ = 2^nqubits - 1
    K_matrix = zeros(ComplexF64, max_occ+1, max_occ+1)
    @inbounds for n in 0:max_occ
        K_matrix[n+1, n+1] = exp(-1im * χ * t * n^2)
    end
    return _matrix_to_cluster_operator(K_matrix, cluster_sites)
end

"""
    expect_photon_number(psi::BMPS{<:ITensorMPS.MPS,PseudoSite}, mode::Int)

Compute photon number expectation value for a specific mode.

Arguments:
- psi::BMPS: State in PseudoSite representation
- mode::Int: Mode index (1 to n_modes)

Returns:
- Float64: ⟨n⟩ for the specified mode
"""
function expect_photon_number(psi::BMPS{<:ITensorMPS.MPS,<:PseudoSite}, mode::Int)
    alg = psi.alg
    sites = ITensorMPS.siteinds(psi.mps)
    cluster_sites = _mode_cluster(sites, alg, mode)
    n_op = _number_qubit(cluster_sites)
    n_psi = ITensors.apply(n_op, psi.mps)
    return real(ITensorMPS.inner(psi.mps, n_psi))
end

"""
Build hopping MPO for adjacent modes using direct construction.
Handles â†ᵢ âᵢ₊₁ term.
"""
