"""
Bosonic Matrix Product State wrapper that supports different algorithms.
Contains an underlying `ITensorMPS.MPS` object and algorithm specification.

Fields:
- mps::M: The underlying `ITensorMPS.MPS` object
- alg::A: Algorithm specification (`Truncated`, `PseudoSite`, or `LocalBasis`)
"""
mutable struct BMPS{M<:ITensorMPS.MPS,A<:MabsAlg}
    mps::M
    alg::A
end

"""
    BMPS(mps::ITensorMPS.MPS, alg::Truncated)

Create a `BMPS` from an existing `MPS` using the Truncated algorithm.

Arguments:
- mps::ITensorMPS.MPS: Input matrix product state
- alg::Truncated: Algorithm specification

Returns:
- BMPS: Wrapped bosonic MPS
"""
function BMPS(mps::ITensorMPS.MPS, alg::Truncated)
    return BMPS{typeof(mps), typeof(alg)}(mps, alg)
end

function BMPS(sites::Vector{<:ITensors.Index}, alg::Truncated, states::Vector)
    mps = ITensorMPS.productMPS(sites, states)
    return BMPS{typeof(mps), typeof(alg)}(mps, alg)
end

ITensorMPS.siteinds(bmps::BMPS) = ITensorMPS.siteinds(bmps.mps)
ITensorMPS.maxlinkdim(bmps::BMPS) = ITensorMPS.maxlinkdim(bmps.mps)
ITensorMPS.linkind(bmps::BMPS, i::Int) = ITensorMPS.linkind(bmps.mps, i)
ITensorMPS.siteind(bmps::BMPS, i::Int) = ITensorMPS.siteind(bmps.mps, i)
Base.eltype(bmps::BMPS) = eltype(bmps.mps[1])  
Base.length(bmps::BMPS) = length(bmps.mps)

for f in [
    :(ITensorMPS.findsite),
    :(ITensorMPS.findsites),
    :(ITensorMPS.firstsiteinds),
    :(ITensorMPS.expect),
    :(LinearAlgebra.norm),
    :(ITensorMPS.lognorm),
    :(Base.collect),
    :(Base.size)
]
    @eval ($f)(bmps::BMPS{<:ITensorMPS.MPS,<:MabsAlg}) = ($f)(bmps.mps)
    @eval ($f)(bmps::BMPS{<:ITensorMPS.MPS,<:MabsAlg}, args...; kwargs...) = ($f)(bmps.mps, args...; kwargs...)
end
for f in [
    :(ITensors.prime),
    :(ITensors.swapprime),
    :(ITensors.setprime),
    :(ITensors.noprime),
    :(ITensors.dag)
]
    @eval ($f)(bmps::BMPS{<:ITensorMPS.MPS,<:MabsAlg}) = BMPS(($f)(bmps.mps), bmps.alg)
end


Base.copy(bmps::BMPS) = BMPS(copy(bmps.mps), bmps.alg)
Base.deepcopy(bmps::BMPS) = BMPS(deepcopy(bmps.mps), bmps.alg)
Base.iterate(bmps::BMPS) = Base.iterate(bmps.mps)
Base.iterate(bmps::BMPS, state) = Base.iterate(bmps.mps, state)
Base.eachindex(bmps::BMPS) = Base.eachindex(bmps.mps)
Base.getindex(bmps::BMPS, i) = bmps.mps[i]
Base.setindex!(bmps::BMPS, val, i) = (bmps.mps[i] = val)
Base.firstindex(bmps::BMPS) = Base.firstindex(bmps.mps)
Base.lastindex(bmps::BMPS) = Base.lastindex(bmps.mps)

function LinearAlgebra.normalize!(bmps::BMPS{<:ITensorMPS.MPS,<:MabsAlg}; kwargs...)
    LinearAlgebra.normalize!(bmps.mps; kwargs...)
    return bmps
end

function LinearAlgebra.normalize(bmps::BMPS{<:ITensorMPS.MPS,<:MabsAlg}; kwargs...)
    normalized_mps = LinearAlgebra.normalize(bmps.mps; kwargs...)
    return BMPS(normalized_mps, bmps.alg)
end

function ITensorMPS.orthogonalize!(bmps::BMPS{<:ITensorMPS.MPS,<:MabsAlg}, j::Int; kwargs...)
    ITensorMPS.orthogonalize!(bmps.mps, j; kwargs...)
    return bmps
end

function ITensorMPS.orthogonalize(bmps::BMPS{<:ITensorMPS.MPS,<:MabsAlg}, j::Int; kwargs...)
    orthog_mps = ITensorMPS.orthogonalize(bmps.mps, j; kwargs...)
    return BMPS(orthog_mps, bmps.alg)
end

function ITensorMPS.truncate(bmps::BMPS{<:ITensorMPS.MPS,<:MabsAlg}; kwargs...)
    truncated_mps = ITensorMPS.truncate(bmps.mps; kwargs...)
    return BMPS(truncated_mps, bmps.alg)
end

function ITensorMPS.truncate!(bmps::BMPS{<:ITensorMPS.MPS,<:MabsAlg}; kwargs...)
    ITensorMPS.truncate!(bmps.mps; kwargs...)
    return bmps
end

function Base.:(+)(
    bmps1::BMPS{<:ITensorMPS.MPS,<:MabsAlg}, 
    bmps2::BMPS{<:ITensorMPS.MPS,<:MabsAlg}; 
    kwargs...
)
    bmps1.alg == bmps2.alg || throw(ArgumentError(ALGORITHM_MISMATCH_ERROR))
    result_mps = Base.:(+)(bmps1.mps, bmps2.mps; kwargs...)
    return BMPS(result_mps, bmps1.alg)
end

function ITensorMPS.add(
    bmps1::BMPS{<:ITensorMPS.MPS,<:MabsAlg}, 
    bmps2::BMPS{<:ITensorMPS.MPS,<:MabsAlg}; 
    kwargs...
)
    return Base.:(+)(bmps1, bmps2; kwargs...)
end

function ITensorMPS.outer(
    bmps1::BMPS{<:ITensorMPS.MPS,<:MabsAlg}, 
    bmps2::BMPS{<:ITensorMPS.MPS,<:MabsAlg}; 
    kwargs...
)
    bmps1.alg == bmps2.alg || throw(ArgumentError(ALGORITHM_MISMATCH_ERROR))
    outer_result = ITensorMPS.outer(bmps1.mps, bmps2.mps; kwargs...)
    return BMPO(outer_result, bmps1.alg)
end

function LinearAlgebra.dot(
    bmps1::BMPS{<:ITensorMPS.MPS,<:MabsAlg}, 
    bmps2::BMPS{<:ITensorMPS.MPS,<:MabsAlg}; 
    kwargs...
)
    bmps1.alg == bmps2.alg || throw(ArgumentError(ALGORITHM_MISMATCH_ERROR))
    return LinearAlgebra.dot(bmps1.mps, bmps2.mps; kwargs...)
end

function ITensorMPS.inner(
    bmps1::BMPS{<:ITensorMPS.MPS,<:MabsAlg}, 
    bmps2::BMPS{<:ITensorMPS.MPS,<:MabsAlg}; 
    kwargs...
)
    bmps1.alg == bmps2.alg || throw(ArgumentError(ALGORITHM_MISMATCH_ERROR))
    return ITensorMPS.inner(bmps1.mps, bmps2.mps; kwargs...)
end

"""
    BMPS(mps::ITensorMPS.MPS, alg::PseudoSite)

Create BMPS from existing MPS using PseudoSite algorithm.
"""
function BMPS(mps::ITensorMPS.MPS, alg::PseudoSite)
    length(mps) % alg.nmodes == 0 || 
        throw(ArgumentError("MPS length $(length(mps)) must be divisible by nmodes $(alg.nmodes)"))
    n_expected = alg.nmodes * _nqubits_per_mode(mps, alg)
    length(mps) == n_expected || 
        throw(ArgumentError("MPS length $(length(mps)) doesn't match expected $n_expected"))
    
    return BMPS{typeof(mps), typeof(alg)}(mps, alg)
end

"""
    BMPS(sites::Vector{<:ITensors.Index}, states::Vector, alg::PseudoSite)

Create a bosonic matrix product state in the pseudo-site representation.

Arguments:
- sites::Vector{ITensors.Index}: Qubit site indices
- states::Vector: Bosonic occupation numbers for each mode
- alg::PseudoSite: Algorithm specification

Returns:
- BMPS: Product state in the pseudo-site representation
"""
function BMPS(sites::Vector{<:ITensors.Index}, states::Vector, alg::PseudoSite)
    nqubits = _nqubits_per_mode(sites, alg)
    n_expected = alg.nmodes * nqubits
    length(sites) == n_expected || 
        throw(ArgumentError("Sites length $(length(sites)) must match expected $n_expected"))
    length(states) == alg.nmodes || 
        throw(ArgumentError("Number of states $(length(states)) must match modes $(alg.nmodes))"))
    qubit_states = Vector{Int}(undef, n_expected) 
    qubit_state_buffer = Vector{Int}(undef, nqubits)
    idx = 1
    
    @inbounds for (mode_idx, n) in enumerate(states)
        n <= 2^nqubits - 1 || 
            throw(ArgumentError("State $n exceeds maximum $(2^nqubits - 1)"))
        _fock_to_qubit!(qubit_state_buffer, n, nqubits)
        copyto!(qubit_states, idx, qubit_state_buffer, 1, nqubits)
        idx += nqubits
    end
    mps = ITensorMPS.productMPS(sites, qubit_states)
    return BMPS{typeof(mps), typeof(alg)}(mps, alg)
end