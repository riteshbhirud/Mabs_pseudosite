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

function BMPS(sites::Vector{<:ITensors.Index}, states::Vector, alg::Truncated)
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
    :(Base.length),
    :(Base.size)
]
    @eval ($f)(bmps::BMPS{<:ITensorMPS.MPS,Truncated}) = ($f)(bmps.mps)
    @eval ($f)(bmps::BMPS{<:ITensorMPS.MPS,Truncated}, args...; kwargs...) = ($f)(bmps.mps, args...; kwargs...)
end
for f in [
    :(ITensors.prime),
    :(ITensors.swapprime),
    :(ITensors.setprime),
    :(ITensors.noprime),
    :(ITensors.dag)
]
    @eval ($f)(bmps::BMPS{<:ITensorMPS.MPS,Truncated}) = BMPS(($f)(bmps.mps), bmps.alg)
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
    bmps1.alg == bmps2.alg || throw(ArgumentError("Algorithms must match"))
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
    bmps1.alg == bmps2.alg || throw(ArgumentError("Algorithms must match"))
    outer_result = ITensorMPS.outer(bmps1.mps, bmps2.mps; kwargs...)
    return BMPO(outer_result, bmps1.alg)
end

function LinearAlgebra.dot(
    bmps1::BMPS{<:ITensorMPS.MPS,<:MabsAlg}, 
    bmps2::BMPS{<:ITensorMPS.MPS,<:MabsAlg}; 
    kwargs...
)
    bmps1.alg == bmps2.alg || throw(ArgumentError("Algorithms must match"))
    return LinearAlgebra.dot(bmps1.mps, bmps2.mps; kwargs...)
end

function ITensorMPS.inner(
    bmps1::BMPS{<:ITensorMPS.MPS,<:MabsAlg}, 
    bmps2::BMPS{<:ITensorMPS.MPS,<:MabsAlg}; 
    kwargs...
)
    bmps1.alg == bmps2.alg || throw(ArgumentError("Algorithms must match"))
    return ITensorMPS.inner(bmps1.mps, bmps2.mps; kwargs...)
end

"""
    BMPS(mps::ITensorMPS.MPS, alg::PseudoSite)

Create BMPS from existing MPS using PseudoSite algorithm.
"""
function BMPS(mps::ITensorMPS.MPS, alg::PseudoSite)
    n_expected = alg.n_modes * n_qubits_per_mode(alg)
    length(mps) == n_expected || 
        throw(ArgumentError("MPS length $(length(mps)) doesn't match expected $n_expected"))
    
    return BMPS{typeof(mps), typeof(alg)}(mps, alg)
end

"""
    BMPS(sites::Vector{<:ITensors.Index}, states::Vector, alg::PseudoSite)

Create product state BMPS in PseudoSite representation.

Arguments:
- sites::Vector{ITensors.Index}: Should be alg.sites
- states::Vector: Bosonic occupation numbers for each mode
- alg::PseudoSite: Algorithm specification

Returns:
- BMPS: Product state in quantics representation
"""
function BMPS(sites::Vector{<:ITensors.Index}, states::Vector, alg::PseudoSite)
    n_expected = alg.n_modes * n_qubits_per_mode(alg)
    length(sites) == n_expected || 
        throw(ArgumentError("Sites length $(length(sites)) must match expected $n_expected"))
    length(states) == alg.n_modes || 
        throw(ArgumentError("Number of states $(length(states)) must match modes $(alg.n_modes))"))
    n_qubits = n_qubits_per_mode(alg)
    binary_states = Vector{Int}(undef, n_expected) 
    idx = 1
    for (mode_idx, n) in enumerate(states)
        n <= alg.fock_cutoff || 
            throw(ArgumentError("State $n exceeds maximum $(alg.fock_cutoff)"))
        binary_state = Mabs.decimal_to_binary_state(n, n_qubits)
        copyto!(binary_states, idx, binary_state, 1, n_qubits)
        idx += n_qubits
    end
    mps = ITensorMPS.productMPS(sites, binary_states)
    return BMPS{typeof(mps), typeof(alg)}(mps, alg)
end
for f in [
    :(ITensorMPS.findsite),
    :(ITensorMPS.findsites),
    :(ITensorMPS.firstsiteinds),
    :(ITensorMPS.expect),
    :(LinearAlgebra.norm),
    :(ITensorMPS.lognorm),
    :(Base.collect),
    :(Base.length),
    :(Base.size)
]
    @eval ($f)(bmps::BMPS{<:ITensorMPS.MPS,<:PseudoSite}) = ($f)(bmps.mps)
    @eval ($f)(bmps::BMPS{<:ITensorMPS.MPS,<:PseudoSite}, args...; kwargs...) = 
        ($f)(bmps.mps, args...; kwargs...)
end

for f in [
    :(ITensors.prime),
    :(ITensors.swapprime),
    :(ITensors.setprime),
    :(ITensors.noprime),
    :(ITensors.dag)
]
    @eval ($f)(bmps::BMPS{<:ITensorMPS.MPS,<:PseudoSite}) = BMPS(($f)(bmps.mps), bmps.alg)
end
