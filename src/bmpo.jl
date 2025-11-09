"""
Bosonic Matrix Product Operator wrapper that supports different algorithms.
Contains an underlying `ITensorMPS.MPO` object and algorithm specification.

Fields:
- mpo::M: The underlying ITensorMPS.MPO object
- alg::A: Algorithm specification (`Truncated`, `PseudoSite`, or `LocalBasis`)
"""
struct BMPO{M<:ITensorMPS.MPO,A<:MabsAlg}
    mpo::M
    alg::A
end

"""
    BMPO(mpo::ITensorMPS.MPO, alg::Truncated)

Create a `BMPO`` from an existing `MPO` using the Truncated algorithm.

Arguments:
- mpo::ITensorMPS.MPO: Input matrix product operator
- alg::Truncated: Algorithm specification

Returns:
- BMPO: Wrapped bosonic MPO
"""
function BMPO(mpo::ITensorMPS.MPO, alg::Truncated)
    return BMPO{typeof(mpo), typeof(alg)}(mpo, alg)
end

"""
    BMPO(opsum::ITensors.OpSum, sites::Vector{<:ITensors.Index}, alg::Truncated)

Create a `BMPO` directly from an OpSum and sites using the Truncated algorithm.

Arguments:
- opsum::ITensors.OpSum: Operator sum specification
- sites::Vector{<:ITensors.Index}: Vector of site indices  
- alg::Truncated: Algorithm specification

Returns:
- BMPO: Bosonic MPO constructed from OpSum
"""
function BMPO(
    opsum::ITensors.OpSum, 
    sites::Vector{<:ITensors.Index}, 
    alg::Truncated
)
    mpo = ITensorMPS.MPO(opsum, sites)
    return BMPO{typeof(mpo), typeof(alg)}(mpo, alg)
end

ITensorMPS.siteinds(bmpo::BMPO) = ITensorMPS.siteinds(bmpo.mpo)
ITensorMPS.maxlinkdim(bmpo::BMPO) = ITensorMPS.maxlinkdim(bmpo.mpo)
Base.eltype(bmpo::BMPO) = eltype(bmpo.mpo)
Base.length(bmpo::BMPO) = length(bmpo.mpo)
Base.copy(bmpo::BMPO) = BMPO(copy(bmpo.mpo), bmpo.alg)
Base.deepcopy(bmpo::BMPO) = BMPO(deepcopy(bmpo.mpo), bmpo.alg)
ITensorMPS.linkind(bmpo::BMPO, i::Int) = ITensorMPS.linkind(bmpo.mpo, i)
ITensorMPS.siteind(bmpo::BMPO, i::Int) = ITensorMPS.siteind(bmpo.mpo, i)
Base.iterate(bmpo::BMPO) = Base.iterate(bmpo.mpo)
Base.iterate(bmpo::BMPO, state) = Base.iterate(bmpo.mpo, state)
Base.eachindex(bmpo::BMPO) = Base.eachindex(bmpo.mpo)
Base.getindex(bmpo::BMPO, i) = bmpo.mpo[i]
Base.setindex!(bmpo::BMPO, val, i) = (bmpo.mpo[i] = val)
Base.firstindex(bmpo::BMPO) = Base.firstindex(bmpo.mpo)
Base.lastindex(bmpo::BMPO) = Base.lastindex(bmpo.mpo)
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
    @eval ($f)(bmpo::BMPO{<:ITensorMPS.MPO,<:MabsAlg}) = ($f)(bmpo.mpo)
    @eval ($f)(bmpo::BMPO{<:ITensorMPS.MPO,<:MabsAlg}, args...; kwargs...) = ($f)(bmpo.mpo, args...; kwargs...)
end
for f in [
    :(ITensors.prime),
    :(ITensors.swapprime),
    :(ITensors.setprime),
    :(ITensors.noprime),
    :(ITensors.dag),
]
    @eval ($f)(bmpo::BMPO{<:ITensorMPS.MPO,<:MabsAlg}) = BMPO(($f)(bmpo.mpo), bmpo.alg)
end

function ITensorMPS.truncate(bmpo::BMPO{<:ITensorMPS.MPO,<:MabsAlg}; kwargs...)
    truncated_mpo = ITensorMPS.truncate(bmpo.mpo; kwargs...)
    return BMPO(truncated_mpo, bmpo.alg)
end

function ITensorMPS.truncate!(bmpo::BMPO{<:ITensorMPS.MPO,<:MabsAlg}; kwargs...)
    ITensorMPS.truncate!(bmpo.mpo; kwargs...)
    return bmpo
end

function Base.:(+)(
    bmpo1::BMPO{<:ITensorMPS.MPO,<:MabsAlg}, 
    bmpo2::BMPO{<:ITensorMPS.MPO,<:MabsAlg}; 
    kwargs...
)
    bmpo1.alg == bmpo2.alg || throw(ArgumentError(ALGORITHM_MISMATCH_ERROR))
    return ITensorMPS.add(bmpo1, bmpo2; kwargs...)
end

function ITensorMPS.add(
    bmpo1::BMPO{<:ITensorMPS.MPO,<:MabsAlg}, 
    bmpo2::BMPO{<:ITensorMPS.MPO,<:MabsAlg}; 
    kwargs...
)
    bmpo1.alg == bmpo2.alg || throw(ArgumentError(ALGORITHM_MISMATCH_ERROR))
    result_mpo = ITensorMPS.add(bmpo1.mpo, bmpo2.mpo; kwargs...)
    return BMPO(result_mpo, bmpo1.alg)
end

function ITensors.contract(
    bmpo::BMPO{<:ITensorMPS.MPO,<:MabsAlg}, 
    bmps::BMPS{<:ITensorMPS.MPS,<:MabsAlg}; 
    kwargs...
)
    bmpo.alg == bmps.alg || throw(ArgumentError(ALGORITHM_MISMATCH_ERROR))
    result_mps = ITensors.contract(bmpo.mpo, bmps.mps; kwargs...)
    return BMPS(result_mps, bmps.alg)
end

function ITensors.apply(
    bmpo::BMPO{<:ITensorMPS.MPO,<:MabsAlg}, 
    bmps::BMPS{<:ITensorMPS.MPS,<:MabsAlg}; 
    kwargs...
)
    bmpo.alg == bmps.alg || throw(ArgumentError(ALGORITHM_MISMATCH_ERROR))
    result_mps = ITensors.apply(bmpo.mpo, bmps.mps; kwargs...)
    return BMPS(result_mps, bmps.alg)
end

function ITensorMPS.outer(
    bmpo1::BMPO{<:ITensorMPS.MPO,<:MabsAlg}, 
    bmpo2::BMPO{<:ITensorMPS.MPO,<:MabsAlg}; 
    kwargs...
)
    bmpo1.alg == bmpo2.alg || throw(ArgumentError(ALGORITHM_MISMATCH_ERROR))
    outer_result = ITensorMPS.outer(bmpo1.mpo, bmpo2.mpo; kwargs...)
    return BMPO(outer_result, bmpo1.alg)
end

function LinearAlgebra.dot(
    bmpo1::BMPO{<:ITensorMPS.MPO,<:MabsAlg}, 
    bmpo2::BMPO{<:ITensorMPS.MPO,<:MabsAlg}; 
    kwargs...
)
    bmpo1.alg == bmpo2.alg || throw(ArgumentError(ALGORITHM_MISMATCH_ERROR))
    return LinearAlgebra.dot(bmpo1.mpo, bmpo2.mpo; kwargs...)
end

function ITensorMPS.inner(
    bmpo1::BMPO{<:ITensorMPS.MPO,<:MabsAlg}, 
    bmpo2::BMPO{<:ITensorMPS.MPO,<:MabsAlg}; 
    kwargs...
)
    bmpo1.alg == bmpo2.alg || throw(ArgumentError(ALGORITHM_MISMATCH_ERROR))
    return ITensorMPS.inner(bmpo1.mpo, bmpo2.mpo; kwargs...)
end

"""
    BMPO(mpo::ITensorMPS.MPO, alg::PseudoSite)

Create BMPO from existing MPO using the pseudo-site algorithm.
"""
function BMPO(mpo::ITensorMPS.MPO, alg::PseudoSite)
    n_expected = alg.nmodes * _nqubits_per_mode(mpo, alg)
    length(mpo) == n_expected || 
        throw(ArgumentError("MPO length $(length(mpo)) doesn't match expected $n_expected"))
    
    return BMPO{typeof(mpo), typeof(alg)}(mpo, alg)
end

"""
    BMPO(opsum::ITensors.OpSum, sites::Vector{<:ITensors.Index}, alg::PseudoSite)

Create BMPO from OpSum using the pseudo-site algorithm.

NOTE: Only supports simple single-mode operators (N, Id).
Multi-site operators (Adag, A) are not yet supported via OpSum.
Use explicit operator construction instead.
"""
function BMPO(opsum::ITensors.OpSum, sites::Vector{<:ITensors.Index}, alg::PseudoSite)
    n_expected = alg.n_modes * _nqubits_per_mode(alg)
    length(sites) == n_expected || 
        throw(ArgumentError("Sites length $(length(sites)) must match expected $n_expected"))
    qubit_opsum = _qubit_opsum(opsum, sites, alg)
    mpo = ITensorMPS.MPO(qubit_opsum, sites)
    return BMPO{typeof(mpo), typeof(alg)}(mpo, alg)
end

"""
    _qubit_opsum(opsum::ITensors.OpSum, alg::PseudoSite)

Convert bosonic OpSum to a qubit OpSum.
Currently supports: `N` (number operator), `Id` (identity operator)
"""
function _qubit_opsum(
    opsum::ITensors.OpSum, sites::Vector{<:ITensors.Index}, alg::PseudoSite
)
    qubit_opsum = ITensors.OpSum()
    opsum_terms = opsum.data
    nqubits = _nqubits_per_mode(sites, alg)
    @inbounds for term in opsum_terms
        coeff = term.coef
        ops = term.ops
        sites_in_term = term.sites
        @inbounds for (op_name, site_idx) in zip(ops, sites_in_term)
            if op_name == "N"
                @inbounds for i in 1:nqubits
                    weight = coeff * 2^(i-1)
                    global_qubit_idx = (site_idx - 1) * nqubits + i
                    qubit_opsum += weight, "N", global_qubit_idx
                end
            elseif op_name == "Id"
                continue
            else
                error("Operator '$op_name' not supported in PseudoSite OpSum. " *
                      "Supported: N, Id. " *
                      "For other operators, construct them explicitly using qubit maps.")
            end
        end
    end
    return qubit_opsum
end