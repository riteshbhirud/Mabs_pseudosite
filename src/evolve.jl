"""
    tebd(psi::BMPS, gate::ITensors.ITensor; kwargs...)
    tebd(psi::BMPS, gates::Vector{ITensors.ITensor}; kwargs...)

Perform time evolution using TEBD algorithm.

Arguments:
- psi::BMPS: Input bosonic MPS
- gate/gates: Time evolution gate(s) to apply

Keyword Arguments:
- kwargs...: Passed directly to `ITensors.apply` (e.g., maxdim, cutoff)

Returns:
- BMPS: Evolved bosonic MPS
"""
function tebd(
    psi::BMPS{<:ITensorMPS.MPS,<:MabsAlg}, 
    gate::ITensors.ITensor; 
    kwargs...
)
    evolved_mps = ITensors.apply(gate, psi.mps; kwargs...)
    return BMPS(evolved_mps, psi.alg)
end

function tebd(
    psi::BMPS{<:ITensorMPS.MPS,<:MabsAlg}, 
    gates::Vector{ITensors.ITensor}; 
    kwargs...
)
    evolved_mps = ITensors.apply(gates, psi.mps; kwargs...)
    return BMPS(evolved_mps, psi.alg)
end

"""
    tdvp(psi::BMPS, H::BMPO, dt::Number; kwargs...)

Perform time evolution using Time Dependent Variational Principle (TDVP) algorithm.

Arguments:
- psi::BMPS: Input bosonic MPS
- H::BMPO: Hamiltonian as bosonic MPO
- dt::Number: Time step

Keyword Arguments:
- kwargs...: All keyword arguments are passed directly to `ITensorMPS.tdvp`
  Common options include:
  - nsweeps::Int: Number of TDVP sweeps (default depends on ITensorMPS)
  - cutoff::Float64: Truncation cutoff (default depends on ITensorMPS)
  - maxdim::Int: Maximum bond dimension (default depends on ITensorMPS)
  - normalize::Bool: Normalize after evolution (default depends on ITensorMPS)

Returns:
- BMPS: Evolved bosonic MPS
"""
function tdvp(
    psi::BMPS{<:ITensorMPS.MPS,<:MabsAlg}, 
    H::BMPO{<:ITensorMPS.MPO,<:MabsAlg}, 
    dt::Number; 
    kwargs...
)
    psi.alg == H.alg || throw(ArgumentError(ALGORITHM_MISMATCH_ERROR))
    
    evolved_mps = ITensorMPS.tdvp(
        H.mpo, 
        dt,  
        psi.mps; 
        kwargs...
    )
    return BMPS(evolved_mps, psi.alg)
end