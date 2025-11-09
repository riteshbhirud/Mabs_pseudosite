"""
    dmrg(
        H::BMPO{<:ITensorMPS.MPO,<:MabsAlg}, 
        psi0::BMPS{<:ITensorMPS.MPS,<:MabsAlg}; 
        nsweeps::Int, 
        kwargs...
    )

Perform DMRG calculation to find ground state of bosonic Hamiltonian.

Arguments:
- H::BMPO: Bosonic Hamiltonian as matrix product operator
- psi0::BMPS: Initial guess for ground state
- nsweeps::Int: Number of DMRG sweeps to perform

Keyword Arguments:
- kwargs...: Additional parameters passed to ITensorMPS.dmrg

Returns:
- Tuple: (energy::Real, psi_gs::BMPS) - ground state energy and state
"""
function dmrg(
    H::BMPO{<:ITensorMPS.MPO,<:MabsAlg}, 
    psi0::BMPS{<:ITensorMPS.MPS,<:MabsAlg}; 
    nsweeps::Int,
    kwargs...
)
    H.alg == psi0.alg || throw(ArgumentError(ALGORITHM_MISMATCH_ERROR))
    energy, converged_mps = ITensorMPS.dmrg(H.mpo, psi0.mps; nsweeps, kwargs...)
    return energy, BMPS(converged_mps, psi0.alg)
end