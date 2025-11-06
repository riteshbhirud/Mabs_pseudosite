
# helper function for safe factorial calculation to avoid performance issues.
function _safe_factorial(n::Int)
    if n <= 20
        return factorial(n)
    else
        return factorial(big(n))
    end
end

"""
    create(site::ITensors.Index, alg::MabsAlg)

Create the bosonic creation operator (raising operator) for a given site.

Arguments:
- site::ITensors.Index: Site index with bosonic tag
- alg::MabsAlg: Bosonic MPS algorithm

Returns:
- ITensors.ITensor: Creation operator tensor
"""
function create(site::ITensors.Index, alg::Truncated)
    max_occ = ITensors.dim(site) - 1
    op = ITensors.ITensor(ComplexF64, site', site)
    @inbounds for n in 0:(max_occ-1)
        op[n+2, n+1] = sqrt(n+1)
    end
    return op
end

"""
    destroy(site::ITensors.Index, alg::MabsAlg)

Create the bosonic annihilation operator (lowering operator) for a given site.

Arguments:
- site::ITensors.Index: Site index with bosonic tag
- alg::MabsAlg: Bosonic MPS algorithm

Returns:
- ITensors.ITensor: Annihilation operator tensor
"""
function destroy(site::ITensors.Index, alg::Truncated)
    max_occ = ITensors.dim(site) - 1
    op = ITensors.ITensor(ComplexF64, site', site)
    @inbounds for n in 1:max_occ
        op[n, n+1] = sqrt(n)
    end
    return op
end

"""
    number(site::ITensors.Index, alg::Truncated)

Create the bosonic number operator for a given site.

Arguments:
- site::ITensors.Index: Site index with bosonic tag
- alg::MabsAlg: Bosonic MPS algorithm

Returns:
- ITensors.ITensor: Number operator tensor
"""
function number(site::ITensors.Index, alg::Truncated)
    max_occ = ITensors.dim(site) - 1
    op = ITensors.ITensor(ComplexF64, site', site)
    @inbounds for n in 0:max_occ
        op[n+1, n+1] = n
    end
    return op
end

"""
    displace(site::ITensors.Index, alg::MabsAlg, α::Number)

Create the displacement operator `D(α) = exp(α*a† - α*a)` for a given site.

Arguments:
- site::ITensors.Index: Site index with bosonic tag
- alg::MabsAlg: Bosonic MPS algorithm
- α::Number: Displacement amplitude (can be complex)

Returns:
- ITensors.ITensor: Displacement operator tensor
"""
function displace(site::ITensors.Index, alg::Truncated, α::Number)
    #  G = α*a† - α*a
    a_dag = create(site, alg)
    a = destroy(site, alg)
    generator = α * a_dag - conj(α) * a
    op = ITensors.exp(generator)
    return op
end

"""
    squeeze(site::ITensors.Index, alg::MabsAlg, ξ::Number)

Create the squeezing operator `S(ξ) = exp(0.5*(ξ*a†² - ξ*a²))` for a given site.
Uses direct matrix element construction for numerical stability.

Arguments:
- site::ITensors.Index: Site index with bosonic tag
- alg::MabsAlg: Bosonic MPS algorithm
- ξ::Number: Squeezing parameter (can be complex)

Returns:
- ITensors.ITensor: Squeezing operator tensor
"""
function squeeze(site::ITensors.Index, alg::Truncated, ξ::Number)
    max_occ = ITensors.dim(site) - 1
    op = ITensors.ITensor(ComplexF64, site', site)
    r = abs(ξ)
    φ = angle(ξ)
    @inbounds for n in 0:max_occ
        @inbounds for m in 0:max_occ
            if (n + m) % 2 == 0 
                k_max = min(n, m) ÷ 2
                element = 0.0 + 0.0im
                @inbounds for k in 0:k_max
                    coeff = sqrt(_safe_factorial(n) * _safe_factorial(m)) / 
                           (_safe_factorial(k) * _safe_factorial((n-2k)) * _safe_factorial((m-2k)))
                    coeff *= (-0.5 * tanh(r) * exp(2im*φ))^k / sqrt(cosh(r))
                    if n == m && k == 0
                        coeff /= sqrt(cosh(r))  #.. additional factor for diagonal terms
                    end
                    element += coeff
                end
                op[m+1, n+1] = convert(ComplexF64, element)
            end
        end
    end
    return op
end

"""
    kerr(site::ITensors.Index, alg::MabsAlg, χ::Real, t::Real)

Create the Kerr evolution operator `exp(-i*χ*t*n²)` for a given site.

Arguments:
- site::ITensors.Index: Site index with bosonic tag
- alg::MabsAlg: Bosonic MPS algorithm
- χ::Real: Kerr nonlinearity strength
- t::Real: Evolution time

Returns:
- ITensors.ITensor: Kerr evolution operator tensor
"""
function kerr(site::ITensors.Index, alg::Truncated, χ::Real, t::Real)
    max_occ = ITensors.dim(site) - 1
    op = ITensors.ITensor(ComplexF64, site', site)
    @inbounds for n in 0:max_occ
        phase = exp(-1im * χ * t * n^2)
        op[n+1, n+1] = phase
    end
    return op
end

"""
    harmonic_chain(sites::Vector{<:ITensors.Index}, alg::MabsAlg, ω::Real, J::Real)

Build MPO for a chain of harmonic oscillators with optional nearest-neighbor coupling.
Here the Hamiltonian is `H = Σᵢ ω*nᵢ + J*Σᵢ (aᵢ†aᵢ₊₁ + aᵢaᵢ₊₁†)`.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of bosonic site indices
- alg::MabsAlg: Bosonic MPS algorithm
- ω::Real: Harmonic oscillator frequency
- J::Real: Nearest-neighbor hopping strength

Returns:
- BMPO: Matrix product operator for harmonic chain
"""
function harmonic_chain(sites::Vector{<:ITensors.Index}, alg::Truncated, ω::Real, J::Real)
    opsum = ITensors.OpSum()
    @inbounds for i in eachindex(sites)
        opsum += ω, "N", i
    end
    if J != 0.0
        @inbounds for i in 1:(length(sites)-1)
            opsum += J, "Adag", i, "A", i+1
            opsum += J, "A", i, "Adag", i+1
        end
    end
    mpo = ITensorMPS.MPO(opsum, sites)
    return BMPO(mpo, Truncated())
end

"""
    kerr(sites::Vector{<:ITensors.Index}, alg::MabsAlg, ω::Real, χ::Real)

Build MPO for a chain of Kerr oscillators.
Here the Hamiltonian is `H = Σᵢ (ω*nᵢ + χ*nᵢ²)`

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of bosonic site indices
- alg::MabsAlg: Bosonic MPS algorithm
- ω::Real: Linear frequency
- χ::Real: Kerr nonlinearity strength

Returns:
- BMPO: Matrix product operator for Kerr chain
"""
function kerr(sites::Vector{<:ITensors.Index}, alg::Truncated, ω::Real, χ::Real)
    opsum = ITensors.OpSum()
    @inbounds for i in eachindex(sites)
        opsum += ω, "N", i
        opsum += χ, "N", i, "N", i
    end
    mpo = ITensorMPS.MPO(opsum, sites)

    return BMPO(mpo, Truncated())
end