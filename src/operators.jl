# helper function for safe factorial calculation to avoid performance issues.
function _safe_factorial(n::Int)
    if n <= 20
        return factorial(n)
    else
        return factorial(big(n))
    end
end

"""
    create(site::ITensors.Index, alg::Truncated)

Create the bosonic creation operator (raising operator) for a given site.

Arguments:
- site::ITensors.Index: Site index with bosonic tag
- alg::Truncated: Truncated algorithm specification

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
    destroy(site::ITensors.Index, alg::Truncated)

Create the bosonic annihilation operator (lowering operator) for a given site.

Arguments:
- site::ITensors.Index: Site index with bosonic tag
- alg::Truncated: Truncated algorithm specification

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
- alg::Truncated: Truncated algorithm specification

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
    displace(site::ITensors.Index, alg::Truncated, α::Number)

Create the displacement operator `D(α) = exp(α*a† - α*a)` for a given site.

Arguments:
- site::ITensors.Index: Site index with bosonic tag
- alg::Truncated: Truncated algorithm specification
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
    squeeze(site::ITensors.Index, alg::Truncated, ξ::Number)

Create the squeezing operator `S(ξ) = exp(0.5*(ξ*a†² - ξ*a²))` for a given site.
Uses direct matrix element construction for numerical stability.

Arguments:
- site::ITensors.Index: Site index with bosonic tag
- alg::Truncated: Truncated algorithm specification
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
                        coeff /= sqrt(cosh(r))  
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
    kerr(site::ITensors.Index, alg::Truncated, χ::Real, t::Real)

Create the Kerr evolution operator `exp(-i*χ*t*n²)` for a given site.

Arguments:
- site::ITensors.Index: Site index with bosonic tag
- alg::Truncated: Truncated algorithm specification
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
    harmonic_chain(sites::Vector{<:ITensors.Index}, alg::Truncated, ω::Real, J::Real)

Build MPO for a chain of harmonic oscillators with optional nearest-neighbor coupling.
Here the Hamiltonian is `H = Σᵢ ω*nᵢ + J*Σᵢ (aᵢ†aᵢ₊₁ + aᵢaᵢ₊₁†)`.

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of bosonic site indices
- alg::Truncated: Truncated algorithm specification
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
    kerr(sites::Vector{<:ITensors.Index}, alg::Truncated, ω::Real, χ::Real)

Build MPO for a chain of Kerr oscillators.
Here the Hamiltonian is `H = Σᵢ (ω*nᵢ + χ*nᵢ²)`

Arguments:
- sites::Vector{<:ITensors.Index}: Vector of bosonic site indices
- alg::Truncated: Truncated algorithm specification
- ω::Real: Linear frequency
- χ::Real: Kerr nonlinearity strength

Returns:
- BMPO: Matrix product operator for Kerr chain
"""
function kerr_hamiltonian(sites::Vector{<:ITensors.Index}, alg::Truncated, ω::Real, χ::Real)

    opsum = ITensors.OpSum()
    @inbounds for i in eachindex(sites)
        opsum += ω, "N", i
        opsum += χ, "N", i, "N", i
    end
    mpo = ITensorMPS.MPO(opsum, sites)

    return BMPO(mpo, Truncated())
end

# PseudoSite operators - Mode-indexed interface (recommended)
"""
    create(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int)

Create bosonic creation operator for a specific mode in the pseudo-site representation.

This is the recommended interface for typical usage. The function automatically extracts
the qubit cluster corresponding to the specified mode. If you have already extracted 
the cluster sites, you can use `create(cluster_sites, alg)` instead.

# Arguments
- sites::Vector{<:ITensors.Index}: All qubit site indices for the system
- alg::PseudoSite: Algorithm specification
- mode::Int: Bosonic mode index (`1` to `nmodes`)

# Returns
- ITensors.ITensor: Creation operator `â†` for the specified mode

# Example
```julia
alg = PseudoSite(2)  # 2 modes
sites = [Index(2, "Qubit,n=\$i") for i in 1:6]  # 3 qubits per mode

# Recommended: Let function extract cluster
a_dag = create(sites, alg, 1)  # Creation op for mode 1

# Alternative: Manual cluster extraction (see convenience methods below)
cluster = sites[1:3]  # First 3 qubits = mode 1
a_dag = create(cluster, alg)  # Same result
```
"""
function create(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int)
    cluster_sites = _mode_cluster(sites, alg, mode)
    return _create_qubit(cluster_sites)
end

"""
    destroy(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int)

Create bosonic annihilation operator for a specific mode in the pseudo-site representation.

This is the recommended interface for typical usage. See `create(sites, alg, mode)` 
for detailed documentation and usage patterns.

# Arguments
- sites::Vector{<:ITensors.Index}: All qubit site indices for the system
- alg::PseudoSite: Algorithm specification
- mode::Int: Bosonic mode index (`1` to `nmodes`)

# Returns
- ITensors.ITensor: Annihilation operator `â` for the specified mode
"""
function destroy(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int)
    cluster_sites = _mode_cluster(sites, alg, mode)
    return _destroy_qubit(cluster_sites)
end

"""
    number(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int)

Create bosonic number operator for a specific mode in the pseudo-site representation.

This is the recommended interface for typical usage. See `create(sites, alg, mode)` 
for detailed documentation and usage patterns.

# Arguments
- sites::Vector{<:ITensors.Index}: All qubit site indices for the system
- alg::PseudoSite: Algorithm specification
- mode::Int: Bosonic mode index (`1` to `nmodes`)

# Returns
- ITensors.ITensor: Number operator `n̂` for the specified mode
"""
function number(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int)
    cluster_sites = _mode_cluster(sites, alg, mode)
    return _number_qubit(cluster_sites)
end

"""
    displace(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int, α::Number)

Create displacement operator for a specific mode in the pseudo-site representation.

This is the recommended interface for typical usage. See `create(sites, alg, mode)` 
for detailed documentation and usage patterns.

# Arguments
- sites::Vector{<:ITensors.Index}: All qubit site indices for the system
- alg::PseudoSite: Algorithm specification
- mode::Int: Bosonic mode index (`1` to `nmodes`)
- α::Number: Displacement amplitude (can be complex)

# Returns
- ITensors.ITensor: Displacement operator D(α) = exp(α â† - α* â) for the specified mode
"""
function displace(
    sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int, α::Number
)
    cluster_sites = _mode_cluster(sites, alg, mode)
    return _displace_qubit(cluster_sites, α)
end

"""
    squeeze(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int, ξ::Number)

Create squeeze operator for a specific mode in the pseudo-site representation.

This is the recommended interface for typical usage. See `create(sites, alg, mode)` 
for detailed documentation and usage patterns.

# Arguments
- sites::Vector{<:ITensors.Index}: All qubit site indices for the system
- alg::PseudoSite: Algorithm specification
- mode::Int: Bosonic mode index (`1` to `nmodes`)
- ξ::Number: Squeezing parameter (can be complex)

# Returns
- ITensors.ITensor: Squeeze operator S(ξ) = exp(½(ξ â†² - ξ* â²)) for the specified mode
"""
function squeeze(
    sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int, ξ::Number
)
    cluster_sites = _mode_cluster(sites, alg, mode)
    return _squeeze_qubit(cluster_sites, ξ)
end

"""
    kerr(sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int, χ::Real, t::Real)

Create Kerr evolution operator for a specific mode in the pseudo-site representation.

This is the recommended interface for typical usage. See `create(sites, alg, mode)` 
for detailed documentation and usage patterns.

# Arguments
- sites::Vector{<:ITensors.Index}: All qubit site indices for the system
- alg::PseudoSite: Algorithm specification
- mode::Int: Bosonic mode index (`1` to `nmodes`)
- χ::Real: Kerr nonlinearity strength
- t::Real: Evolution time

# Returns
- ITensors.ITensor: Kerr operator exp(-i χ t n̂²) for the specified mode
"""
function kerr(
    sites::Vector{<:ITensors.Index}, alg::PseudoSite, mode::Int, χ::Real, t::Real
)
    cluster_sites = _mode_cluster(sites, alg, mode)
    return _kerr_qubit(cluster_sites, χ, t)
end

# PseudoSite operators - Cluster-based convenience interface
"""
    create(cluster_sites::Vector{<:ITensors.Index}, alg::PseudoSite)

Create bosonic creation operator given pre-extracted qubit cluster sites.

This is a convenience method for when you have already extracted the cluster sites
for a specific mode. For typical usage, prefer `create(sites, alg, mode)` which 
handles cluster extraction automatically.

# Arguments
- cluster_sites::Vector{<:ITensors.Index}: Qubit sites for a single mode cluster
- alg::PseudoSite: Algorithm specification

# Returns
- ITensors.ITensor: Creation operator `â†` for the mode

# Example
```julia
alg = PseudoSite(2)
sites = [Index(2, "Qubit,n=\$i") for i in 1:6]

# Manual cluster extraction
cluster1 = sites[1:3]  # Mode 1
a_dag = create(cluster1, alg)
```
"""
function create(cluster_sites::Vector{<:ITensors.Index}, alg::PseudoSite)
    return _create_qubit(cluster_sites)
end

"""
    destroy(cluster_sites::Vector{<:ITensors.Index}, alg::PseudoSite)

Create bosonic annihilation operator given pre-extracted qubit cluster sites.

Convenience method for pre-extracted clusters. See `create(cluster_sites, alg)` 
for detailed documentation.
"""
function destroy(cluster_sites::Vector{<:ITensors.Index}, alg::PseudoSite)
    return _destroy_qubit(cluster_sites)
end

"""
    number(cluster_sites::Vector{<:ITensors.Index}, alg::PseudoSite)

Create bosonic number operator given pre-extracted qubit cluster sites.

Convenience method for pre-extracted clusters. See `create(cluster_sites, alg)` 
for detailed documentation.
"""
function number(cluster_sites::Vector{<:ITensors.Index}, alg::PseudoSite)
    return _number_qubit(cluster_sites)
end

"""
    displace(cluster_sites::Vector{<:ITensors.Index}, alg::PseudoSite, α::Number)

Create displacement operator given pre-extracted qubit cluster sites.

Convenience method for pre-extracted clusters. See `create(cluster_sites, alg)` 
for detailed documentation.
"""
function displace(cluster_sites::Vector{<:ITensors.Index}, alg::PseudoSite, α::Number)
    return _displace_qubit(cluster_sites, α)
end

"""
    squeeze(cluster_sites::Vector{<:ITensors.Index}, alg::PseudoSite, ξ::Number)

Create squeeze operator given pre-extracted qubit cluster sites.

Convenience method for pre-extracted clusters. See `create(cluster_sites, alg)` 
for detailed documentation.
"""
function squeeze(cluster_sites::Vector{<:ITensors.Index}, alg::PseudoSite, ξ::Number)
    return _squeeze_qubit(cluster_sites, ξ)
end

"""
    kerr(cluster_sites::Vector{<:ITensors.Index}, alg::PseudoSite, χ::Real, t::Real)

Create Kerr evolution operator given pre-extracted qubit cluster sites.

Convenience method for pre-extracted clusters. See `create(cluster_sites, alg)` 
for detailed documentation.
"""
function kerr(cluster_sites::Vector{<:ITensors.Index}, alg::PseudoSite, χ::Real, t::Real)
    return _kerr_qubit(cluster_sites, χ, t)
end

"""
    harmonic_chain(sites::Vector{<:ITensors.Index}, alg::PseudoSite, ω::Real, J::Real; kwargs...)

Build harmonic chain Hamiltonian in the pseudo-site representation.
H = Σᵢ ω*nᵢ + J*Σᵢ (aᵢ†aᵢ₊₁ + h.c.)

Arguments:
- sites::Vector{<:ITensors.Index}: Qubit site indices
- alg::PseudoSite: Algorithm specification
- ω::Real: Harmonic oscillator frequency
- J::Real: Nearest-neighbor hopping strength

Keyword Arguments:
- kwargs...: Additional parameters passed to `ITensorMPS.add` (e.g., `cutoff`, `maxdim`)
"""
function harmonic_chain(sites::Vector{<:ITensors.Index}, alg::PseudoSite, ω::Real, J::Real; kwargs...)
    nqubits = _nqubits_per_mode(sites, alg)
    n_expected = alg.nmodes * nqubits
    length(sites) == n_expected || throw(ArgumentError("Sites must match algorithm"))
    
    opsum = ITensors.OpSum()
    @inbounds for mode in 1:alg.nmodes
        @inbounds for i in 1:nqubits
            weight = ω * 2^(i-1)
            global_idx = (mode - 1) * nqubits + i
            opsum += weight, "N", global_idx
        end
    end
    
    H = ITensorMPS.MPO(opsum, sites)
    if J != 0.0
        @inbounds for mode in 1:(alg.nmodes - 1)
            H_hop = _hopping_mpo(sites, alg, mode, J)
            H = ITensorMPS.add(H, H_hop; kwargs...)
        end
    end
    return BMPO(H, alg)
end

"""
    kerr_hamiltonian(sites::Vector{<:ITensors.Index}, alg::PseudoSite, ω::Real, χ::Real)

Build Kerr nonlinearity Hamiltonian in the pseudo-site representation.
H = Σᵢ (ω*nᵢ + χ*nᵢ²)

Arguments:
- sites::Vector{<:ITensors.Index}: Qubit site indices
- alg::PseudoSite: Algorithm specification
- ω::Real: Linear frequency
- χ::Real: Kerr nonlinearity strength

Returns:
- BMPO: Matrix product operator for Kerr Hamiltonian

# Example
```julia
alg = PseudoSite(3)
sites = [Index(2, "Qubit,n=\$i") for i in 1:9]
H = kerr_hamiltonian(sites, alg, 1.0, 0.1)  # H = Σᵢ(nᵢ + 0.1nᵢ²)
```
"""
function kerr_hamiltonian(sites::Vector{<:ITensors.Index}, alg::PseudoSite, ω::Real, χ::Real)
    nqubits = _nqubits_per_mode(sites, alg)
    n_expected = alg.nmodes * nqubits
    length(sites) == n_expected || throw(ArgumentError("Sites must match algorithm"))
    
    opsum = ITensors.OpSum()
    @inbounds for mode in 1:alg.nmodes
        @inbounds for i in 1:nqubits
            weight = ω * 2^(i-1)
            global_idx = (mode - 1) * nqubits + i
            opsum += weight, "N", global_idx
        end
        @inbounds for i in 1:nqubits
            @inbounds for j in 1:nqubits
                weight = χ * 2^(i + j - 2)
                idx_i = (mode - 1) * nqubits + i
                idx_j = (mode - 1) * nqubits + j
                if i == j
                    opsum += weight, "N", idx_i
                else
                    opsum += weight, "N", idx_i, "N", idx_j
                end
            end
        end
    end
    mpo = ITensorMPS.MPO(opsum, sites)
    return BMPO(mpo, alg)
end

function ITensors.op(::ITensors.OpName"N", ::ITensors.SiteType"Qubit", s::ITensors.Index)
    return ITensors.ITensor(ComplexF64[0.0 0.0; 0.0 1.0], s', s)
end

function ITensors.op(::ITensors.OpName"Id", ::ITensors.SiteType"Qubit", s::ITensors.Index)
    return ITensors.ITensor(ComplexF64[1.0 0.0; 0.0 1.0], s', s)
end

function ITensors.op(::ITensors.OpName"X", ::ITensors.SiteType"Qubit", s::ITensors.Index)
    return ITensors.ITensor(ComplexF64[0.0 1.0; 1.0 0.0], s', s)
end

function ITensors.op(::ITensors.OpName"Z", ::ITensors.SiteType"Qubit", s::ITensors.Index)
    return ITensors.ITensor(ComplexF64[1.0 0.0; 0.0 -1.0], s', s)
end

function ITensors.op(::ITensors.OpName"Y", ::ITensors.SiteType"Qubit", s::ITensors.Index)
    return ITensors.ITensor(ComplexF64[0.0 -im; im 0.0], s', s)
end