using Test
using Mabs
using ITensors
using ITensorMPS
using LinearAlgebra

@testset "PseudoSite Algorithm Tests" begin
    
    @testset "PseudoSite Algorithm Construction" begin
        @testset "Basic construction" begin
            alg = PseudoSite(2)
            @test alg isa PseudoSite
            @test alg.nmodes == 2
            
            alg_single = PseudoSite(1)
            @test alg_single.nmodes == 1
            
            alg_many = PseudoSite(10)
            @test alg_many.nmodes == 10
        end
        
        @testset "Algorithm equality" begin
            alg1 = PseudoSite(3)
            alg2 = PseudoSite(3)
            alg3 = PseudoSite(4)
            
            @test alg1 == alg2
            @test !(alg1 == alg3)
        end
    end
    
    @testset "Helper Functions" begin
        @testset "_nqubits_per_mode" begin
            alg = PseudoSite(2)
            
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
            @test Mabs._nqubits_per_mode(sites, alg) == 3
            
            sites2 = [ITensors.Index(2, "Qubit,n=$i") for i in 1:8]
            @test Mabs._nqubits_per_mode(sites2, alg) == 4
            
            mps = ITensorMPS.random_mps(sites)
            @test Mabs._nqubits_per_mode(mps, alg) == 3
        end
        
        @testset "_fock_to_qubit!" begin
            nqubits = 3
            buffer = zeros(Int, nqubits)
            
            Mabs._fock_to_qubit!(buffer, 0, nqubits)
            @test buffer == [1, 1, 1]
            
            Mabs._fock_to_qubit!(buffer, 1, nqubits)
            @test buffer == [2, 1, 1]
            
            Mabs._fock_to_qubit!(buffer, 2, nqubits)
            @test buffer == [1, 2, 1]
            
            Mabs._fock_to_qubit!(buffer, 3, nqubits)
            @test buffer == [2, 2, 1]
            
            Mabs._fock_to_qubit!(buffer, 5, nqubits)
            @test buffer == [2, 1, 2]
            
            Mabs._fock_to_qubit!(buffer, 7, nqubits)
            @test buffer == [2, 2, 2]
            
            @test_throws ArgumentError Mabs._fock_to_qubit!(buffer, -1, nqubits)
            
            @test_throws ArgumentError Mabs._fock_to_qubit!(buffer, 8, nqubits)
        end
        
        @testset "_qubit_to_fock" begin
            @test Mabs._qubit_to_fock([1, 1, 1]) == 0
            
            @test Mabs._qubit_to_fock([2, 1, 1]) == 1
            
            @test Mabs._qubit_to_fock([1, 2, 1]) == 2
            
            @test Mabs._qubit_to_fock([2, 2, 1]) == 3
            
            @test Mabs._qubit_to_fock([2, 1, 2]) == 5
            
            @test Mabs._qubit_to_fock([2, 2, 2]) == 7
            
            @test_throws ArgumentError Mabs._qubit_to_fock([0, 1, 1])
            @test_throws ArgumentError Mabs._qubit_to_fock([3, 1, 1])
        end
        
        @testset "_fock_to_qubit! and _qubit_to_fock roundtrip" begin
            nqubits = 4
            max_occ = 2^nqubits - 1
            buffer = zeros(Int, nqubits)
            
            for n in 0:max_occ
                Mabs._fock_to_qubit!(buffer, n, nqubits)
                n_recovered = Mabs._qubit_to_fock(buffer)
                @test n_recovered == n
            end
        end
        
        @testset "_mode_cluster" begin
            alg = PseudoSite(3)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:9]
            
            cluster1 = Mabs._mode_cluster(sites, alg, 1)
            @test length(cluster1) == 3
            @test cluster1 == sites[1:3]
            
            cluster2 = Mabs._mode_cluster(sites, alg, 2)
            @test length(cluster2) == 3
            @test cluster2 == sites[4:6]
            
            cluster3 = Mabs._mode_cluster(sites, alg, 3)
            @test length(cluster3) == 3
            @test cluster3 == sites[7:9]
            
            @test_throws ArgumentError Mabs._mode_cluster(sites, alg, 0)
            @test_throws ArgumentError Mabs._mode_cluster(sites, alg, 4)
        end
        
        @testset "_mode_indices" begin
            alg = PseudoSite(2)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
            
            indices1 = Mabs._mode_indices(sites, alg, 1)
            @test indices1 == 1:3
            
            indices2 = Mabs._mode_indices(sites, alg, 2)
            @test indices2 == 4:6
            
            @test_throws ArgumentError Mabs._mode_indices(sites, alg, 0)
            @test_throws ArgumentError Mabs._mode_indices(sites, alg, 3)
        end
    end
    
    @testset "BMPS Construction" begin
        @testset "Basic BMPS from MPS" begin
            alg = PseudoSite(2)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
            mps = ITensorMPS.random_mps(sites)
            
            bmps = BMPS(mps, alg)
            @test bmps isa BMPS{<:ITensorMPS.MPS, PseudoSite}
            @test length(bmps) == 6
            @test bmps.alg == alg
            
            wrong_sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:5]
            wrong_mps = ITensorMPS.random_mps(wrong_sites)
            @test_throws ArgumentError BMPS(wrong_mps, alg)
        end
        
        @testset "Product state construction" begin
            alg = PseudoSite(2)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
            
            states = [3, 5]
            bmps = BMPS(sites, states, alg)
            
            @test bmps isa BMPS{<:ITensorMPS.MPS, PseudoSite}
            @test length(bmps) == 6
            @test abs(norm(bmps) - 1.0) < 1e-10
            
            @test_throws ArgumentError BMPS(sites, [1], alg)
            @test_throws ArgumentError BMPS(sites, [1, 2, 3], alg)
            
            @test_throws ArgumentError BMPS(sites, [8, 1], alg)
        end
        
        @testset "BMPS properties and operations" begin
            alg = PseudoSite(2)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
            bmps = random_bmps(sites, alg)
            
            @test siteinds(bmps) == sites
            @test length(bmps) == 6
            @test maxlinkdim(bmps) >= 1
            @test eltype(bmps) <: Number
            
            bmps_copy = copy(bmps)
            @test bmps_copy !== bmps
            @test bmps_copy.mps !== bmps.mps
            @test bmps_copy.alg == bmps.alg
            
            bmps_deep = deepcopy(bmps)
            @test bmps_deep !== bmps
            @test bmps_deep.mps !== bmps.mps
            
            normalize!(bmps)
            @test abs(norm(bmps) - 1.0) < 1e-10
            
            bmps_normed = normalize(bmps_copy)
            @test abs(norm(bmps_normed) - 1.0) < 1e-10
            @test bmps_normed !== bmps_copy
        end
    end
    
    @testset "BMPO Construction" begin
        @testset "Basic BMPO from MPO" begin
            alg = PseudoSite(2)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
            
            opsum = ITensors.OpSum()
            for i in 1:6
                opsum += 1.0, "N", i
            end
            mpo = ITensorMPS.MPO(opsum, sites)
            
            bmpo = BMPO(mpo, alg)
            @test bmpo isa BMPO{<:ITensorMPS.MPO, PseudoSite}
            @test length(bmpo) == 6
            @test bmpo.alg == alg
            
            wrong_sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:5]
            wrong_opsum = ITensors.OpSum()
            for i in 1:5
                wrong_opsum += 1.0, "N", i
            end
            wrong_mpo = ITensorMPS.MPO(wrong_opsum, wrong_sites)
            @test_throws ArgumentError BMPO(wrong_mpo, alg)
        end
        
        @testset "BMPO from OpSum" begin
            alg = PseudoSite(2)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
            
            opsum = ITensors.OpSum()
            for i in 1:alg.nmodes
                opsum += 1.0, "N", i
            end
            
            bmpo = BMPO(opsum, sites, alg)
            @test bmpo isa BMPO{<:ITensorMPS.MPO, PseudoSite}
            @test length(bmpo) == 6
            
            bad_opsum = ITensors.OpSum()
            bad_opsum += 1.0, "Adag", 1, "A", 2
            @test_throws ErrorException BMPO(bad_opsum, sites, alg)
        end
        
        @testset "BMPO properties" begin
            alg = PseudoSite(2)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
            opsum = ITensors.OpSum()
            for i in 1:alg.nmodes
                opsum += 1.0, "N", i
            end
            
            bmpo = BMPO(opsum, sites, alg)
            
            @test [s[end] for s in siteinds(bmpo)] == sites
            @test length(bmpo) == 6
            @test eltype(bmpo) == ITensors.ITensor  
            
            bmpo_copy = copy(bmpo)
            @test bmpo_copy !== bmpo
            @test bmpo_copy.mpo !== bmpo.mpo
        end
    end
    
    @testset "State Construction" begin
        @testset "Vacuum state" begin
            alg = PseudoSite(3)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:9]
            
            vac = vacuumstate(sites, alg)
            @test vac isa BMPS{<:ITensorMPS.MPS, PseudoSite}
            @test length(vac) == 9
            @test abs(norm(vac) - 1.0) < 1e-10
            
            for mode in 1:alg.nmodes
                n_op = number(sites, alg, mode)
                n_psi = ITensors.apply(n_op, vac.mps)
                n_exp = real(ITensors.inner(vac.mps, n_psi))
                @test abs(n_exp) < 1e-10
            end
        end
        
        @testset "Random BMPS" begin
            alg = PseudoSite(2)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
            
            psi = random_bmps(sites, alg)
            @test psi isa BMPS{<:ITensorMPS.MPS, PseudoSite}
            @test length(psi) == 6
            @test maxlinkdim(psi) >= 1
            
            psi_large = random_bmps(sites, alg; linkdims=4)
            @test maxlinkdim(psi_large) >= 1
        end
        
        @testset "Coherent state - single amplitude" begin
            alg = PseudoSite(2)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
            α = 1.0
            
            psi_coherent = coherentstate(sites, alg, α)
            @test psi_coherent isa BMPS{<:ITensorMPS.MPS, PseudoSite}
            @test abs(norm(psi_coherent) - 1.0) < 1e-8
            
            alg_small = PseudoSite(2)
            sites_small = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
            psi_small = coherentstate(sites_small, alg_small, 0.5)
            @test abs(norm(psi_small) - 1.0) < 1e-8
        end
        
        @testset "Coherent state - multiple amplitudes" begin
            alg = PseudoSite(3)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:9]
            αs = [0.5, 1.0, 0.3]
            
            psi = coherentstate(sites, alg, αs)
            @test psi isa BMPS{<:ITensorMPS.MPS, PseudoSite}
            @test abs(norm(psi) - 1.0) < 1e-8
            
            @test_throws ArgumentError coherentstate(sites, alg, [0.1, 0.2])
        end
        
        @testset "Coherent state - displacement method" begin
            alg = PseudoSite(2)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:8]
            α = 1.5
            
            psi = coherentstate(sites, alg, α; maxdim=20)
            @test psi isa BMPS{<:ITensorMPS.MPS, PseudoSite}
            @test abs(norm(psi) - 1.0) < 1e-8
        end
    end
    
    
    @testset "Operator Construction - Mode Interface" begin
        alg = PseudoSite(2)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
        
        @testset "Creation operator" begin
            a_dag = create(sites, alg, 1)
            @test a_dag isa ITensors.ITensor
            
            cluster1 = sites[1:3]
            @test ITensors.hasinds(a_dag, cluster1'..., cluster1...)
            
            a_dag2 = create(sites, alg, 2)
            @test a_dag2 isa ITensors.ITensor
        end
        
        @testset "Annihilation operator" begin
            a = destroy(sites, alg, 1)
            @test a isa ITensors.ITensor
            
            cluster1 = sites[1:3]
            @test ITensors.hasinds(a, cluster1'..., cluster1...)
        end
        
        @testset "Number operator" begin
            n = number(sites, alg, 1)
            @test n isa ITensors.ITensor
            
            cluster1 = sites[1:3]
            @test ITensors.hasinds(n, cluster1'..., cluster1...)
        end
        
        @testset "Displacement operator" begin
            α = 0.5 + 0.3im
            D = displace(sites, alg, 1, α)
            @test D isa ITensors.ITensor
            
            cluster1 = sites[1:3]
            @test ITensors.hasinds(D, cluster1'..., cluster1...)
        end
        
        @testset "Squeeze operator" begin
            ξ = 0.5
            S = squeeze(sites, alg, 1, ξ)
            @test S isa ITensors.ITensor
            
            cluster1 = sites[1:3]
            @test ITensors.hasinds(S, cluster1'..., cluster1...)
        end
        
        @testset "Kerr operator" begin
            χ = 0.1
            t = 0.5
            K = kerr(sites, alg, 1, χ, t)
            @test K isa ITensors.ITensor
            
            cluster1 = sites[1:3]
            @test ITensors.hasinds(K, cluster1'..., cluster1...)
        end
    end
    
    @testset "Operator Construction - Cluster Interface" begin
        alg = PseudoSite(2)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
        cluster1 = sites[1:3]
        
        @testset "Creation operator" begin
            a_dag = create(cluster1, alg)
            @test a_dag isa ITensors.ITensor
            @test ITensors.hasinds(a_dag, cluster1'..., cluster1...)
        end
        
        @testset "Annihilation operator" begin
            a = destroy(cluster1, alg)
            @test a isa ITensors.ITensor
            @test ITensors.hasinds(a, cluster1'..., cluster1...)
        end
        
        @testset "Number operator" begin
            n = number(cluster1, alg)
            @test n isa ITensors.ITensor
            @test ITensors.hasinds(n, cluster1'..., cluster1...)
        end
        
        @testset "Displacement operator" begin
            D = displace(cluster1, alg, 0.5)
            @test D isa ITensors.ITensor
            @test ITensors.hasinds(D, cluster1'..., cluster1...)
        end
        
        @testset "Squeeze operator" begin
            S = squeeze(cluster1, alg, 0.3)
            @test S isa ITensors.ITensor
            @test ITensors.hasinds(S, cluster1'..., cluster1...)
        end
        
        @testset "Kerr operator" begin
            K = kerr(cluster1, alg, 0.1, 0.5)
            @test K isa ITensors.ITensor
            @test ITensors.hasinds(K, cluster1'..., cluster1...)
        end
    end
    
    @testset "Hamiltonian Construction" begin
        @testset "Harmonic chain" begin
            alg = PseudoSite(3)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:9]
            
            H = harmonic_chain(sites, alg, 1.0, 0.0)
            @test H isa BMPO{<:ITensorMPS.MPO, PseudoSite}
            @test length(H) == 9
            
            H_hop = harmonic_chain(sites, alg, 1.0, 0.5)
            @test H_hop isa BMPO{<:ITensorMPS.MPO, PseudoSite}
            @test length(H_hop) == 9
            
            wrong_sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:8]
            @test_throws ArgumentError harmonic_chain(wrong_sites, alg, 1.0, 0.0)
        end
        
        @testset "Kerr Hamiltonian" begin
            alg = PseudoSite(2)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
            
            H = kerr_hamiltonian(sites, alg, 1.0, 0.1)
            @test H isa BMPO{<:ITensorMPS.MPO, PseudoSite}
            @test length(H) == 6
            
            wrong_sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:5]
            @test_throws ArgumentError kerr_hamiltonian(wrong_sites, alg, 1.0, 0.1)
        end
    end
    
    @testset "Operator Physics - Matrix Elements" begin
        alg = PseudoSite(1)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:3]
        
        @testset "Creation operator matrix elements" begin
            a_dag = create(sites, alg, 1)
            
            for n in 0:6
                state_n = BMPS(sites, [n], alg)
                state_np1 = BMPS(sites, [n+1], alg)
                
                a_dag_n = ITensors.apply(a_dag, state_n.mps)
                overlap = ITensors.inner(state_np1.mps, a_dag_n)
                
                expected = sqrt(n + 1)
                @test abs(abs(overlap) - expected) < 1e-10
            end
        end
        
        @testset "Annihilation operator matrix elements" begin
            a = destroy(sites, alg, 1)
            
            for n in 1:7
                state_n = BMPS(sites, [n], alg)
                state_nm1 = BMPS(sites, [n-1], alg)
                
                a_n = ITensors.apply(a, state_n.mps)
                overlap = ITensors.inner(state_nm1.mps, a_n)
                
                expected = sqrt(n)
                @test abs(abs(overlap) - expected) < 1e-10
            end
            
            state_0 = BMPS(sites, [0], alg)
            a_0 = ITensors.apply(a, state_0.mps)
            @test norm(a_0) < 1e-12
        end
        
        @testset "Number operator eigenvalues" begin
            n_op = number(sites, alg, 1)
            
            for n in 0:7
                state_n = BMPS(sites, [n], alg)
                n_psi = ITensors.apply(n_op, state_n.mps)
                
                expectation = real(ITensors.inner(state_n.mps, n_psi))
                @test abs(expectation - n) < 1e-10
            end
        end
    end
    
    @testset "Bosonic Commutation Relations" begin
        alg = PseudoSite(1)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:3]
        
        a = destroy(sites, alg, 1)
        a_dag = create(sites, alg, 1)
        
        @testset "Commutator [a, a†] = 1" begin
            for n in 0:6
                state_n = BMPS(sites, [n], alg)
                
                adag_a_psi = ITensors.apply(a, state_n.mps)
                adag_a_psi = ITensors.apply(a_dag, adag_a_psi)
                exp1 = real(ITensors.inner(state_n.mps, adag_a_psi))
                
                a_adag_psi = ITensors.apply(a_dag, state_n.mps)
                a_adag_psi = ITensors.apply(a, a_adag_psi)
                exp2 = real(ITensors.inner(state_n.mps, a_adag_psi))
                
                commutator = exp2 - exp1
                @test abs(commutator - 1.0) < 1e-10
            end
        end
        
        @testset "Number operator n = a†a" begin
            n_op = number(sites, alg, 1)
            
            for n in 0:6
                state_n = BMPS(sites, [n], alg)
                
                n_psi = ITensors.apply(n_op, state_n.mps)
                n_exp = real(ITensors.inner(state_n.mps, n_psi))
                
                adag_a_psi = ITensors.apply(a, state_n.mps)
                adag_a_psi = ITensors.apply(a_dag, adag_a_psi)
                adag_a_exp = real(ITensors.inner(state_n.mps, adag_a_psi))
                
                @test abs(n_exp - adag_a_exp) < 1e-10
            end
        end
    end
    
    @testset "Coherent State Physics" begin
        alg = PseudoSite(1)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:4]
        
        @testset "Photon number expectation" begin
            α = 2.0
            psi_coherent = coherentstate(sites, alg, α)
            
            n_op = number(sites, alg, 1)
            n_psi = ITensors.apply(n_op, psi_coherent.mps)
            n_expectation = real(ITensors.inner(psi_coherent.mps, n_psi))
            
            expected = abs2(α)
            @test abs(n_expectation - expected) < 0.5
        end
        
        @testset "Coherent state is eigenstate of annihilation operator" begin
            α = 1.0 + 0.5im
            psi_coherent = coherentstate(sites, alg, α)
            
            a = destroy(sites, alg, 1)
            a_psi = ITensors.apply(a, psi_coherent.mps; maxdim=20)
            
            overlap = ITensors.inner(psi_coherent.mps, a_psi)
            
            @test abs(overlap - α) < 0.2
        end
        
        @testset "Multi-mode coherent states" begin
            alg_multi = PseudoSite(2)
            sites_multi = [ITensors.Index(2, "Qubit,n=$i") for i in 1:8]
            αs = [1.0, 1.5]
            
            psi = coherentstate(sites_multi, alg_multi, αs)
            @test abs(norm(psi) - 1.0) < 1e-8
            
            for mode in 1:2
                n_op = number(sites_multi, alg_multi, mode)
                n_psi = ITensors.apply(n_op, psi.mps)
                n_exp = real(ITensors.inner(psi.mps, n_psi))
                
                expected = abs2(αs[mode])
                @test abs(n_exp - expected) < 0.5
            end
        end
    end
    
    @testset "Displacement Operator Physics" begin
        alg = PseudoSite(1)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:3]
        
        @testset "Displacement of vacuum creates coherent state" begin
            α = 0.8
            
            vac = vacuumstate(sites, alg)
            D = displace(sites, alg, 1, α)
            D_vac = ITensors.apply(D, vac.mps; maxdim=10)
            normalize!(D_vac)
            
            coherent = coherentstate(sites, alg, α)
            
            overlap = abs(ITensors.inner(coherent.mps, D_vac))
            @test overlap > 0.90
        end
        
        @testset "Displacement composition D(α)D(β) = D(α+β)e^(iℑ(α*β̄))" begin
            α = 0.5 + 0.2im
            β = 0.3 - 0.1im
            
            D_alpha = displace(sites, alg, 1, α)
            D_beta = displace(sites, alg, 1, β)
            D_sum = displace(sites, alg, 1, α + β)
            
            vac = vacuumstate(sites, alg)
            
            psi1 = ITensors.apply(D_beta, vac.mps; maxdim=10)
            psi1 = ITensors.apply(D_alpha, psi1; maxdim=10)
            normalize!(psi1)
            
            psi2 = ITensors.apply(D_sum, vac.mps; maxdim=10)
            normalize!(psi2)
            
            overlap = abs(ITensors.inner(psi1, psi2))
            @test overlap > 0.95
        end
    end
    
    @testset "Squeeze Operator Physics" begin
        alg = PseudoSite(1)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:3]
        
        @testset "Squeezing vacuum" begin
            ξ = 0.3
            S = squeeze(sites, alg, 1, ξ)
            
            vac = vacuumstate(sites, alg)
            squeezed = ITensors.apply(S, vac.mps; maxdim=10)
            
            @test abs(norm(squeezed) - 1.0) < 0.5 
            
            overlap_with_vac = abs(ITensors.inner(vac.mps, squeezed))
            @test overlap_with_vac < 0.99
        end
    end
    
    @testset "Kerr Evolution Physics" begin
        alg = PseudoSite(1)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:3]
        
        @testset "Number states acquire phases" begin
            χ = 0.1
            t = 0.5
            K = kerr(sites, alg, 1, χ, t)
            
            for n in 0:5
                state_n = BMPS(sites, [n], alg)
                K_n = ITensors.apply(K, state_n.mps)
                
                overlap = ITensors.inner(state_n.mps, K_n)
                expected_phase = exp(-1im * χ * t * n^2)
                
                @test abs(overlap - expected_phase) < 1e-10
            end
        end
    end
    
    @testset "DMRG Ground State Finding" begin
        alg = PseudoSite(2)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
        
        @testset "Harmonic oscillator chain ground state" begin
            ω = 1.0
            H = harmonic_chain(sites, alg, ω, 0.0)
            
            psi0 = random_bmps(sites, alg; linkdims=4)
            
            energy, psi_gs = Mabs.dmrg(H, psi0; nsweeps=10, maxdim=20, cutoff=1e-10)
            
            @test energy isa Real
            @test psi_gs isa BMPS{<:ITensorMPS.MPS, PseudoSite}
            @test abs(norm(psi_gs) - 1.0) < 1e-8
            
            @test energy < 2.0
        end
        
        @testset "DMRG with Kerr Hamiltonian" begin
            H = kerr_hamiltonian(sites, alg, 1.0, 0.1)
            psi0 = random_bmps(sites, alg; linkdims=4)
            
            energy, psi_gs = Mabs.dmrg(H, psi0; nsweeps=5, maxdim=15, cutoff=1e-10)
            
            @test energy isa Real
            @test psi_gs isa BMPS{<:ITensorMPS.MPS, PseudoSite}
        end
        
        @testset "DMRG algorithm mismatch error" begin
            alg_wrong = PseudoSite(3)
            sites_wrong = [ITensors.Index(2, "Qubit,n=$i") for i in 1:9]
            H_wrong = harmonic_chain(sites_wrong, alg_wrong, 1.0, 0.0)
            
            psi0 = random_bmps(sites, alg; linkdims=4)
            
            @test_throws ArgumentError Mabs.dmrg(H_wrong, psi0; nsweeps=2)
        end
    end
    
    @testset "Time Evolution - TEBD" begin
        alg = PseudoSite(1)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:3]
        
        @testset "Number operator evolution" begin
            ω = 1.0
            dt = 0.1
            n = 2
            
            state_n = BMPS(sites, [n], alg)
            normalize!(state_n)
            
            n_op = number(sites, alg, 1)
            gate = exp(-1im * ω * dt * n_op)
            
            state_evolved = tebd(state_n, gate)
            
            overlap = ITensors.inner(state_n.mps, state_evolved.mps)
            expected_phase = exp(-1im * ω * dt * n)
            
            @test abs(overlap - expected_phase) < 1e-10
            @test abs(norm(state_evolved) - 1.0) < 1e-10
        end
        
        @testset "TEBD with multiple gates" begin
            psi = random_bmps(sites, alg; linkdims=4)
            normalize!(psi)
            
            gates = ITensors.ITensor[]
            n_op = number(sites, alg, 1)
            push!(gates, exp(-1im * 0.01 * n_op))
            
            psi_evolved = tebd(psi, gates)
            @test psi_evolved isa BMPS{<:ITensorMPS.MPS, PseudoSite}
            @test abs(norm(psi_evolved) - 1.0) < 1e-8
        end
    end
    
    @testset "Time Evolution - TDVP" begin
        alg = PseudoSite(2)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
        
        @testset "Basic TDVP evolution" begin
            H = harmonic_chain(sites, alg, 1.0, 0.5)
            psi0 = random_bmps(sites, alg; linkdims=4)
            normalize!(psi0)
            
            dt = 0.01
            psi_evolved = Mabs.tdvp(psi0, H, -1im * dt; cutoff=1e-10)
            
            @test psi_evolved isa BMPS{<:ITensorMPS.MPS, PseudoSite}
            @test psi_evolved !== psi0
        end
        
        @testset "TDVP algorithm mismatch error" begin
            alg_wrong = PseudoSite(3)
            sites_wrong = [ITensors.Index(2, "Qubit,n=$i") for i in 1:9]
            H_wrong = harmonic_chain(sites_wrong, alg_wrong, 1.0, 0.0)
            
            psi = random_bmps(sites, alg; linkdims=4)
            
            @test_throws ArgumentError Mabs.tdvp(psi, H_wrong, 0.01)
        end
    end
    
    @testset "Arithmetic Operations" begin
        alg = PseudoSite(2)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
        
        @testset "BMPS addition" begin
            psi1 = random_bmps(sites, alg; linkdims=2)
            psi2 = random_bmps(sites, alg; linkdims=2)
            
            psi_sum = psi1 + psi2
            @test psi_sum isa BMPS{<:ITensorMPS.MPS, PseudoSite}
            
            alg_wrong = PseudoSite(3)
            sites_wrong = [ITensors.Index(2, "Qubit,n=$i") for i in 1:9]
            psi_wrong = random_bmps(sites_wrong, alg_wrong)
            
            @test_throws ArgumentError psi1 + psi_wrong
        end
        
        @testset "BMPS inner products" begin
            psi1 = random_bmps(sites, alg; linkdims=2)
            psi2 = random_bmps(sites, alg; linkdims=2)
            normalize!(psi1)
            normalize!(psi2)
            
            overlap1 = dot(psi1, psi2)
            @test overlap1 isa Number
            @test isfinite(overlap1)
            
            overlap2 = ITensorMPS.inner(psi1, psi2)
            @test overlap2 isa Number
            @test abs(overlap1 - overlap2) < 1e-10
            
            self_overlap = dot(psi1, psi1)
            @test abs(self_overlap - 1.0) < 1e-10
        end
        
        @testset "BMPO-BMPS operations" begin
            H = harmonic_chain(sites, alg, 1.0, 0.0)
            psi = random_bmps(sites, alg; linkdims=4)
            
            result = ITensors.contract(H, psi)
            @test result isa BMPS{<:ITensorMPS.MPS, PseudoSite}
            
            result2 = ITensors.apply(H, psi; maxdim=10)
            @test result2 isa BMPS{<:ITensorMPS.MPS, PseudoSite}
        end
        
        @testset "BMPO addition" begin
            opsum1 = ITensors.OpSum()
            opsum2 = ITensors.OpSum()
            for i in 1:alg.nmodes
                opsum1 += 1.0, "N", i
                opsum2 += 0.5, "N", i
            end
            
            H1 = BMPO(opsum1, sites, alg)
            H2 = BMPO(opsum2, sites, alg)
            
            H_sum = H1 + H2
            @test H_sum isa BMPO{<:ITensorMPS.MPO, PseudoSite}
        end
    end
    
    @testset "Advanced BMPS Operations" begin
        alg = PseudoSite(2)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
        
        @testset "Orthogonalization" begin
            psi = random_bmps(sites, alg; linkdims=4)
            
            for j in 1:6
                psi_ortho = ITensorMPS.orthogonalize(psi, j)
                @test psi_ortho isa BMPS{<:ITensorMPS.MPS, PseudoSite}
                @test abs(norm(psi_ortho) - norm(psi)) < 1e-10
            end
            
            psi_copy = copy(psi)
            ITensorMPS.orthogonalize!(psi_copy, 3)
            @test psi_copy isa BMPS{<:ITensorMPS.MPS, PseudoSite}
        end
        
        @testset "Truncation" begin
            psi = random_bmps(sites, alg; linkdims=10)
            
            psi_truncated = ITensorMPS.truncate(psi; maxdim=5)
            @test psi_truncated isa BMPS{<:ITensorMPS.MPS, PseudoSite}
            @test maxlinkdim(psi_truncated) <= 5
            
            psi_copy = copy(psi)
            ITensorMPS.truncate!(psi_copy; maxdim=5)
            @test maxlinkdim(psi_copy) <= 5
        end
        
        @testset "Outer product" begin
            psi1 = random_bmps(sites, alg; linkdims=2)
            psi2 = random_bmps(sites, alg; linkdims=2)
            
            rho = ITensorMPS.outer(prime(psi1), psi2)
            @test rho isa BMPO{<:ITensorMPS.MPO, PseudoSite}
            @test length(rho) == 6
        end
        
        @testset "Indexing" begin
            psi = random_bmps(sites, alg; linkdims=2)
            
            @test psi[1] isa ITensors.ITensor
            @test psi[end] isa ITensors.ITensor
            
            count = 0
            for tensor in psi
                @test tensor isa ITensors.ITensor
                count += 1
            end
            @test count == 6
        end
    end
    
    @testset "Multi-mode Physics" begin
        alg = PseudoSite(2)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
        
        @testset "Independent mode operations" begin
            psi = BMPS(sites, [2, 3], alg)
            
            n1 = number(sites, alg, 1)
            n1_psi = ITensors.apply(n1, psi.mps)
            exp1 = real(ITensors.inner(psi.mps, n1_psi))
            @test abs(exp1 - 2) < 1e-10
            
            n2 = number(sites, alg, 2)
            n2_psi = ITensors.apply(n2, psi.mps)
            exp2 = real(ITensors.inner(psi.mps, n2_psi))
            @test abs(exp2 - 3) < 1e-10
        end
        
        @testset "Mode-specific operations don't affect other modes" begin
            psi = BMPS(sites, [1, 2], alg)
            
            a1_dag = create(sites, alg, 1)
            psi_new = ITensors.apply(a1_dag, psi.mps)
            normalize!(psi_new)
            
            n2 = number(sites, alg, 2)
            n2_psi = ITensors.apply(n2, psi_new)
            exp2 = real(ITensors.inner(psi_new, n2_psi))
            @test abs(exp2 - 2) < 1e-8
        end
    end
    
    @testset "Edge Cases and Error Handling" begin
        alg = PseudoSite(2)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
        
        @testset "Zero amplitude coherent state" begin
            psi = coherentstate(sites, alg, 0.0)
            vac = vacuumstate(sites, alg)
            
            overlap = abs(ITensors.inner(psi.mps, vac.mps))
            @test overlap > 0.99
        end
        
        @testset "Very small displacement" begin
            α = 1e-11
            psi = coherentstate(sites, alg, α)
            vac = vacuumstate(sites, alg)
            
            overlap = abs(ITensors.inner(psi.mps, vac.mps))
            @test overlap > 0.99
        end
        
        @testset "Algorithm mismatch errors" begin
            alg2 = PseudoSite(3)
            sites2 = [ITensors.Index(2, "Qubit,n=$i") for i in 1:9]
            
            psi1 = random_bmps(sites, alg)
            psi2 = random_bmps(sites2, alg2)
            
            @test_throws ArgumentError psi1 + psi2
            @test_throws ArgumentError dot(psi1, psi2)
            @test_throws ArgumentError ITensorMPS.inner(psi1, psi2)
        end
    end
    
    @testset "Qubit Operator Definitions" begin
        @testset "ITensors.op definitions" begin
            site = ITensors.Index(2, "Qubit")
            
            n_op = ITensors.op("N", site)
            @test n_op isa ITensors.ITensor
            @test n_op[site'=>1, site=>1] == 0.0
            @test n_op[site'=>2, site=>2] == 1.0
            
            id_op = ITensors.op("Id", site)
            @test id_op isa ITensors.ITensor
            @test id_op[site'=>1, site=>1] == 1.0
            @test id_op[site'=>2, site=>2] == 1.0
            
            x_op = ITensors.op("X", site)
            @test x_op isa ITensors.ITensor
            
            z_op = ITensors.op("Z", site)
            @test z_op isa ITensors.ITensor
            
            y_op = ITensors.op("Y", site)
            @test y_op isa ITensors.ITensor
        end
    end
    
    @testset "Performance and Scaling" begin
        @testset "Bond dimension growth" begin
            alg = PseudoSite(2)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
            
            psi = random_bmps(sites, alg; linkdims=2)
            @test maxlinkdim(psi) <= 2
            
            H = harmonic_chain(sites, alg, 1.0, 0.5)
            psi_evolved = Mabs.tdvp(psi, H, 0.01; cutoff=1e-12)
            
            @test maxlinkdim(psi_evolved) >= maxlinkdim(psi)
        end
        
        @testset "Different qubit cluster sizes" begin
            for nqubits in [2, 3, 4]
                alg = PseudoSite(1)
                sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:nqubits]
                max_occ = 2^nqubits - 1
                
                psi = vacuumstate(sites, alg)
                @test abs(norm(psi) - 1.0) < 1e-10
                
                psi_max = BMPS(sites, [max_occ], alg)
                @test abs(norm(psi_max) - 1.0) < 1e-10
            end
        end
    end

    @testset "Comprehensive Operator Matrix Construction" begin
        alg = PseudoSite(1)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:3]
        
        @testset "Internal matrix construction helpers" begin
            matrix = zeros(ComplexF64, 8, 8)
            for i in 1:8
                matrix[i, i] = Float64(i-1)
            end
            
            cluster = sites
            op = Mabs._matrix_to_cluster_operator(matrix, cluster)
            
            @test op isa ITensors.ITensor
            @test ITensors.hasinds(op, cluster'..., cluster...)
            
            for n in 0:7
                buffer = zeros(Int, 3)
                Mabs._fock_to_qubit!(buffer, n, 3)
                
                idx_spec = Pair{ITensors.Index, Int}[]
                for (i, site) in enumerate(cluster)
                    push!(idx_spec, site' => buffer[i])
                    push!(idx_spec, site => buffer[i])
                end
                
                val = op[idx_spec...]
                @test abs(val - Float64(n)) < 1e-12
            end
        end
        
        @testset "set_cluster_matrix_element!" begin
            cluster = sites
            op = ITensors.ITensor(ComplexF64, cluster'..., cluster...)
            
            bra = [2, 1, 1]
            ket = [1, 1, 1]
            value = 1.5 + 0.5im
            
            Mabs.set_cluster_matrix_element!(op, cluster, bra, ket, value)
            
            idx_spec = Pair{ITensors.Index, Int}[]
            for (i, site) in enumerate(cluster)
                push!(idx_spec, site' => bra[i])
                push!(idx_spec, site => ket[i])
            end
            
            @test abs(op[idx_spec...] - value) < 1e-12
        end
    end
    
    @testset "Displacement Operator Matrix Elements" begin
        alg = PseudoSite(1)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:3]
        
        @testset "_displace_matrix_element" begin
            α = 0.5 + 0.3im
            
            elem_00 = Mabs._displace_matrix_element(0, 0, α)
            @test abs(elem_00 - 1.0) < 1e-12
            
            elem_10 = Mabs._displace_matrix_element(1, 0, α)
            @test abs(elem_10 - α) < 1e-12
            
            elem_01 = Mabs._displace_matrix_element(0, 1, α)
            @test abs(elem_01 - (-conj(α))) < 1e-12
            
            α_real = 1.0
            elem_real = Mabs._displace_matrix_element(2, 0, α_real)
            expected = α_real^2 / sqrt(2.0) * sqrt(2.0 / 1.0)
            @test abs(elem_real - expected) < 1e-12
        end
        
        @testset "Displacement operator completeness" begin
            α = 0.5
            D = displace(sites, alg, 1, α)
            
            @test D isa ITensors.ITensor
            
            vac = vacuumstate(sites, alg)
            D_vac = ITensors.apply(D, vac.mps; maxdim=10)
            normalize!(D_vac)
            
            @test abs(norm(D_vac) - 1.0) < 1e-8
        end
    end
    
    @testset "Squeezing Operator Matrix Elements" begin
        alg = PseudoSite(1)
        sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:3]
        
        @testset "_squeezing_matrix_element" begin
            r = 0.3
            φ = π/4
            
            elem_00 = Mabs._squeezing_matrix_element(0, 0, r, φ)
            expected_00 = 1.0 / cosh(r)
            @test abs(elem_00 - expected_00) < 1e-12
            
            S = squeeze(sites, alg, 1, r * exp(1im * φ))
            
            state_1 = BMPS(sites, [1], alg)
            state_2 = BMPS(sites, [2], alg)
            
            S_state_1 = ITensors.apply(S, state_1.mps)
            overlap_12 = ITensors.inner(state_2.mps, S_state_1)
            
            @test abs(overlap_12) < 1e-10
        end
    end

    @testset "Production-ready Integration Tests" begin
        @testset "Full quantum circuit simulation" begin
            alg = PseudoSite(2)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:6]
            
            psi = vacuumstate(sites, alg)
            
            D1 = displace(sites, alg, 1, 0.5)
            psi.mps = ITensors.apply(D1, psi.mps; maxdim=15)
            normalize!(psi.mps)
            S2 = squeeze(sites, alg, 2, 0.3)
            psi.mps = ITensors.apply(S2, psi.mps; maxdim=15)
            normalize!(psi.mps)
            H = harmonic_chain(sites, alg, 1.0, 0.2)
            psi = Mabs.tdvp(psi, H, 0.01; cutoff=1e-10)
            
            @test abs(norm(psi) - 1.0) < 0.1
            
            n1 = number(sites, alg, 1)
            n1_exp = real(ITensors.inner(psi.mps, ITensors.apply(n1, psi.mps)))
            @test n1_exp >= 0
            @test isfinite(n1_exp)
        end
        
        @testset "Error correction: cat state preparation" begin
            alg = PseudoSite(1)
            sites = [ITensors.Index(2, "Qubit,n=$i") for i in 1:4]
            
            α = 1.5
            psi_plus = coherentstate(sites, alg, α)
            psi_minus = coherentstate(sites, alg, -α)
            
            psi_cat = add(psi_plus, psi_minus; maxdim=30)
            normalize!(psi_cat)
            
            @test abs(norm(psi_cat) - 1.0) < 1e-8
            
            @test norm(psi_cat.mps) > 0
        end
    end
end