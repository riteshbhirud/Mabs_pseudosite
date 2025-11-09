using Test
using Mabs
using ITensors
using ITensorMPS
using LinearAlgebra

@testset "Truncated Algorithm Tests" begin
    
    @testset "Basic BMPS Operations" begin
        N = 4
        max_occ = 5
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        psi = random_bmps(sites, Truncated())
        @test length(psi) == N
        @test maxlinkdim(psi) > 0
        @test siteinds(psi) == sites
        @test eltype(psi) <: Number  
        
        psi_copy = copy(psi)
        @test psi_copy !== psi
        @test psi_copy.mps !== psi.mps
        
        psi_deepcopy = deepcopy(psi)
        @test psi_deepcopy !== psi
        @test psi_deepcopy.mps !== psi.mps
    end
    
    @testset "Matrix Product State Construction" begin
        N = 3
        max_occ = 4
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        vac = vacuumstate(sites, Truncated())
        @test length(vac) == N
        @test abs(norm(vac) - 1.0) < 1e-10  
        
        states = [1, 2, 3] 
        psi = BMPS(sites, Truncated(), states)
        @test length(psi) == N
        @test abs(norm(psi) - 1.0) < 1e-10
    end
    
    @testset "Coherent State Construction" begin
        max_occ = 10
        site = ITensors.siteinds("Boson", 1; dim=max_occ+1)
        α = 0.5 + 0.3im
        
        coherent_mps = coherentstate(site, Truncated(), α)
        @test length(coherent_mps) == 1
        @test abs(norm(coherent_mps) - 1.0) < 1e-8  
    end
    
    @testset "Operator Construction" begin
        max_occ = 5
        sites = ITensors.siteinds("Boson", 1; dim=max_occ+1)
        site = sites[1]
        
        a_dag = create(site, Truncated())
        a = destroy(site, Truncated())
        n = number(site, Truncated())

        @test a_dag isa ITensors.ITensor
        @test a isa ITensors.ITensor  
        @test n isa ITensors.ITensor
        
        vac = vacuumstate(sites, Truncated())
        one_photon = BMPS(sites, Truncated(), [2]) 
        two_photon = BMPS(sites, Truncated(), [3])  
        
        a_dag_vac = ITensors.apply(a_dag, vac.mps)
        normalize!(a_dag_vac)
        overlap_1 = abs(ITensors.inner(a_dag_vac, one_photon.mps))
        @test overlap_1 > 0.99  
        
        a_one = ITensors.apply(a, one_photon.mps)
        normalize!(a_one)
        overlap_0 = abs(ITensors.inner(a_one, vac.mps))
        @test overlap_0 > 0.99
        
        n_one = ITensors.apply(n, one_photon.mps)
        expectation_n = real(ITensors.inner(one_photon.mps, n_one))
        @test abs(expectation_n - 1.0) < 1e-8 
    end
    
    @testset "Hamiltonian Construction" begin
        N = 4
        max_occ = 6
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        H_harmonic = harmonic_chain(sites, Truncated(), 1.0, 0.1)
        @test H_harmonic isa BMPO{<:ITensorMPS.MPO,Truncated}
        @test length(H_harmonic) == N
        
        H_kerr = kerr(sites, Truncated(), 1.0, 0.05)
        @test H_kerr isa BMPO{<:ITensorMPS.MPO,Truncated}
        @test length(H_kerr) == N
    end

    @testset "BMPO OpSum Constructor" begin
        N = 3
        max_occ = 5
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
    
        @testset "Direct OpSum to BMPO construction" begin
            opsum = ITensors.OpSum()
            for i in 1:N
                opsum += 1.0, "N", i  
            end
            for i in 1:(N-1)
                opsum += 0.5, "Adag", i, "A", i+1  
                opsum += 0.5, "A", i, "Adag", i+1
            end
            
            H_direct = BMPO(opsum, sites, Truncated())
            
            @test H_direct isa BMPO{<:ITensorMPS.MPO,Truncated}
            @test length(H_direct) == N
            mpo_unprimed_sites = [s[end] for s in siteinds(H_direct)] 
            @test mpo_unprimed_sites == sites
            @test H_direct.alg isa Truncated
            
            mpo_manual = ITensorMPS.MPO(opsum, sites)
            H_manual = BMPO(mpo_manual, Truncated())
            
            @test length(H_direct) == length(H_manual)
            @test siteinds(H_direct) == siteinds(H_manual)
            @test maxlinkdim(H_direct) == maxlinkdim(H_manual)
        end
    
        @testset "OpSum constructor with different Hamiltonians" begin
            opsum_harmonic = ITensors.OpSum()
            for i in 1:N
                opsum_harmonic += 2.0, "N", i
            end
            
            H_harmonic = BMPO(opsum_harmonic, sites, Truncated())
            @test H_harmonic isa BMPO{<:ITensorMPS.MPO,Truncated}
            @test length(H_harmonic) == N
            
            opsum_hopping = ITensors.OpSum()
            for i in 1:(N-1)
                opsum_hopping += 1.0, "Adag", i, "A", i+1
                opsum_hopping += 1.0, "A", i, "Adag", i+1
            end
            
            H_hopping = BMPO(opsum_hopping, sites, Truncated())
            @test H_hopping isa BMPO{<:ITensorMPS.MPO,Truncated}
            @test length(H_hopping) == N
            
            opsum_single = ITensors.OpSum()
            opsum_single += 3.0, "N", 1
            opsum_single += 1.5, "N", 2
            
            H_single = BMPO(opsum_single, sites, Truncated())
            @test H_single isa BMPO{<:ITensorMPS.MPO,Truncated}
            @test length(H_single) == N
        end
    
        @testset "OpSum BMPO with DMRG" begin
            opsum = ITensors.OpSum()
            for i in 1:N
                opsum += 1.0, "N", i
            end
            
            H = BMPO(opsum, sites, Truncated())
            psi0 = random_bmps(sites, Truncated(); linkdims=4)
            
            result = Mabs.dmrg(H, psi0; nsweeps=2, maxdim=20, cutoff=1e-10)
            
            if result isa Tuple
                energy, psi_gs = result
                @test energy isa Real
                @test psi_gs isa BMPS{<:ITensorMPS.MPS,Truncated}
            else
                psi_gs = result
                @test psi_gs isa BMPS{<:ITensorMPS.MPS,Truncated}
            end
        end
    
        @testset "OpSum BMPO expectation values" begin
            opsum = ITensors.OpSum()
            opsum += 1.0, "N", 1  
            
            H = BMPO(opsum, sites, Truncated())
            
            psi = BMPS(sites, Truncated(), [2, 1, 1])  
            normalize!(psi)
            
            expectation_val = real(ITensors.inner(psi.mps, ITensors.apply(H.mpo, psi.mps)))
            @test abs(expectation_val - 1.0) < 1e-10
        end
    end
    
    @testset "DMRG Ground State Calculation" begin
        N = 2  
        max_occ = 6
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        H = harmonic_chain(sites, Truncated(), 1.0, 0.0)
        psi0 = random_bmps(sites, Truncated())
        
        energy, psi_gs = Mabs.dmrg(H, psi0; nsweeps=8, maxdim=100, cutoff=1e-10, noise=1e-11)
        
        @test energy isa Real
        @test psi_gs isa BMPS{<:ITensorMPS.MPS,Truncated}
        @test abs(energy - N/2) < 1.0
    end

    @testset "Time Evolution" begin
        N = 2
        max_occ = 4 
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        psi0 = vacuumstate(sites, Truncated())
        
        dt = 0.001  
        id_gate = ITensors.op("Id", sites[1])
        
        psi_evolved = tebd(psi0, id_gate)
        @test psi_evolved isa BMPS{<:ITensorMPS.MPS,Truncated}
        @test psi_evolved !== psi0
        
        psi_evolved = copy(psi0)
        psi_evolved = tebd(psi_evolved, id_gate)
        @test abs(norm(psi_evolved) - 1.0) < 1e-10
    end
    
    @testset "TDVP Evolution" begin
        N = 2
        max_occ = 4
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        H = harmonic_chain(sites, Truncated(), 1.0, 1.0)
        psi0 = random_bmps(sites, Truncated(); linkdims=4)
        normalize!(psi0)
        
        dt = 0.01
        
        psi_evolved = Mabs.tdvp(psi0, H, -1im * dt; cutoff=1e-8)
        @test psi_evolved isa BMPS{<:ITensorMPS.MPS,Truncated}
        @test psi_evolved !== psi0
    end
    
    @testset "Arithmetic Operations" begin
        N = 2
        max_occ = 4
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        psi1 = random_bmps(sites, Truncated(); linkdims=2)
        psi2 = random_bmps(sites, Truncated(); linkdims=2)
        
        psi_sum = psi1 + psi2
        @test psi_sum isa BMPS{<:ITensorMPS.MPS,Truncated}
        
        overlap = dot(psi1, psi2)
        @test overlap isa Number
        @test isfinite(overlap)
        
        H = harmonic_chain(sites, Truncated(), 1.0, 1.0)
        normalize!(psi1)
        expectation_val = real(ITensors.inner(psi1.mps, ITensors.apply(H.mpo, psi1.mps)))
        @test expectation_val isa Real
        @test isfinite(expectation_val)
    end
    
    @testset "Multi-mode Coherent States" begin
        single_sites = ITensors.siteinds("Boson", 1; dim=7)
        psi_single = coherentstate(single_sites, Truncated(), 0.5)
        @test length(psi_single) == 1
        @test abs(norm(psi_single) - 1.0) < 1e-8
        
        multi_sites = ITensors.siteinds("Boson", 3; dim=7)
        psi_uniform = coherentstate(multi_sites, Truncated(), 0.3)
        @test length(psi_uniform) == 3
        @test abs(norm(psi_uniform) - 1.0) < 1e-8
        
        αs = [0.2, 0.4 + 0.1im, 0.1]
        psi_mixed = coherentstate(multi_sites, Truncated(), αs)
        @test length(psi_mixed) == 3
        @test abs(norm(psi_mixed) - 1.0) < 1e-8
        
        @test_throws ErrorException coherentstate(multi_sites, Truncated(), [0.1, 0.2])  
    end

    @testset "Operator Matrix Elements" begin
        max_occ = 3
        sites = ITensors.siteinds("Boson", 1; dim=max_occ+1)
        site = sites[1]
        
        a_dag = create(site, Truncated())
        a = destroy(site, Truncated())
        
        state_0 = BMPS(sites, Truncated(), [1]) 
        state_1 = BMPS(sites, Truncated(), [2])  
        state_2 = BMPS(sites, Truncated(), [3])  
        
        result_01 = ITensors.apply(a_dag, state_0.mps)
        overlap_01 = abs(ITensors.inner(state_1.mps, result_01))
        @test abs(overlap_01 - 1.0) < 1e-10  
        
        result_12 = ITensors.apply(a_dag, state_1.mps)
        overlap_12 = abs(ITensors.inner(state_2.mps, result_12))
        @test abs(overlap_12 - sqrt(2)) < 1e-10
        
        result_10 = ITensors.apply(a, state_1.mps)
        overlap_10 = abs(ITensors.inner(state_0.mps, result_10))
        @test abs(overlap_10 - 1.0) < 1e-10
    end
    
    @testset "Displacement Operator" begin
        max_occ = 8
        sites = ITensors.siteinds("Boson", 1; dim=max_occ+1)
        α = 0.5  
        
        D = displace(sites[1], Truncated(), α)
        @test D isa ITensors.ITensor
        
        @test ITensors.hasinds(D, sites[1]', sites[1])
        
        vac = vacuumstate(sites, Truncated())
        displaced_vac = ITensors.apply(D, vac.mps)
        
        overlap = abs(ITensors.inner(vac.mps, displaced_vac))
        @test overlap < 1.0  
        @test overlap > 0.0  
    end

    @testset "TEBD Tests" begin
    
        @testset "TEBD Evolution" begin
            N = 2
            max_occ = 4
            sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
            
            @testset "Basic evolution with identity gate" begin
                psi = vacuumstate(sites, Truncated())
                id_gate = ITensors.op("Id", sites[1])
                
                psi_evolved = tebd(psi, id_gate)
                @test psi_evolved isa BMPS{<:ITensorMPS.MPS,Truncated}
                @test abs(norm(psi_evolved) - 1.0) < 1e-10
            end
        
            @testset "Evolution preserves normalization" begin
                psi = random_bmps(sites, Truncated(); linkdims=2)
                normalize!(psi)
                n_op = number(sites[1], Truncated())
                dt = 0.01
                gate = exp(-1im * dt * n_op)
                
                psi_evolved = tebd(psi, gate)
                @test abs(norm(psi_evolved) - 1.0) < 1e-8
            end
        end
    
        @testset "Backwards Compatibility" begin
            N = 2
            max_occ = 4
            sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
            
            psi = random_bmps(sites, Truncated(); linkdims=4)
            normalize!(psi)
            
            id_gate = ITensors.op("Id", sites[1])
            
            psi_evolved = tebd(psi, id_gate)
            @test psi_evolved isa BMPS{<:ITensorMPS.MPS,Truncated}
            @test psi_evolved !== psi
            
            psi_evolved = copy(psi)
            psi_evolved = tebd(psi_evolved, id_gate)
            @test abs(norm(psi_evolved) - 1.0) < 1e-10
        end
    end

    @testset "BMPS Advanced Operations" begin
        N = 3
        max_occ = 5
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        @testset "Orthogonalization" begin
            psi = random_bmps(sites, Truncated(); linkdims=4)
            
            for j in 1:N
                psi_ortho = orthogonalize(psi, j)
                @test psi_ortho isa BMPS{<:ITensorMPS.MPS,Truncated}
                @test abs(norm(psi_ortho) - norm(psi)) < 1e-10
            end
            
            psi_copy = copy(psi)
            orthogonalize!(psi_copy, 2)
            @test psi_copy isa BMPS{<:ITensorMPS.MPS,Truncated}
        end
        
        @testset "Truncation" begin
            psi = random_bmps(sites, Truncated(); linkdims=10)
            
            psi_truncated = truncate(psi; maxdim=5)
            @test psi_truncated isa BMPS{<:ITensorMPS.MPS,Truncated}
            @test maxlinkdim(psi_truncated) <= 5
            
            psi_copy = copy(psi)
            truncate!(psi_copy; maxdim=5)
            @test maxlinkdim(psi_copy) <= 5
        end
        
        @testset "Addition with truncation" begin
            psi1 = random_bmps(sites, Truncated(); linkdims=3)
            psi2 = random_bmps(sites, Truncated(); linkdims=3)
            
            psi_sum = psi1 + psi2
            @test psi_sum isa BMPS{<:ITensorMPS.MPS,Truncated}
            
            psi_sum_trunc = add(psi1, psi2; maxdim=5, cutoff=1e-10)
            @test maxlinkdim(psi_sum_trunc) <= 5
        end
        
        @testset "Outer product" begin
            psi1 = random_bmps(sites, Truncated(); linkdims=2)
            psi2 = random_bmps(sites, Truncated(); linkdims=2)
            
            rho = outer(prime(psi1), psi2)
            @test rho isa BMPO{<:ITensorMPS.MPO,Truncated}
            @test length(rho) == N
        end
        
        @testset "Indexing and iteration" begin
            psi = random_bmps(sites, Truncated(); linkdims=2)
            
            @test psi[1] isa ITensors.ITensor
            @test psi[end] isa ITensors.ITensor
            
            count = 0
            for tensor in psi
                @test tensor isa ITensors.ITensor
                count += 1
            end
            @test count == N
            
            old_tensor = psi[1]
            psi[1] = old_tensor
            @test psi[1] == old_tensor
        end
    end

    @testset "BMPO Advanced Operations" begin
        N = 3
        max_occ = 5
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        @testset "BMPO copy operations" begin
            opsum = ITensors.OpSum()
            for i in 1:N
                opsum += 1.0, "N", i
            end
            H = BMPO(opsum, sites, Truncated())
            
            H_copy = copy(H)
            @test H_copy !== H
            @test H_copy.mpo !== H.mpo
            
            H_deep = deepcopy(H)
            @test H_deep !== H
            @test H_deep.mpo !== H.mpo
        end
        
        @testset "BMPO truncation" begin
            opsum = ITensors.OpSum()
            for i in 1:N
                opsum += 1.0, "N", i
            end
            H = BMPO(opsum, sites, Truncated())
            
            H_trunc = truncate(H; maxdim=10)
            @test H_trunc isa BMPO{<:ITensorMPS.MPO,Truncated}
            
            H_copy = copy(H)
            truncate!(H_copy; maxdim=10)
            @test H_copy isa BMPO{<:ITensorMPS.MPO,Truncated}
        end
        
        @testset "BMPO addition" begin
            opsum1 = ITensors.OpSum()
            opsum2 = ITensors.OpSum()
            for i in 1:N
                opsum1 += 1.0, "N", i
                opsum2 += 0.5, "N", i
            end
            
            H1 = BMPO(opsum1, sites, Truncated())
            H2 = BMPO(opsum2, sites, Truncated())
            
            H_sum = H1 + H2
            @test H_sum isa BMPO{<:ITensorMPS.MPO,Truncated}
            
            H_sum_trunc = add(H1, H2; maxdim=20)
            @test H_sum_trunc isa BMPO{<:ITensorMPS.MPO,Truncated}
        end
        
        @testset "BMPO-BMPS operations" begin
            opsum = ITensors.OpSum()
            for i in 1:N
                opsum += 1.0, "N", i
            end
            H = BMPO(opsum, sites, Truncated())
            psi = random_bmps(sites, Truncated(); linkdims=4)
            
            result = contract(H, psi)
            @test result isa BMPS{<:ITensorMPS.MPS,Truncated}
            
            result2 = ITensors.apply(H, psi; maxdim=10)
            @test result2 isa BMPS{<:ITensorMPS.MPS,Truncated}
        end
        
        @testset "BMPO indexing" begin
            opsum = ITensors.OpSum()
            for i in 1:N
                opsum += 1.0, "N", i
            end
            H = BMPO(opsum, sites, Truncated())
            
            @test H[1] isa ITensors.ITensor
            @test H[end] isa ITensors.ITensor
            
            count = 0
            for tensor in H
                @test tensor isa ITensors.ITensor
                count += 1
            end
            @test count == N
        end
    end

    @testset "Operator Completeness" begin
        max_occ = 5
        sites = ITensors.siteinds("Boson", 1; dim=max_occ+1)
        site = sites[1]
    
        @testset "Squeeze operator" begin
            ξ = 0.5
            S = squeeze(site, Truncated(), ξ)
            @test S isa ITensors.ITensor
            @test ITensors.hasinds(S, site', site)
            
            vac = vacuumstate(sites, Truncated())
            squeezed = ITensors.apply(S, vac.mps)
            @test squeezed isa ITensorMPS.MPS
        end
    
        @testset "Kerr single-site operator" begin
            χ = 0.1
            t = 0.5
            K = kerr(site, Truncated(), χ, t)
            @test K isa ITensors.ITensor
            @test ITensors.hasinds(K, site', site)
            
            for n in 0:max_occ
                expected_phase = exp(-1im * χ * t * n^2)
                actual = K[site'=>(n+1), site=>(n+1)]
                @test abs(actual - expected_phase) < 1e-12
            end
        end
    
        @testset "ITensor op definitions" begin
            n_op = ITensors.op("N", site)
            @test n_op isa ITensors.ITensor
            
            a_dag_op = ITensors.op("Adag", site)
            @test a_dag_op isa ITensors.ITensor
            
            a_op = ITensors.op("A", site)
            @test a_op isa ITensors.ITensor
        end
    
        @testset "_safe_factorial" begin
            @test Mabs._safe_factorial(5) == 120
            @test Mabs._safe_factorial(0) == 1
            @test Mabs._safe_factorial(20) == factorial(20)
            @test Mabs._safe_factorial(21) isa BigInt
        end
    end

    @testset "TEBD/TDVP Edge Cases" begin
        N = 2
        max_occ = 5
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        
        @testset "TEBD with multiple gates" begin
            psi = random_bmps(sites, Truncated(); linkdims=4)
            normalize!(psi)
            
            gates = ITensors.ITensor[]
            for i in 1:N
                n_op = number(sites[i], Truncated())
                gate = exp(-1im * 0.01 * n_op)
                push!(gates, gate)
            end
            
            psi_evolved = tebd(psi, gates)
            @test psi_evolved isa BMPS{<:ITensorMPS.MPS,Truncated}
            @test abs(norm(psi_evolved) - 1.0) < 1e-8
        end
        
        @testset "TEBD with truncation" begin
            psi = random_bmps(sites, Truncated(); linkdims=10)
            gate = ITensors.op("Id", sites[1])
            
            psi_evolved = tebd(psi, gate; maxdim=5, cutoff=1e-10)
            @test psi_evolved isa BMPS{<:ITensorMPS.MPS,Truncated}
        end
        
        @testset "TDVP with parameters" begin
            H = harmonic_chain(sites, Truncated(), 1.0, 1.0)
            psi = random_bmps(sites, Truncated(); linkdims=4)
            normalize!(psi)
            
            psi_evolved = Mabs.tdvp(psi, H, 0.1; cutoff=1e-10)
            @test psi_evolved isa BMPS{<:ITensorMPS.MPS,Truncated}
        end
    end

    @testset "Error Handling" begin
        N = 2
        max_occ = 5
        sites1 = ITensors.siteinds("Boson", N; dim=max_occ+1)
        sites2 = ITensors.siteinds("Boson", N; dim=max_occ+1)  
        
        @testset "Dimension mismatches" begin
            psi1 = random_bmps(sites1, Truncated())
            psi2 = random_bmps(sites2, Truncated())
            
            @test_throws Exception psi1 + psi2
        end
        
        @testset "Invalid coherent state inputs" begin
            @test_throws ErrorException coherentstate(sites1, Truncated(), [0.1])
        end
        
        @testset "DMRG with incompatible types" begin
            H1 = harmonic_chain(sites1, Truncated(), 1.0, 1.0)
            psi2 = random_bmps(sites2, Truncated())
            
            @test_throws Exception dmrg(H1, psi2; nsweeps=2)
        end
    end

    @testset "Bosonic Commutation Relations" begin
        max_occ = 10
        sites = ITensors.siteinds("Boson", 1; dim=max_occ+1)
        site = sites[1]
        a = destroy(site, Truncated())
        a_dag = create(site, Truncated())
        @testset "Commutator on Fock states" begin
            for n in 0:5
                psi_n = BMPS(sites, Truncated(), [n+1])
            
                adag_a_psi = ITensors.apply(a, psi_n.mps)
                adag_a_psi = ITensors.apply(a_dag, adag_a_psi)
                exp1 = real(ITensors.inner(psi_n.mps, adag_a_psi))
                @test abs(exp1 - n) < 1e-10
                
                a_adag_psi = ITensors.apply(a_dag, psi_n.mps)
                a_adag_psi = ITensors.apply(a, a_adag_psi)
                exp2 = real(ITensors.inner(psi_n.mps, a_adag_psi))
                @test abs(exp2 - (n+1)) < 1e-10
                
                commutator_value = exp2 - exp1
                @test abs(commutator_value - 1.0) < 1e-10
            end
        end
        
        @testset "Number operator relation" begin
            for n in 0:5
                psi_n = BMPS(sites, Truncated(), [n+1])
                n_op = number(site, Truncated())
                n_psi = ITensors.apply(n_op, psi_n.mps)
                n_exp = real(ITensors.inner(psi_n.mps, n_psi))
                adag_a_psi = ITensors.apply(a, psi_n.mps)
                adag_a_psi = ITensors.apply(a_dag, adag_a_psi)
                adag_a_exp = real(ITensors.inner(psi_n.mps, adag_a_psi))
                @test abs(n_exp - adag_a_exp) < 1e-10
            end
        end
    end

    @testset "Coherent State Physics" begin
        max_occ = 10
        sites = ITensors.siteinds("Boson", 1; dim=max_occ+1)
        α = 2.0 + 1.0im
        
        psi_coherent = coherentstate(sites, Truncated(), α)
        
        @testset "Photon number expectation" begin
            n_expectation = real(ITensorMPS.expect(psi_coherent.mps, "N"; sites=1))
            @test abs(n_expectation - abs2(α)) < 0.2
        end
        
        @testset "Normalized state" begin
            @test abs(norm(psi_coherent) - 1.0) < 1e-8
        end
        
        @testset "Non-zero for all relevant Fock states" begin
            a_op = destroy(sites[1], Truncated())
            a_psi = ITensors.apply(a_op, psi_coherent.mps)
            @test norm(a_psi) > 0 
            
            a_dag_op = create(sites[1], Truncated())
            adag_psi = ITensors.apply(a_dag_op, psi_coherent.mps)
            @test norm(adag_psi) > 0  
        end
    end

    @testset "DMRG Energy Accuracy" begin
        N = 4
        max_occ = 10
        sites = ITensors.siteinds("Boson", N; dim=max_occ+1)
        ω = 1.0
        H = harmonic_chain(sites, Truncated(), ω, 0.0)
        psi0 = random_bmps(sites, Truncated(); linkdims=10)
        
        energy, psi_gs = Mabs.dmrg(H, psi0; nsweeps=20, maxdim=50, cutoff=1e-12)
        
        expected_energy = 0.0  
        @test abs(energy - expected_energy) < 0.01
    end

    @testset "Operator Action Accuracy" begin
        max_occ = 10
        sites = ITensors.siteinds("Boson", 1; dim=max_occ+1)
        
        @testset "Number operator eigenvalues" begin
            n_op = number(sites[1], Truncated())
            
            for n in 0:5
                psi_n = BMPS(sites, Truncated(), [n+1])
                n_psi = ITensors.apply(n_op, psi_n.mps)
                
                expectation = real(ITensors.inner(psi_n.mps, n_psi))
                @test abs(expectation - n) < 1e-12
            end
        end
        
        @testset "Ladder operator norms" begin
            a = destroy(sites[1], Truncated())
            a_dag = create(sites[1], Truncated())
            
            for n in 1:5
                psi_n = BMPS(sites, Truncated(), [n+1])
                
                a_psi = ITensors.apply(a, psi_n.mps)
                @test abs(norm(a_psi) - sqrt(n)) < 1e-10
                
                adag_psi = ITensors.apply(a_dag, psi_n.mps)
                @test abs(norm(adag_psi) - sqrt(n+1)) < 1e-10
            end
        end
    end

    @testset "Time Evolution Accuracy" begin
        max_occ = 8
        sites = ITensors.siteinds("Boson", 1; dim=max_occ+1)
        
        @testset "Number operator evolution" begin
            ω = 1.0
            dt = 0.5
            n = 2
            
            psi_n = BMPS(sites, Truncated(), [n+1])
            normalize!(psi_n)
            
            n_op = number(sites[1], Truncated())
            gate = exp(-1im * dt * ω * n_op)
            
            psi_evolved = tebd(psi_n, gate)
            
            overlap = ITensors.inner(psi_n.mps, psi_evolved.mps)
            expected_phase = exp(-1im * ω * dt * n)
            @test abs(overlap - expected_phase) < 1e-10
        end
    end

    @testset "Operator Action Accuracy" begin
        max_occ = 10
        sites = ITensors.siteinds("Boson", 1; dim=max_occ+1)
        
        @testset "Number operator eigenvalues" begin
            n_op = number(sites[1], Truncated())
            
            for n in 0:5
                psi_n = BMPS(sites, Truncated(), [n+1])
                n_psi = ITensors.apply(n_op, psi_n.mps)
                
                expectation = real(ITensors.inner(psi_n.mps, n_psi))
                @test abs(expectation - n) < 1e-12
            end
        end
        
        @testset "Ladder operator norms" begin
            a = destroy(sites[1], Truncated())
            a_dag = create(sites[1], Truncated())
            
            for n in 1:5
                psi_n = BMPS(sites, Truncated(), [n+1])
                
                a_psi = ITensors.apply(a, psi_n.mps)
                @test abs(norm(a_psi) - sqrt(n)) < 1e-10
                
                adag_psi = ITensors.apply(a_dag, psi_n.mps)
                @test abs(norm(adag_psi) - sqrt(n+1)) < 1e-10
            end
        end
    end

    @testset "Time Evolution Accuracy" begin
        max_occ = 8
        sites = ITensors.siteinds("Boson", 1; dim=max_occ+1)
        
        @testset "Number operator evolution phase" begin
            ω = 1.0
            dt = 0.5
            n = 2
            psi_n = BMPS(sites, Truncated(), [n+1])
            normalize!(psi_n)
            n_op = number(sites[1], Truncated())
            gate = exp(-1im * dt * ω * n_op)
            psi_evolved = tebd(psi_n, gate)
            overlap = ITensors.inner(psi_n.mps, psi_evolved.mps)
            expected_phase = exp(-1im * ω * dt * n)
            @test abs(overlap - expected_phase) < 1e-10
        end
    end
end