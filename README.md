Matrix product state backend for simulating bosonic systems.

## Interface Plans

State methods:
```
vacuumstate
thermalstate
coherentstate
squeezedstate
eprstate
fockstate
gkpstate
catstate
```
Gate methods:
```
displace
squeeze
twosqueeze
kerr
cubic
number
create
destroy
beamsplitter
```
Measurement methods:
```
homodyne
generaldyne
heterodyne
count (photon counting, maybe there's a better name)
threshold (zero photons or nonzero photons)
```
Possible backend interface (inspired by Makie.jl themes):
```
with_backend(backend) do
  # insert bosonic simulation
end
```
Possible names for backends could be `GaussianBackend(::SymplecticBasis)` and `MPSBackend(; alg, maxdim, cutoff, nsites)`.

## References

### Review Articles
- Stolpp, J., Köhler, T., Manmana, S. R., Jeckelmann, E., Heidrich-Meisner, F., and Paeckel, S., “Comparative Study of State-of-the-Art Matrix-Product-State Methods for Lattice Models with Large Local Hilbert Spaces”, <i>arXiv e-prints</i>, https://doi.org/10.48550/arXiv.2011.07412.
- Jeckelmann, E. and Fehske, H., “Exact numerical methods for electron-phonon problems”, <i>Nuovo Cimento Rivista Serie</i>, vol. 30, no. 6, pp. 259–292, 2007. https://doi.org/10.1393/ncr/i2007-10021-y.
- Magnifico, G., Cataldi, G., Rigobello, M. et al., “Tensor networks for lattice gauge theories beyond one dimension”, <i>Commun Phys 8, 322 (2025). https://doi.org/10.1038/s42005-025-02125-x.

### Pseudo-site Approach
- Jeckelmann, E. and White, S. R., “Density-matrix renormalization-group study of the polaron problem in the Holstein model”, <i>Physical Review B</i>, vol. 57, no. 11, pp. 6376–6385, 1998. https://doi.org/10.1103/PhysRevB.57.6376.
- Zhao, P.-Y., Ding, K., and Yang, S., “Chebyshev pseudosite matrix product state approach for the spectral functions of electron-phonon coupling systems”, <i>Physical Review Research</i>, vol. 5, no. 2, Art. no. 023026, 2023. https://10.1103/PhysRevResearch.5.023026.

### Local-Basis Optimization
- Vinther, J. and Kastoryano, M. J., “Variational tensor network simulation of Gaussian boson sampling and beyond”, <i>Physical Review A</i>, vol. 112, no. 2, Art. no. 022605, 2025. https://doi.org/10.1103/z463-7gqy.
- Brockt, C., Dorfner, F., Vidmar, L., Heidrich-Meisner, F., and Jeckelmann, E., “Matrix-product-state method with a dynamical local basis optimization for bosonic systems out of equilibrium”, <i>Physical Review B</i>, vol. 92, no. 24, Art. no. 241106, 2015. https://doi.org/10.1103/PhysRevB.92.241106.
- Guo, C., Weichselbaum, A., von Delft, J., and Vojta, M., “Critical and Strong-Coupling Phases in One- and Two-Bath Spin-Boson Models”, <i>Physical Review Letters</i>, vol. 108, no. 16, Art. no. 160401, 2012. https://doi.org/10.1103/PhysRevLett.108.160401.
- Wong, H. and Chen, Z.-D., “Density matrix renormalization group approach to the spin-boson model”, <i>Physical Review B</i>, vol. 77, no. 17, Art. no. 174305, 2008. https://doi.org/10.1103/PhysRevB.77.174305.

### Projected-Purified DMRG
- Köhler, T., Stolpp, J., and Paeckel, S., “Efficient and flexible approach to simulate low-dimensional quantum lattice models with large local Hilbert spaces”, <i>SciPost Physics</i>, vol. 10, no. 3, Art. no. 058, 2021. https://doi.org/10.21468/SciPostPhys.10.3.058.


