# InverseAlgos.jl

Docs: 
[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ginvlab.github.io/InverseAlgos.jl/stable)
[![Docs Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://ginvlab.github.io/InverseAlgos.jl/dev)
[![Documentation](https://github.com/GinvLab/InverseAlgos.jl/actions/workflows/Documentation.yml/badge.svg)](https://github.com/GinvLab/InverseAlgos.jl/actions/workflows/Documentation.yml)

Probabilistic and deterministic inverse algorithms for Geophysical problems and beyond.
 
`InverseAlgos` is an unbrella package currently including three sub-modules:
- `MCSamplers`: (Hamiltonian) Monte Carlo sampling algorithms (formerly part of HMCLab)
- `Optimizers`: deterministic gradient-based descent algorithms
- `KronLinInv`: Kronecker-product-based least squares inversion under Gaussian and separability assumptions
