# InverseAlgos.jl

Docs: 
[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ginvlab.github.io/InverseAlgos.jl/stable)
[![Docs Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://ginvlab.github.io/InverseAlgos.jl/dev)
[![Documentation](https://github.com/GinvLab/InverseAlgos.jl/actions/workflows/Documentation.yml/badge.svg)](https://github.com/GinvLab/InverseAlgos.jl/actions/workflows/Documentation.yml)

> [!WARNING]
> **Work in progress!**

Probabilistic and deterministic inverse algorithms for geophysical problems and beyond.
 
`InverseAlgos` is an unbrella package currently including three sub-modules:
- `Samplers`: (Hamiltonian) Monte Carlo sampling algorithms (formerly part of HMCLab)
- `Optimizers`: deterministic gradient-based algorithms, currently including l-BFGS and Gauss-Newton algos
- `KronLinInv`: Kronecker-product-based least squares inversion under Gaussian and separability assumptions


See

* Andrea Zunino, Lars Gebraad, Alessandro Ghirotto, Andreas Fichtner (2023), **HMCLab: a framework for solving diverse geophysical inverse problems using the Hamiltonian Monte Carlo method**, Geophysical Journal International, Volume 235, Issue 3, Pages 2979–2991, [https://doi.org/10.1093/gji/ggad403](https://doi.org/10.1093/gji/ggad403)

* Andrea Zunino, Klaus Mosegaard (2019), **An efficient method to solve large linearizable inverse problems under Gaussian and separability assumptions**, *Computers & Geosciences*. ISSN 0098-3004, [https://doi.org/10.1016/j.cageo.2018.09.005](https://doi.org/10.1016/j.cageo.2018.09.005).


