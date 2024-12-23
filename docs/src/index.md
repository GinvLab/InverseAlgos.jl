```@meta
```
# [InverseAlgos](@id InverseAlgos_header)


```@contents
Pages = ["index.md","samplers.md","optimizers.md","kronlininv.md"]
Depth = 2
```

## Overview

Probabilistic and deterministic inverse algorithms for Geophysical problems and beyond. [InverseAlgos]() contains two main sub-modules: 

- [Samplers](@ref) for probabilistic (Monte Carlo) inversions, and,

- [Optimizers](@ref) for deterministic inversion (optimization).

- [KronLinInv](@ref) for linear inversion under Gaussian and separability assumptions


## Installation

To install the package simple enter into the package manager mode in Julia by typing "`]`" at the REPL prompt and add the `GinvLab` registry as
```
(@v1.8) pkg> registry add https://github.com/GinvLab/GinvLabRegistry
```
Then install the package with 
```
(v1.8) pkg> add InverseAlgos
```
The package will be automatically downloaded from the web and installed.

