```@meta
```
# [InverseAlgos](@id InverseAlgos_header)


```@contents
Pages = ["index.md","mcsamplers.md","optimizers.md"]
Depth = 3
```

## Overview

Probabilistic and deterministic inverse algorithms for Geophysical problems and beyond. [InverseAlgos]() contains two main sub-modules: 

- [MCSamplers](@ref) for probabilistic (Monte Carlo) inversions, and,

- [Optimizers](@ref) for deterministic inversion (optimization).


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

