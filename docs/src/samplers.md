```@meta
```

# Samplers


```@contents
Pages = ["index.md"]
Depth = 3
```

## Quick overview

`Samplers` contains sampling algorithms to solve inverse problems. It focuses mainly on the Hamiltonian Monte Carlo (HMC) algorithm and its variants. 
Currently the following samplers are implemented:

 * the "standard" HMC algorithm with optional constraints as described in [^4] and [^5], and
 * the No U-Turn (NUTS) algorithm as described in [^8].
 * the Extended Metropolis algorithm as described in [^9].
 
See the [User guide](@ref) section for an explanation of the basics of `Samplers` and some minimal usage examples.


## Theoretical background 

### Probabilistic approach to inverse problems

In the probabilistic approach, an inverse problem essentially represents an indirect measurement where the knowledge about the observed data and model parameters is completely expressed in terms of probabilities [^1]. Within such formalism, the general solution to the inverse problem is a probability density function (PDF), i.e., the posterior PDF (see [^1] and [^2] for a detailed explanation).
 Typically, the posterior PDF cannot be computed analytically, therefore sampling methods come into play. The result is thus not a single ``optimal'' model, but a collection of plausible models which may feature significant differences while they are generally able to explain the observed data and are compatible with the prior information. This reflects the nature of the inverse problem: if different scenarios are plausible from the point of view of the observed data, then the probabilistic solution will aim at reflecting that.
 
The posterior PDF is constructed from the combination of two pieces of separate information: 1) the prior knowledge on the model parameters, expressed by the PDF $\rho(\mathbf{m})$, where $\mathbf{m}$ represents the model parameters and 2) the information provided by the experiment, described by $L(\mathbf{m})$. The posterior distribution, under certain fairly wide assumptions, is then given by [^2] [^3]:
```math
\begin{align}
  \sigma(\mathbf{m}) = k \, \rho(\mathbf{m}) \, L(\mathbf{m}).
\end{align}
```
Since $\sigma(\mathbf{m})$ is a PDF, it requires to evaluate the relevant integrals to find features of interest. For example, calculating the expected model given the data requires evaluating the following integral
```math
\begin{equation}
  \mathbb{E} \left[ \mathbf{m} \right] = \int_{M} \mathbf{m} \, \sigma(\mathbf{m}) \, \mathrm{d}\mathbf{m},
\end{equation}
```
where $M$ represents the whole model space.

Finally, to calculate some arbitrary function $\phi(\mathbf{m})$ of $\mathbf{m}$, we can use the following relationship once we have obtained a collection of samples from the posterior distribution:
```math
\begin{equation}
 \int_{\mathrm{M}} \phi(\mathbf{m}) \sigma(\mathbf{m}) \, \mathrm{d} \mathbf{m} \approx \frac{1}{N} \sum_{i=1}^N \phi(\mathbf{m}_{(i)})
\end{equation}
```
where $N$ is the number of available samples


### Basic HMC algorithm

HMC constructs a Markov chain over an ``n``-dimensional probability density function ``\sigma(\mathbf{m})`` using classical Hamiltonian mechanics. The algorithm regards the current state ``\mathbf{m}`` of the Markov chain as the location of a physical particle in an ``n``-dimensional space ``\mathcal{M}`` (i.e., model or parameter space). It moves under the influence of a potential energy, ``U``, which is defined as
```math
\begin{align}
U(\mathbf{m})=-\ln{ \left( \sigma(\mathbf{m}) \right) }.
\end{align}
```
To complete the physical system, the state of the Markov chain needs to be artificially augmented with momentum variables ``\mathbf{p}`` and a generalized mass for every dimension pair. The collection of resulting masses is contained in a positive definite mass matrix ``\mathbf{M}`` of dimension ``n \times n``. The momenta and the mass matrix define the kinetic energy of the particle as
```math
\begin{align}
    K(\mathbf{p})=\frac{1}{2} \mathbf{p}^T \mathbf{M}^{-1} \mathbf{p}.
\end{align}
```
In the HMC algorithm, the momenta ``\mathbf{p}`` are drawn randomly from a multivariate Gaussian with covariance matrix ``\mathbf{M}`` (the mass matrix). The sum of  the location-dependent potential and momentum-dependent kinetic energy constitute the total energy, or Hamiltonian, of the system
```math
\begin{align}
    H(\mathbf{m},\mathbf{p})=U(\mathbf{m})+K(\mathbf{p}).
\end{align}
```
The Hamiltonian dynamics are governed by the following equations,
```math
\begin{align}
    \frac{\partial\mathbf{m}}{\partial\tau} = \frac{\partial H}{\partial\mathbf{p}},\quad \frac{\partial\mathbf{p}}{\partial\tau} = - \frac{\partial H}{\partial\mathbf{m}} \, ,
\end{align}
```
which determine the position of the particle as a function of time ``\tau``. This time ``\tau`` is artificial just like the mass matrix, it has no connection to the actual physics of the inverse problem at hand.

We can simplify Hamilton's equations using the fact that kinetic and potential energy depend only on momentum and location, respectively, to obtain
```math
\begin{align}
    \frac{\partial\mathbf{m}}{\partial\tau} = \mathbf{M}^{-1} \mathbf{p}, \quad \frac{\partial\mathbf{p}}{\partial\tau} = - \frac{\partial U}{\partial\mathbf{m}} \, .
\end{align}
```
Evolving ``\mathbf{m}`` over time ``\tau`` generates another possible state of the system with new position ``\mathbf{\tilde{m}}``, momentum ``\mathbf{\tilde{p}}``, potential energy ``\tilde{U}``, and kinetic energy ``\tilde{K}``. Due to the conservation of energy, the Hamiltonian is equal in both states, i.e., ``U+K = \tilde{U} + \tilde{K}``. Successively drawing random momenta and evolving the system generates a distribution of the possible states of the system. Thereby, HMC samples the joint momentum and model space, referred to as phase space. As we are not interested in the momentum component of phase space, we marginalize over the momenta by simply dropping them. This results in samples drawn from ``\sigma(\mathbf{m})``.

If one could solve Hamilton's equations exactly, every proposed state (after burn-in) would be a valid sample of ``\sigma(\mathbf{m})``. Since Hamilton's equations for non-linear forward models cannot be solved analytically, the system must be integrated numerically. Suitable integrators are symplectic, meaning that time reversibility, phase space partitioning and volume preservation are satisfied [^4][^5]. We employ the leapfrog method as described in [^4]. However, the Hamiltonian is generally not preserved exactly when explicit time-stepping schemes are used (e.g., [^6]). Therefore, the time evolution generates samples not exactly proportional to the original distribution. A Metropolis-Hastings correction step is therefore applied at the end of numerical integration.

In summary, at each iteration, samples are generated starting from a randomly drawn model ``\mathbf{m}`` in the following way:

   1. ``\mathbf{p}`` according to the Gaussian with mean ``\mathbf{0}`` and covariance matrix ``\mathbf{M}``;  
   2. Compute the Hamiltonian ``H`` of model ``\mathbf{m}`` with momenta ``\mathbf{p}``;  
   3. Propagate ``\mathbf{m}`` and ``\mathbf{p}`` for some time ``\tau`` to ``\tilde{\mathbf{m}}`` and     ``\tilde{\mathbf{p}}``, using the discretized version of Hamilton's equations and a suitable numerical integrator;   
   4. Compute the Hamiltonian ``\tilde{H}`` of model ``\tilde{\mathbf{m}}`` with momenta ``\mathbf{\tilde{p}}``;  
   5. Accept the proposed move ``\mathbf{m} \rightarrow \tilde{\mathbf{m}}`` with probability `` p_\text{accept} = \min \left( 1, \exp ( H-\tilde{H} ) \right)\,. ``
   6. If accepted, use (and count) ``\tilde{\mathbf{m}}`` as the new state. Otherwise, keep (and count) the previous state. Then return to point 1.  

The mass matrix ``\mathbf{M}`` is one of the important tuning parameters of the HMC algorithm; details on its meaning and suggestions for tuning can be found in [^5][^7]. Moreover, employing the discrete leapfrog integrator implies that there are two additional parameters that need to be tuned, namely the time step ``\epsilon``
and the number of iterations ``L`` [^4].

## User guide

### Monte Carlo simulations

All kinds of simulations are started using the function [`runMC`](@ref Samplers.runMC). The arguments passed to [`runMC`](@ref Samplers.runMC) determine which algorithm will be used, number of iterations, the output files, etc.. The signature of the function is:
```julia
runMC(
    pars::Dict,
    MCpar::AbstractMCParams,
    mstart::Vector{Float64};
    likelihood,
    prior
)
```
The argument `pars` is a dictionary containing a set of generic parameters for the simulation, such as the maximum number of iterations, the name of the simulation, the directory used to write the output, etc. See the documentation of [`runMC`](@ref Samplers.runMC) for more details. 
The argument `MCpar`is a Julia structure which depends on the algorithm intended to be employed. Its type, in fact, determines which algorithm will be used to perform the sampling. As such, the parameters that need to be specified for each algorithm are different, and so the "fields" of the `MCpar` structure. 
Currently the following `AbstractMCParams` structures (types) are defined:
  * [`HMCParams`](@ref Samplers.HMCParams) for the "standard" HMC algorithm with optional constraints as described in [^4] and [^5];
  * [`NUTSParams`](@ref Samplers.NUTSParams) for the No U-Turn (NUTS) algorithm as described in [^8];
  * [`ExtMetropParams`](@ref Samplers.ExtMetropParams) for the Extended Metropolis algorithm as described in [^9].
See their documentation for details.
The argument `mstart` is a vector containing the starting model for the simulation. The model has to be in the form of a 1D vector for the computations to work, i.e., physically 3D models need to be "unrolled" to 1D.
The last two arguments, `likelihood` and `prior` are related to the two PDFs composing the posterior PDF as seen above, i.e., the prior PDF ``\rho(\mathbf{m})`` and the likelihood PDF ``L(\mathbf{m})``. However,  `likelihood` and `prior` refer in practice to the **negative logarithm** of those, namely ``- \ln( \rho(\mathbf{m}))`` and the likelihood PDF ``- \ln (L(\mathbf{m})``.

Once the above parameters have been set, the simulation can be started by issuing, for instance, something like  
```julia
runMC(pars,MCpar,mstart,likelihood=mylikelihood,prior=myprior)
```

#### Output

The output of a simulation is written to two files:
 1. `<name of the simulation>_inp.jld2`, and
 2. `<name of the simulation>_outp.h5`.
The first file contains all the input parameters of the simulation which completely specify the problem under study. These include the prior and likelihood models, the generic simulation parameters, etc. Such file is saved in the `.jld2` format (using the package `JLD2`), which is a HDF5-compatible format capable of saving the Julia structures and retrieving them later on.
The second file contains mainly the results of the simulation, namely the saved models, some statistical info, etc.. Such file is saved in the `.h5` format (using the package `HDF5`).

The full output can be read and saved in an appropriate structure by using the function [`readMCoutput`](@ref Samplers.readMCoutput). The signature of this function is
```julia
readMCoutput(
    datadir::String,
    simname::String;
    istart,
    iend
) 
```
The argument `datadir` specifies the directory in which the output is saved and `simname` the name of the simulation. The optional parameters `istart` and `iend` specify which subset of the saved models to retrieve.

#### Interrupting and continuing simulations

This package allows you to interrupt the simulation at any time by simply killing the Julia process. That should not create problems to the output files even in case of a crash of the program. The output files can then be used to continue (or re-start) the simulation from the point where it was interrupted by using the function [`contrunMC`](@ref Samplers.contrunMC). This function needs as the first argument a dictionary containing the generic simulation parameters (see above). Moreover, this function allows for a continuation using the same exact parameters used before or, by additionally passing a new instance of `AbstractMCParams` of the appropriate type (e.g., a [`HMCParams`](@ref Samplers.HMCParams) struct) to continue the previous simulation with new algorithm-specific parameters.

The output files are written using the Single Writer Multiple Reader (SWMR) capabilities of the HDF5 file format, so it is easy while the simulation is running to launch a different Julia instance and, from there, to plot or analyze the results so far. The SWMR will guarantee that only the initial process will the able to write to the file, while allowing others to read the data.

Finally, in case the output file containing the saved models gets corrupted because of wrong HDF5 "status flags", the function  [`clear_stflags_hdf5`](@ref Samplers.clear_stflags_hdf5) provides a way to clear them and then continue the simulation using [`contrunMC`](@ref Samplers.contrunMC).


### Example 1: Sampling a 2D Gaussian with the "standard" HMC algorithm

In the following we show a simple example of how to construct a custom problem and run [`Samplers`](@ref) to sample the target distribution. In this case we aim at sampling a simple 2D Gaussian probability density function (PDF), which corresponds to having a Gaussian likelihood function and a forward model which is simply the identity matrix. In other words, the forward model is given by
```math
\mathbf{G} = 
\begin{bmatrix}
1 & 0 \\
0 & 1 
\end{bmatrix} 
\, .
```
while the misfit function (negative logarithm of the PDF) by
```math
U(\mathbf{m}) =  \frac{1}{2} \, (\mathbf{G} \mathbf{m}-\mathbf{d})^{\mathrm{T}} \mathbf{C}_{\mathrm{D}}^{-1} (\mathbf{G} \mathbf{m}-\mathbf{d}) \, ,
```
where ``\mathbf{m}`` are the model parameters (the unknowns) and ``\mathbf{d}`` the vector of observed (or measured) data, i.e., in this example, the mean of the Gaussian PDF.

First we load some needed packages
```@example gauss1
using LinearAlgebra
using InverseAlgos.Samplers  # sampling algos
using CairoMakie  # for plotting
```
Then the problem to solve is defined by using a custom type (`struct`) named `GaussProb`:
```@example gauss1
struct GaussProb
    mean::Vector
    invcov::Matrix

    function GaussProb(mean,invcov)
        @assert isposdef(invcov)
        @assert issymmetric(invcov)
        return new(mean,invcov)
    end
end
```
The above defines a structure (type) which contains two fields: `mean`, representing the mean of the Gaussian pdf and `invcov`, representing the inverse of the covariance matrix (i.e., the precision matrix). These two pieces of information completely specify our problem. 

We now need to provide functions to compute the value of the negative logarithm of probability density function, i.e,  ``U(\mathbf{m})``, and its gradient with respect to the model parameters ``\mathbf{m}``. 
To do so, we make the `struct` callable (sometimes called a "functor") with a function that computes either the negative natural logarithm of the pdf or the gradient of the latter for an input vector `x` depending on the value of the argument `kind`:
```@example gauss1
function (gp::GaussProb)(x,kind) 
   if kind==:nlogpdf 
	   # negative logarithm of the pdf
       dif = x .- gp.mean
       pd = 1/2 * dot(dif, (gp.invcov * dif))
       return pd
   elseif kind==:gradnlogpdf 
	   # gradient of neg. log. of the pdf
       gra = gp.invcov * x
       return gra
   else
       error("Wrong argument 'kind'. It can be either ':nlogpdf' or ':gradnlogpdf'.")
   end
end 
```
An important thing here is that in order to use the above function with [`Samplers`](@ref) the signature of the function must be `(x::Vector{<:Real},kind::Symbol)` where `x` is a vector and kind is a `Symbol`. Moreover the *only* possible values for `kind` are either `:nlogpdf` which evaluates the negative logarithm of the likelihood functional or `:gradnlogpdf` which computes its gradient. 

Computing the value of the negative logarithm of the pdf and its gradient it is all that is needed for setting up a HMC inversion (except for additional tuning parameters). Computing the negative logarithm of the pdf essentially corresponds to evaluating the misfit functional.

We now define the values for the mean (representing the observed data ``\mathbf{d}`` and inverse covariance ``\mathbf{C}_{\mathrm{D}}^{-1}`` and instantiate our likelihood model:
```@example gauss1
mea = [0.2,-0.3] # then mean value
invcov = inv([1.5 0.2; 0.2 2.3]) # the inverse covariance matrix
likelihood = GaussProb(mea,invcov) # instantiate a GaussProb struct
nothing # hide
```
	
Next we set up the HMC simulation (sampling) by first defining the starting model and the parameters needed for HMC. We define the starting model as
```@example gauss1
mstart = [5.0,-3.0]
nothing # hide
```
and the precision matrix as a diagonal one
```@example gauss1
imM = Diagonal(1.0*I,length(mstart))
LchoM = cholesky(inv(imM)).L
@show imM
@show LchoM
nothing # hide
```
Now we need to set the parameters relative to the chosen HMC algorithm, in this case the plain version of HMC. Moreover, we need to define whether the model parameters will be constrained or not (`false` in our case), the values for `ϵ` (type \epsilon-TAB in Julia) and `L`, i.e., the step length and number of iterations for the leap-frog integration
```@example gauss1
constrained = false
ϵ = 0.25 
L = 10
nothing # hide
```
We thus set all the needed parameters for a "standard" HMC simulation, therefore we can instantiate the [`HMCParams`](@ref Samplers.HMCParams) structure
```@example gauss1
hmcpars = HMCParams(invmassM=imM,
                    LcholmassM=LchoM,
                    constrained=constrained,
                    ϵ=ϵ,
                    L=L)
nothing # hide
```
The type of the above structure also defines the algorithm that will be used for sampling, i.e., [`HMCParams`](@ref Samplers.HMCParams) implies the use of the classic HMC algorithm, while, e.g., [`NUTSParams`](@ref Samplers.NUTSParams) implies the use of the NUTS algorithm.

Finally, we can set in a dictionary the general parameters about the simulation, i.e., the simulation name, the maximum number of iterations, the output directory, etc..
```@example gauss1
pars = Dict()
pars["simname"] = "gauss1" # simulation name
pars["maxiter"] = 5000     # maximum number of iterations
pars["outdir"] = "results" # output directory
pars["saveevery"] = 5      # save every 2 models to disk, e.g., thinning the chain
pars["infoevery"] = 500    # print info every 10 iterations
pars["stdout"] = "oneline" # hide
nothing # hide
```

We can now start the simulation using the generic function [`runMC`](@ref Samplers.runMC):
```@example gauss1
pars["stdout"] = "oneline" # hide
runMC(pars,hmcpars,mstart,likelihood=likelihood)
```

We can load the full results of the simulation (using [`readMCoutput`](@ref Samplers.readMCoutput)), which will store the outputs in a structure of type [`Samplers.HMCOutput`](@ref Samplers.readMCoutput). See the documentation of [`Samplers.HMCOutput`](@ref Samplers.readMCoutput) for an exaplanation of the various fields.
```@example gauss1
outhmc = readMCoutput("results","gauss1")
```

Now we can plot some quantities of interest. As a very simple example, here we plot the values of the samples ``\mathbf{m}`` as ``(x,y)`` coordinates and the potential energy (misfit) as a function of iterations. 

```@example gauss1

userprob = outhmc.userprob

mods = outhmc.mods
iter = outhmc.iter
ucur = outhmc.ucur

fig = Figure()

ax1 = Axis(fig[1:2,1],xlabel="x",ylabel="y")
scatter!(ax1,mods[1,:],mods[2,:])
scatter!(ax1,mea[1],mea[2],color="red")

ax2 = Axis(fig[3,1],xlabel="iteration",ylabel="Pot. energy U")
lines!(ax2,iter,ucur)

fig
```


### Example 2: Sampling a 2D Gaussian with the NUTS algorithm

Firstly, we repeat the same initial steps than above to set up the problem.
```@example gauss2
using LinearAlgebra
using InverseAlgos.Samplers  # sampling algos
using CairoMakie  # for plotting

struct GaussProb
    mean::Vector
    invcov::Matrix

    function GaussProb(mean,invcov)
        @assert isposdef(invcov)
        @assert issymmetric(invcov)
        return new(mean,invcov)
    end
end

function (gp::GaussProb)(x,kind) 
   if kind==:nlogpdf 
	   # negative logarithm of the pdf
       dif = x .- gp.mean
       pd = 1/2 * dot(dif, (gp.invcov * dif))
       return pd
   elseif kind==:gradnlogpdf 
	   # gradient of neg. log. of the pdf
       gra = gp.invcov * x
       return gra
   else
       error("Wrong argument 'kind'. It can be either ':nlogpdf' or ':gradnlogpdf'.")
   end
end 
```
We define the observed data, i.e., mean value in this case, the inverse of the covariance function, instantiate the `GaussProb` and define the starting model.
```@example gauss2
mea = [0.2,-0.3] # then mean value
invcov = inv([1.5 0.2; 0.2 2.3]) # the inverse covariance matrix
likelihood = GaussProb(mea,invcov) # instantiate a GaussProb struct

mstart = [0.0,-3.0]
nothing # hide
```
This time we intend to use the NUTS algorithm, so we start by defining the required parameters for the struct [`NUTSParams`](@ref Samplers.NUTSParams) as follows
```@example gauss2
imM = Diagonal(1.0*I,length(mstart))  # inverse of the mass matrix
LchoM = cholesky(inv(imM)).L          # lower triangular matrix from Cholesky decomposition of the mass matrix
constrained = false                   # constrained problem?
δ = 0.65                              # target acceptance rate for dual averaging
Δmax = 1000.0                         # maximum change in the Hamiltonian value
niteradapt = 500                      # number of iterations for adaptation  
ϵmax = 0.25                           # maximum value for ϵ (type \epsilon TAB in Julia)
maxtreeheight = 4                     # maximum height of the balanced tree for NUTS
nothing # hide
```
See the documentation for [`NUTSParams`](@ref Samplers.NUTSParams) for more details. Now we instantiate a [`NUTSParams`](@ref Samplers.NUTSParams) structure:
```@example gauss2
hmcpars = NUTSParams(invmassM=imM,
                     LcholmassM=LchoM,
                     constrained=constrained,
                     δ=δ,
                     Δmax=Δmax,
                     niteradapt=niteradapt,
                     ϵmax=ϵmax,
                     maxtreeheight=maxtreeheight )
```
Now set in a dictionary the general parameters about the simulation
```@example gauss2
pars = Dict()
pars["simname"] = "gauss2" # simulation name
pars["maxiter"] = 5000     # maximum number of iterations
pars["outdir"] = "results" # output directory
pars["saveevery"] = 5      # save every 2 models to disk, e.g., thinning the chain
pars["infoevery"] = 500    # print info every 10 iterations
pars["stdout"] = "oneline" # hide
nothing # hide
```
and then run the simulation using NUTS
```@example gauss2
pars["stdout"] = "oneline" # hide
runMC(pars,hmcpars,mstart,likelihood=likelihood)
```
Load the full results using [`readMCoutput`](@ref Samplers.readMCoutput)
```@example gauss2
outhmc = readMCoutput("results","gauss2")
```
and plot some results
```@example gauss2
userprob = outhmc.userprob

mods = outhmc.mods
iter = outhmc.iter
ucur = outhmc.ucur

fig = Figure()

ax1 = Axis(fig[1:2,1],xlabel="x",ylabel="y")
scatter!(ax1,mods[1,:],mods[2,:])
scatter!(ax1,mea[1],mea[2],color="red")

ax2 = Axis(fig[3,1],xlabel="iteration",ylabel="Pot. energy U")
lines!(ax2,iter,ucur)

fig
```


### Example 3: Sampling a 2D Gaussian with the Extended Metropolis algorithm

In this example we aim at sampling a 2D Gaussian using also a prior model with the Extended Metropolis (E-M) algorithm.
First we load the needed packages
```@example gauss3
using LinearAlgebra
using InverseAlgos.Samplers  # sampling algos
using CairoMakie  # for plotting
```
We then setup the problem by creating two structures, one for the prior `GaussPrior` and one for the likelihood `GaussLikelihood`.
The algorithm requires a function to draw samples from the prior, since it is used to create the new proposed state. We implement that in the struct `GaussPrior` as shown below
```@example gauss3
struct GaussPrior
    mea::Vector
    Lchol::AbstractMatrix

    function GaussPrior(mea,invcov)
        @assert isposdef(invcov)
        @assert issymmetric(invcov)
        return new(mea,invcov)
    end
end

function (gp::GaussPrior)(x,kind)
   if kind==:drawsample
       # draw sample
       zdev = randn(size(vec(x)))
       sa = gp.Lchol * zdev + gp.mea
       return sa
   else
       error("Wrong argument 'kind'")
   end
end
```
Next we define the likelihood with the struct `GaussLikelihood`. In this case we only need a function to compute the value of the negative-logarithm of the likelihood PDF. No derivatives are required with this algorithm.
```@example gauss3
struct GaussLikelihood
    mean::Vector
    invcov::Matrix

    function GaussLikelihood(mean,invcov)
        @assert isposdef(invcov)
        @assert issymmetric(invcov)
        return new(mean,invcov)
    end
end

function (gp::GaussLikelihood)(x,kind)
   if kind==:nlogpdf
	   # negative logarithm of the pdf
       dif = x .- gp.mean
       pd = 1/2 * dot(dif, (gp.invcov * dif))
       return pd
   else
       error("Wrong argument 'kind'")
   end
end
```
The prior and likelihood can now be instantiated once the means and covariance matrices are set.
```@example gauss3
mea1 = [-0.1,-0.5]
Cm = [2.5 0.0; 0.0 3.0]
Lchol = cholesky(Cm).L
prior = GaussPrior(mea1,Lchol) # instantiate a GaussProb struct

mea2 = [0.2,-0.3] # then mean value
invcov = inv([1.5 0.2; 0.2 2.3]) # the inverse covariance matrix
likelihood = GaussLikelihood(mea2,invcov) # instantiate a GaussProb struct
```
The starting model is set to
```@example gauss3
mstart = [5.0,-3.0]
```
Now, the structure [`ExtMetropParams`](@ref Samplers.ExtMetropParams) for the E-M algorithm need to be instantiated. In this case, such structure requires no parameters to be specified.
```@example gauss3
mcpars = ExtMetropParams()
```
Finally, the generic parameters for the simulation are specified
```@example gauss3
pars = Dict()
pars["simname"] = "metrop1" # simulation name
pars["maxiter"] = 5000      # maximum number of iterations
pars["outdir"] = "results"  # output directory
pars["saveevery"] = 5       # save every 2 models to disk, e.g., thinning the chain
pars["infoevery"] = 500     # print info every 10 iterations
nothing # hide
```

At this point, we can run the simulation
```@example gauss3
pars["stdout"] = "oneline" # hide
runMC(pars,mcpars,mstart,likelihood=likelihood,prior=prior)
```
then collect the output
```@example gauss3
outmc = readMCoutput("results","metrop1")
```
And finally plot some results
```@example gauss3
userprob = outmc.userprob

mods = outmc.mods
iter = outmc.iter
nllk = outmc.nllk

fig = Figure()

ax1 = Axis(fig[1:2,1],xlabel="x",ylabel="y")
scatter!(ax1,mods[1,:],mods[2,:])

ax2 = Axis(fig[3,1],xlabel="iteration",ylabel="Pot. energy U")
lines!(ax2,iter,nllk)

fig
```


[^1]: Tarantola, A., & Valette, B. (1982). Inverse problems = quest for information. J. Geophys., 50 , 159–170.

[^2]: Mosegaard, K., & Tarantola, A. (1995). Monte Carlo sampling of solutions to inverse problems. Journal of Geophysical Research: Solid Earth, 100 (B7),12431–12447. doi: 10.1029/94JB03097

[^3]: Tarantola, A. (2005). Inverse Problem Theory and Methods for Model Parameter Estimation. Society for Industrial and Applied Mathematics. doi: 10.1137/1.9780898717921

[^4]: Neal, R. (2012). MCMC using Hamiltonian dynamics. Handbook of Markov Chain Monte Carlo. doi: 10.1201/b10905-6

[^5]: Fichtner, A., Zunino, A., & Gebraad, L. (2019). Hamiltonian Monte Carlo solution of tomographic inverse problems. Geophysical Journal International , 216 (2), 1344–1363. doi: 10.1093/gji/ggy496

[^6]: Simo, J. C., Tarnow, N., & Wong, K. K. (1992). Exact energy-momentum conserving algorithms and symplectic schemes for nonlinear dynamics. Computer Methods in Applied Mechanics and Engineering, 100 (1), 63–116. doi: 10.1016/0045-7825(92)90115-Z

[^7]: Fichtner, A., Zunino, A., Gebraad, L., & Boehm, C. (2021). Autotuning Hamiltonian Monte Carlo for efficient generalized nullspace exploration. Geophysical Journal International , 227 (2), 941–968. doi: 10.1093/gji/ggab270

[^8]: Hoffman, M. D., & Gelman, A. (2014). The No-U-turn sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. The Journal of Machine Learning Research, 15 (1), 1593–1623.

[^9]: Mosegaard, K., & Tarantola, A. (1995).Monte Carlo sampling of solutions to inverse problems. Journal of Geophysical Research: Solid Earth, 100 (B7), 12431–12447. doi: 10.1029/94JB03097

## Public API

```@docs
InverseAlgos.Samplers
Samplers.runMC
Samplers.contrunMC
Samplers.HMCParams
Samplers.NUTSParams
Samplers.ExtMetropParams
Samplers.clear_stflags_hdf5
Samplers.readMCoutput
Samplers.AvEpsFromHMCIter
Samplers.AvEpsFromTree
```

## Reference, functions not exported
```@autodocs
Modules = [InverseAlgos.Samplers]
Public = false
Private = true
Order = [:type,:function]
```

