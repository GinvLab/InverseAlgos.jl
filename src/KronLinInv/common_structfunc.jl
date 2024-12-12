

##==========================================================
@doc raw"""
    FwdOps

A structure containing the three forward model matrices  G1, G2, G3, where 
`` \mathbf{G} =  \mathbf{G_1} \otimes \mathbf{G_2} \otimes \mathbf{G_3} ``
"""
struct FwdOps
    G1::Array{Float64,2}
    G2::Array{Float64,2}
    G3::Array{Float64,2} 
end

##------------------------
@doc raw"""
    CovMats

A structure containing the six covariance matrices `Cm1, Cm2, Cm3, Cd1, Cd2, Cd3`,
where
`` \mathbf{C}_{\rm{M}} = \mathbf{C}_{\rm{M}}^{\rm{x}} \otimes 
\mathbf{C}_{\rm{M}}^{\rm{y}} \otimes \mathbf{C}_{\rm{M}}^{\rm{z}} ``  and
``\quad \mathbf{C}_{\rm{D}} = \mathbf{C}_{\rm{D}}^{\rm{x}} \otimes 
\mathbf{C}_{\rm{D}}^{\rm{y}} \otimes \mathbf{C}_{\rm{D}}^{\rm{z}} ``
"""
struct CovMats
    Cd1::Array{Float64,2}
    Cd2::Array{Float64,2}
    Cd3::Array{Float64,2} 
    Cm1::Array{Float64,2} 
    Cm2::Array{Float64,2} 
    Cm3::Array{Float64,2} 
end

##------------------------
@doc raw"""
    KLIFactors

A structure containing all the factors necessary to perform further calculations with KronLinInv,
as, for instance, computations of the posterior mean model or the posterior covariance matrix.
The structure includes:

* `U1, U2, U3`:  `` \mathbf{U}_1 ``, `` \mathbf{U}_2 ``, `` \mathbf{U}_3  ``  of  `` F_{\sf{A}} `` 
* `invlambda`:  ``\big( \mathbf{I} + \mathbf{\Lambda}_1\!  \otimes \! \mathbf{\Lambda}_2 \!  \otimes\!  \mathbf{\Lambda}_3 \big)^{-1}``  of ``F_{\sf{B}} ``
* `iUCm1, iUCm2, iUCm3`:  ``\mathbf{U}_1^{-1} \mathbf{C}_{\rm{M}}^{\rm{x}}``,
  ``\mathbf{U}_2^{-1}  \mathbf{C}_{\rm{M}}^{\rm{y}}``,
  ``\mathbf{U}_2^{-1}  \mathbf{C}_{\rm{M}}^{\rm{z}}`` of  `` F_{\sf{C}} `` 
* `iUCmGtiCd1, iUCmGtiCd1, iUCmGtiCd1`:  
  ``\mathbf{U}_1^{-1} \mathbf{C}_{\rm{M}}^{\rm{x}}
  (\mathbf{G}^{\rm{x}})^{\sf{T}}(\mathbf{C}_{\rm{D}}^{\rm{x}})^{-1}``,
  `` \mathbf{U}_2^{-1} \mathbf{C}_{\rm{M}}^{\rm{y}}
  (\mathbf{G}^{\rm{y}})^{\sf{T}} (\mathbf{C}_{\rm{D}}^{\rm{y}})^{-1}``,
  `` \mathbf{U}_3^{-1} \mathbf{C}_{\rm{M}}^{\rm{z}}
  (\mathbf{G}^{\rm{z}})^{\sf{T}} (\mathbf{C}_{\rm{D}}^{\rm{z}})^{-1}``
  of  ``F_{\sf{D}}``
"""
struct KLIFactors
    U1::Array{Float64,2}
    U2::Array{Float64,2}
    U3::Array{Float64,2}
    invlambda::Array{Float64,1}
    iUCm1::Array{Float64,2}
    iUCm2::Array{Float64,2}
    iUCm3::Array{Float64,2}
    iUCmGtiCd1::Array{Float64,2}
    iUCmGtiCd2::Array{Float64,2}
    iUCmGtiCd3::Array{Float64,2}
end


##==========================================================

@doc raw"""
     kli_calcfactors(Gfwd::FwdOps,Covs::CovMats)

Computes the factors necessary to solve the inverse problem. 
  
The factors are the ones to be stored to subsequently calculate posterior
  mean and covariance. First an eigen decomposition is performed, to get
 
```math
   \mathbf{U}_1 \mathbf{\Lambda}_1  \mathbf{U}_1^{-1}  
   = \mathbf{C}_{\rm{M}}^{\rm{x}} (\mathbf{G}^{\rm{x}})^{\sf{T}}
  (\mathbf{C}_{\rm{D}}^{\rm{x}})^{-1} \mathbf{G}^{\rm{x}}
```
 
```math
 \mathbf{U}_2 \mathbf{\Lambda}_2  \mathbf{U}_2^{-1}
  =  \mathbf{C}_{\rm{M}}^{\rm{y}} (\mathbf{G}^{\rm{y}})^{\sf{T}}
  (\mathbf{C}_{\rm{D}}^{\rm{y}})^{-1} \mathbf{G}^{\rm{y}}
```
 
```math
  \mathbf{U}_3 \mathbf{\Lambda}_3  \mathbf{U}_3^{-1}
  = \mathbf{C}_{\rm{M}}^{\rm{z}} (\mathbf{G}^{\rm{z}})^{\sf{T}}
  (\mathbf{C}_{\rm{D}}^{\rm{z}})^{-1} \mathbf{G}^{\rm{z}} 
```
 
  The principal factors involved in the computation of the posterior covariance and
  mean are:
 
```math
  F_{\sf{A}} =  \mathbf{U}_1 \otimes \mathbf{U}_2 \otimes \mathbf{U}_3 
```
 
```math
  F_{\sf{B}} = \big( 
  \mathbf{I} + \mathbf{\Lambda}_1 \! \otimes \! \mathbf{\Lambda}_2 \!
  \otimes \! \mathbf{\Lambda}_3 
  \big)^{-1} 
```
 
```math
  F_{\sf{C}} =
  \mathbf{U}_1^{-1}  \mathbf{C}_{\rm{M}}^{\rm{x}} \otimes 
  \mathbf{U}_2^{-1} \mathbf{C}_{\rm{M}}^{\rm{y}} \otimes 
  \mathbf{U}_3^{-1} \mathbf{C}_{\rm{M}}^{\rm{z}} 
```
 
```math
  F_{\sf{D}} =   
  \left( \mathbf{U}_1^{-1}  \mathbf{C}_{\rm{M}}^{\rm{x}} (\mathbf{G}^{\rm{x}})^{\sf{T}}
  (\mathbf{C}_{\rm{D}}^{\rm{x}})^{-1} \right) \!    \otimes 
  \left( \mathbf{U}_2^{-1} \mathbf{C}_{\rm{M}}^{\rm{y}}  (\mathbf{G}^{\rm{y}})^{\sf{T}}
  (\mathbf{C}_{\rm{D}}^{\rm{y}})^{-1}  \right)   \!   
  \otimes  \left( \mathbf{U}_3^{-1} \mathbf{C}_{\rm{M}}^{\rm{z}}
  (\mathbf{G}^{\rm{z}})^{\sf{T}} (\mathbf{C}_{\rm{D}}^{\rm{z}})^{-1} \right)
```

Uses LAPACK.sygvd!(), see <http://www.netlib.org/lapack/lug/node54.html>.
  Reduces a real symmetric-definite generalized eigenvalue problem to
  the standard form. \n
  ``B A z = \lambda z`` 	B = LLT 	C = LT A L 	z = L y
  - A is symmetric
  - B is symmetric, positive definite
  
  
# Arguments
- `Gfwd`: a [`FwdOps`](@ref) structure containing the three forward model matrices  G1, G2 and G3, where 
     `` \mathbf{G} =  \mathbf{G_1} \otimes \mathbf{G_2} \otimes \mathbf{G_3} ``
- `Covs`: a [`CovMats`](@ref) structure containing the six covariance matrices ``\mathbf{C}_{\rm{M}} = \mathbf{C}_{\rm{M}}^{\rm{x}} \otimes \mathbf{C}_{\rm{M}}^{\rm{y}} \otimes \mathbf{C}_{\rm{M}}^{\rm{z}}`` and ``\mathbf{C}_{\rm{D}} = \mathbf{C}_{\rm{D}}^{\rm{x}} \otimes \mathbf{C}_{\rm{D}}^{\rm{y}} \otimes \mathbf{C}_{\rm{D}}^{\rm{z}} ``

# Returns
- A [`KLIFactors`](@ref) structure containing all the "factors" necessary to perform further calculations with KronLinInv,
    as, for instance, computations of the posterior mean model or the posterior covariance matrix. 
    The structure includes:

    * `U1, U2, U3`:  `` \mathbf{U}_1 ``, `` \mathbf{U}_2 ``,
      `` \mathbf{U}_3  ``  of  `` F_{\sf{A}} `` 
    * `invlambda`:  `` F_{\sf{B}} ``
    * `iUCm1, iUCm2, iUCm3`:  ``\mathbf{U}_1^{-1} \mathbf{C}_{\rm{M}}^{\rm{x}}``,
       ``\mathbf{U}_2^{-1}  \mathbf{C}_{\rm{M}}^{\rm{y}}``,
        ``\mathbf{U}_2^{-1}  \mathbf{C}_{\rm{M}}^{\rm{z}}`` of  `` F_{\sf{C}} `` 
    * `iUCmGtiCd1, iUCmGtiCd1, iUCmGtiCd1`:  `` \mathbf{U}_1^{-1}
        \mathbf{C}_{\rm{M}}^{\rm{x}}
        (\mathbf{G}^{\rm{x}})^{\sf{T}}(\mathbf{C}_{\rm{D}}^{\rm{x}})^{-1}  ``,
        `` \mathbf{U}_2^{-1} \mathbf{C}_{\rm{M}}^{\rm{y}}
        (\mathbf{G}^{\rm{y}})^{\sf{T}} (\mathbf{C}_{\rm{D}}^{\rm{y}})^{-1}``,
        `` \mathbf{U}_3^{-1} \mathbf{C}_{\rm{M}}^{\rm{z}}
        (\mathbf{G}^{\rm{z}})^{\sf{T}} (\mathbf{C}_{\rm{D}}^{\rm{z}})^{-1} ``
        of  `` F_{\sf{D}} `` 

"""
function kli_calcfactors(Gfwd::FwdOps,Covs::CovMats)

    
    ##----------------
    ## Check positive definiteness of covariance matrices
    fiecou = fieldcount(CovMats)
    for i=1:fiecou
        C = getfield(Covs,i)
        if isposdef( C )==false
            fnam = string(fieldname(CovMats,i))
            error("\n kli_calcfactors(): $(fnam) is not positive definite. Aborting. \n")
        end
    end
    
    ##----------------
    Cm1,Cm2,Cm3 = Covs.Cm1,Covs.Cm2,Covs.Cm3
    Cd1,Cd2,Cd3 = Covs.Cd1,Covs.Cd2,Covs.Cd3
    G1,G2,G3 = Gfwd.G1,Gfwd.G2,Gfwd.G3
    
    ##----------------
    iCdG1 = Cd1 \ G1
    iCdG2 = Cd2 \ G2
    iCdG3 = Cd3 \ G3
    GtiCd1 = transpose(G1) * iCdG1
    GtiCd2 = transpose(G2) * iCdG2
    GtiCd3 = transpose(G3) * iCdG3
    GtiCdG1 = transpose(G1) * iCdG1 
    GtiCdG2 = transpose(G2) * iCdG2 
    GtiCdG3 = transpose(G3) * iCdG3 

    
    ## If itype = 3, the problem to solve is B * A * x = lambda * x.
    itype = 3 ## the problem to solve is B * A * x = lambda * x.
    uplo = 'L'
    jobz = 'V'
    
    ## Finds the generalized eigenvalues (jobz = N) or eigenvalues and eigenvectors (jobz = V)
    ## of a symmetric matrix A and symmetric positive-definite matrix B. If uplo = U, the upper
    ## triangles of A and B are used. If uplo = L, the lower triangles of A and B are used. If
    ## itype = 1, the problem to solve is A * x = lambda * B * x. If itype = 2, the problem to
    ## solve is A * B * x = lambda * x. If itype = 3, the problem to solve is
    ## B * A * x = lambda * x.
    ## sygvd!(itype::Integer,jobz::Char,uplo::Char,
    ##        A::StridedMatrix{$elty},B::StridedMatrix{$elty})
    ## print 'Calculating fa' etc

    lambda1,U1,LcholB1 = LAPACK.sygvd!(itype,jobz,uplo,GtiCdG1,copy(Cm1))
    lambda2,U2,LcholB2 = LAPACK.sygvd!(itype,jobz,uplo,GtiCdG2,copy(Cm2))
    lambda3,U3,LcholB3 = LAPACK.sygvd!(itype,jobz,uplo,GtiCdG3,copy(Cm3))

    
    ## println('Calculating fb')
    argfb = 1.0 .+ krondiag(lambda1,krondiag(lambda2,lambda3))
    vecdfac = 1.0./argfb
    
    ## println('Calculating fc')
    iUCm1 = U1 \ Cm1 
    iUCm2 = U2 \ Cm2
    iUCm3 = U3 \ Cm3

    ## println('Calculating fd')
    fd11 = Cd1 \ Matrix{Float64}(I,size(Cd1,1),size(Cd1,2)) #eye(Cd1)
    fd22 = Cd2 \ Matrix{Float64}(I,size(Cd2,1),size(Cd2,2)) ##eye(Cd2) 
    fd33 = Cd3 \ Matrix{Float64}(I,size(Cd3,1),size(Cd3,2)) ##eye(Cd3)

    iUCmGtiCd1 = iUCm1 * transpose(G1) * fd11 
    iUCmGtiCd2 = iUCm2 * transpose(G2) * fd22
    iUCmGtiCd3 = iUCm3 * transpose(G3) * fd33

    factors = KLIFactors(U1,U2,U3, vecdfac, iUCm1,iUCm2,iUCm3, iUCmGtiCd1,iUCmGtiCd2,iUCmGtiCd3)
    return  factors
end



@doc raw"""
    kli_posteriormean(klifac::KLIFactors,Gfwd::FwdOps,mprior::Array{Float64,1},
                      dobs::Array{Float64,1}; parall::Symbol=:serial)

Computes the posterior mean model for a distributed memory setup.

# Arguments
- `klifac`: a structure containing the required "factors" previously computed with 
    the function `kli_calcfactors()`. It includes
    * `U1, U2, U3` ``\mathbf{U}_1``, ``\mathbf{U}_2``, ``\mathbf{U}_3``  of  ``F_{\sf{A}}``
    * `diaginvlambda` ``F_{\sf{B}}``
    * `iUCmGtiCd1, iUCmGtiCd2, iUCmGtiCd3` ``\mathbf{U}_1^{-1}
      \mathbf{C}_{\rm{M}}^{\rm{x}}
      (\mathbf{G}^{\rm{x}})^{\sf{T}}(\mathbf{C}_{\rm{D}}^{\rm{x}})^{-1} ``,
      `` \mathbf{U}_2^{-1} \mathbf{C}_{\rm{M}}^{\rm{y}}
      (\mathbf{G}^{\rm{y}})^{\sf{T}} (\mathbf{C}_{\rm{D}}^{\rm{y}})^{-1} ``,
      `` \mathbf{U}_3^{-1} \mathbf{C}_{\rm{M}}^{\rm{z}}
      (\mathbf{G}^{\rm{z}})^{\sf{T}} (\mathbf{C}_{\rm{D}}^{\rm{z}})^{-1} ``
      of  ``F_{\sf{D}} ``
- `FwdOps`: a structure containing the three forward matrices
    * `G1, G2, G3` `` \mathbf{G} = \mathbf{G_1} \otimes \mathbf{G_2} \otimes \mathbf{G_3} ``
- `mprior`: prior model (vector)
- `dobs`:  observed data (vector)
- `parall`: type of parallelization: :serial, ;sharedmem, :distrmem

# Returns
- The posterior mean model (vector)
"""
function kli_posteriormean(klifac::KLIFactors,
                           Gfwd::FwdOps,
                           mprior::Array{Float64,1},
                           dobs::Array{Float64,1} ;
                           parall::Symbol=:serial)

    if parall==:serial
        return posteriormean_serial(klifac,Gfwd,mprior,dobs)
    elseif parall==:threads
        println("Parallelization in shared memory is work in progress")
        return
    elseif parall==:distribmem
        return posteriormean_distrmem(klifac,Gfwd,mprior,dobs)
    else
        error("Wrong `parall` argument.")
    end

    return
end



@doc raw""" 
    kli_blockpostcov(klifac::KLIFactors,astart::Int64,aend::Int64,
                 bstart::Int64,bend::Int64; parall::Symbol=:serial)

Computes a block of the posterior covariance, purely serial version. 

# Arguments
- `klifac`: a structure containing the required "factors" previously computed with 
    the function `kli_calcfactors()`. It includes
    * U1,U2,U3 `` \mathbf{U}_1``, ``\mathbf{U}_2``, ``\mathbf{U}_3`` of ``F_{\sf{A}}``
    * diaginvlambda ``F_{\sf{B}}``
    * iUCm1, iUCm2, iUCm3  ``\mathbf{U}_1^{-1} \mathbf{C}_{\rm{M}}^{\rm{x}} ``,
      ``\mathbf{U}_2^{-1}  \mathbf{C}_{\rm{M}}^{\rm{y}}``,
      ``\mathbf{U}_2^{-1}  \mathbf{C}_{\rm{M}}^{\rm{z}}`` of  ``F_{\sf{C}} `` 
- `Gfwd`: a structure containing the three forward model matrices  G1,G2,G3, where 
     `` \mathbf{G} =  \mathbf{G_1} \otimes \mathbf{G_2} \otimes \mathbf{G_3} ``
- `astart, aend`: indices of the first and last rowa of the requested block
- `bstart, bend`: indices of the first and last columns of the requested block
- `parall`: type of parallelization: :serial, ;sharedmem, :distrmem

# Returns
- The requested block of the posterior covariance.

"""
function kli_blockposteriorcov(klifac::KLIFactors,
                               astart::Int64,aend::Int64,
                               bstart::Int64,bend::Int64 ;
                               parall::Symbol=:serial)

    if parall==:serial
        return blockpostcov_serial(klifac,astart,aend,bstart,bend)

    elseif parall==:sharedmem
        println("Parallelization in shared memory is work in progress")
        return
    elseif parall==:distribmem
        return blockpostcov_distrmem(klifac,astart,aend,bstart,bend)
    else
        error("Wrong `parall` argument.")
    end

    return
end
