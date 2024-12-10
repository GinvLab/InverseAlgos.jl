


##======================================================================

# @doc raw"""
#     posteriormean_distmem(klifac::KLIFactors,Gfwd::FwdOps,mprior::Array{Float64,1},
#                          dobs::Array{Float64,1})

# Computes the posterior mean model for a distributed memory setup.

# # Arguments
# - `klifac`: a structure containing the required "factors" previously computed with 
#     the function `calcfactors()`. It includes
#     * `U1, U2, U3` ``\mathbf{U}_1``, ``\mathbf{U}_2``, ``\mathbf{U}_3``  of  ``F_{\sf{A}}``
#     * `diaginvlambda` ``F_{\sf{B}}``
#     * `iUCmGtiCd1, iUCmGtiCd2, iUCmGtiCd3` ``\mathbf{U}_1^{-1}
#       \mathbf{C}_{\rm{M}}^{\rm{x}}
#       (\mathbf{G}^{\rm{x}})^{\sf{T}}(\mathbf{C}_{\rm{D}}^{\rm{x}})^{-1} ``,
#       `` \mathbf{U}_2^{-1} \mathbf{C}_{\rm{M}}^{\rm{y}}
#       (\mathbf{G}^{\rm{y}})^{\sf{T}} (\mathbf{C}_{\rm{D}}^{\rm{y}})^{-1} ``,
#       `` \mathbf{U}_3^{-1} \mathbf{C}_{\rm{M}}^{\rm{z}}
#       (\mathbf{G}^{\rm{z}})^{\sf{T}} (\mathbf{C}_{\rm{D}}^{\rm{z}})^{-1} ``
#       of  ``F_{\sf{D}} ``
# - `FwdOps`: a structure containing the three forward matrices
#     * `G1, G2, G3` `` \mathbf{G} = \mathbf{G_1} \otimes \mathbf{G_2} \otimes \mathbf{G_3} ``
# - `mprior`: prior model (vector)
# - `dobs`:  observed data (vector)

# # Returns
# - The posterior mean model (vector)
# """
function posteriormean_distrmem(klifac::KLIFactors,Gfwd::FwdOps,mprior::Array{Float64,1},
                                dobs::Array{Float64,1})

    ####================================================
    ##     Parallel version
    ####================================================
    ## get the ids of cores
    idcpus = workers()
    firstwork = idcpus[1]
    numwork = nworkers()
    println("posteriormean(): Parallel run using $numwork workers")

    
    ##--------------
    U1,U2,U3 = klifac.U1,klifac.U2,klifac.U3
    diaginvlambda = klifac.invlambda
    Z1,Z2,Z3 = klifac.iUCmGtiCd1,klifac.iUCmGtiCd2,klifac.iUCmGtiCd3
    G1,G2,G3 = Gfwd.G1,Gfwd.G2,Gfwd.G3

    ## sizes
    Ni = size(Z1,1)
    Nl = size(Z1,2)
    Nj = size(Z2,1)
    Nm = size(Z2,2)
    Nk = size(Z3,1)
    Nn = size(Z3,2)

    ## sizes
    # Nr12 = size(U1,2)*size(U2,2)*size(U3,2)
    # Nc1  = size(Z1,1)*size(Z2,1)*size(Z3,1)
    Na = size(mprior,1)
    Nb = size(dobs,1)

    ##-------------
    av = collect(1:Na)
    bv = collect(1:Nb)
    ## vectors containing all possible indices for 
    ##    row calculations of Kron prod AxBxC
    iv = div.( (av.-1), (Nk.*Nj) ) .+1 
    jv = div.( (av.-1 .-(iv.-1).*Nk.*Nj), Nk ) .+1 
    kv = av.-(jv.-1).*Nk.-(iv.-1).*Nk.*Nj 
    ## vectors containing all possible indices for
    ##    column calculations of Kron prod AxBxC
    lv =  div.( (bv .-1), (Nn.*Nm) ) .+ 1 
    mv =  div.( (bv.-1 .-(lv.-1).*Nn.*Nm), Nn ) .+ 1 
    nv =  bv.-(mv.-1).*Nn-(lv.-1).*Nn.*Nm 
    ##  Gs have different shape than Us !!
    
    ####===================================
 
    ## create the channel for tracking progress
    chanit = RemoteChannel(()->Channel{Tuple{Float64,Float64,Bool}}(1))

    ##################################################
    #             loop 1                             #
    ##################################################
    ## Nb
    scheduling,looping = spreadwork(Nb,numwork,1) ## Nb !!
    everynit = looping[1,2]>100 ? div(looping[1,2],100) : 2
    # init channel
    put!(chanit,(0.0,Inf,false))
    
    ddiff = Array{Float64}(undef,Nb)
    @sync begin
        for ip=1:numwork 
            bstart,bend = looping[ip,1],looping[ip,2]
            # ## distribute work to specific cores
            @async ddiff[bstart:bend] = remotecall_fetch(comp_ddiff,idcpus[ip],
                                                         everynit,firstwork,
                                                         iv,lv,jv,mv,kv,nv,
                                                         G1,G2,G3,mprior,dobs,
                                                         bstart,bend,chanit)
        end

        jobdone = false
        while !jobdone
            wait(chanit)
            perc,reta,jobdone = take!(chanit)
            print("posteriormean(): loop 1/3, $(perc)%; ETA: $reta min     \r") 
        end
    end 

    ##################################################
    #             loop 2                             #
    ##################################################
    ### need to re-loop because full Zh is needed
    # init channel
    put!(chanit,(0.0,Inf,false))

    ## Na
    scheduling,looping = spreadwork(Na,numwork,1) ## Na!!
    
    Zh = Array{Float64}(undef,Na)
    @sync begin
        for ip=1:numwork 
            astart,aend = looping[ip,1],looping[ip,2]
            ## distribute work to specific cores 
            @async Zh[astart:aend] = remotecall_fetch(comp_Zh,idcpus[ip],
                                                      everynit,firstwork,
                                                      iv,lv,jv,mv,kv,nv,
                                                      Z1,Z2,Z3,ddiff,
                                                      astart,aend,chanit)
        end

        jobdone = false
        while !jobdone
            wait(chanit)
            perc,reta,jobdone = take!(chanit)
            print("posteriormean(): loop 2/3, $(perc) %; ETA: $reta min     \r")
        end
    end

    ##################################################
    #             loop 3                             #
    ##################################################
    ## Na
    scheduling,looping = spreadwork(Na,numwork,1) ## Na!!
    # init channel
    put!(chanit,(0.0,Inf,false))
    
    postm = Array{Float64}(undef,Na)
    @sync begin
        for ip=1:numwork 
            astart,aend = looping[ip,1],looping[ip,2]
            ## distribute work to specific cores 
            @async postm[astart:aend] = remotecall_fetch(comp_postm,idcpus[ip],
                                                         everynit,firstwork,
                                                         iv,lv,jv,mv,kv,nv,
                                                         U1,U2,U3,
                                                         diaginvlambda,Zh,
                                                         mprior,astart,aend,chanit)
        end

        jobdone = false
        while !jobdone
            wait(chanit)
            perc,reta,jobdone = take!(chanit)
            print("posteriormean(): loop 3/3, $(perc) %; ETA: $reta min     \r")
        end
    end
    println()
    
    return postm
end

##==========================================================

function comp_ddiff(everynit::Int64 ,firstwork::Int64,
                    iv::Array{Int64,1},lv::Array{Int64,1},jv::Array{Int64,1},
                    mv::Array{Int64,1},kv::Array{Int64,1},nv::Array{Int64,1},
                    G1::Array{Float64,2},G2::Array{Float64,2},G3::Array{Float64,2},
                    mprior::Array{Float64,1},dobs::Array{Float64,1},
                    bstart::Int64,bend::Int64,
                    chanit::RemoteChannel{Channel{Tuple{Float64,Float64,Bool}}})
    ## dobs - dcalc(mprior)
    @assert bend>=bstart
    startt = time()
    myNb = bend-bstart+1
    ddiff = zeros(Float64,myNb)
    Na = length(mprior)
    # loop 
    for b=bstart:bend # b=1:Nb
        myb = b-bstart+1

        # print info
        if myid()==firstwork
            if b<bend && ( (myb%everynit==0) | (myb==5) )
                eta = ( (time()-startt)/float(myb-1) * (myNb-myb+1) ) /60.0
                reta = round(eta,digits=3)
                perc = round(myb/myNb*100.0,digits=3)
                ## put in channel
                put!(chanit,(perc,reta,false))
            elseif b==bend
                # last iteration
                put!(chanit,(100.0,0.0,true))
            end
        end

        ## do computations
        datp = 0.0
        for j=1:Na
            elG = G1[lv[b],iv[j]] * G2[mv[b],jv[j]] * G3[nv[b],kv[j]]
            datp = datp +  mprior[j] * elG
        end        
        ddiff[myb] = dobs[b] - datp
    end
    return ddiff
end

##==========================================================

function comp_Zh(everynit::Int64,firstwork::Int64,
                 iv::Array{Int64,1},lv::Array{Int64,1},jv::Array{Int64,1},
                 mv::Array{Int64,1},kv::Array{Int64,1},nv::Array{Int64,1},
                 Z1::Array{Float64,2},Z2::Array{Float64,2},Z3::Array{Float64,2},
                 ddiff::Array{Float64,1},
                 astart::Int64,aend::Int64,
                 chanit::RemoteChannel{Channel{Tuple{Float64,Float64,Bool}}})
    ## compute Zh
    @assert aend>=astart
    startt = time()
    myNa = aend-astart+1
    Nb = length(ddiff)
    Zh = zeros(Float64,myNa)
    # loop on chunk
    for i=astart:aend #i=1:Na
        mya = i-astart+1

        # print info
        if myid()==firstwork
            if i<aend && ( (mya%everynit==0) | (mya==5) )
                eta = ( (time()-startt)/float(mya-1) * (myNa-mya+1) ) /60.0
                reta = round(eta,digits=3)
                perc = round(mya/myNa*100.0,digits=3)
                ## put in channel
                put!(chanit,(perc,reta,false))
            elseif i==aend
                # last iteration
                put!(chanit,(100.0,0.0,true))
            end
        end

        ## do calculations
        Zh[mya]=0.0
        for j=1:Nb
            tZZ = Z1[iv[i],lv[j]] * Z2[jv[i],mv[j]] * Z3[kv[i],nv[j]]
            Zh[mya] = Zh[mya] + tZZ * ddiff[j]
        end
    end
    return Zh
end        

##==========================================================

function comp_postm(everynit::Int64,firstwork::Int64,
                    iv::Array{Int64,1},lv::Array{Int64,1},jv::Array{Int64,1},
                    mv::Array{Int64,1},kv::Array{Int64,1},nv::Array{Int64,1},
                    U1::Array{Float64,2},U2::Array{Float64,2},U3::Array{Float64,2},
                    diaginvlambda::Array{Float64,1},Zh::Array{Float64,1},
                    mprior::Array{Float64,1},
                    astart::Int64,aend::Int64,
                    chanit::RemoteChannel{Channel{Tuple{Float64,Float64,Bool}}})

    # compute postm
    @assert aend>=astart
    startt = time()
    Na = length(mprior)
    myNa = aend-astart+1
    postm = zeros(Float64,myNa)
    elUDZh = zeros(Float64,myNa)
    # loop on chunk
    ### need to re-loop because full Zh is needed
    for i=astart:aend  #i=1:Na
        mya = i-astart+1

        # print info
        if myid()==firstwork
            if i<aend && ( (mya%everynit==0) | (mya==5) )
                eta = ( (time()-startt)/float(mya-1) * (myNa-mya+1) ) /60.0
                reta = round(eta,digits=3)
                perc = round(mya/myNa*100.0,digits=3)
                ## put in channel
                put!(chanit,(perc,reta,false))
            elseif i==aend
                #  last iteration
                put!(chanit,(100.0,0.0,true))
            end
        end
        ## do calculations
        ## UD times Zh
        elUDZh[mya] = 0.0
        for j=1:Na
            # element of row of UD
            elrowUD = U1[iv[i],iv[j]] * U2[jv[i],jv[j]] *
                U3[kv[i],kv[j]] * diaginvlambda[j]
            # element of final vector
            elUDZh[mya] = elUDZh[mya] + elrowUD * Zh[j]
        end

        ## element of the posterior mean
        postm[mya] = mprior[i] + elUDZh[mya] # sum(bigmatrow.*ddiff)
    end
    return postm
end

##============================================================================

# @doc raw""" 
#     blockpostcov_distrmem(klifac::KLIFactors,astart::Int64,aend::Int64,
#                  bstart::Int64,bend::Int64 )

# Computes a block of the posterior covariance for a distributed memory setup. 

# # Arguments
# - `klifac`: a structure containing the required "factors" previously computed with 
#     the function `calcfactors()`. It includes
#     * U1,U2,U3 `` \mathbf{U}_1``, ``\mathbf{U}_2``, ``\mathbf{U}_3`` of ``F_{\sf{A}}``
#     * diaginvlambda ``F_{\sf{B}}``p
#     * iUCm1, iUCm2, iUCm3  ``\mathbf{U}_1^{-1} \mathbf{C}_{\rm{M}}^{\rm{x}} ``,
#       ``\mathbf{U}_2^{-1}  \mathbf{C}_{\rm{M}}^{\rm{y}}``,
#       ``\mathbf{U}_2^{-1}  \mathbf{C}_{\rm{M}}^{\rm{z}}`` of  ``F_{\sf{C}} `` 
# - `astart, aend`: indices of the first and last rowa of the requested block
# - `bstart, bend`: indices of the first and last columns of the requested block

# # Returns
# - The requested block of the posterior covariance.

# """
function blockpostcov_distrmem(klifac::KLIFactors,
                               astart::Int64,aend::Int64,
                               bstart::Int64,bend::Int64 )

    ####================================================
    ##     Parallel version
    ####================================================
    ## get the id of cores
    idcpus = workers()
    firstwork = idcpus[1]
    numwork = nworkers()
    println("blockpostcov(): Parallel run using $numwork workers")

    ##--------------
    U1,U2,U3 = klifac.U1,klifac.U2,klifac.U3
    diaginvlambda = klifac.invlambda
    iUCm1,iUCm2,iUCm3 = klifac.iUCm1,klifac.iUCm2,klifac.iUCm3

    ##-----------------
    Ni = size(U1,1)
    Nl = size(U1,2)
    Nj = size(U2,1)
    Nm = size(U2,2)
    Nk = size(U3,1)
    Nn = size(U3,2)
    Na = Ni*Nj*Nk
    Nb = Nl*Nm*Nn
    
    ## check limits of requested block
    if astart<1 || aend>Na || astart>aend || bstart<1 || bend>Na || bstart>bend
        error("blockpostcov(): Wrong size of the requested block array.")
    end 

    ##-----------------
    av = collect(1:Na)
    ## vectors containing all possible indices for 
    ##    row calculations of Kron prod AxBxC
    iv =  div.( (av.-1), (Nk.*Nj) ) .+1 
    jv =  div.( (av.-1 .-(iv.-1).*Nk.*Nj), Nk ) .+ 1 
    kv =  av.-(jv.-1).*Nk.-(iv.-1).*Nk.*Nj 

    nci = aend-astart+1
    ncj = bend-bstart+1
    postC = Array{Float64}(undef,nci,ncj)

    ##-----------------
    startt = time()
  
    ## spread work on rows (Na)
    scheduling,looping = spreadwork(nci,numwork,1) ## Nb !!

    ## create the channel for tracking progress
    chanit = RemoteChannel(()->Channel{Tuple{Float64,Float64,Bool}}(1))
    # init channel
    put!(chanit,(0.0,Inf,false))
    
    @sync begin
        for ip=1:numwork 
            astart,aend = looping[ip,1],looping[ip,2]
            @async  postC[astart:aend,:] = remotecall_fetch(comp_rowsblockpostC,idcpus[ip],firstwork,
                                                            U1,U2,U3,diaginvlambda,
                                                            iUCm1,iUCm2,iUCm3,iv,jv,kv,astart,aend,bstart,bend,chanit)
        end

        jobdone = false
        while !jobdone
            wait(chanit)
            perc,reta,jobdone = take!(chanit)
            print("blockpostcov(): $(perc) %; ETA: $reta min     \r")
        end
    end 

    println()
    return postC
end
    
##==========================================================

function comp_rowsblockpostC(firstwork::Int64,U1::Array{Float64,2},U2::Array{Float64,2},U3::Array{Float64,2},
                             diaginvlambda::Array{Float64,1},
                             iUCm1::Array{Float64,2},iUCm2::Array{Float64,2},iUCm3::Array{Float64,2},
                             iv::Array{Int64,1},jv::Array{Int64,1},kv::Array{Int64,1},
                             astart::Int64,aend::Int64,bstart::Int64,bend::Int64,
                             chanit::RemoteChannel{Channel{Tuple{Float64,Float64,Bool}}})
    
    @assert aend>=astart
    @assert bend>=bstart
    ##-----------------
    Ni = size(U1,1)
    Nj = size(U2,1)
    Nk = size(U3,1)
    Na = Ni*Nj*Nk

    startt = time()
    myNa = aend-astart+1
    nci = aend-astart+1
    ncj = bend-bstart+1
    rowspostC =  zeros(Float64,nci,ncj)
    row2  = Array{Float64}(undef,Na)

    for a=astart:aend
        mya = a-astart+1
        
        # print info
        if myid()==firstwork
            if a<aend && ( (mya%100==0) | (mya==5) )
                eta = ( (time()-startt)/float(mya-1) * (myNa-mya+1) ) /60.0
                reta = round(eta,digits=3)
                perc = round(mya/myNa*100.0,digits=3)
                ## put in channel
                put!(chanit,(perc,reta,false))
            elseif a==aend
                # last iteration
                put!(chanit,(100.0,0.0,true))
            end
        end

        ## calculate one row of first two factors
        ## row of  Kron prod AxBxC times a diag matrix (fb)
        ## row2 =  U1(iv(a),lv) * U2(jv(a),mv) * U3(kv(a),nv) * diaginvlambda
        for q=1:Na
            row2[q] = U1[iv[a],iv[q]] * U2[jv[a],jv[q]] * U3[kv[a],kv[q]] * diaginvlambda[q]
        end
        
        for b=bstart:bend
            myb = b-bstart+1
            
            rowspostC[mya,myb] = 0.0
            for p=1:Na
                ## calculate one column of fc
                col1 = iUCm1[iv[p],iv[b]] * iUCm2[jv[p],jv[b]] * iUCm3[kv[p],kv[b]]
                ## calculate one element 
                rowspostC[mya,myb] = rowspostC[mya,myb] + row2[p] * col1
            end
        end
    end
    return rowspostC
end

##===============================================================
"""
    spreadwork(nit::Int64,nunits::Int64,startpoint::Int64)

Distribute work among workers. 
"""
function spreadwork(nit::Int64,nunits::Int64,startpoint::Int64)

    if (nit<=nunits)
        println(myid(),": spredwork(): nit<=nunits")
        println(nit,nunits)
        error()
    end 
    # allocate
    scheduling = Array{Int64,1}(undef,nunits)
    looping = Array{Int64,2}(undef,nunits,2)

    ## compute distribution of workload for Nb
    nitcpu = div(nit,nunits) # nit/nunits
    scheduling[1:end] .= nitcpu
    resto = mod(nit,nunits)
    
    ## spread the remaining workload
    scheduling[1:resto] = scheduling[1:resto] .+ 1
    looping[1,:] = [startpoint,startpoint+scheduling[1]-1] 
    for i=2:nunits
        looping[i,1] = sum(scheduling[1:i-1]) + startpoint
        looping[i,2] = sum(scheduling[1:i]) + startpoint - 1
    end 
    
    return scheduling,looping
end 

##==========================================================

"""
    krondiag(a::Array{Float64,1},b::Array{Float64,1}) 

Kronecker product of two diagonal matrices.
Returns only the diagonal as a vector.
"""
function krondiag(a::Array{Float64,1},b::Array{Float64,1}) 

    ni=size(a,1)
    nj=size(b,1)
    c=zeros(Float64,ni*nj)
    k=1
    for i=1:ni 
        c[k:k+nj-1] = a[i]*b
        k += nj
    end
    return c
end

##==========================================================
##==========================================================


# !!!==================================================
#   !-------------------------------------------------
#   !>  @brief <b> Computes a band of the posterior covariance. </b>
#   !> See http://www.netlib.org/lapack/lug/node124.html
#   !> @param[in] U1,U2,U3  `` \mathbf{U}_1 ``, `` \mathbf{U}_2 ``,
#   !>     `` \mathbf{U}_3  ``  of  `` F_{\sf{A}} `` 
#   !> @param[in] diaginvlambda  `` F_{\sf{B}} ``
#   !> @param[in] iUCm1,iUCm2,iUCm3  ``\mathbf{U}_1^{-1} \mathbf{C}_{\rm{M}}^{\rm{x}}``,
#   !>      ``\mathbf{U}_2^{-1}  \mathbf{C}_{\rm{M}}^{\rm{y}}``,
#   !>       ``\mathbf{U}_2^{-1}  \mathbf{C}_{\rm{M}}^{\rm{z}}`` of  `` F_{\sf{C}} `` 
#   !> @param[in] lowdiag,updiag  Lower and upper diagonal number of requested band
#   !> @param[out] postC  band of the posterior covariance stored following Lapack convention
#   !
#   !
#   !-------------------------------------------------  
#   subroutine bandpostcov(U1,U2,U3, diaginvlambda, &
#        iUCm1,iUCm2,iUCm3, lowdiag, updiag, bandpostC) 
#     !!
#     !! Calculate a band of the posterior covariance
#     !! 
#     real(dp),intent(in) :: U1(:,:),U2(:,:),U3(:,:),iUCm1(:,:), &
#          iUCm2(:,:),iUCm3(:,:)
#     !! diaginvlambda = (I + lam1 x lam2 x lam3 )^-1
#     real(dp),intent(in) :: diaginvlambda(:) !! diagonal/vector central factor
#     real(dp),intent(inout) :: bandpostC(:,:)
#     integer,intent(in) :: lowdiag, updiag
    
#     integer :: a,b
#     real(dp),allocatable :: row2(:),col1(:),recvrow(:),bandpostCmpi(:)
#     integer :: Nr12,Nc1
#     integer :: Ni,Nj,Nk,Nl,Nm,Nn,Na,Nb    

#     real(dp) :: firststartt,startt
#     integer,allocatable :: av(:)!,bv(:)
#     integer,allocatable :: iv(:),jv(:),kv(:) !,lv(:),mv(:),nv(:) 
#     integer :: p,aband,aend,bband,astart,d,bband1,bband2,ntota,everynit,mytotit
#     integer,allocatable :: scheduling(:),looping(:,:)
#     character(len=30) :: loopinfo
#     character(len=12) :: dnum
    
#     Ni = size(U1,1)
#     Nl = size(U1,2)
#     Nj = size(U2,1)
#     Nm = size(U2,2)
#     Nk = size(U3,1)
#     Nn = size(U3,2)
#     Na = Ni*Nj*Nk 
#     Nb = Nl*Nm*Nn    

#     if (Na /= Nb) then
#        write(*,*) '(Na /= Nb)', Na,Nb
#        stop
#     end if
#     if ( (updiag>=Na) .or. (lowdiag>=Na) .or. (lowdiag<0) .or. (updiag<0) ) then
#        write(*,*) "(updiag<Na) .or. (lowdiag<Na)"
#        write(*,*) "updiag",updiag,"Na",Na,"lowdiag",lowdiag,"Na",Na
#        stop
#     end if

#     !! vectorize row and col calculations for Kron prod AxBxC
#     allocate(av(Na),iv(Na),jv(Na),kv(Na))
#     !allocate(bv(Nb),lv(Nb),mv(Nb),nv(Nb))
#     forall(p = 1:Na) av(p) = p
#     !forall(p = 1:Nb) bv(p) = p

#     !! vectors containing all possible indices for 
#     !!    row calculations of Kron prod AxBxC
#     iv(:) =  (av-1)/(Nk*Nj)+1 
#     jv(:) =  (av-1-(iv-1)*Nk*Nj)/Nk+1 
#     kv(:) =  av-(jv-1)*Nk-(iv-1)*Nk*Nj 
#     !! vectors containing all possible indices for
#     !!    column calculations of Kron prod AxBxC
#     ! lv =  (bv-1)/(Nn*Nm) + 1 
#     ! mv =  (bv-1-(lv-1)*Nn*Nm)/Nn + 1 
#     ! nv =  bv-(mv-1)*Nn-(lv-1)*Nn*Nm
    
#     !! allocate stuff
#     Nr12 = size(U1,2)* size(U2,2)* size(U3,2)
#     Nc1  = size(iUCm1,1)* size(iUCm2,1)* size(iUCm3,1)
#     allocate(row2(Nr12),col1(Nc1))

#     ! Lapack: http://www.netlib.org/lapack/lug/node124.html
#     ! aij is stored in AB(ku+1+i-j,j) for max(1,j-ku) <= i <= \min(m,j+kl).
#     !------------
#     ! Diagonals of a matrix
#     ! i + d = j
#     ! main diag d = 0
#     ! upper d > 0
#     ! lower d < 0
#     ! diagonals of a matrix and indices of related band matrix
#     ! ONLY for square matrix

#     !!---------------------------------
#     firststartt = MPI_Wtime()
#     ! !! calculate scheduling for a
#     ! call spreadwork(ntota,numcpus,scheduling,looping,1)
#     ! allocate(postcmpi(scheduling(myrank+1),ntotb),recvrow(ntotb),sendrow(ntotb))
#     ! mytotit = looping(myrank+1,2)-looping(myrank+1,1)+1

#     allocate(bandpostCmpi(size(bandpostC,2)), recvrow(size(bandpostC,2)))
#     ! initialize postC
    
#     bandpostC = 0.0_dp       

#     do d=-lowdiag,updiag
#        call para_barrier()
       
#        !!--------------------------
#        if (d<0) then
#           astart = abs(d)+1
#        else
#           astart = 1
#        end if
#        if (d>0) then
#           aend = Na-abs(d)
#        else
#           aend = Na
#        end if
#        !print*,'diagonal',d

#        !! calculate scheduling for a
#        ntota = aend-astart+1
#        call spreadwork(ntota,numcpus,scheduling,looping,astart)
#        if (ntota<1000) then
#           everynit = ntota/100
#        else
#           everynit = ntota/1000
#        end if
       
#        write(dnum,"(i9)") d
#        loopinfo = "bandpostcov()  diag "//trim(dnum)//" "
#        startt = MPI_Wtime()
#        if (myrank==masterrank) write(OUTPUT_UNIT,*)
#        !! indices of normal matrix
#        bandpostCmpi(:) = 0.0_dp
#        aband = updiag+1-d !! aband is constant within this loop
#        do a=looping(myrank+1,1),looping(myrank+1,2) !!astart,aend
#           !! info
#           if ( (myrank==masterrank) .and. (mod(a,everynit)==0) ) then
#              call timeinfo(scheduling(myrank+1),a-astart+1,startt,loopinfo)
#           end if

#           !if ( (myrank==masterrank) .and. (mod(a,250)==0) ) print*,"d",d,"a",a
#           !!--------------------------------------------------------
#           b = a+d
#           !! indices of the band matrix
#           !aband = updiag+1+a-b
#           !bband = b
#           !!--------------------------------------------------------
                         
#           !!----------------------------------------------------------------
#           ! row first two factors
#           row2(:) = diaginvlambda * U1(iv(a),iv) * U2(jv(a),jv) * U3(kv(a),kv)

#           !! calculate one row of first TWO factors
#           !!call columnAxBxC(Ni,Nj,Nk,Nl,Nm,Nn, iUCm1,iUCm2,iUCm3,b,col1)
#           col1(:) = iUCm1(iv,iv(b)) * iUCm2(jv,jv(b)) * iUCm3(kv,kv(b))

#           !! calculate one element of the posterior covariance
#           !! store it in the band storage format
#           !!bandpostC(aband,bband) = sum(row2*col1)
#           bandpostCmpi(b) = sum(row2*col1)
#           !print*, a,b,aband,bband,bandpostC(aband,bband)
#           !!----------------------------------------------------------------
          
#        end do ! a=astart,aend
       
#        !! send/recv rows                                                      
#        !! collect all results in masterrank
       
#        bband1 = looping(myrank+1,1)+d !!b = a+d
#        bband2 = looping(myrank+1,2)+d
       
#        call para_gathv1ddp(scheduling, looping(:,1)-1, masterrank, &
#             bandpostCmpi(bband1:bband2), recvrow )
#        if (myrank==masterrank) bandpostC(aband,:) = recvrow
      
       
#     end do

#     end subroutine bandpostcov
           
##==========================================================
