
#########################################

"""
$(TYPEDSIGNATURES)

Calculate a matrix-vector product in parallel.
"""
function paramatvec(Amat::Array{Future,1},p::Vector{Float64},grptask::Array{Int,2})

    function domul(Afut::Future,p::Vector{Float64})
        A = fetch(Afut)
        y = A*p
    end

    nrows = length(p)
    wks = workers()
    y = Array{Float64,1}(undef,nrows)
    @sync for i=1:size(grptask,1)
        #@async yf[i] = remotecall(domul,wks[i],Arows[i],x)
        @async y[grptask[i,1]:grptask[i,2]] = remotecall_fetch(domul,wks[i],Amat[i],p)
    end

    return y
end

##################################################

"""
$(TYPEDSIGNATURES)

Load from file and in parallel the mass matrix (Cholesky low triangular) and its inverse.
"""
function paraloadmassM!(HMCpar::AbstractHMCParams)

    ## Read the data in parallel
    println(" Loading mass matrix data to workers")

    function geth5data(flname::String,i1::Int,i2::Int,key::String)
        #println("Reading file $flname, dataset $key, rows $i1 to $i2")
        fid = h5open(flname, "r")
        dset = fid[key]
        Arows = dset[i1:i2,:]
        close(fid)
        return Arows
    end

    flname = HMCpar.massMfile

    ## get the number of rows of the invmassM
    nrows = nothing
    h5open(flname,"r") do fid
        nrows = size(fid["invmassM"],1)
    end

    grptask = distribwork(nrows,HMCpar.numwks)
    wks = workers()
    @assert nworkers()==HMCpar.numwks
    #@show grptask

    kiM = "invmassM"
    kLM = "LcholmassM"
         
    # get the Future(s) for later parallel calculations
    @sync for i=1:HMCpar.numwks    
        @async HMCpar.invmassM[i]   = remotecall(geth5data,wks[i],flname,grptask[i,1],grptask[i,2],kiM)
        @async HMCpar.LcholmassM[i] = remotecall(geth5data,wks[i],flname,grptask[i,1],grptask[i,2],kLM)
    end

    return grptask
end


#########################################


"""
$(TYPEDSIGNATURES)

Subdivide computations among workers.
"""
function distribwork(nelem::Integer,nwks::Integer)
    ## calculate how to subdivide the srcs among the workers
    if nelem>=nwks
        dis = div(nelem,nwks)
        grpsizes = dis*ones(Int64,nwks)        
        resto = mod(nelem,nwks)
        if resto>0
            ## add the reminder 
            grpsizes[1:resto] .+= 1
        end
    else
        ## if more workers than sources use only necessary workers
        grpsizes = ones(Int64,nelem)        
    end
    ## now set the indices for groups of srcs
    grpsrc = zeros(Int64,length(grpsizes),2)
    grpsrc[:,1] = cumsum(grpsizes).-grpsizes.+1
    grpsrc[:,2] = cumsum(grpsizes)
    return grpsrc
end

#############################################################
