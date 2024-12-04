
##############################################################


"""
$(TYPEDSIGNATURES)

Save the initial input parameters and model to a .jld2 and HDF5 files or read values from a 
    previous simulation.
Uses the SWMR mode of HDF5.
"""
function initsavestuff!(pars::Dict,nlogppd::NLogPostPDF,MCpar::AbstractMCParams,
                        mstart::Vector{Float64} )
                       
    #---------------------------------------
    if !(haskey(pars,"outdir"))
        pars["outdir"] = ""
    end
    #---------------------------------------
    if !("stdout" in keys(pars))
        pars["stdout"] = nothing
    end
    
    # create outdir if it does not exist
    if !(isdir(pars["outdir"]))
        mkpath(pars["outdir"])
    end

    ## output file names
    outfile_inp = joinpath(pars["outdir"],pars["simname"]*"_inp.jld2")
    outfile_out = joinpath(pars["outdir"],pars["simname"]*"_outp.h5")

    ### if the program crashes, from the terminal do
    ##$  h5clear -s "filename"

    ## determine which kind of simulation/algo is used
    mcalgo = String(MCpar.algo)
    if typeof(MCpar)<:AbstractHMCParams
        simtype = :HMC
    elseif typeof(MCpar)<:AbstractMeHaParams
        simtype = :MeHa
    end

    
    if pars["continueold"]==true

        ## HDF5 swmr=true to enable (Single Writer Multiple Reader)
        ## open in r+ mode
        fid_h5 = h5open(outfile_out,"r+",swmr=true)

        ## check that simulation id is the same for .h5 and .jld2
        pars["simid_h5"] = string(read(fid_h5["simulid"]))
        if pars["simid_h5"] != pars["simid_jld2"]
            close(fid_h5)
            error("The simulation id (UUID) is not the same for input (.jld2) and output (.h5) files.")
        end

        # saved models
        nsav = convert(Int64,size(fid_h5["mods"],2))
        # accepted models
        nacc = convert(Int64,fid_h5["nacc"][end])
        # initial potential energy
        if simtype==:HMC
            Ucur = convert(Float64,fid_h5["ucur_m"][end])
        elseif simtype==:MeHa
            if algo=="ExMetrop"
                Ucur = convert(Float64,fid_h5["nllk_m"][end])
            end
        end
        ## nsaved is a reference to Int64, so that it's passed
        ##   by reference(?) and mutated inside savestuff!()
        nsaved = Ref{Int64}(nsav)
        
        ## previous iterations (from previous simulation)
        previt = convert(Int64,fid_h5["iter"][end])
  
        ## remove the previous endtime in case the program crashes
        if haskey(fid_h5,"endtime")
            HDF5.delete_object(fid_h5,"endtime")
        end

        if pars["maxiter"]<=previt
            error("Maximum number of iterations (par[\"maxiter\"]) <= number of previous iterations. Aborting.")
        end

        ##------------------------------------------------------------------------------------------
        ## only for problems with checks during forward/gradient computation, e.g. polygonal bodies
        if isdefined(nlogppd.likelihood,:dofwdchecks) && nlogppd.likelihood.dofwdchecks
            if nlogppd.likelihood.probkind==:polygonalbodies
                nlogppd.likelihood(fid_h5,"readh5bodyindices")
            end
        end
        ##------------------------------------------------------------------------------------------

            
    elseif pars["continueold"]==false
       
        ## brute force rm old fileS!!!
        # println("Deleting $outfile_inp, $outfile_out")
        # rm(outfile_inp,force=true)
        # rm(outfile_out,force=true)

        if isfile(outfile_inp) || isfile(outfile_out)
            error(" Output files \"$outfile_inp\" and/or \"$outfile_out\" already exist. Aborting.")
        end

        # unique simulation id
        simulid = string(uuid1())

        ## write the input parameters file
        JLD2.jldopen(outfile_inp, "a+") do fjd
            ##inpgrp = JLD2.Group(fjd, "input")
            fjd["mcparams"] = MCpar
            fjd["problemparams"] = nlogppd
            fjd["simulid"] = simulid
            fjd["startingmodel"] = mstart
            if haskey(pars,"refmodel")
                if typeof(pars["refmodel"])==Vector{Float64}
                    println(" Synthetic test, saving refence model to HDF5 file.")
                    fjd["refmodel"] = pars["refmodel"]
                else
                    @show typeof(pars["refmodel"])
                    error("Provided reference model pars[\"refmod\"] (optional) is not a Vector{Float64}.")
                end
            end
        end
        
        ## Switch to pure HDF5 for output/results
        ## HDF5 swmr=true to enable (Single Writer Multiple Reader) 
        ## open in w mode
        fid_h5 = h5open(outfile_out,"cw",swmr=true)
        fid_h5["starttime"] = Dates.format(Dates.now(),"yyyy-mm-dd_HH:MM:SS")
        # set the simulation id (UUID)
        fid_h5["simulid"] = simulid
        fid_h5["algo"] = String(MCpar.algo)

        # accepted models
        nacc = 0
        # saved models
        ## nsaved is a reference to Int64, so that it's passed
        ##   by reference(?) and mutated inside savestuff!()
        nsaved = Ref{Int64}(0)

        if simtype==:HMC
            # initial potential energy
            Ucur = calcU(nlogppd,mstart)
        elseif simtype==:MeHa
            if MCpar.algo==:ExtMetrop
                Ucur = calcnllk(nlogppd,mstart)
             else
                error("initsavestuff!(): Wrong algo: $(MCpar.algo)")
            end
        else
            error("initsavestuff!(): Wrong simtype...")
        end

        ## previous iterations (from previous simulation)
        previt=0

    end

        
    if pars["continueold"] && MCpar.algo===:NUTS
        if haskey(pars,"new_HMCpar") && (pars["new_HMCpar"]==true)
            ## in case new HMCpar has been specified
            ϵ = MCpar.ϵinimax
        else
            ## else get it from last iteration
            ϵ = convert(Float64,fid_h5["epsilon"][end])
        end
        ## if NUTS then return also ϵ
        return fid_h5,nacc,nsaved,Ucur,previt,ϵ
    else
        return fid_h5,nacc,nsaved,Ucur,previt
    end
end
   
#######################################################################

"""
$(TYPEDSIGNATURES)

Save models and parameters in a HDF5 file while the HMC is running.
Uses the SWMR mode of HDF5.
"""
function savestuff!(fid_h5::HDF5.File,ntobesaved::Int64,pars::Dict,nsaved::Base.RefValue{Int64}, 
                    it::Int64,m::Vector{Float64},nacc::Int64,ucurval::Float64 ;
                    llkprival::Union{AbstractVector,Nothing}=nothing,
                    algo::Symbol, nustats::Union{NUTSstats,Nothing}=nothing,fwdmod=nothing)
                    
    saveevery = pars["saveevery"]    
    nmpar = size(m,1)

    ##  outgrpname = "output"
    if algo==:NUTS || algo==:plainHMC
        namenlppd = "ucur"
    elseif algo==:ExtMetrop
        namenlppd = "nllk"
    end

    # first iteration
    if it==1 && (pars["continueold"]==false) 
        
        @assert pars["maxiter"]>=saveevery
        maxiter = pars["maxiter"]

        ## create group for output/results
        # gout = g_create(fid_h5,outgrpname)
        
        ### init
        ## WARNING: it saves the models as Float32 to save space...
        #@show ntobesaved
        # create_dataset(parent, name, dtype, (dims, max_dims), "chunk", (chunk_dims), [lcpl, dcpl, dapl])
        ##                                   -1 is equivalent to typemax(Hsize)

        ## LIMITED max dims to number of iterations/saveevery
        # it_m_h5 = HDF5.create_dataset(fid_h5, "iter_m", Int64, ((1,), (ntobesaved,)), "chunk", (ntobesaved,) )
        # mod_h5 = HDF5.create_dataset(fid_h5, "mods", Float32,   ((nmpar,1), (nmpar,ntobesaved)),"chunk", (nmpar,1) )
        # ucur_m_h5 = HDF5.create_dataset(fid_h5, "ucur_m", Float64, ((1,), (ntobesaved,)),"chunk", (ntobesaved,))        
        # it_h5 = HDF5.create_dataset(fid_h5, "iter", Int64, ((1,), (maxiter,)),"chunk", (maxiter,)  )
        # ucur_h5 = HDF5.create_dataset(fid_h5, namenlppd, Float64, ((1,), (maxiter,)),"chunk", (maxiter,)  )
        # nacc_h5 = HDF5.create_dataset(fid_h5, "nacc", Int64, ((1,), (maxiter,)),"chunk", (maxiter,)  )
        # if caldat!=nothing
        #     ndobs = size(caldat[:],1)
        #     dcalc_h5 = HDF5.create_dataset(fid_h5, "dcalc_m", Float32,((ndobs,1), (ndobs,ntobesaved)),"chunk", (ndobs,1) )
        # end
        
        ## *UNLIMITED* max dims to number of iterations/saveevery
        chunksize1d = 1000
        it_m_h5 = HDF5.create_dataset(fid_h5, "iter_m", Int64, ((1,), (-1,)), chunk=(chunksize1d,) )
        mod_h5 = HDF5.create_dataset(fid_h5, "mods", Float32,   ((nmpar,1), (nmpar,-1)),chunk=(nmpar,1) )
        ucur_m_h5 = HDF5.create_dataset(fid_h5, namenlppd*"_m", Float64, ((1,), (-1,)),chunk=(chunksize1d,))        
        it_h5 = HDF5.create_dataset(fid_h5, "iter", Int64, ((1,), (-1,)),chunk=(chunksize1d,)  )
        ucur_h5 = HDF5.create_dataset(fid_h5, namenlppd, Float64, ((1,), (-1,)),chunk=(chunksize1d,)  )
        if !ismissing(llkprival[1])
            llk_h5 = HDF5.create_dataset(fid_h5, "likelihoodval", Float64, ((1,), (-1,)),chunk=(chunksize1d,)  )
        end
        if !ismissing(llkprival[2])
            prior_h5 = HDF5.create_dataset(fid_h5, "priorval", Float64, ((1,), (-1,)),chunk=(chunksize1d,)  )
        end
        nacc_h5 = HDF5.create_dataset(fid_h5, "nacc", Int64, ((1,), (-1,)),chunk=(chunksize1d,)  )
        if algo===:NUTS
            ϵ_h5 = HDF5.create_dataset(fid_h5, "epsilon", Float64, ((1,), (-1,)),chunk=(chunksize1d,)  )
            treeheight_h5 = HDF5.create_dataset(fid_h5, "treeheight", Int64, ((1,), (-1,)),chunk=(chunksize1d,)  )
            ϵbar_h5 = HDF5.create_dataset(fid_h5, "epsilonbar", Float64, ((1,), (-1,)),chunk=(chunksize1d,)  )
            accatlfstep_h5 = HDF5.create_dataset(fid_h5, "accepted_at_lfstep", Int64, ((1,), (-1,)),chunk=(chunksize1d,)  )
            lfsteps_h5 = HDF5.create_dataset(fid_h5, "lfsteps", Int64, ((1,), (-1,)),chunk=(chunksize1d,)  )
        end

        # close(gout)
    end

    ##====================================
    ## first open the group for output
    #gout =  g_open(fid_h5,outgrpname)
    
    # it
    it_h5 = HDF5.open_dataset(fid_h5,"iter")
    HDF5.set_extent_dims(it_h5,(it,))
    it_h5[it] = it
    # ucur
    ucur_h5 = HDF5.open_dataset(fid_h5,namenlppd)
    HDF5.set_extent_dims(ucur_h5,(it,))
    ucur_h5[it] = ucurval
    if !ismissing(llkprival[1])
        # likelihood value
        llk_h5 = HDF5.open_dataset(fid_h5,"likelihoodval")
        HDF5.set_extent_dims(llk_h5,(it,))
        llk_h5[it] = llkprival[1]
    end
    if !ismissing(llkprival[2])
        # prior value
        prior_h5 = HDF5.open_dataset(fid_h5,"priorval")
        HDF5.set_extent_dims(prior_h5,(it,))
        prior_h5[it] = llkprival[2]
    end
    # nacc
    nacc_h5 = HDF5.open_dataset(fid_h5,"nacc")
    HDF5.set_extent_dims(nacc_h5,(it,))
    nacc_h5[it] = nacc    

    ## if NUTS then save also ϵ
    if algo===:NUTS
        # ϵ
        ϵ_h5 = HDF5.open_dataset(fid_h5,"epsilon")
        HDF5.set_extent_dims(ϵ_h5,(it,))
        ϵ_h5[it] = nustats.ϵ
        # treeheight
        treeheight_h5 = HDF5.open_dataset(fid_h5,"treeheight")
        HDF5.set_extent_dims(treeheight_h5,(it,))
        treeheight_h5[it] = nustats.treeheight
        # ϵbar
        epsilonbar_h5 = HDF5.open_dataset(fid_h5,"epsilonbar")
        HDF5.set_extent_dims(epsilonbar_h5,(it,))
        epsilonbar_h5[it] = nustats.ϵbar
        # lfsteps
        lfsteps_h5 = HDF5.open_dataset(fid_h5,"lfsteps")
        HDF5.set_extent_dims(lfsteps_h5,(it,))
        lfsteps_h5[it] = nustats.lfsteps
        # accepted at leap-frog step
        accatlfstep_h5 = HDF5.open_dataset(fid_h5,"accepted_at_lfstep")
        HDF5.set_extent_dims(accatlfstep_h5,(it,))
        accatlfstep_h5[it] = nustats.accatlfstep
    end
    
    ##====================================
    if (it%saveevery==0) | (it==1)
        nsaved[] = nsaved[]+1
        sit = nsaved[]
        
        # iterations
        ##println(">>> H5: $(fid_h5)")
        it_m_h5 = HDF5.open_dataset(fid_h5,"iter_m")
        HDF5.set_extent_dims(it_m_h5,(sit,))
        it_m_h5[sit] = it
        # models
        mod_h5 = HDF5.open_dataset(fid_h5,"mods")
        HDF5.set_extent_dims(mod_h5,(nmpar,nsaved[]))
        mod_h5[:,sit] = m  ## <<<===  EXP/LOG
        # ucur_m
        ucur_m_h5 = HDF5.open_dataset(fid_h5,namenlppd*"_m")
        HDF5.set_extent_dims(ucur_m_h5,(sit,))
        ucur_m_h5[sit] = ucurval ## <<<===  EXP/LOG
        

        ##------------------------------------------------------------------------------------------
        ## only for problems with checks during forward/gradient computation, e.g. polygonal bodies
        if fwdmod!=nothing
            if isdefined(fwdmod,:dofwdchecks) && fwdmod.dofwdchecks
                if fwdmod.probkind==:polygonalbodies
                    fwdmod(fid_h5,it,sit,"savebodyindices")
                end
            end
        end
        ##------------------------------------------------------------------------------------------

        # # caldat
        # if caldat!=nothing
        #     dcalc_m_h5 = HDF5.open_dataset(fid_h5,"dcalc_m")
        #     set_extent_dims(dcalc_m_h5,(ndobs,nsaved[]))
        #     dcalc_m_h5[:,sit] = caldat[:]
        # end
 
    end

    ##====================================
    ## close the output group
    #close(gout)
    
    ### flush all the stuff such that (hopefully) the hdf5 swmr will work...
    herr = HDF5.h5f_flush(fid_h5,HDF5.H5F_SCOPE_GLOBAL)
  
    ### if the program crashes, from the terminal do
    ##$  h5clear -s "filename" 
    
    ## At last iteration, close the HDF5 file
    if pars["maxiter"]==it #nsaved[]==ntobesaved
        fid_h5["endtime"] = Dates.format(Dates.now(),"yyyy-mm-dd_HH:MM:SS")
        close(fid_h5)
    end
    return nothing
end

#######################################################################


"""
$(TYPEDSIGNATURES)

Print info during the HMC run to stdout or file.
"""
function printiterinfo(starttime,it,maxiter,naccvec,actualit,Ucur,pars)

    elsecs = time()-starttime           
    eltime = "ELA: $(hmsstr(elsecs))"
    estt = (elsecs/it)*(maxiter-actualit)            
    esttime = "ETA: $(hmsstr(estt))"
    accrat = naccvec[it]/actualit*100.0
    itsecratio =  @sprintf("it/s: %.2f",round(it/elsecs; digits=2))

    if it>11
        acclast10 = naccvec[it]-naccvec[it-10]
    else
        acclast10 = missing
    end
    if it>101
        acclast100 = naccvec[it]-naccvec[it-100]
    else
        acclast100 = missing
    end
    lastnacc = "  acc.last10: "*string(acclast10*10)*"%   acc.last100: "*string(acclast100)*"%"

    if pars["stdout"]=="file"
        if it==1
            println("Info is being written to the file 'iterinfo.txt'")

            line = @sprintf("Iter.#: %d  Acc.: %3.2f%% Ucur: %.3f ",actualit,accrat,Ucur)
            open("iterinfo.txt", "w") do io
                write(io,"\n ====================================")
                write(io,"\n      Hamiltonian Monte Carlo      ")
                write(io,"\n ====================================\n")
                if pars["continueold"]
                    write(io," Continuing previous simulation $(pars["simname"]) \n")
                else
                    write(io," Running simulation $(pars["simname"]) \n")
                end
                write(io, line*eltime*"  "*esttime*"  "*itsecratio*" \n")
            end
        else
            line = @sprintf("Iter.#: %d  Acc.: %3.2f%% Ucur: %.3f ",actualit,accrat,Ucur)
            open("iterinfo.txt", "a") do io
                write(io, line*eltime*"  "*esttime*"  "*itsecratio*" \n")
            end
        end

        
    elseif pars["stdout"]=="oneline"        
        #############################
        ## one line output to stdout
        #############################
        if it==1
            line = @sprintf("\e[31m\e[1m Iter.#: %d  \e[0mAcc.: %3.2f%%  \033[36m\e[1mUcur: %.3f \e[0m ",actualit,accrat,Ucur)
            print(line*eltime*"  "*esttime*"    ")
            flush(stdout)
        else
            # \e[1F \e[0J
            line = @sprintf("\r\e[31m\e[1m Iter.#: %d  \e[0mAcc.: %3.2f%%  \033[36m\e[1mUcur: %.3f \e[0m ",actualit,accrat,Ucur)
            #line = @sprintf("\e[0J\e[31m\e[1m Iter.#: %d  \e[0mAcc.: %3.2f%%  \033[36m\e[1mUcur: %.3f \e[0m \e[1F",actualit,accrat,Ucur)
            print(line*eltime*"  "*esttime*"    ")
            flush(stdout)
        end

    else
        #############################
        ## default output to stdout
        #############################
        #line = @sprintf("\e[1K \r\e[31m\e[1m Iter.#: %d  \e[0mAcc.: %3.2f%%  \033[36m\e[1mUcur: %.3f \e[0m ",actualit,accrat,Ucur)

        if it==1
            line = @sprintf("\e[31m\e[1m Iter.#: %d  \e[0mAcc.: %3.2f%%  \033[36m\e[1mUcur: %.3f \e[0m ",actualit,accrat,Ucur)
            print(line*eltime*"  "*esttime*"  \n  "*itsecratio*" "*lastnacc)
            flush(stdout)
        else
            # \e[1F \e[0J
            line = @sprintf("\e[1F\e[0J\e[31m\e[1m Iter.#: %d  \e[0mAcc.: %3.2f%%  \033[36m\e[1mUcur: %.3f \e[0m ",actualit,accrat,Ucur)
            #line = @sprintf("\e[0J\e[31m\e[1m Iter.#: %d  \e[0mAcc.: %3.2f%%  \033[36m\e[1mUcur: %.3f \e[0m \e[1F",actualit,accrat,Ucur)
            print(line*eltime*"  "*esttime*"  \n  "*itsecratio*" "*lastnacc)
            flush(stdout)
        end

    end
    return nothing
end

#############################################################################################################

"""
$(TYPEDSIGNATURES)

Compute elapsed hours, minutes and seconds for given elapsed seconds.
"""
function hmsstr(elsecs)
    hours = floor(elsecs / 3600)
    mins = floor(elsecs / 60 % 60)
    secs = floor(elsecs % 60)
    line = @sprintf "%dh:%dm:%ds" hours mins secs
    return line
end

#############################################################################################################
