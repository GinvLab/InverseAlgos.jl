


function savestuff!(outfile::String,overwriteoutput::Bool,k::Integer,misfnew::Real,xnew::Vector{<:Real})

    nmpar = length(xnew)

    if k==0

        if overwriteoutput==true
            fid_h5 = h5open(outfile,"w") #,swmr=true)
        else
            fid_h5 = h5open(outfile,"cw") #,swmr=true)
        end
        misf_h5 = HDF5.create_dataset(fid_h5, "misfit", Float64, ((1,), (-1,)),chunk=(100,)  )
        mods_h5 = HDF5.create_dataset(fid_h5, "mods", Float64, ((nmpar,1), (nmpar,-1)),chunk=(nmpar,1) )

        HDF5.set_extent_dims(misf_h5,(k+1,))
        misf_h5[k+1] = misfnew

        HDF5.set_extent_dims(mods_h5,(nmpar,k+1))
        mods_h5[:,k+1] = xnew

        close(fid_h5)

    else

        fid_h5 = h5open(outfile,"r+")

        misf_h5 = HDF5.open_dataset(fid_h5,"misfit")
        HDF5.set_extent_dims(misf_h5,(k+1,))
        misf_h5[k+1] = misfnew

        mods_h5 = HDF5.open_dataset(fid_h5,"mods")
        HDF5.set_extent_dims(mods_h5,(nmpar,k+1))
        mods_h5[:,k+1] = xnew

        close(fid_h5)

    end

    return
end


