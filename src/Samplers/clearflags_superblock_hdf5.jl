


##using CRC32c

###==========================================

    #   BIN    DEC
    #    0 	0
    #    1 	1
    #   10	2
    #   11	3
    #  100	4
    #  101	5
    #  110	6
    #  111	7
    # 1000	8


#############################################################

"""
 $(TYPEDSIGNATURES)

An *unsafe* but useful way to clear the status flags in the HDF5 superblock in 
 case the HDF5 file is not closed properly. That can happen when a simulation is 
 suddently interrupted before reaching the maximum number of iterations.
"""
function clear_stflags_hdf5(flname::String)
    
    ## https://support.hdfgroup.org/HDF5/doc/H5.format.html

    iof = open(flname,"r+")

    headerNOTfound = true
    HEADER_LEN::Int64 = 0

    maxcount = 10
    counter=0
    while headerNOTfound
        counter+=1      

        # Signature, attempting to skip header
        seek(iof,HEADER_LEN)
        signature = read(iof, UInt64)
        #@show signature
        ## signature of HDF5 files
        ## 0x0a1a0a0d46444889 is the HDF5 files signature
        hdf5signature = htol(0x0a1a0a0d46444889)

        if signature==hdf5signature

            ## exit while loop at the end
            headerNOTfound=false

            # Version
            version = read(iof, UInt8)
            ## make sure it's version 3 of header, otherwise the
            ##   the file might end up being corrupted
            @assert version==0x03
            #@show version

            # Size of offsets and size of lengths
            size_of_offsets = read(iof, UInt8)
            size_of_lengths = read(iof, UInt8)
            #@show size_of_offsets
            #@show size_of_lengths

            if size_of_offsets==0x08
                sizeread_addresses = UInt64
            elseif size_of_offsets==0x04
                sizeread_addresses = UInt32
            else
                close(iof)
                error("clear_stflags_hdf5(): Undefined size_of_offsets. Aborting.")
            end

            # File consistency flags
            cons_flags_position = position(iof)
            file_consistency_flags = read(iof, UInt8)
            #@show file_consistency_flags

            # Addresses
            base_address = read(iof, UInt64)
            superblock_extension_address = read(iof, sizeread_addresses)
            end_of_file_address = read(iof, UInt64)
            root_group_object_header_address = read(iof, sizeread_addresses)

            # CHECKSUM
            position_chksum = position(iof)
            cheksum_supblock = read(iof, UInt32) 
            #@show position_chksum

            ###==========================================
            ## compute CHECKSUM
            nbytes = position_chksum-HEADER_LEN
            
            ## crc32c(io::IO, [nb::Integer,] crc::UInt32=0x00000000)
            # chksum = CRC32c.crc32c(seek(iof,HEADER_LEN),nbytes)
            # @show chksum

            ## position iof right after the HEADER
            ## get all the superblock excluding the checksum as an
            ##  vector of UInt8
            vec8 = read( seek(iof,HEADER_LEN), nbytes)

            ## Compute checksum using the Lookup3 algo from JLD2
            chksum_look3 = JLD2.Lookup3.hash(vec8,1,nbytes)
            @assert cheksum_supblock==chksum_look3
           
            ###==========================================
            ## START MODIFY STUFF

            # clear the consistency flags
            seek(iof,cons_flags_position)
            write(iof,UInt8(0x00))

            ## compute new checksum
            vec8 = read( seek(iof,HEADER_LEN), nbytes)
            new_chksum_look3 = JLD2.Lookup3.hash(vec8,1,nbytes)
            ## write new checksum
            seek(iof,position_chksum)
            write(iof,UInt32(new_chksum_look3))

            ## END MODIFY STUFF
            ###==========================================

        else

            if counter>=maxcount
                close(iof)
                error("clear_stflags_hdf5(): Cannot find valid HDF5 signature. Aborting.")
            end

            HEADER_LEN += 512
            
        end
    end

    ## close io stream
    close(iof)
    ###==========================================

    return
end

#############################################################
