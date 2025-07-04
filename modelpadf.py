
import numpy as np
#import array
import os
import params.paramsMODEL as params
import fast_model_padf2 as fmp
import multiprocessing as mp
import copy

def thread_subcell_model_padf(cells, j, modelp, method, return_dict): 

        
        padfsum_evens = np.zeros( (modelp.nr, modelp.nr, modelp.nth))
        padfsum_odds  = np.zeros( (modelp.nr, modelp.nr, modelp.nth)) 
        #sphvol_evens = np.zeros( (modelp.nr, modelp.nthvol, modelp.phivol))
        #sphvol_odds  = np.zeros( (modelp.nr, modelp.nthvol, modelp.phivol)) 

        for icell, cell in enumerate(cells):
            print( f"Processor {j}; cell {icell+1} / {len(cells)}") 
            ia, ib, ic = cell[0], cell[1], cell[2] 
            modelp.limits = np.array([ia*modelp.subcellsize,(ia+1)*modelp.subcellsize, \
                        ib*modelp.subcellsize,(ib+1)*modelp.subcellsize,ic*modelp.subcellsize,(ic+1)*modelp.subcellsize])   

            if method == 'serial':
                fmp.ModelPadfCalculator.run_fast_serial_calculation(modelp)
            elif method=='matrix':
                fmp.ModelPadfCalculator.run_fast_matrix_calculation(modelp)
            elif method=='histogram':
                fmp.ModelPadfCalculator.run_histogram_calculation(modelp)   
            elif method == 'spharmonic':
                fmp.ModelPadfCalculator.run_spherical_harmonic_calculation(modelp)   
            else:
                print(" <method> parameter is not one of the valid options: serial, matrix, histogram, spharmonic")

            print( f"Processor {j}", np.max(modelp.rolling_Theta_evens)) 
            padfsum_evens += np.copy(modelp.rolling_Theta_evens)
            padfsum_odds += np.copy(modelp.rolling_Theta_odds)
            #sphvol_evens += np.copy(modelp.sphvol_evens)
            #sphvol_odds += np.copy(modelp.sphvol_odds)

        #return_dict[j] = [padfsum_evens, padfsum_odds, sphvol_evens, sphvol_odds]
        return_dict[j] = [padfsum_evens, padfsum_odds]



if __name__ == '__main__':
    #
    # set up parameter class
    #
    p = params.paramsMODEL()
    #print("pypadf version ",p.version,"\n")

    #
    # Read input parameters from a file
    #
    p.read_config_file()


    # modelp = pmp.ModelPADF()
    modelp = fmp.ModelPadfCalculator()

    #
    # This is the directory that contains the root directory for input/output files
    # (don't forget the final "/". If using windows may need "\\" for each one.)
    #
    modelp.root = p.path_to_string(p.outpath.parents[0])+"/" 

    #
    # this is a directory for this specific project
    #
    modelp.project = p.outpath.name+"/"
    modelp.outpath = p.outpath.name+"/"

    #
    # a meaningful sample tag
    #
    modelp.tag = p.tag

    #
    # name of .xyz file containing all the atomic coordinates
    #
    modelp.use_supercell = p.use_supercell
    modelp.subjectxyz = p.subjectxyz
    if p.use_supercell == True:
        modelp.supercellxyz = p.supercellxyz
    else:
        modelp.supercellxyz = p.subjectxyz 

    #modelp.subject_number = -1

    # probe radius: defines the neighbourhood around each atom to correlate
    modelp.rmax = p.rmax

    # "Number of real-space radial samples"
    modelp.nr = p.nr

    # number of angular bins in the final PADF function
    modelp.nth = p.nth
    modelp.nthvol = p.nthvol
    modelp.phivol = p.nphivol

    # min and max spherical harmonic order to use in the 'spharmonic' calculation
    modelp.nlmin = p.nlmin
    modelp.nl    = p.nl

    # Scale the radial correlations by this power, i.e. r^(r_power)
    modelp.r_power = p.r_power

    modelp.convergence_check_flag = p.check_convergence
    modelp.convergence_target = p.convergence_target

    # Use the atomic weights
    modelp.use_atom_weights = p.use_atom_weights


    '''
    Calculation mode.
    'rrprime' :     Calculate the r = r' slice

    # Under development:
    'rrtheta' :     Calculate slices through Theta(r,r',theta)
                    slice frequency given by probe_theta_bin
    'stm'     :     Calculate Theta(r,r',theta) and send directly to a 
                    numpy matrix (Straight-To-Matrix). Can't be rebinned
                    at a later date
    '''
    modelp.mode = p.mode

    #
    # set processor number
    #
    # implemented for spherical harmonic calc only
    modelp.nthreads = p.nthreads
    modelp.processor_num = p.nthreads


    #
    # set the level of output.
    # 0 - clean up the project folder
    # 1 - leave the intermediate arrays etc
    # will be extended to logging output in the future
    modelp.verbosity = p.verbosity

    modelp.subcellsize = p.subcellsize

    #
    # save parameters to file
    #
    modelp.write_all_params_to_file()

    #
    # Run the calculation!
    #
    # pmp.ModelPADF.run(modelp)
    if modelp.subcellsize < 0.0:
        print( "Calculation will NOT be split into subcells")

        if p.method == 'serial':
            fmp.ModelPadfCalculator.run_fast_serial_calculation(modelp)
        elif p.method=='matrix':
            fmp.ModelPadfCalculator.run_fast_matrix_calculation(modelp)
        elif p.method=='histogram':
            fmp.ModelPadfCalculator.run_histogram_calculation(modelp)   
        elif p.method == 'spharmonic':
            fmp.ModelPadfCalculator.run_spherical_harmonic_calculation(modelp)   
        else:
            print(" <method> parameter is not one of the valid options: serial, matrix, histogram, spharmonic")
    else:

        modelp.processor_num = 1 #HANDLE THE THREADING IN THIS SCRIPT, ENURE FAST MODEL PADF DOES NOT USE THREADING
        modelp.parameter_check()
        modelp.write_all_params_to_file()
        modelp.verbosity = 0

        modelp.subject_atoms, modelp.extended_atoms = modelp.subject_target_setup()  # Sets up the atom positions of the subject set and supercell
        modelp.box_dimensions_from_subject_atoms()
        na, nb, nc = int(modelp.x_wid/modelp.subcellsize), int(modelp.x_wid/modelp.subcellsize), int(modelp.x_wid/modelp.subcellsize)

        cellindices = []
        for ia in range(na): 
            for ib in range(nb): 
                for ic in range(nc): 
                    cellindices.append( [ia,ib,ic])

        nsubcells = len(cellindices)        
        print( "len(cellindices), na, nb, nc, modelp.x_wid", len(cellindices), na, nb, nc, modelp.x_wid)

        if p.nthreads > 1:
            manager = mp.Manager()            
            return_dict = manager.dict()
    
            cells_per_proc = np.max( [nsubcells // p.nthreads,1])
            remainder = nsubcells%p.nthreads
            print( "cells per proc", cells_per_proc)

            processes = []
            for iproc in range(p.nthreads):
                    if iproc < remainder:
                        cells = cellindices[iproc*cells_per_proc:(iproc+1)*cells_per_proc+1]
                    else:
                        cells = cellindices[iproc*cells_per_proc:(iproc+1)*cells_per_proc]
                    pr = mp.Process(target=thread_subcell_model_padf, \
                                    args=(cells,iproc,copy.deepcopy(modelp),p.method,return_dict))
                    pr.start()
                    processes.append(pr)

            print("joining")
            for pr in processes:
                pr.join()


            print( "Finished the multiprocessing part; Next: sum and save the output") 
            
            modelp.rolling_Theta_evens = return_dict[0][0]+ return_dict[0][1]
            modelp.rolling_Theta_odds = 0.0*return_dict[0][1]
            #modelp.sphvol_evens = return_dict[0][2]+ return_dict[0][3]
            #modelp.sphvol_odds = 0.0*return_dict[0][3]

            for iproc in range(1,p.nthreads):
                if iproc%2==0:
                    modelp.rolling_Theta_evens += return_dict[iproc][0]+return_dict[iproc][1]
                    #modelp.sphvol_evens += return_dict[iproc][2] +  return_dict[iproc][3]
                else:
                    modelp.rolling_Theta_odds += return_dict[iproc][0]+return_dict[iproc][1]
                    #modelp.sphvol_odds += return_dict[iproc][2]+ return_dict[iproc][3]
            
                 
            #np.save(modelp.root + modelp.project + modelp.tag + '_sphvol_odds_tiled', modelp.sphvol_odds)
            #np.save(modelp.root + modelp.project + modelp.tag + '_sphvol_evens_tiled', modelp.sphvol_evens)

 
        else:
           #single thread
            for ia in range(na): 
                for ib in range(nb): 
                    for ic in range(nc): 
                        modelp.limits = np.array([ia*modelp.subcellsize,(ia+1)*modelp.subcellsize, \
                                    ib*modelp.subcellsize,(ib+1)*modelp.subcellsize,ic*modelp.subcellsize,(ic+1)*modelp.subcellsize])   

                        if p.method == 'serial':
                            fmp.ModelPadfCalculator.run_fast_serial_calculation(modelp)
                        elif p.method=='matrix':
                            fmp.ModelPadfCalculator.run_fast_matrix_calculation(modelp)
                        elif p.method=='histogram':
                            fmp.ModelPadfCalculator.run_histogram_calculation(modelp)   
                        elif p.method == 'spharmonic':
                            fmp.ModelPadfCalculator.run_spherical_harmonic_calculation(modelp)   
                        else:
                            print(" <method> parameter is not one of the valid options: serial, matrix, histogram, spharmonic")

                        if (ia==0)and(ib==0)and(ic==0):
                            padfsum_evens = np.copy(modelp.rolling_Theta_evens)
                            padfsum_odds = np.copy(modelp.rolling_Theta_odds)
                        else:
                            padfsum_evens += np.copy(modelp.rolling_Theta_evens)
                            padfsum_odds += np.copy(modelp.rolling_Theta_odds)
       
            modelp.rolling_Theta_evens = padfsum_evens
            modelp.rolling_Theta_odds  = padfsum_odds
 
    np.save(modelp.root + modelp.project + modelp.tag + '_mPADF_total_sum_tiled', modelp.rolling_Theta_odds + modelp.rolling_Theta_evens)
    np.save(modelp.root + modelp.project + modelp.tag + '_mPADF_odds_sum_tiled', modelp.rolling_Theta_odds)
    np.save(modelp.root + modelp.project + modelp.tag + '_mPADF_evens_sum_tiled', modelp.rolling_Theta_evens)
           
