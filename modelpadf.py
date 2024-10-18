
import numpy as np
#import array
import os
import params.paramsMODEL as params
import fast_model_padf as fmp


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

    #
    # a meaningful sample tag
    #
    modelp.tag = p.tag

    #
    # name of .xyz file containing all the atomic coordinates
    #
    modelp.subject_atoms = p.subjectxyz
    if p.use_supercell == True:
        modelp.supercell_atoms = p.supercellxyz
    else:
        modelp.supercell_atoms = p.supercellxyz 

    #modelp.subject_number = -1

    # probe radius: defines the neighbourhood around each atom to correlate
    modelp.rmax = p.rmax

    # "Number of real-space radial samples"
    modelp.nr = p.nr

    # number of angular bins in the final PADF function
    modelp.nth = p.nth
    modelp.nthvol = p.nthvol
    modelp.phivol = p.nphivol

    # Scale the radial correlations by this power, i.e. r^(r_power)
    modelp.r_power = p.r_power

    modelp.convergence_check_flag = p.check_convergence
    modelp.convergence_target = p.convergence_target

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
    # NOT IMPLEMENTED
    modelp.processor_num = p.nthreads


    #
    # set the level of output.
    # 0 - clean up the project folder
    # 1 - leave the intermediate arrays etc
    # will be extended to logging output in the future
    modelp.verbosity = p.verbosity

    #
    # save parameters to file
    #
    modelp.write_all_params_to_file()

    #
    # Run the calculation!
    #
    # pmp.ModelPADF.run(modelp)
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


