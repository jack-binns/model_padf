"""
modelpadf.py

Compute a pair-angle distribution function directly from an atomic model

Authors:
Jack Binns
Patrick Adams
Andrew Martin
"""

import numpy as np
import os
import params.paramsMODEL as params
import fast_model_padf2 as fmp
import multiprocessing as mp
import copy
import time



if __name__ == '__main__':

    start = time.time()

    #
    # set up parameter class
    #
    p = params.paramsMODEL()
    #print("pypadf version ",p.version,"\n")

    print(" MODELPADF.py ")
    print( "version :", p.version, "\n") 

    #
    # Read input parameters from a file
    #
    p.read_config_file()


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
    'stm'     :     Calculate 3D Theta(r,r',theta)
    
    'rrprime' :     Calculate the r = r' slice

    # Under development:
    'rrtheta' :     Calculate slices through Theta(r,r',theta)
                    slice frequency given by probe_theta_bin
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
    modelp.calculate_model_padf(method=p.method, nthreads=p.nthreads)


    print( "Model PADF calculation took ", time.time()-start, "seconds")

