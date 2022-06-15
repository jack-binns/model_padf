"""
Model PADF Runner

@author: andrewmartin, jack-binns
"""
import numpy as np

import parallel_model_padf_0p94_unstable as pmp

# import multiprocessing as mp

if __name__ == '__main__':
    #
    # create an instance of the class with all parameters
    #
    modelp = pmp.ModelPADF()

    #
    # This is the directory that contains the root directory for input/output files
    # (don't forget the final "/". If using windows may need "\\" for each one.)
    #
    modelp.root = "C:\\rmit\\dlc\\model_padf\\"

    #
    # this is a directory for this specific project
    #
    modelp.project = f"k_phase_5disks\\"

    #
    # a meaningful sample tag
    #
    modelp.tag = "dlc_5disks"

    #
    # Subject atoms embedded in supercell
    #
    modelp.subject_atoms = "dlc_5disks.xyz"  # the cif containing the asymmetric unit
    # Selects the number of probe atoms selected from asymmetric unit. -1 selects all
    modelp.subject_number = -1

    #
    # Supercell
    #
    modelp.supercell_atoms = "dlc_5disks.xyz"  # the xyz file contains the cartesian coords of the crystal
    # modelp.xyz_name = "system_clo_alpha.xyz"  # the xyz file contains the cartesian coords of the crystal
    # structure expanded to include r_probe

    # probe radius: defines the neighbourhood around each atom to correlate
    modelp.rmax = 40.0

    # "Number of real-space radial samples"
    modelp.nr = 120

    # number of angular bins in the final PADF function
    modelp.nth = 180

    '''
        Convergence mode.
        Set flag to True to take advantage of convergence routines
        set convergence_target to the desired cosine similarity between
        loop n-1 and n
        '''
    modelp.convergence_check_flag = False
    modelp.convergence_target = 1e-10

    '''
        Calculation mode.
        'rrprime' :     Calculate the r = r' slice
        '''

    modelp.mode = 'stm'

    #
    # save parameters to file
    #
    modelp.write_all_params_to_file()

    #
    # set processor number
    #
    modelp.processor_num = 2

    #
    # set the level of output.
    # 0 - clean up the project folder
    # 1 - leave the intermediate arrays etc
    # will be extended to logging output in the future
    modelp.verbosity = 0

    #
    # Run the calculation!
    #
    pmp.ModelPADF.run(modelp)
