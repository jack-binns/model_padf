"""
Model PADF Runner

@author: andrewmartin, jack-binns
"""
import numpy as np

import parallel_model_padf_0p92 as pmp

# import multiprocessing as mp

if __name__ == '__main__':
    #
    # create an instance of the class with all parameters
    #
    j = 0
    probes = [25.0]
    # for probe in np.arange(2.5, 25.5, 2.0):
    for probe in probes:

        modelp = pmp.ModelPADF()

        #
        # This is the directory that contains the root directory for input/output files
        # (don't forget the final "/". If using windows may need "\\" for each one.)
        #
        modelp.root = "E:\\RMIT\\mofs_saw\\HKUST1_mPADF\\"

        #
        # this is a directory for this specific project
        #
        modelp.project = f"defect_n\\"

        #
        # a meaningful sample tag
        #
        modelp.tag = "hkust1_defect"

        #
        # name of .xyz file containing all the atomic coordinates
        #
        modelp.xyz_name = "hkust1_defect_sc.xyz"  # the xyz file contains the cartesian coords of the crystal
        # modelp.xyz_name = "system_clo_alpha.xyz"  # the xyz file contains the cartesian coords of the crystal
        # structure expanded to include r_probe

        #
        # crystallographic file with unit cell information
        #
        modelp.subject_atom_name = "hkust1_defect_as.xyz"  # the cif containing the asymmetric unit
        # Selects the number of probe atoms selected from asymmetric unit. -1 selects all
        modelp.subject_number = -1

        #
        # Unit cell parameters (a,b,c,alpha,beta,gamma)
        #
        modelp.ucds = [800.00, 800.00, 800.00, 90.0000, 90.0000, 90.0000]

        # probe radius: defines the neighbourhood around each atom to correlate
        modelp.rmax = probe

        # "Number of real-space radial samples"
        modelp.nr = 100

        # number of angular bins in the final PADF function
        modelp.nth = 180

        # Scale the radial correlations by this power, i.e. r^(r_power)
        modelp.r_power = 2

        '''
        Convergence mode.
        Set flag to True to take advantage of convergence routines
        set convergence_target to the desired cosine similarity between
        loop n-1 and n
        '''
        modelp.convergence_check_flag = False
        modelp.convergence_target = 0.999

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
        # modelp.mode = 'rrprime'
        modelp.mode = 'stm'

        #
        # Include four body terms (True/False)
        #
        modelp.fourbody = False

        #
        # save parameters to file
        #
        modelp.write_all_params_to_file()

        #
        # set processor number
        #
        modelp.processor_num = 5

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
        j = j + 1