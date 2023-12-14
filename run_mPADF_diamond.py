"""
Model PADF Runner

@author: andrewmartin, jack-binns
"""
import numpy as np

#import parallel_model_padf_0p94 as pmp
import fast_model_padf as fmp

# import multiprocessing as mp

if __name__ == '__main__':
    #
    # create an instance of the class with all parameters
    #
    j = 0
    probes = [40.0]
    # for probe in np.arange(2.5, 25.5, 2.0):
    for probe in probes:

        # modelp = pmp.ModelPADF()
        modelp = fmp.ModelPadfCalculator()

        #
        # This is the directory that contains the root directory for input/output files
        # (don't forget the final "/". If using windows may need "\\" for each one.)
        #
        modelp.root = "/Users/andrewmartin/cloudstor/Work/Research/SF/Projects/2023/padf_models/diamond/"

        #
        # this is a directory for this specific project
        #
        modelp.project = f"results/"

        #
        # a meaningful sample tag
        #
        modelp.tag = "diamond9MATRIX"

        #
        # name of .xyz file containing all the atomic coordinates
        #
        modelp.supercell_atoms = "diamond_9_9_9.xyz"  # the xyz file contains the cartesian coords of the crystal
        # modelp.xyz_name = "system_clo_alpha.xyz"  # the xyz file contains the cartesian coords of the crystal
        # structure expanded to include r_probe

        #
        # crystallographic file with unit cell information
        #
        modelp.subject_atoms = "diamond_1_1_1.xyz"  # the cif containing the asymmetric unit
        # Selects the number of probe atoms selected from asymmetric unit. -1 selects all
        modelp.subject_number = -1

        # probe radius: defines the neighbourhood around each atom to correlate
        modelp.rmax = probe

        # "Number of real-space radial samples"
        modelp.nr = 256

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
        modelp.convergence_check_flag = True
        modelp.convergence_target = 0.99

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
        # set processor number
        #
        modelp.processor_num = 4
        #modelp.update_pool()
        #
        # set the level of output.
        # 0 - clean up the project folder
        # 1 - leave the intermediate arrays etc
        # will be extended to logging output in the future
        modelp.verbosity = 0

        #
        # save parameters to file
        #
        modelp.write_all_params_to_file()

        #
        # Run the calculation!
        #
        # pmp.ModelPADF.run(modelp)
        #fmp.ModelPadfCalculator.run_fast_serial_calculation(modelp)
        fmp.ModelPadfCalculator.run_fast_matrix_calculation(modelp)
        #fmp.ModelPadfCalculator.run_fast_mp_calculation(modelp)
        j = j + 1
