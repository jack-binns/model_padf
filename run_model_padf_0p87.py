"""
Model PADF Runner

@author: andrewmartin, jack-binns
"""

import parallel_model_padf_0p87 as pmp

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
    # modelp.root = "/home/jack/python/model_padf/"
    modelp.root = "G:\\python\\model_padf\\ben\\"
    # modelp.root = '/media/jack/Storage/python/model_padf/'

    #
    # this is a directory for this specific project
    #
    modelp.project = "para_testing\\"

    #
    # name of .xyz file containing all the atomic coordinates
    #
    modelp.xyz_name = "para_testing_dlattice.xyz"  # the xyz file contains the cartesian coords of the crystal
    # structure expanded to include r_probe

    #
    # crystallographic file with unit cell information
    #
    modelp.subject_atom_name = "para_testing_edit.cif"  # the cif containing the asymmetric unit. This often needs to
    # be edited for PDB CIFs hence '_edit' here
    # modelp.subject_number = 48  # the number of random atoms in the subject atoms group
    modelp.subject_number = -1
    # that will be used as probe atoms

    #
    # Unit cell parameters (a,b,c,alpha,beta,gamma)
    #
    modelp.ucds = [100, 100, 100, 90.0000, 90.0000, 90.0000]

    # probe radius: defines the neighbourhood around each atom to correlate
    modelp.r_probe = 20.0

    # angular pixel (bin) width in the final PADF function
    modelp.angular_bin = 2.0

    # radial pixel (bin) width in the final PADF function
    modelp.r_dist_bin = 0.1

    # Scale the radial correlations by this power, i.e. r^(r_power)
    modelp.r_power = 2

    # If you wish to compute the final PADFs from a
    # partial data set use this flag and input the loop
    # at which the data is converged (check the _cosim.dat plot)
    modelp.convergence_check_flag = True
    modelp.convergence_target = 0.999999

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
    modelp.mode = 'rrprime'
    # modelp.mode = 'stm'

    '''
    theta bin size for constructing full r, r', theta matrix
    '''
    modelp.probe_theta_bin = 20.0

    #
    # Include four body terms (True/False)
    #
    modelp.fourbody = True

    #
    # save parameters to file
    #
    modelp.write_all_params_to_file()

    #
    # set processor number
    #
    modelp.processor_num = 6

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
