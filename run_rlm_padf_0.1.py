"""
Model PADF Runner

@author: andrewmartin, jack-binns
"""

import rlm_padf_01 as pmp

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
    # modelp.root = "G:\\python\\model_padf\\ben\\"
    modelp.root = "G:\\RMIT\\cluster_simulation\\"
    # modelp.root = '/media/jack/Storage/python/model_padf/'

    #
    # this is a directory for this specific project
    #
    modelp.project = "n_3\\"

    #
    # name of .xyz file containing all the atomic coordinates
    #
    modelp.xyz_name = "n3_cluster.xyz"  # the xyz file contains the cartesian coords of the crystal
    # structure expanded to include r_probe

    #
    # crystallographic file with unit cell information
    #
    modelp.subject_atom_name = "n3_cluster.xyz"  # the cif containing the asymmetric unit
    # Selects the number of probe atoms selected from asymmetric unit. -1 selects all
    modelp.subject_number = -1

    #
    # Unit cell parameters (a,b,c,alpha,beta,gamma)
    #
    modelp.ucds = [200, 200, 200, 90.0000, 90.0000, 90.0000]

    # probe radius: defines the neighbourhood around each atom to correlate
    modelp.r_probe = 200.0

    # angular pixel (bin) width in the final PADF function
    modelp.angular_bin = 2.0

    # radial pixel (bin) width in the final PADF function
    modelp.r_dist_bin = 1.0

    # Scale the radial correlations by this power, i.e. r^(r_power)
    modelp.r_power = 2

    '''
    Convergence mode.
    Set flag to True to take advantage of convergence routines
    set convergence_target to the desired cosine similarity between
    loop n-1 and n
    '''
    modelp.convergence_check_flag = True
    modelp.convergence_target = 0.9999

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
    modelp.verbosity = 1

    #
    # Run the calculation!
    #
    pmp.ModelPADF.run(modelp)
