"""
Batch Model PADF Runner

@authors: andrewmartin, jack-binns
"""

import parallel_model_padf_0p89 as pmp

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
    # modelp.root = "G:\\RMIT\\cov2-spike\\"
    modelp.root = "F:\\euxfel_2304\\model_padf\\"
    # modelp.root = "G:\\RMIT\\indexing_padf\\"
    # modelp.root = "G:\\RMIT\\orientation\\work\\lcp_simulation\\m_padf\\"
    # modelp.root = '/media/jack/Storage/python/model_padf/'

    #
    # this is a directory for this specific project
    #
    # modelp.project = "pm-3m\\"
    # modelp.project = "closed\\"
    # modelp.project = "1vds_300res_5nm\\"
    modelp.project = "1vds_100res_5nm\\"
    # modelp.project = "DESRES-Trajectory_sarscov2-11021566-structure\\"
    # modelp.project = "para_testing\\"

    #
    # name of .xyz file containing all the atomic coordinates
    #
    modelp.xyz_name = "1vds_super_100res.xyz"  # the xyz file contains the cartesian coords of the crystal
    # modelp.xyz_name = "system_clo_alpha.xyz"  # the xyz file contains the cartesian coords of the crystal
    # structure expanded to include r_probe

    #
    # crystallographic file with unit cell information
    #
    modelp.subject_atom_name = "1vds_ghst_100res.xyz"  # the cif containing the asymmetric unit
    # Selects the number of probe atoms selected from asymmetric unit. -1 selects all
    modelp.subject_number = -1

    #
    # Unit cell parameters (a,b,c,alpha,beta,gamma)
    #
    modelp.ucds = [79.192, 79.192, 37.933, 90.0000, 90.0000, 90.0000]

    # probe radius: defines the neighbourhood around each atom to correlate
    modelp.r_probe = 100.0

    # angular pixel (bin) width in the final PADF function
    modelp.angular_bin = 1.0

    # radial pixel (bin) width in the final PADF function
    modelp.r_dist_bin = 0.1

    # Scale the radial correlations by this power, i.e. r^(r_power)
    modelp.r_power = 2

    '''
    Convergence mode.
    Set flag to True to take advantage of convergence routines
    set convergence_target to the desired cosine similarity between
    loop n-1 and n
    '''
    modelp.convergence_check_flag = True
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

    '''
    theta bin size for constructing full r, r', theta matrix
    '''
    modelp.probe_theta_bin = 20.0

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
    modelp.processor_num = 4

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
