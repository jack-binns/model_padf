"""

Runner script for automating model PADF calculations

controller module generates frames and passes variables from here into
the fast_model_padf module.

TO DO:
 - parallelise
 - total calculation timings
 - total calculation contact tracking

"""

import controller

if __name__ == '__main__':
    num_frames = 10

    cont = controller.MPADFController()
    cont.root = 'C:\\rmit\\dlc\\model_padf\\md\\'
    cont.project = '192DLC\\'
    cont.tag = '192DLC'

    # probe radius: defines the neighbourhood around each atom to correlate
    cont.rmax = 20.0
    # Number of real-space radial samples
    cont.nr = 128
    # number of angular bins in the final PADF function
    cont.nth = 180

    cont.frame_number = num_frames
    cont.convergence_target = 0.9
    cont.com_cluster_flag = True
    cont.com_radius = 10.0

    # Generate the file paths
    cont.generate_calculation_plan()
    cont.run_serial_mPADF_calc()
