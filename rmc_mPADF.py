"""
RMC Model PADF Runner

@author: andrewmartin, jack-binns
"""
import matplotlib.pyplot as plt
import numpy as np

import parallel_model_padf_0p92 as pmp
import rmc_utils

# import multiprocessing as mp
import utils

if __name__ == '__main__':
    #
    # create an instance of the class with all parameters
    #
    j = 0
    jit_vec = 0.025
    chisq = 0
    rmc_runs = 1000
    best_j = 0
    best_sim = 1000
    sims = []
    while j < rmc_runs:
        modelp = pmp.ModelPADF()
        modelp.root = "E:\\RMIT\\mofs_saw\\ZIF8_mPADF"
        modelp.project = f"ordered\\"
        modelp.tag = f"ucube_{j}"
        modelp.nth = 180
        modelp.r_power = 2
        modelp.subject_number = -1
        modelp.rmax = 5
        modelp.nr = 10
        modelp.mode = 'stm'
        modelp.fourbody = True
        # modelp.write_all_params_to_file()
        modelp.convergence_check_flag = False
        modelp.processor_num = 8
        modelp.verbosity = 0
        if j == 0:
            print(f'Iteration {j}...')
            modelp.xyz_name = "ucube_target.xyz"
            modelp.subject_atom_name = "ucube_target.xyz"
            pmp.ModelPADF.run(modelp)  # generates target mPADF
            rmccell = rmc_utils.rmcCell()
            rmccell.ucds = [5.0, 5.0, 5.0, 90, 90, 90]
            rmccell.atom_number = 6
            rmccell.formula = [('Rb', 8)]
            # rmccell.gen_seed()
            # rmccell.write_xyz(f'{modelp.root}{modelp.project}ucube_{j}.xyz')
            # now let's generate the first j mPADF
            modelp.tag = f"iter_{j}"
            modelp.xyz_name = f'ucube_{j}.xyz'
            modelp.subject_atom_name = f'ucube_{j}.xyz'
            pmp.ModelPADF.run(modelp)  # generate iter 0 mPADF
            ar0 = np.load(f'{modelp.root}{modelp.project}ucube_0_mPADF_total_sum.npy')
            ar1 = np.load(f'{modelp.root}{modelp.project}iter_{j}_mPADF_total_sum.npy')
            # plt.plot(ar0[:, 0], ar0[:, 1])
            # plt.plot(ar1[:, 0], ar1[:, 1])
            # plt.show()
            iter_sim = utils.cossim_measure(ar0[:, 1], ar1[:, 1])
            # iter_sim = utils.calc_rfactor(ar0, ar1)
            best_sim = iter_sim
            sims.append(iter_sim)
            print(f'Iteration {j}  ---   sim = {iter_sim}')
        else:
            print(f'Iteration {j}...')
            # Read in previous iteration
            itmc = rmc_utils.rmcCell()
            itmc.read_xyz(f'{modelp.root}{modelp.project}ucube_{best_j}.xyz')
            itmc.jitter(jit_mag=jit_vec)
            itmc.write_xyz(f'{modelp.root}{modelp.project}ucube_{j}.xyz')
            # now let's generate the first j mPADF
            modelp.tag = f"iter_{j}"
            modelp.xyz_name = f'ucube_{j}.xyz'
            modelp.subject_atom_name = f'ucube_{j}.xyz'
            pmp.ModelPADF.run(modelp)  # generate iter mPADF
            ar0 = np.load(f'{modelp.root}{modelp.project}ucube_0_mPADF_total_sum.npy')
            ar1 = np.load(f'{modelp.root}{modelp.project}iter_{j}_mPADF_total_sum.npy')
            # plt.plot(ar0[:, 0], ar0[:, 1])
            # plt.plot(ar1[:, 0], ar1[:, 1])
            # plt.show()
            iter_sim = utils.cossim_measure(ar0[:, 1], ar1[:, 1])
            # iter_sim = utils.calc_rfactor(ar0, ar1)
            sims.append(iter_sim)
            print(f' {iter_sim} > {best_sim} ? ')
            if iter_sim > best_sim:
                best_j = j
                best_sim = iter_sim
            print(f'Iteration {j}  ---   sim = {iter_sim}')
            print(f'Best j  ---  {best_j}')

        j = j + 1
        print(f'Best sim  ---   sim = {best_sim}')
    plt.plot(sims)
    plt.show()
