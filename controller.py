"""
Model PADF MD controller

@author: jack-binns, andrewmartin
"""
import glob
import shutil

import numpy as np

import utils

import fast_model_padf as fmp
import random
import matplotlib.animation


class MPADFController:

    def __init__(self, root='', project='', tag='', rmin=0, rmax=-1, nr=-1, nth=-1):
        # File control variables
        self.root = root
        self.project = project
        self.tag = tag
        self.subject_set_manifest = []
        self.supercell_set_manifest = []
        self.frame_number = 0

        # Calculation variables
        self.rmin = rmin
        self.rmax = rmax
        self.nr = nr
        self.nth = nth
        self.seed = 888

        # Meta variables for whole run
        self.total_counts = 0

        self.convergence_target = 1.0
        self.com_cluster_flag = False
        self.com_radius = 0.0
        self.verbosity = 1

    def generate_calculation_plan(self, stringtag='_sc.xyz'):
        """
        Organise the input xyz files for the serial mPADF caluclations
        :param stringtag: suffix tag for the relevant xyz - should be set if you're using results from clustering
        calculations
        :return:
        """
        for k in range(self.frame_number):
            print(f'{k} / {self.frame_number}')
            self.subject_set_manifest.append(f'{self.tag}_frame_{k}{stringtag}')
            self.supercell_set_manifest.append(f'{self.tag}_frame_{k}{stringtag}')
        manifests = list(zip(self.supercell_set_manifest, self.subject_set_manifest))
        random.seed(self.seed)
        random.shuffle(manifests)
        self.supercell_set_manifest, self.subject_set_manifest = zip(*manifests)
        # print(self.subject_set_manifest[0])
        # print(self.supercell_set_manifest[0])

    def consolidate_md_results(self, clean_folder: bool = False,
                               animate: bool = False,
                               total_string_tag: str = '*_mPADF_total_sum.npy',
                               odd_string_tag: str = '*_mPADF_odds_sum.npy',
                               even_string_tag: str = '*_mPADF_evens_sum.npy'):
        """
        Combine the mPADFs from each frame of an MD trajectory and combine them. Options to clean the folder up and make
        a movie.
        :param clean_folder: bool, delete files in work folder once results have been consolidated
        :param animate: bool, generate a movie for presentations
        :param total_string_tag: str, suffix to denote the total sum from each MD frame,
        unlikely to be changed from default value
        :param odd_string_tag:  str, suffix to denote the odd contributions from each MD frame,
        unlikely to be changed from default value
        :param even_string_tag:  str, suffix to denote the even contributions from each MD frame,
        unlikely to be changed from default value
        :return:
        """
        mpadf_list = utils.sorted_nicely(glob.glob(f'{self.root}{self.project}\\{total_string_tag}'))
        odds_list = glob.glob(f'{self.root}{self.project}\\{odd_string_tag}')
        evens_list = glob.glob(f'{self.root}{self.project}\\{even_string_tag}')
        # print(mpadf_list[0], odds_list[0], evens_list[0])

        shutil.copyfile(src=f'{self.root}{self.project}{self.tag}_0_mPADF_param_log.txt',
                        dst=f'{self.root}{self.project}{self.tag}_trajectory_mPADF_param_log.txt')

        trajectory_sum = np.zeros((self.nr, self.nr, self.nth))
        trajectory_odds = np.zeros((self.nr, self.nr, self.nth))
        trajectory_evens = np.zeros((self.nr, self.nr, self.nth))

        for k, frame_padf in enumerate(mpadf_list):
            frame_sum = np.load(frame_padf)
            frame_odd = np.load(odds_list[k])
            frame_even = np.load(evens_list[k])
            trajectory_sum += frame_sum
            trajectory_odds += frame_odd
            trajectory_evens += frame_even
            self.total_counts += np.sum(np.sum(np.sum(frame_sum, axis=2), axis=1), axis=0)
            # Add verbosity check and delete frame mPADFs

        np.save(f'{self.root}{self.project}{self.tag}_trajectory_mPADF_total_sum.npy', trajectory_sum)
        np.save(f'{self.root}{self.project}{self.tag}_trajectory_mPADF_odds_sum.npy', trajectory_odds)
        np.save(f'{self.root}{self.project}{self.tag}_trajectory_mPADF_evens_sum.npy', trajectory_evens)

        print(f'<controller.consolidate_md_results> total trajectory intensity {self.total_counts}')

    def run_serial_mPADF_calc(self, starting_frame: int = 0):
        print(f'<controller.run_serial_mPADF_calc> Beginning mPADF calculation on MD trajectory')
        for k in np.arange(start=starting_frame, stop=self.frame_number):
            print(f'<controller.run_serial_mPADF_calc> Starting MD frame {k}')
            mpc = fmp.ModelPadfCalculator()
            mpc.root = self.root
            mpc.project = self.project
            mpc.tag = f'{self.tag}_{k}'
            mpc.supercell_atoms = self.supercell_set_manifest[k]
            mpc.subject_atoms = self.subject_set_manifest[k]
            mpc.rmax = self.rmax
            mpc.nr = self.nr
            mpc.nth = self.nth
            mpc.verbosity = self.verbosity
            mpc.convergence_target = self.convergence_target
            mpc.convergence_check_flag = True
            mpc.com_cluster_flag = True
            mpc.com_radius = self.com_radius
            mpc.write_all_params_to_file()
            fmp.ModelPadfCalculator.run_fast_serial_calculation(mpc)
