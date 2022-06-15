"""
Model PADF MD controller

@author: andrewmartin, jack-binns
"""
import fast_model_padf as fmp
import glob


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

        self.convergence_target = 1.0
        self.com_cluster_flag = False
        self.com_radius = 0.0

    def generate_calculation_plan(self):
        for k in range(self.frame_number):
            print(f'{k} / {self.frame_number}')
            self.subject_set_manifest.append(f'{self.tag}_frame_{k}.xyz')
            self.supercell_set_manifest.append(f'{self.tag}_frame_{k}.xyz')
        print(self.subject_set_manifest)
        print(self.supercell_set_manifest)

    def run_serial_mPADF_calc(self):
        print(f'<controller.run_serial_mPADF_calc> Beginning mPADF calculation on MD trajectory')
        for k in range(self.frame_number):
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
            mpc.convergence_target = self.convergence_target
            mpc.convergence_check_flag = True
            mpc.com_cluster_flag = True
            mpc.com_radius = self.com_radius
            mpc.write_all_params_to_file()
            fmp.ModelPadfCalculator.run_fast_serial_calculation(mpc)
