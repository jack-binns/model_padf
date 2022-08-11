"""
Fast Model PADF Calculator

@author: andrewmartin, jack-binns
"""
import shutil
import numpy as np
import time
import multiprocessing as mp
import math as m
import matplotlib.pyplot as plt
import os
import utils as u


class ModelPadfCalculator:

    def __init__(self):

        self.root = ""
        self.project = ""
        self.tag = ""
        self.supercell_atoms = ""  # the xyz file contains the cartesian coords of the crystal structure expanded
        # to include r_probe
        self.subject_atoms = ""  # the cif containing the asymmetric unit
        # probe radius
        self.rmin = 0.0
        self.rmax = 10.0
        self.nr = 100
        self.r_dist_bin = 1.0
        self.nth = 180
        self.angular_bin = 1.0
        self.r_power = 2
        self.convergence_check_flag = False
        self.r12_reflection = True
        self.mode = 'stm'
        self.dimension = 3
        self.processor_num = 2
        self.Pool = mp.Pool(self.processor_num)
        self.loops = 0
        self.verbosity = 0
        self.Theta = np.zeros(0)
        self.rolling_Theta = np.zeros(0)
        self.rolling_Theta_odds = np.zeros(0)
        self.rolling_Theta_evens = np.zeros(0)
        self.n2_contacts = []
        self.interatomic_vectors = []
        self.subject_atoms = []
        self.subject_number = 1
        self.raw_extended_atoms = []
        self.extended_atoms = []
        self.loop_similarity_array = []
        self.convergence_target = 1.0
        self.converged_loop = 0
        self.converged_flag = False
        self.com_cluster_flag = False
        self.com_radius = 0.0
        self.total_contribs = 0
        self.calculation_time = 0.0
        self.percent_milestones = np.zeros(0)
        self.iteration_times = np.zeros(0)

    def parameter_check(self):
        """
        Check and print calculation parameters and a welcome
        """
        self.r_dist_bin = self.rmax / self.nr
        self.angular_bin = 180 / self.nth
        print('.......................................')
        print('....Atomistic Model PADF Calculator....')
        print('.......................................')
        print(f'<parameter_check>: Real space parameters...')
        print(f'<parameter_check>: rmax : {self.rmax} ')
        print(f'<parameter_check>: nr : {self.nr} ')
        print(f'<parameter_check>: r_dist_bin : {self.r_dist_bin} ')
        print(f'<parameter_check>: Angular parameters...')
        print(f'<parameter_check>: nth : {self.nth}')
        print(f'<parameter_check>: angular_bin : {self.angular_bin}')
        print(f'<parameter_check>: model PADF dimensions: {self.nr, self.nr, self.nth}')

    def get_dimension(self):
        """
        Sets the dimension of the calculation. Mostly just important if you're calculating
        a slice rather than the full r, r', theta matrix.
        :return:
        """
        if self.mode == 'rrprime':
            print("<get_dimension>: Calculating r = r' slice")
            self.dimension = 2
        elif self.mode == 'rrtheta':
            print("<get_dimension>: Calculating r, r', theta slices")
            # run_rrtheta(model_padf_instance)
            pass
        elif self.mode == 'stm':
            print("<get_dimension>: Calculating Theta(r,r',theta) directly...")
            self.dimension = 3
        return self.dimension

    def write_all_params_to_file(self):
        """
        Writes all the input parameters to a log file
        :param name:
        :param script:
        :return:
        """
        path = self.root + self.project
        if not os.path.isdir(path):
            print('<write_all_params_to_file>: Moving files...')
            os.mkdir(path)
            src = self.root + self.supercell_atoms
            dst = self.root + self.project + self.supercell_atoms
            shutil.copy(src, dst)
            src = self.root + self.subject_atoms
            dst = self.root + self.project + self.subject_atoms
            shutil.copy(src, dst)
        else:
            f = open(self.root + self.project + f'{self.tag}_mPADF_param_log.txt', 'w')
            f.write("# log of input parameters for model PADF calculation\n")
            f.write('["model PADF"]\n')
            a = self.__dict__
            for d, e in a.items():
                f.write(d + " = " + str(e) + "\n")
            f.close()

    def write_calculation_summary(self):
        """
        Writes out a summary of the calculation
        :return:
        """
        with open(self.root + self.project + f'{self.tag}_calculation_log.txt', 'w') as f:
            f.write(f'Calculation time: {self.calculation_time} s\n')
            f.write(f'Total number of interatomic vectors {len(self.interatomic_vectors)}\n')
            f.write(f'Total number of atoms in system {len(self.extended_atoms)}\n')
            f.write(f'Total number of contributing contacts {self.total_contribs}\n')
        np.savetxt(self.root + self.project + f'{self.tag}_similarity_log.txt', np.array(self.loop_similarity_array))

    def subject_target_setup(self):
        """
        Handlers to read in the subject atoms (a.k.a. asymmetric unit) and the extended atoms (environment)
        :return:
        """
        print(f'<subject_target_setup> Reading in subject set...')
        self.subject_atoms = u.read_xyz(
            f'{self.root}{self.project}{self.subject_atoms}')  # Read the full asymmetric unit
        if self.com_cluster_flag:
            self.subject_atoms = self.clean_subject_atoms()

        print(f'<subject_target_setup> Reading in extended atom set...')
        self.raw_extended_atoms = u.read_xyz(
            f'{self.root}{self.project}{self.supercell_atoms}')  # Take in the raw environment atoms
        self.extended_atoms = self.clean_extended_atoms()  # Trim to the atoms probed by the subject set
        # if self.com_cluster_flag:
        #     self.output_cluster_xyz()       ## WRITE OUT THE CLUSTER GEOMETRIES
        u.output_reference_xyz(self.subject_atoms, path=f'{self.root}{self.project}{self.tag}_clean_subject_atoms.xyz')
        u.output_reference_xyz(self.extended_atoms,
                               path=f'{self.root}{self.project}{self.tag}_clean_extended_atoms.xyz')
        return self.subject_atoms, self.extended_atoms

    def cycle_assessment(self, k, start_time):
        # Measure internal convergence
        if k > 1:
            loop_cos = u.cossim_measure(self.rolling_Theta_odds, self.rolling_Theta_evens)
            # print(f"{loop_cos=}")
            self.loop_similarity_array.append([k, loop_cos])
            if self.verbosity == 1:
                print(
                    f"| {k} / {len(self.interatomic_vectors)} | Odd/even cosine similarity == {loop_cos}")
            if k % 100 == 0:
                print(
                    f"| {k} / {len(self.interatomic_vectors)} | Odd/even cosine similarity == {loop_cos}")
                # foo = np.array(self.loop_similarity_array)
                # plt.plot(foo[:, 0], foo[:, 1], '-')
                # plt.show()
        else:
            loop_cos = 0.0
        if loop_cos >= self.convergence_target:
            self.converged_flag = True
        # Estimate remaining time:
        cycle_time = time.time() - start_time
        self.iteration_times[k] = cycle_time
        time_remaining = np.mean(np.array(self.iteration_times[:k + 1])) * (len(self.iteration_times) - k)
        # print(cycle_time)
        # print(time_remaining)
        if self.verbosity == 1:
            if time_remaining > 3600:
                print(
                    f"| {k} / {len(self.interatomic_vectors)} | Estimate {round(time_remaining / 3600, 3)} hr remaining")
            else:
                print(f"| {k} / {len(self.interatomic_vectors)} | Estimate {round(time_remaining, 3)} s remaining")

    def generate_empty_theta(self, shape):
        """
        Sets up the empty Theta matrix
        :return: Empty numpy array Theta
        """
        if shape == 2:
            Theta = np.zeros((int(self.rmax / self.r_dist_bin), int(m.pi / m.radians(self.angular_bin))))
            # print("Creating empty Theta slice :", Theta.shape)
        elif shape == 3:
            Theta = np.zeros(
                (int(self.rmax / self.r_dist_bin), int(self.rmax / self.r_dist_bin),
                 int(m.pi / m.radians(self.angular_bin))))
        else:
            print("<generate_empty_theta>: WARNING: Please supply Theta dimension")
            Theta = np.zeros(0)
        # print("Creating empty Theta :", Theta.shape)
        return Theta

    def clean_extended_atoms(self):
        """
        Trims the length of the extended atoms to the set probed by
        the r_probe and asymmetric unit
        :return:
        """
        print(f'<fast_model_padf.clean_extended_atoms> Trimming atom sets to rmax')
        clean_ex = []
        cluster_subject = []
        if not self.com_cluster_flag:
            for ex_atom in self.raw_extended_atoms:
                for as_atom in self.subject_atoms:
                    diff = u.fast_vec_difmag(ex_atom[0], ex_atom[1], ex_atom[2], as_atom[0], as_atom[1], as_atom[2])
                    if abs(diff) <= self.rmax:
                        clean_ex.append(ex_atom)
                        break
                    else:
                        continue
        elif self.com_cluster_flag:
            x_com = np.mean(self.subject_atoms[:, 0])
            y_com = np.mean(self.subject_atoms[:, 1])
            z_com = np.mean(self.subject_atoms[:, 2])
            print(f'center of mass at {[x_com, y_com, z_com]}')
            for ex_atom in self.raw_extended_atoms:
                diff = u.fast_vec_difmag(ex_atom[0], ex_atom[1], ex_atom[2], x_com, y_com, z_com)
                if abs(diff) <= 2 * self.rmax:
                    clean_ex.append(ex_atom)
                else:
                    continue
            for s_atom in self.subject_atoms:
                diff = u.fast_vec_difmag(s_atom[0], s_atom[1], s_atom[2], x_com, y_com, z_com)
                if abs(diff) <= self.rmax:
                    cluster_subject.append(s_atom)
                else:
                    continue
        clean_ex = np.array(clean_ex)
        print(
            f"<clean_extended_atoms>: Extended atom set has been reduced to {len(clean_ex)} atoms within {self.rmax} radius")
        return np.array(clean_ex)

    def clean_subject_atoms(self):
        """
        Trims the length of the extended atoms to the set probed by
        the r_probe and asymmetric unit
        :return:
        """
        print(f'<fast_model_padf.clean_extended_atoms> Trimming atom sets to rmax {len(self.subject_atoms)} atoms')
        cluster_subject = []
        x_com = np.mean(self.subject_atoms[:, 0])
        y_com = np.mean(self.subject_atoms[:, 1])
        z_com = np.mean(self.subject_atoms[:, 2])
        print(f'center of mass at {[x_com, y_com, z_com]} {self.com_radius}')
        for s_atom in self.subject_atoms:
            diff = u.fast_vec_difmag(s_atom[0], s_atom[1], s_atom[2], x_com, y_com, z_com)
            if abs(diff) <= self.com_radius:
                cluster_subject.append(s_atom)
            else:
                continue
        cluster_subject = np.array(cluster_subject)
        print(
            f"<clean_subject_atoms>: Subject atom set has been reduced to {len(cluster_subject)} atoms within {self.com_radius} radius")
        return np.array(cluster_subject)

    def bin_cor_vec_to_theta(self, cor_vec, fz, array):
        """
        Bin and then add the correlation vector to the
        chunk array
        :param fz: product of atomic nmbers < first approx to Z-weighting
        :param cor_vec: correlation vector length 2 or 3
        :param array: Theta chunk
        :return:
        """
        r_yard_stick = np.arange(self.r_dist_bin, self.rmax + self.r_dist_bin, self.r_dist_bin)
        th_yard_stick = np.arange(0, m.pi, m.radians(self.angular_bin))
        if self.dimension == 2:
            r1_index = (np.abs(r_yard_stick - cor_vec[0])).argmin()
            th_index = (np.abs(th_yard_stick - cor_vec[-1])).argmin()
            index_vec = [r1_index, th_index]
            array[index_vec[0], index_vec[1]] = array[index_vec[0], index_vec[1]] + fz
            if self.r12_reflection:
                array[index_vec[1], index_vec[0]] = array[index_vec[1], index_vec[0]] + fz
        elif self.dimension == 3:
            r1_index = (np.abs(r_yard_stick - cor_vec[0])).argmin()
            r2_index = (np.abs(r_yard_stick - cor_vec[1])).argmin()
            th_index = (np.abs(th_yard_stick - cor_vec[-1])).argmin()
            array[r1_index, r2_index, th_index] = array[r1_index, r2_index, th_index] + fz
            if self.r12_reflection:
                array[r2_index, r1_index, th_index] = array[r2_index, r1_index, th_index] + fz

    def calc_padf_frm_iav(self, k, r_ij):
        start = time.time()
        # print(f'<calc_padf_frm_iav>: Starting calculation on thread {k}...')
        fb_hit_count = 0
        for r_xy in self.interatomic_vectors:
            if np.array_equal(r_ij, r_xy):
                continue
            theta = u.fast_vec_angle(r_ij[0], r_ij[1], r_ij[2], r_xy[0], r_xy[1], r_xy[2])
            fprod = r_ij[4] * r_xy[4]
            self.bin_cor_vec_to_theta([r_ij[3], r_xy[3], theta], fprod, self.rolling_Theta)
            if k % 2 == 0:
                self.bin_cor_vec_to_theta([r_ij[3], r_xy[3], theta], fprod, self.rolling_Theta_evens)
            else:
                self.bin_cor_vec_to_theta([r_ij[3], r_xy[3], theta], fprod, self.rolling_Theta_odds)
            fb_hit_count += 1
        end = time.time()
        # print(f"<calc_padf_frm_iav>: Thread {k} execution time = ", end - start, " seconds")
        # print(f"<calc_padf_frm_iav>: Thread {k} added {fb_hit_count} four-body contacts")
        # Save the Theta array as is:
        self.total_contribs += fb_hit_count
        # print(self.total_contribs, fb_hit_count)
        # np.save(self.root + self.project + self.tag + '_Theta_' + str(k), Theta)

    def pair_dist_calculation(self):
        print(f'<pair_dist_calculation> Calculating pairwise interatomic distances...')
        # interatomic_vectors
        for k, a_i in enumerate(self.subject_atoms):
            if k % int(len(self.subject_atoms) * 0.05) == 0:
                print(f"{k} / {len(self.subject_atoms)}")
            for a_j in self.extended_atoms:
                if not np.array_equal(a_i, a_j):
                    mag_r_ij = u.fast_vec_difmag(a_i[0], a_i[1], a_i[2], a_j[0], a_j[1], a_j[2])
                    r_ij = u.fast_vec_subtraction(a_i[0], a_i[1], a_i[2], a_j[0], a_j[1], a_j[2])
                    r_ij.append(mag_r_ij)
                    r_ij.append(a_i[3] * a_j[3])
                    # print(f'r_ij : {r_ij}')
                    if mag_r_ij < 0.8:
                        print(f'<pair_dist_calculation> Warning: Unphysical interatomic distances detected:')
                        print(f'<pair_dist_calculation> {a_i} {a_j} are problematic')
                    self.n2_contacts.append(mag_r_ij)
                    self.interatomic_vectors.append(r_ij)
        np.array(self.n2_contacts)
        print(f'<pair_dist_calculation> {len(self.interatomic_vectors)} interatomic vectors')
        np.savetxt(self.root + self.project + self.tag + '_atomic_pairs.txt', self.n2_contacts)
        np.save(self.root + self.project + self.tag + '_interatomic_vectors.npy', self.interatomic_vectors)
        # np.savetxt(self.root + self.project + self.tag + '_interatomic_vectors.txt', self.interatomic_vectors)
        print(f'<pair_dist_calculation> ... interatomic distances calculated')
        pdf_r_range = np.arange(start=0, stop=self.rmax, step=(self.r_dist_bin / 10))
        n_atoms = len(self.extended_atoms)
        n_atom_density = n_atoms / ((4 / 3) * np.pi * self.rmax ** 3)
        print(f'<pair_dist_calculation> Calculating pair distribution function...')
        print(f'<pair_dist_calculation> N_atoms = {n_atoms}')
        print(f'<pair_dist_calculation> Atomic number density = {n_atom_density} AA^-3 (or nm^-3)')
        print(f'<pair_dist_calculation> Constructing PDF...')
        adpf_in_hist = np.histogram(self.n2_contacts, bins=pdf_r_range)
        adfr_r = adpf_in_hist[1][1:]
        adfr_int = adpf_in_hist[0][:]
        adfr_corr = np.zeros(adfr_int.shape)
        for k, rb in enumerate(adfr_r):
            adfr_corr[k] = n_atom_density * (1 / (4 * np.pi * rb ** 2 * n_atoms)) * adfr_int[k]
        pdf_arr = np.column_stack((adfr_r, adfr_corr))
        print(f'<pair_dist_calculation> PDF written to: ')
        np.savetxt(self.root + self.project + self.tag + '_PDF.txt', pdf_arr)
        np.savetxt(self.root + self.project + self.tag + '_APDF.txt', np.column_stack((adfr_r, adfr_int)))
        print(f"{self.root + self.project + self.tag + '_PDF.txt'}")
        print(f"{self.root + self.project + self.tag + '_APDF.txt'}")
        return self.interatomic_vectors

    def trim_interatomic_vectors_to_probe(self):
        """
        Removes all interatomic vectors with length outside range r_{min} < r < r_{max}
        :return:
        """
        print(f'<trim_interatomic_vectors_to_probe> Before trimming : {len(self.interatomic_vectors)} vectors')
        print(self.interatomic_vectors[0])
        a = np.array(self.interatomic_vectors)
        b = a[a[:, 3] < self.rmax]
        c = b[b[:, 3] > self.rmin]
        print(f'<trim_interatomic_vectors_to_probe> ..after trimming to < self.rmax : {len(b)} vectors')
        print(f'<trim_interatomic_vectors_to_probe> ..after trimming to > self.rmin : {len(c)} vectors')
        print(f'<trim_interatomic_vectors_to_probe> ..after trimming : {len(c)} vectors')
        self.interatomic_vectors = c
        np.save(self.root + self.project + self.tag + '_interatomic_vectors_trim.npy', self.interatomic_vectors)

    def run_fast_serial_calculation(self):
        global_start = time.time()
        self.parameter_check()
        self.write_all_params_to_file()
        self.subject_atoms, self.extended_atoms = self.subject_target_setup()  # Sets up the atom positions of the subject set and supercell
        self.dimension = self.get_dimension()  # Sets the target dimension (somewhat redundant until I get the fast r=r' mode set up)
        """
        Here we do the chunking for sending to threads
        """
        self.interatomic_vectors = self.pair_dist_calculation()  # Calculate all the interatomic vectors.
        self.trim_interatomic_vectors_to_probe()  # Trim all the interatomic vectors to the r_probe limit
        self.percent_milestones = np.linspace(start=0, stop=len(self.interatomic_vectors), num=10)
        self.iteration_times = np.zeros(len(self.interatomic_vectors))
        # print(self.iteration_times.shape)
        [int(j) for j in self.percent_milestones]
        # print(f'{self.percent_milestones=}')
        np.random.shuffle(self.interatomic_vectors)  # Shuffle list of vectors
        print(
            f'<fast_model_padf.run_fast_serial_calculation> Total interatomic vectors: {len(self.interatomic_vectors)}')
        # Set up the rolling PADF arrays
        self.rolling_Theta = np.zeros((self.nr, self.nr, self.nth))
        self.rolling_Theta_odds = np.zeros((self.nr, self.nr, self.nth))
        self.rolling_Theta_evens = np.zeros((self.nr, self.nr, self.nth))
        # Here we loop over interatomic vectors
        print(f'<fast_model_padf.run_fast_serial_calculation> Working...')
        for k, subject_iav in enumerate(self.interatomic_vectors):
            k_start = time.time()
            self.calc_padf_frm_iav(k=k, r_ij=subject_iav)
            self.cycle_assessment(k=k, start_time=k_start)
            if self.converged_flag:
                break

        # Save the rolling PADF arrays
        np.save(self.root + self.project + self.tag + '_mPADF_total_sum', self.rolling_Theta)
        np.save(self.root + self.project + self.tag + '_mPADF_odds_sum', self.rolling_Theta_odds)
        np.save(self.root + self.project + self.tag + '_mPADF_evens_sum', self.rolling_Theta_evens)

        self.calculation_time = time.time() - global_start
        print(
            f"<fast_model_padf.run_fast_serial_calculation> run_fast_serial_calculation run time = {self.calculation_time} seconds")
        print(
            f"<fast_model_padf.run_fast_serial_calculation> Total contributing contacts (for normalization) = {self.total_contribs}")
        self.write_calculation_summary()
        # Plot diagnostics
        self.loop_similarity_array = np.array(self.loop_similarity_array)

        # plt.plot(self.loop_similarity_array[:, 0], self.loop_similarity_array[:, 1], '-')
        # plt.show()

# if __name__ == '__main__':
#     modelp = ModelPadfCalculator()
