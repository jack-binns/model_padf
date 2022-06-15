"""
Parallel Model PADF Calculator

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


class ModelPADF:

    def __init__(self):

        self.root = ""
        self.project = ""
        self.tag = ""
        self.supercell_atoms = ""  # the xyz file contains the cartesian coords of the crystal structure expanded
        # to include r_probe
        self.subject_atoms = ""  # the cif containing the asymmetric unit
        self.ucds = [1.0, 1.0, 1.0, 90.0000, 90.0000, 90.0000]
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
        self.mode = 'rrprime'
        self.dimension = 2
        self.processor_num = 2
        self.Pool = mp.Pool(self.processor_num)
        self.loops = 0
        self.verbosity = 0
        self.Theta = np.zeros(0)
        self.rolling_Theta = np.zeros(0)
        self.n2_contacts = []
        self.interatomic_vectors = []
        self.subject_set = []
        self.subject_number = 1
        self.raw_extended_atoms = []
        self.extended_atoms = []
        self.loop_similarity_array = []
        self.convergence_target = 1.0
        self.converged_loop = 0
        self.total_contribs = 0
        self.calculation_time = 0.0

    """
    Handlers for parallelization
    no such thing as "import parallel"!
    |/|/|/|/|/|/|/|/|/|/|/|/|/|/|/|/|/|/|/|/|/
    """

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['Pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    """
    ^^^
    Handlers for parallelization
    """

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

    def clean_project_folder(self):
        """
        Cleans up the Theta and Theta_loop files that are generated through the calculation
        :return:
        """
        print("<clean_project_folder>: Cleaning work folder...")
        if self.verbosity == 0:
            os.remove(self.root + self.project + self.tag + '_Theta_0.npy')
        else:
            if self.converged_loop > 0:
                for i in range(1, int(self.converged_loop) + 1):
                    os.remove(self.root + self.project + self.tag + '_Theta_loop_' + str(i) + '.npy')

                for j in range(int(self.processor_num)):
                    os.remove(self.root + self.project + self.tag + '_Theta_' + str(j) + '.npy')
            else:
                for i in range(1, int(self.loops) + 2):
                    os.remove(self.root + self.project + self.tag + '_Theta_loop_' + str(i) + '.npy')

                for j in range(int(self.processor_num)):
                    os.remove(self.root + self.project + self.tag + '_Theta_' + str(j) + '.npy')

    def write_calculation_summary(self):
        """
        Writes out a summary of the calculation
        :return:
        """
        with open(self.root + self.project + f'{self.tag}_calculation_log.txt', 'w') as f:
            f.write(f'Calculation time: {self.calculation_time} s\n')
            f.write(f'Total number of interatomic vectors {len(self.interatomic_vectors)}\n')
            f.write(f'Total number of atoms in system {len(self.extended_atoms)}\n')
        np.savetxt(self.root + self.project + f'{self.tag}_similarity_log.txt', np.array(self.loop_similarity_array))

    def filter_subject_set(self):
        """
        Shuffles and trims the subject atoms (a.k.a. asymmetric unit) on the basis of the subject number
        in the setup file.
        Also shuffles
        :return:
        """
        print(f"<filter_subject_set>: Selecting subset of {self.subject_number} subject atoms ")
        np.random.shuffle(self.subject_set)
        self.subject_set = self.subject_set[:self.subject_number]
        print(f"<filter_subject_set>: Subject set now includes {len(self.subject_set)} atoms ")
        return self.subject_set

    def subject_target_setup(self):
        """
        Handlers to read in the subject atoms (a.k.a. asymmetric unit) and the extended atoms (environment)
        :return:
        """
        self.subject_set = u.subject_atom_reader(self.root + self.project + self.subject_atoms,
                                                 self.ucds)  # Read the full asymmetric unit
        if self.subject_number > 0:
            self.subject_set = self.filter_subject_set()  # Filter the subject set using self.subject_number
        self.raw_extended_atoms = u.read_xyz(
            self.root + self.project + self.supercell_atoms)  # Take in the raw environment atoms
        self.extended_atoms = self.clean_extended_atoms()  # Trim to the atoms probed by the subject set
        # print(self.extended_atoms[0])
        # print(self.subject_set[0])
        return self.subject_set, self.extended_atoms

    def sum_loop_arrays(self, loop):
        """
        Sum up the theta npy's for the loops
        up to loop
        :param loop: loop at which to perform the sum
        :return:
        """
        SumTheta = self.generate_empty_theta(self.dimension)
        if self.convergence_check_flag is False:
            for j in np.arange(1, int(loop) + 2, 1):
                chunk_Theta = np.load(self.root + self.project + self.tag + '_Theta_loop_' + str(j) + '.npy')
                SumTheta = np.add(SumTheta, chunk_Theta)
        else:
            for j in np.arange(1, int(loop) + 1, 1):
                chunk_Theta = np.load(self.root + self.project + self.tag + '_Theta_loop_' + str(j) + '.npy')
                SumTheta = np.add(SumTheta, chunk_Theta)
        if self.dimension == 2:
            np.save(self.root + self.project + self.tag + '_slice_total_sum', SumTheta)
        elif self.dimension == 3:
            np.save(self.root + self.project + self.tag + '_mPADF_total_sum', SumTheta)
        return SumTheta

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
        clean_ex = []
        for ex_atom in self.raw_extended_atoms:
            for as_atom in self.subject_set:
                diff = u.fast_vec_difmag(ex_atom[0], ex_atom[1], ex_atom[2], as_atom[0], as_atom[1], as_atom[2])
                if abs(diff) <= self.rmax:
                    clean_ex.append(ex_atom)
                    break
                else:
                    continue
        clean_ex = np.array(clean_ex)
        print(
            f"<clean_extended_atoms>: Extended atom set has been reduced to {len(clean_ex)} atoms within {self.rmax} radius")
        return np.array(clean_ex)

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

    def parallel_pool_accounting(self, loop_number):
        """
        Sums arrays together for each cycle
        :param loop_number: loop id
        :return:
        """
        if loop_number == 1:
            self.rolling_Theta = self.generate_empty_theta(self.dimension)
            for i in range(int(self.processor_num)):
                chunk_Theta = np.load(self.root + self.project + self.tag + '_Theta_' + str(i) + '.npy')
                self.rolling_Theta = np.add(self.rolling_Theta, chunk_Theta)
            np.save(self.root + self.project + self.tag + '_rolling_Theta', self.rolling_Theta)
            np.save(self.root + self.project + self.tag + f'_rolling_Theta_{loop_number}.npy', self.rolling_Theta)

        elif loop_number > 1:
            loop_Theta = self.generate_empty_theta(self.dimension)
            # np.save(f'{self.root}{self.project}{self.tag}_rolling_Theta_{loop_number-1}', self.rolling_Theta)
            for i in range(int(self.processor_num)):
                chunk_Theta = np.load(self.root + self.project + self.tag + '_Theta_' + str(i) + '.npy')
                loop_Theta = np.add(loop_Theta, chunk_Theta)
            self.rolling_Theta = np.add(self.rolling_Theta, loop_Theta)
            np.save(self.root + self.project + self.tag + f'_rolling_Theta_{loop_number}.npy', self.rolling_Theta)
            np.save(self.root + self.project + self.tag + f'_rolling_Theta.npy', self.rolling_Theta)
            # if self.verbosity == 0:
            #     os.remove(f'{self.root}{self.project}{self.tag}_rolling_Theta_{loop_number-2}.npy')

    def prelim_padf_correction(self, raw_padf):
        if self.dimension == 2:
            th = np.outer(np.ones(raw_padf.shape[0]), np.arange(raw_padf.shape[1]))
            ith = np.where(th > 0.0)
            raw_padf[ith] *= 1.0 / np.abs(np.sin(np.pi * th[ith] / float(raw_padf.shape[1])) + 1e-3)
            r = np.outer(np.arange(raw_padf.shape[0]), np.ones(raw_padf.shape[1])) * self.rmax / float(
                raw_padf.shape[0])
            ir = np.where(r > 1.0)
            raw_padf[ir] *= 1.0 / (r[ir] ** self.r_power)  # radial correction (edit this line [default value: **4])
            return raw_padf
        elif self.dimension == 3:
            data = np.zeros((raw_padf.shape[0], raw_padf.shape[-1]))
            for i in np.arange(
                    raw_padf.shape[0]):  # loop over NumPy array to return evenly spaced values within a given interval
                data[i, :] = raw_padf[i, i, :]
            data += data[:, ::-1]
            th = np.outer(np.ones(data.shape[0]), np.arange(data.shape[1]))
            ith = np.where(th > 0.0)
            data[ith] *= 1.0 / np.abs(np.sin(np.pi * th[ith] / float(data.shape[1])) + 1e-3)
            r = np.outer(np.arange(data.shape[0]), np.ones(data.shape[1])) * self.rmax / float(data.shape[0])
            ir = np.where(r > 1.0)
            data[ir] *= 1.0 / (r[ir] ** self.r_power)  # radial correction (edit this line [default value: **4])
            return data

    def report_cossim(self):
        print("---------------------------------")
        print("Loop num        cosine similarity")
        print("---------------------------------")
        for n, i in enumerate(self.loop_similarity_array):
            # if n % 100 == 0:
            print(f'{i[0]}         {i[1]}')
        print("---------------------------------")

    def convergence_check(self, loop_number):
        if loop_number == 1:
            print("<convergence_check>: No convergence check in loop 1")
            self.loop_similarity_array.append([loop_number, 0.0])
            # print(f"{self.loop_similarity_array=}")
            return 0.0
        else:
            # calculate the n-minus_padf
            n_minus_padf = np.load(f'{self.root}{self.project}{self.tag}_rolling_Theta_{loop_number - 1}.npy')
            n_minus_padf_corr = self.prelim_padf_correction(n_minus_padf)
            # calculate the n_padf
            n_padf = np.load(self.root + self.project + self.tag + '_rolling_Theta.npy')
            n_padf_corr = self.prelim_padf_correction(n_padf)
            # Normalize both arrays
            n_minus_padf_normal = n_minus_padf_corr / np.linalg.norm(n_minus_padf_corr)
            n_padf_normal = n_padf_corr / np.linalg.norm(n_padf_corr)
            loop_cos = u.cossim_measure(n_padf_normal, n_minus_padf_normal)
            self.loop_similarity_array.append([loop_number, loop_cos])
            self.report_cossim()
            # print(f"{self.loop_similarity_array=} {loop_number}")
            if self.verbosity == 0 and loop_number > 2:
                # print('REACHING DELETION')
                # print(f'{self.root}{self.project}{self.tag}_rolling_Theta_{loop_number-2}.npy')
                os.remove(f'{self.root}{self.project}{self.tag}_rolling_Theta_{loop_number - 2}.npy')
            return loop_cos

    def calc_padf_frm_iav(self, k, r_ij):
        start = time.time()
        print(f'<calc_padf_frm_iav>: Starting calculation on thread {k}...')
        fb_hit_count = 0
        Theta = self.generate_empty_theta(3)
        # print(f"Thread {k}: Beginning calculation")
        for r_xy in self.interatomic_vectors:
            if np.array_equal(r_ij, r_xy):
                continue
            theta = u.fast_vec_angle(r_ij[0], r_ij[1], r_ij[2], r_xy[0], r_xy[1], r_xy[2])
            fprod = r_ij[4] * r_xy[4]
            self.bin_cor_vec_to_theta([r_ij[3], r_xy[3], theta], fprod, Theta)
            fb_hit_count += 1
        end = time.time()
        print(f"<calc_padf_frm_iav>: Thread {k} execution time = ", end - start, " seconds")
        print(f"<calc_padf_frm_iav>: Thread {k} added {fb_hit_count} four-body contacts")
        # Save the Theta array as is:
        self.total_contribs = self.total_contribs + fb_hit_count
        # print(self.total_contribs, fb_hit_count)
        np.save(self.root + self.project + self.tag + '_Theta_' + str(k), Theta)

    def pair_dist_calculation(self):
        print(f'<pair_dist_calculation> Calculating pairwise interatomic distances...')
        # interatomic_vectors
        for a_i in self.subject_set:
            for a_j in self.extended_atoms:
                if not np.array_equal(a_i, a_j):
                    # print(a_i)
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

    def trim_interatomic_vectors_to_probe(self):
        print(f'DEBUG <trim_interatomic_vectors_to_probe> Before trimming : {len(self.interatomic_vectors)} vectors')
        print(self.interatomic_vectors[0])
        a = np.array(self.interatomic_vectors)
        b = a[a[:, 3] < self.rmax]
        c = b[b[:, 3] > self.rmin]
        print(f'DEBUG <trim_interatomic_vectors_to_probe> ..after trimming to < self.rmax : {len(b)} vectors')
        print(f'DEBUG <trim_interatomic_vectors_to_probe> ..after trimming to > self.rmin : {len(c)} vectors')
        print(f'DEBUG <trim_interatomic_vectors_to_probe> ..after trimming : {len(c)} vectors')
        self.interatomic_vectors = c
        np.save(self.root + self.project + self.tag + '_interatomic_vectors_trim.npy', self.interatomic_vectors)

    def run(self):
        """
        Runs the Straight-To-Matrix model PADF calculation
        :return:
        """
        start = time.time()
        self.parameter_check()
        self.write_all_params_to_file()
        self.subject_set, self.extended_atoms = self.subject_target_setup()
        self.dimension = self.get_dimension()
        """
        Here we do the chunking for sending to threads
        """
        self.pair_dist_calculation()  # Calculate all the interatomic vectors.
        self.trim_interatomic_vectors_to_probe()  # Trim all the interatomic vectors to the r_probe limit
        np.random.shuffle(self.interatomic_vectors)  # Shuffle the list of interatomic vectors
        # print(f'DEBUG {len(self.interatomic_vectors)}')
        self.loops = int(
            len(self.interatomic_vectors) / self.processor_num)  # The number of loops for complete calculation
        # chunked add_bodies_to_theta:
        for loop_id in np.arange(1, int(self.loops) + 2, 1):
            print(str(loop_id) + " / " + str(int(self.loops) + 1))
            cluster_vectors = self.interatomic_vectors[
                              (
                                      loop_id - 1) * self.processor_num:loop_id * self.processor_num]  # Create a chunk of to send out
            # [print(f'DEBUG: cluster vec {x} : {vec}') for x, vec in enumerate(cluster_vectors)]
            if self.mode == 'rrprime':
                processes = [mp.Process(target=self.calc_padf_frm_iav, args=(i, cl_vec)) for i, cl_vec in
                             enumerate(cluster_vectors)]
            elif self.mode == 'stm':
                processes = [mp.Process(target=self.calc_padf_frm_iav, args=(i, cl_vec)) for i, cl_vec in
                             enumerate(cluster_vectors)]
            else:
                print("ERROR: Missing mode")
                break
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            # Crunch the npy arrays together for this loop
            self.parallel_pool_accounting(loop_id)

            # Check for convergence if required
            if self.convergence_check_flag:
                # print(f"{self.convergence_check_flag} {loop_id=}")
                loop_convergence = self.convergence_check(loop_id)
                if loop_id > 2:
                    # print(f"{self.convergence_check_flag} {loop_id=}")
                    # print(f'loop array {self.loop_similarity_array} {loop_id}')
                    print(
                        f'similarity change :: {loop_convergence - self.loop_similarity_array[loop_id - 2][1]}  target :: {self.convergence_target}')
                    if abs(loop_convergence - self.loop_similarity_array[loop_id - 2][1]) < self.convergence_target:
                        # print(f'DEBUG loop convergence {loop_convergence}  prev  {self.loop_similarity_array[loop_id-2][1]}')
                        print("<run> Calculation converged at loop ", loop_id)
                        self.converged_loop = loop_id
                        break
                    if loop_id == int(self.loops) + 2:
                        self.converged_loop = loop_id
                        break
                    else:
                        continue
                else:
                    continue

        # Crunch the npy arrays together for all loops:
        if self.convergence_check_flag:
            if self.converged_loop == 0:
                BigTheta = self.rolling_Theta
            else:
                print(f"<run> converged_loop: {self.converged_loop}")

                BigTheta = self.rolling_Theta
                print(f'<run> Theta shape: {BigTheta.shape}')
        else:
            BigTheta = self.rolling_Theta
        end = time.time()
        # Check to see if the folder should be cleaned
        if self.verbosity == 0:
            self.clean_project_folder()
        np.save(self.root + self.project + self.tag + '_mPADF_total_sum', BigTheta)
        self.calculation_time = end - start
        print(f"<run> Total run time = {end - start} seconds")
        print(f"<run> Total contributing contacts (for normalization) = {self.total_contribs}")
        if self.dimension == 2:
            plt.imshow(BigTheta)
            plt.show()
        self.write_calculation_summary()


if __name__ == '__main__':
    modelp = ModelPADF()
