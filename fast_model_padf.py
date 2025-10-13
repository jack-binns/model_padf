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
import pyshtools as pysh
from scipy.special import spherical_jn, legendre


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
        self.nthvol = 180
        self.angular_bin = 1.0
        self.r_power = 2
        self.convergence_check_flag = False
        self.r12_reflection = True
        self.mode = 'stm'
        self.dimension = 3
        self.processor_num = 2
        self.chunksize = 50
        #self.Pool = mp.Pool(self.processor_num)
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
        self.use_supercell = True

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
        print(f'DEBUG <subject_target_setup>', self.root)
        print(f'DEBUG <subject_target_setup>', self.project)
        print(f'DEBUG <subject_target_setup>', self.supercell_atoms)
        self.raw_extended_atoms = u.read_xyz(
            f'{self.root}{self.project}{self.supercell_atoms}')  # Take in the raw environment atoms
        if self.use_supercell:
            self.extended_atoms = self.clean_extended_atoms()  # Trim to the atoms probed by the subject set
        else:
            self.extended_atoms = np.copy(self.subject_atoms)
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
            for i, ex_atom in enumerate(self.raw_extended_atoms):
                #print( i+1, "/", len(self.raw_extended_atoms)) 
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

    def select_subcell(self,atomlist,limits):
        selected = []
        xmin, xmax = limits[0],limits[1]
        ymin, ymax = limits[2],limits[3]
        zmin, zmax = limits[4],limits[5]
        for v in atomlist:
             if (v[0]>xmin)and(v[0]<xmax)and(v[1]>ymin)and(v[1]<ymax)and(v[2]>zmin)and(v[2]<zmax):
                selected.append(v)
        return np.array(selected)

    def reduce_subject_atoms_to_subcell(self,limits):
        """Select a subcell from the atom list"""

        selected = []
        xmin, xmax = limits[0],limits[1]
        ymin, ymax = limits[2],limits[3]
        zmin, zmax = limits[4],limits[5]
        for v in self.subject_atoms:
             if (v[0]>xmin)and(v[0]<xmax)and(v[1]>ymin)and(v[1]<ymax)and(v[2]>zmin)and(v[2]<zmax):
                selected.append(v)
        self.subject_atoms = np.array(selected)

    def box_dimensions_from_subject_atoms():
            self.x_min = np.min(self.subject_atoms[:, 0])
            self.y_min = np.min(self.subject_atoms[:, 1])
            self.z_min = np.min(self.subject_atoms[:, 2])

            self.x_max = np.max(self.subject_atoms[:, 0])
            self.y_max = np.max(self.subject_atoms[:, 1])
            self.z_max = np.max(self.subject_atoms[:, 2])

            self.x_wid = self.x_max - self.x_min
            self.y_wid = self.y_max - self.y_min
            self.z_wid = self.z_max - self.z_min

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

    def pair_dist_calculation(self,writefreq=1000, outputpairs=False):
        print(f'<pair_dist_calculation> Calculating pairwise interatomic distances...')
        # interatomic_vectors
        iv_list = []
        for k, a_i in enumerate(self.subject_atoms):
            if k % int( writefreq) == 0:
                print(f"{k} / {len(self.subject_atoms)}")
            """ 
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
            """
            # alternative matrix version
            diff = self.extended_atoms - a_i
            mags = np.sqrt(np.sum(diff*diff,1))
            igood = np.where( (mags>0.8) )
            #print( diff[igood,:].shape, mags[igood].shape, self.extended_atoms[igood,3].shape, "shapes")
            chunk_a_i = np.concatenate( (np.squeeze(diff[igood,:]), np.squeeze(mags[igood]).reshape((mags[igood].shape[0],1)), self.extended_atoms[igood,3].T*a_i[3]),axis=1)
            #iv_list += list(chunk_a_i)
            iv_list.append(chunk_a_i)

            #if k==0:
            #    ivectors = np.copy(chunk_a_i)
            #else:
            #    ivectors = np.concatenate( (ivectors, chunk_a_i), axis=0) 
        
        input("Just pausing for a moment:")
        iv_list = np.array(iv_list)
        self.interatomic_vectors = iv_list #np.concatenate( iv_list, axis=0) #ivectors       
        print(self.interatomic_vectors.shape)
        input("Just pausing for a moment:")
        exit() 
        self.n2_contacts = self.interatomic_vectors[:,2] #ivectors[:,2]

        np.array(self.n2_contacts)
        print(f'<pair_dist_calculation> {len(self.interatomic_vectors)} interatomic vectors')
        if outputpairs: np.savetxt(self.root + self.project + self.tag + '_atomic_pairs.txt', self.n2_contacts)
        if outputpairs: np.save(self.root + self.project + self.tag + '_interatomic_vectors.npy', self.interatomic_vectors)
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


    #
    #  The core of the matrix correlation and histogramming routine
    #
# calculate vec dot vec_prime for every pair of vectors
  

    def calc_padf_frm_iav_matrix(self,vectors,vectors2,nr=100,nth=100):
          corr = vectors@vectors2.T
      
          # get the norm of all vectors and make them into a norm*norm_prime matrix
          norms   = np.sqrt(np.sum(vectors**2,1))
          norms2  = np.sqrt(np.sum(vectors2**2,1))
          #print("min max norms",np.min(norms), np.max(norms),np.min(norms2),np.max(norms2), np.min(corr),np.max(corr)) 
          norms_c = np.outer(norms,norms2.T)
          #print("min max norms_c", np.min(norms_c), np.max(norms_c))
      
          # the cosine(theta) for each pair of vectors
          div = corr/norms_c
          div[div>1] = 1
          div[div<-1] = -1
 
          #print("min max div", np.min(div), np.max(div))
          # the angle between each pair of vectors
          angles = div #np.arccos(div)
          #print("min max angles", np.min(angles), np.max(angles))
      
          # the final step would be to use array masking to histogram the calculation...
          nvec  = len(vectors)
          nvec2 = len(vectors2)
          rrth_coords = np.array([np.outer(norms,np.ones(nvec2).T).flatten(),
                      np.outer(np.ones(nvec),norms2).flatten(),
                      angles.flatten() ])
          #print(nr, nth, np.min(rrth_coords), np.max(rrth_coords))
          padf, edges = np.histogramdd( rrth_coords.T, bins=(nr,nr,nth), range=((0,self.rmax),(0,self.rmax),(-1,1))) #(0,np.pi)))
          #print(edges[0], )
          return padf, edges


    #
    # A matrix implementation of the model calculation   
    #
    #
    def run_fast_matrix_calculation(self,alreadysetup=False):
        global_start = time.time()
        if not alreadysetup:
            self.parameter_check()
            self.write_all_params_to_file()
            self.subject_atoms, self.extended_atoms = self.subject_target_setup()  # Sets up the atom positions of the subject set and supercell
        self.dimension = self.get_dimension()  # Sets the target dimension (somewhat redundant until I get the fast r=r' mode set up)
        """
        Here we do the chunking for sending to threads
        """
        self.interatomic_vectors = self.pair_dist_calculation(10)  # Calculate all the interatomic vectors.
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
        chunksize = np.min([self.chunksize,len(self.interatomic_vectors)])
        nchunks = len(self.interatomic_vectors)//chunksize
        print(f'<fast_model_padf.run_fast_matrix_calculation> Working...')
        global_start = time.time()
        for k in range(nchunks):
            print(f'Correlating chunk index {k+1}/{nchunks}')
            vectors = self.interatomic_vectors[k*chunksize:(k+1)*chunksize,:3]
            for k2 in range(nchunks):
                #print(f'Correlating chunk index {k+1}/{nchunks}  {k2}/{nchunks}')
                vectors2 = self.interatomic_vectors[k2*chunksize:(k2+1)*chunksize,:3]
                padftmp, edges = self.calc_padf_frm_iav_matrix(vectors,vectors2,nr=self.nr,nth=self.nth)
                if (k*chunksize+k2)%2==0:
                    self.rolling_Theta_evens += padftmp
                else:
                    self.rolling_Theta_odds += padftmp
                self.rolling_Theta += padftmp
        """
        if len(self.interatomic_vectors)%chunksize!=0: 
            vectors  = self.interatomic_vectors[nchunks*chunksize:,:3]
            for k2 in range(nchunks):
                #print(f'Correlating chunk index {k+1}/{nchunks}  {k2}/{nchunks}')
                vectors2 = self.interatomic_vectors[k2*chunksize:(k2+1)*chunksize,:3]
                padftmp, edges = self.calc_padf_frm_iav_matrix(vectors,vectors2,nr=self.nr,nth=self.nth)
                if (nchunks*chunksize+k2)%2==0:
                    self.rolling_Theta_evens += padftmp
                else:
                    self.rolling_Theta_odds += padftmp
                self.rolling_Theta += padftmp
            #last one         
            padftmp, edges = self.calc_padf_frm_iav_matrix(vectors,vectors,nr=self.nr,nth=self.nth)
            if (nchunks*chunksize+nchunks)%2==0:
                self.rolling_Theta_evens += padftmp
            else:
                self.rolling_Theta_odds += padftmp
            self.rolling_Theta += padftmp
        """
        
        self.total_contribs = len(self.interatomic_vectors)**2
        self.calculation_time = time.time() - global_start

        #for k, subject_iav in enumerate(self.interatomic_vectors):
        #    k_start = time.time()
        #    self.calc_padf_frm_iav(k=k, r_ij=subject_iav)
        #    self.cycle_assessment(k=k, start_time=k_start)
        #    if self.converged_flag:
        #        break

        # Save the rolling PADF arrays
        np.save(self.root + self.project + self.tag + '_mPADF_total_sum', self.rolling_Theta)
        np.save(self.root + self.project + self.tag + '_mPADF_odds_sum', self.rolling_Theta_odds)
        np.save(self.root + self.project + self.tag + '_mPADF_evens_sum', self.rolling_Theta_evens)

        self.calculation_time = time.time() - global_start
        print(
            f"<fast_model_padf.run_fast_model_calculation> run_fast_model_calculation run time = {self.calculation_time} seconds")
        print(
            f"<fast_model_padf.run_fast_model_calculation> Total contributing contacts (for normalization) = {self.total_contribs}")
        self.write_calculation_summary()
        # Plot diagnostics
        self.loop_similarity_array = np.array(self.loop_similarity_array)

        # plt.plot(self.loop_similarity_array[:, 0], self.loop_similarity_array[:, 1], '-')
        # plt.show()


    def calc_sphvol_from_interatomic_vectors(self,vectors,nr=50,nth=90,nphi=90, tol=5e-2):
      
          # get the norm of all vectors and make them into a norm*norm_prime matrix
          norms   = np.sqrt(np.sum(vectors**2,1))
          
          # the final step would be to use array masking to histogram the calculation...
          nvec  = len(vectors)

          sphv = vectors*0.0
          sphv[:,0] = norms
          inorm = np.where( norms > tol )
          sphv[inorm,1] = np.arccos( vectors[inorm,2]/norms[inorm])

          ix = np.where( (np.abs(vectors[:,0])>tol)*(vectors[:,0]>0) )   #norm of x above thresh; and x positive
          sphv[ix,2] = np.pi/2   + np.arctan( vectors[ix,1]/vectors[ix,0])

          ix = np.where( (np.abs(vectors[:,0])>tol)*(vectors[:,0]<0) )   #norm of x above thresh; and x negative
          sphv[ix,2] = 3*np.pi/2 + np.arctan( vectors[ix,1]/vectors[ix,0])


          ix = np.where( (np.abs(vectors[:,0])<tol)*(vectors[:,1]>0) )   #norm of x below thresh; and y positive
          sphv[ix,2] = 3*np.pi/2
          
          ix = np.where( (np.abs(vectors[:,0])<tol)*(vectors[:,1]<0) )   #norm of x below thresh; and y negative
          sphv[ix,2] = np.pi/2

          #print("vectors")
          #for i in range(nvec): 
          #      if norms[i]>2: print(i, vectors[i], norms[i], sphv[i,:])

          outvol, edges = np.histogramdd( sphv, bins=(nr,nth,nphi), range=((0,self.rmax),(0,np.pi),(0,2*np.pi)))
          #sphv[ix,2] += np.pi
          #outvol2, edges = np.histogramdd( sphv, bins=(nr,nth,nphi), range=((0,self.rmax),(0,np.pi),(0,2*np.pi)))
          #outvol[:,:,nphi//2:] = outvol[:,:,:nphi//2] 
     
          #NEED TO CHECK THE RANGE OF THE ARCCOS AND ARCSIN FUNCTIONS - WHAT ABOUT AFFECT OF THE ARCSIN RANGE???? 
          return outvol, edges

    def calc_volume_correlation_theta_loop(self, ithread, rstart, rstop, sphvol, fsphvol, return_dict):

            coords = np.mgrid[:self.nr,:self.nthvol,:self.phivol]
            print( coords.shape )
            r = coords[0]
            th = coords[1]*np.pi/self.nthvol
            #costh = (2*coords[1]/self.nthvol)-1
            #th = np.arccos( costh )
            phi = coords[2]*2*np.pi/self.phivol
        
            return_dict[ithread] = np.zeros((self.nr, self.nr, self.nth))
            for i in range(rstart,rstop):
                for j in range(self.nthvol):
                    shifted = np.roll(np.roll(sphvol,i,0),j,1)
                    fshifted    = np.fft.fftn( shifted, axes=[2] )
                    outevens = np.real(np.fft.ifftn( fshifted.conjugate()*fsphvol, axes=[2] ))
                                 
                    r_shifted  = np.roll(np.roll(r,i,0),j,1)
                    th_shifted = np.roll(np.roll(th,i,0),j,1)
                    #outevens *= np.sin(th)*np.sin(th_shifted) #sth hack!!!
                    cos_angle = np.cos(th)*np.cos(th_shifted) + np.sin(th)*np.sin(th_shifted)*np.cos(phi)  #here phi stands in for phi1-phi2

                    tmp, be = np.histogramdd( (r.flatten(), r_shifted.flatten(), cos_angle.flatten()), bins=(self.nr,self.nr,self.nth), range=((0,self.nr),(0,self.nr),(-1,1)), weights=outevens.flatten())
                    return_dict[ithread] += tmp


    #
    # A volume and histogram implementation of the model calculation   
    #
    #
    def run_histogram_calculation(self):
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
        np.random.shuffle(self.interatomic_vectors)  # Shuffle list of vectors
        
        self.sphvol_evens = np.zeros((self.nr, self.nthvol, self.phivol))
        self.sphvol_odds  = np.zeros((self.nr, self.nthvol, self.phivol))
        chunksize = np.min([500,len(self.interatomic_vectors)])
        nchunks = len(self.interatomic_vectors)//chunksize
        print(f'<fast_model_padf.run_fast_histogram_calculation> Working...')
        global_start = time.time()
        for k in range(nchunks):
            print(f'Creating 3D pair distributions in spherical coordinates; chunk index {k+1}/{nchunks}')
            vectors = self.interatomic_vectors[k*chunksize:(k+1)*chunksize,:3]
            for k2 in range(nchunks):
                #print(f'Correlating chunk index {k+1}/{nchunks}  {k2}/{nchunks}')
                vectors2 = self.interatomic_vectors[k2*chunksize:(k2+1)*chunksize,:3]
                voltmp, edges = self.calc_sphvol_from_interatomic_vectors(vectors,nr=self.nr,nth=self.nthvol,nphi=self.phivol)
                if (k*chunksize+k2)%2==0:
                    self.sphvol_evens += voltmp
                else:
                    self.sphvol_odds += voltmp

        print(
            f'<fast_model_padf.run_histogram_calculation> Total interatomic vectors: {len(self.interatomic_vectors)}')
        # Set up the rolling PADF arrays
        self.rolling_Theta_odds = np.zeros((self.nr, self.nr, self.nth))
        self.rolling_Theta_evens = np.zeros((self.nr, self.nr, self.nth))
        # Here we loop over interatomic vectors

        fsphvol_evens = np.fft.fftn( self.sphvol_evens, axes=[2])
        fsphvol_odds = np.fft.fftn( self.sphvol_odds, axes=[2])


        coords = np.mgrid[:self.nr,:self.nthvol,:self.phivol]
        print( coords.shape )
        r = coords[0]
        th = coords[1]*np.pi/self.nthvol
        phi = coords[2]*2*np.pi/self.phivol

        # volume correlation
        manager = mp.Manager()            
        return_dict = manager.dict()

        t0 = time.perf_counter()
        tcurrent = t0
        hsum = np.zeros( (self.nr,self.nr,self.nth))
        
        if self.processor_num>1:
            nchunk2 = self.nr//(self.processor_num//2)
        else:
            nchunk2 = self.nr
        processes = [] 

        
        for ithread in range(self.processor_num):     
            oddeven = ithread%2  
            rstart, rstop = (ithread//2)*nchunk2, np.min([((ithread//2)+1)*nchunk2,self.nr])
            print(f"Thread {ithread}; oddeven {oddeven}; rstart {rstart}; rstop {rstop}")
            if oddeven==0:
                p = mp.Process( target=self.calc_volume_correlation_theta_loop, args=(ithread,rstart,rstop,self.sphvol_evens, fsphvol_evens, return_dict))
            else:
                p = mp.Process( target=self.calc_volume_correlation_theta_loop, args=(ithread,rstart,rstop,self.sphvol_odds, fsphvol_odds, return_dict))
            p.start()
            processes.append(p)        
            
        for p in processes:
            p.join()

        for ithread in range(self.processor_num):   
            oddeven = ithread%2
            if oddeven==0:
                self.rolling_Theta_evens += return_dict[ithread]
            else:
                self.rolling_Theta_odds  += return_dict[ithread] 
            
        
        """
        for i in range(self.nr):
            print(i, time.perf_counter()-tcurrent, "s, ;", time.perf_counter()-t0, "s")
            tcurrent = time.perf_counter()
            for j in range(self.nth):
                shifted = np.roll(np.roll(self.sphvol_evens,i,0),j,1)
                fshifted    = np.fft.fftn( shifted, axes=[2] )
                outevens = np.real(np.fft.ifftn( fshifted.conjugate()*fsphvol_evens, axes=[2] ))
                
                shifted = np.roll(np.roll(self.sphvol_odds,i,0),j,1)
                fshifted    = np.fft.fftn( shifted, axes=[2] )
                outodds = np.real(np.fft.ifftn( fshifted.conjugate()*fsphvol_odds, axes=[2] ))
             
                r_shifted  = np.roll(np.roll(r,i,0),j,1)
                th_shifted = np.roll(np.roll(th,i,0),j,1)
                cos_angle = np.cos(th)*np.cos(th_shifted) + np.sin(th)*np.sin(th_shifted)*np.cos(phi)  #here phi stands in for phi1-phi2

                tmp_evens, be = np.histogramdd( (r.flatten(), r_shifted.flatten(), cos_angle.flatten()), bins=(self.nr,self.nr,self.nth), range=((0,self.nr),(0,self.nr),(-1,1)), weights=outevens.flatten())
                tmp_odds, be = np.histogramdd( (r.flatten(), r_shifted.flatten(), cos_angle.flatten()), bins=(self.nr,self.nr,self.nth), range=((0,self.nr),(0,self.nr),(-1,1)), weights=outodds.flatten())
                self.rolling_Theta_evens += tmp_evens
                self.rolling_Theta_odds += tmp_odds
        """

        # Save the rolling PADF arrays
        np.save(self.root + self.project + self.tag + '_mPADF_total_sum', self.rolling_Theta_odds + self.rolling_Theta_evens)
        np.save(self.root + self.project + self.tag + '_mPADF_odds_sum', self.rolling_Theta_odds)
        np.save(self.root + self.project + self.tag + '_mPADF_evens_sum', self.rolling_Theta_evens)

        self.calculation_time = time.time() - global_start
        print(
            f"<fast_model_padf.run_fast_model_calculation> run_fast_model_calculation run time = {self.calculation_time} seconds")
        print(
            f"<fast_model_padf.run_fast_model_calculation> Total contributing contacts (for normalization) = {self.total_contribs}")
        self.write_calculation_summary()
        # Plot diagnostics
        self.loop_similarity_array = np.array(self.loop_similarity_array)

# if __name__ == '__main__':
#     modelp = ModelPadfCalculator()

    #
    # Convert the blrr matrices to the PADF (l->theta)
    #
    def Blrr_to_padf( self, blrr, padfshape, legendre_norm=True ):
        """        
        Transforms B_l(r,r') matrices to padf volume
        using Legendre polynomials        

        Parameters
        ---------
        blrr : blqq object
            input blrr matrices

        padfshape : tuple (floats)
            shape of padf array

        Returns
        -------
        padfout : vol object
            padf volume
        """
        padfout = np.zeros( padfshape )
        for l in np.arange(self.nlmin,self.nl):

             if (l%2)!=0:
                  continue

             s2 = padfout.shape[2]
             z = np.cos( 2.0*np.pi*np.arange(s2)/float(s2) )
             Pn = legendre( int(l) )
             #print("Blrr to padf (1)"+str(l)) #DEBUG
             if legendre_norm:
                p = Pn(z)*np.sqrt(2*l+1)
                #print("blrr to padf; lnorm has been done")
             else:
                p = Pn(z)
                #print("lnorm has not been done")
      
             for i in np.arange(padfout.shape[0]):
                  for j in np.arange(padfout.shape[1]):
                       padfout[i,j,:] += blrr[l,i,j]*p[:]

        return padfout


    #
    # A spherical harmonic version of the model calculation   
    #
    #
    def run_spherical_harmonic_calculation(self):
        global_start = time.time()
        self.parameter_check()
        self.write_all_params_to_file()
        self.subject_atoms, self.extended_atoms = self.subject_target_setup()  # Sets up the atom positions of the subject set and supercell

        if self.subcellsize > 0.0: self.reduce_subject_atoms_to_subcell(self.limits)

        """
        Here we do the chunking for sending to threads
        """
        self.sphvol_evens = np.zeros((self.nr, self.nthvol, self.phivol))
        self.sphvol_odds  = np.zeros((self.nr, self.nthvol, self.phivol))
        print(f'<fast_model_padf.run_fast_spherical_harmonic_calculation> Working...')
        global_start = time.time()
        for i, a_i in enumerate(self.subject_atoms):
                all_interatomic_vectors = self.extended_atoms[:,:3] - np.outer(np.ones(self.extended_atoms.shape[0]),a_i[:3])
                #print( self.extended_atoms.shape, self.extended_atoms[0], a_i, all_interatomic_vectors[0])
                norms = np.sqrt(np.sum(all_interatomic_vectors**2,1))
                inorm = np.where( norms<self.rmax)
                interatomic_vectors = all_interatomic_vectors[inorm]
                chunksize = np.min([500,len(interatomic_vectors)])
                nchunks = len(interatomic_vectors)//chunksize
                if (i%500)==0: print("Subject atoms binned:", i, "/", len(self.subject_atoms), f"time passed {time.time()-global_start}", nchunks, chunksize, len(interatomic_vectors)) 
                for k in range(nchunks):
                    #print(f'Creating 3D pair distributions in spherical coordinates; chunk index {k+1}/{nchunks}', np.max(interatomic_vectors))
                    vectors = interatomic_vectors[k*chunksize:(k+1)*chunksize,:3]
                    voltmp, edges = self.calc_sphvol_from_interatomic_vectors(vectors,nr=self.nr,nth=self.nthvol,nphi=self.phivol)
                    if (i)%2==0:
                        self.sphvol_evens += voltmp
                    else:
                        self.sphvol_odds += voltmp
                 
            
        """
        chunksize = np.min([500,len(self.interatomic_vectors)])
        nchunks = len(self.interatomic_vectors)//chunksize
        self.interatomic_vectors = self.pair_dist_calculation()  # Calculate all the interatomic vectors.
        self.trim_interatomic_vectors_to_probe()  # Trim all the interatomic vectors to the r_probe limit
        np.random.shuffle(self.interatomic_vectors)  # Shuffle list of vectors
        
        print(f'<fast_model_padf.run_fast_spherical_harmonic_calculation> Working...')
        global_start = time.time()
        for k in range(nchunks):
            print(f'Creating 3D pair distributions in spherical coordinates; chunk index {k+1}/{nchunks}')
            vectors = self.interatomic_vectors[k*chunksize:(k+1)*chunksize,:3]
            voltmp, edges = self.calc_sphvol_from_interatomic_vectors(vectors,nr=self.nr,nth=self.nthvol,nphi=self.phivol)
            if (k*chunksize)%2==0:
                self.sphvol_evens += voltmp
            else:
                self.sphvol_odds += voltmp
        """
        #plt.figure()
        #plt.imshow( np.sum(self.sphvol_evens[11:14,:,:],0))
        #plt.show()

        print(
            f'<fast_model_padf.run_spherical_harmonic_calculation> Total interatomic vectors: {len(self.interatomic_vectors)}')
    
        print( "debsug spherical harmonic calc; nl nlmin", self.nl, self.nlmin)

        coeffs_evens = np.zeros( (self.nr, 2, self.nl, self.nl))
        coeffs_odds = np.zeros( (self.nr, 2, self.nl, self.nl))

        for ir in range(self.nr):
            pysh_grid = pysh.shclasses.DHRealGrid(self.sphvol_evens[ir,:,:])
            coeffs_evens[ir,:,:,:] = pysh_grid.expand(csphase=1).coeffs[:,:self.nl, :self.nl]
            pysh_grid = pysh.shclasses.DHRealGrid(self.sphvol_odds[ir,:,:])
            coeffs_odds[ir,:,:,:] = pysh_grid.expand(csphase=1).coeffs[:,:self.nl, :self.nl]

        Blrr_evens = np.zeros( (self.nl, self.nr, self.nr) )
        Blrr_odds  = np.zeros( (self.nl, self.nr, self.nr) )
        for ir in range(self.nr):
            for ir2 in range(self.nr):
                for l in range(self.nl):
                    Blrr_evens[l,ir,ir2] += np.sum( coeffs_evens[ir,:,l,:]*coeffs_evens[ir2,:,l,:])
                    Blrr_odds[l,ir,ir2]  += np.sum( coeffs_odds[ir,:,l,:] *coeffs_odds[ir2,:,l,:] )

        
        coords = np.mgrid[:self.nr,:self.nr,:self.nth]
        print( coords.shape )
        r = coords[0]
        th = coords[2]*2*np.pi/self.nth

        # Set up the rolling PADF arrays
        self.rolling_Theta_odds =  self.Blrr_to_padf( Blrr_odds, (self.nr,self.nr,self.nth)) *np.abs(np.sin(th))
        self.rolling_Theta_evens = self.Blrr_to_padf( Blrr_evens, (self.nr, self.nr, self.nth)) *np.abs(np.sin(th))
        # Here we loop over interatomic vectors


        # Save the rolling PADF arrays
        np.save(self.root + self.project + self.tag + '_mPADF_total_sum', self.rolling_Theta_odds + self.rolling_Theta_evens)
        np.save(self.root + self.project + self.tag + '_mPADF_odds_sum', self.rolling_Theta_odds)
        np.save(self.root + self.project + self.tag + '_mPADF_evens_sum', self.rolling_Theta_evens)

        self.calculation_time = time.time() - global_start
        print(
            f"<fast_model_padf.run_fast_model_calculation> run_fast_model_calculation run time = {self.calculation_time} seconds")
        print(
            f"<fast_model_padf.run_fast_model_calculation> Total contributing contacts (for normalization) = {self.total_contribs}")
        self.write_calculation_summary()
        
# if __name__ == '__main__':
#     modelp = ModelPadfCalculator()
