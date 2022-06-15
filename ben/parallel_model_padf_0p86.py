"""
Parallel Model PADF Calculator

@author: andrewmartin, jack-binns
"""

import numpy as np
import time
import multiprocessing as mp
import numba
import math as m
import matplotlib.pyplot as plt
import os

@numba.njit()
def fast_vec_angle(x1, x2, x3, y1, y2, y3):
    """
    Returns the angle between two vectors
    in range 0 - 90 deg
    :return theta in radians
    """
    mag1 = m.sqrt(x1 ** 2 + x2 ** 2 + x3 ** 2)
    mag2 = m.sqrt(y1 ** 2 + y2 ** 2 + y3 ** 2)
    dot = x1 * y1 + x2 * y2 + x3 * y3
    o = m.acos(dot / (mag1 * mag2))
    if 0.0 <= o < m.pi:
        return o
    else:
        return -1.0


@numba.njit()
def fast_vec_difmag(x1, x2, x3, y1, y2, y3):
    """
    :return: Magnitude of difference between two vectors
    """
    return m.sqrt((y1 - x1) ** 2 + (y2 - x2) ** 2 + (y3 - x3) ** 2)


@numba.njit()
def fast_vec_subtraction(x1, x2, x3, y1, y2, y3):
    """
    Vector subtraction vastly accelerated up by njit
    :return:
    """
    return [(y1 - x1), (y2 - x2), (y3 - x3)]


def make_interaction_sphere(probe, center, atoms):
    sphere = []
    for tar_1 in atoms:
        r_ij = fast_vec_difmag(center[0], center[1], center[2], tar_1[0], tar_1[1], tar_1[2])
        if r_ij != 0.0 and r_ij <= probe:
            sphere.append(tar_1)
    return sphere


def cif_edit_reader(raw, ucds):
    print("Finding the asymmetric unit [cif_edit_reader]...")
    atom_loop_count = 0 # counts the number of _atom_ labels
    atoms = []
    with open(raw, 'r') as foo:
        for line in foo:
            if '_atom_site_' in line:
                atom_loop_count += 1

    with open(raw, 'r') as foo:
        if atom_loop_count == 8:
            for line in foo:
                sploot = line.split()
                if len(sploot) == atom_loop_count:        # VESTA TYPE
                    if sploot[7] != 'H':
                        if "(" in sploot[2]:
                            subsploot = sploot[2].split("(")
                            raw_x = float(subsploot[0])
                        else:
                            raw_x = float(sploot[2])
                        if "(" in sploot[3]:
                            subsploot = sploot[3].split("(")
                            raw_y = float(subsploot[0])
                        else:
                            raw_y = float(sploot[3])
                        if "(" in sploot[4]:
                            subsploot = sploot[4].split("(")
                            raw_z = float(subsploot[0])
                        else:
                            raw_z = float(sploot[4])
                        raw_atom = [float(raw_x * ucds[0]), float(raw_y * ucds[1]),
                                    float(raw_z * ucds[2])]
                        atoms.append(raw_atom)
        elif atom_loop_count == 5:      # AM TYPE
            for line in foo:
                sploot = line.split()
                if len(sploot) == atom_loop_count:
                    if sploot[1][0] != 'H':
                        if "(" in sploot[2]:
                            subsploot = sploot[2].split("(")
                            raw_x = float(subsploot[0])
                        else:
                            raw_x = float(sploot[2])
                        if "(" in sploot[3]:
                            subsploot = sploot[3].split("(")
                            raw_y = float(subsploot[0])
                        else:
                            raw_y = float(sploot[3])
                        if "(" in sploot[4]:
                            subsploot = sploot[4].split("(")
                            raw_z = float(subsploot[0])
                        else:
                            raw_z = float(sploot[4])
                        raw_atom = [float(raw_x * ucds[0]), float(raw_y * ucds[1]),
                                    float(raw_z * ucds[2])]
                        atoms.append(raw_atom)
    print("Asymmetric unit contains ", len(atoms), " atoms found in ", raw)
    np.array(atoms)
    foo.close()
    return atoms


def read_xyz(file):
    print("Finding extended atom set [read_xyz]...")
    raw_x = []
    raw_y = []
    raw_z = []
    with open(file, "r") as xyz:
        for line in xyz:
            splot = line.split()
            if len(splot) == 4:
                raw_x.append(splot[1])
                raw_y.append(splot[2])
                raw_z.append(splot[3])
            elif len(splot) == 3:
                raw_x.append(splot[0])
                raw_y.append(splot[1])
                raw_z.append(splot[2])
    raw_x = [float(x) for x in raw_x]
    raw_y = [float(y) for y in raw_y]
    raw_z = [float(z) for z in raw_z]
    raw_atoms = np.column_stack((raw_x, raw_y, raw_z))
    print("Extended atom set contains ", len(raw_x), " atoms found in " + file)
    return raw_atoms


class ModelPADF:

    def __init__(self):

        self.root = "/Users/andrewmartin/Work/Teaching/2020/ONPS2186/codes/model-padf-master/"

        self.project = "1al1/"
        self.xyz_name = "1al1_ex.xyz"  # the xyz file contains the cartesian coords of the crystal structure expanded
        # to include r_probe
        self.cif_name = "1al1_edit.cif"  # the cif containing the asymmetric unit. This often needs to be edited for
        # PDB CIFs hence '_edit' here

        # self.output_path = root + project + xyz_name[:-4]

        self.ucds = [62.35000, 62.35000, 62.35000, 90.0000, 90.0000, 90.0000]

        # probe radius
        self.r_probe = 10.0
        self.angular_bin = 2.0
        self.r_dist_bin = 0.1
        self.probe_theta_bin = 10.0
        self.r_power = 2
        # If you wish to compute the final PADFs from a
        # partial data set use this flag and input the loop
        # at which the data is converged (check the _cosim.dat plot)
        self.convergence_check_flag = False

        self.mode = 'rrprime'

        self.logname = "parameter_log_file.txt"

        self.processor_num = 2
        self.Pool = mp.Pool(self.processor_num)
        self.loops = 0
        self.verbosity = 0

        self.fourbody = True

        self.Theta = np.zeros(0)

        self.asymm = []
        self.raw_extended_atoms = []
        self.extended_atoms = []

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['Pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    #
    #  Writes all the input parameters to a log file
    #
    def write_all_params_to_file(self, name="None", script="parallel_model_padf_0p2_am.py"):

        if name == "None":
            f = open(self.root + self.project + self.logname, 'w')
        else:
            f = open(name, 'w')
        f.write("# log of input parameters\n")

        if script != "None":
            f.write("# generated by " + script + "\n")

        a = self.__dict__
        for d, e in a.items():
            # list of parameters to exclude
            # if d in []:
            #    print("found one", d)
            #    continue

            # write the parameter to log file
            f.write(d + " = " + str(e) + "\n")

        f.close()

    def clean_project_folder(self):
        print("cleaning work folder...")
        for i in range(1, int(self.loops)+2):
            os.remove(self.root + self.project + self.project[:-1]+ '_Theta_loop_' + str(i) + '.npy')

        for j in range(int(self.processor_num)):
            os.remove(self.root + self.project + self.project[:-1] + '_Theta_' + str(j) + '.npy')


    def run(self):
        """
        Sets the calculation mode
        :return:
        """
        if self.mode == 'rrprime':
            print("Calculating r = r' slice")
            self.run_stm_rrprime()
        elif self.mode == 'rrtheta':
            print("Calculating r, r', theta slices")
            # run_rrtheta(model_padf_instance)
            pass
        elif self.mode == 'stm':
            print("Calculating Theta(r,r',theta) directly...")
            self.run_stm()

    def generate_empty_theta(self, shape):
        """
        Sets up the empty Theta matrix
        :return: Empty numpy array Theta
        """
        if shape == 2:
            Theta = np.zeros((int(self.r_probe / self.r_dist_bin), int(m.pi / m.radians(self.angular_bin))))
            print("Creating empty Theta slice :", Theta.shape)
        elif shape == 3:
            Theta = np.zeros(
                (int(self.r_probe / self.r_dist_bin), int(self.r_probe / self.r_dist_bin),
                 int(m.pi / m.radians(self.angular_bin))))
        else:
            print("Please supply Theta dimension [generate_empty_theta]")
            Theta = np.zeros(0)
        print("Creating empty Theta :", Theta.shape)
        return Theta

    def clean_extended_atoms(self):
        """
        Trims the length of the extended atoms to the set probed by
        the r_probe and asymmetric unit
        :return:
        """
        clean_ex = []
        # print(len(self.asymm))
        # print(len(self.raw_extended_atoms))
        for ex_atom in self.raw_extended_atoms:
            for as_atom in self.asymm:
                diff = fast_vec_difmag(ex_atom[0], ex_atom[1], ex_atom[2], as_atom[0], as_atom[1], as_atom[2])
                if abs(diff) <= self.r_probe:
                    clean_ex.append(ex_atom)
                    break
                else:
                    continue
        clean_ex = np.array(clean_ex)
        print("Extended atom set has been reduced to ", len(clean_ex), " atoms within", self.r_probe, "radius ")
        return np.array(clean_ex)

    def add_cor_vec_to_theta(self, cor_vec, array):
        """
        Adds a
        :param array: Theta matrix
        :param cor_vec: 3d correlation vector, r1, r2, theta
        :return: modified theta
        """
        r_yard_stick = np.arange(self.r_dist_bin, self.r_probe + self.r_dist_bin, self.r_dist_bin)
        th_yard_stick = np.arange(0, m.pi, m.radians(self.angular_bin))
        # find the r1 index:
        r1_index = (np.abs(r_yard_stick - cor_vec[0])).argmin()
        r2_index = (np.abs(r_yard_stick - cor_vec[1])).argmin()
        th_index = (np.abs(th_yard_stick - cor_vec[-1])).argmin()
        # print(self.Theta[index_vec[0], index_vec[1], index_vec[2]])
        array[r1_index, r2_index, th_index] += 1
        # print(self.Theta[index_vec[0], index_vec[1], index_vec[2]])

    def add_cor_vec_to_slice(self, cor_vec, array):
        """
        Adds a
        :param array: Theta matrix
        :param cor_vec: 2d correlation vector, r1,  theta
        :return: modified theta
        """
        r_yard_stick = np.arange(self.r_dist_bin, self.r_probe + self.r_dist_bin, self.r_dist_bin)
        th_yard_stick = np.arange(0, m.pi, m.radians(self.angular_bin))
        # find the r1 index:
        r1_index = (np.abs(r_yard_stick - cor_vec[0])).argmin()
        th_index = (np.abs(th_yard_stick - cor_vec[-1])).argmin()
        index_vec = [r1_index, th_index]
        # print(self.Theta[index_vec[0], index_vec[1], index_vec[2]])
        array[index_vec[0], index_vec[1]] += 1
        # print(self.Theta[index_vec[0], index_vec[1], index_vec[2]])

    def extract_rrprimetheta_slice(self, array):
        print("Extracting Theta(r = r', theta)...")
        pro_Theta = np.zeros(array.shape[1:])
        print("pro_Theta.shape")
        print(pro_Theta.shape)
        for r in range(0, array.shape[0], 1):
            for q in range(0, array.shape[2], 1):
                pro_Theta[r, q] = array[r, r, q]
        plt.imshow(pro_Theta)
        plt.show()
        np.savetxt(self.root + self.project + 'rrp_theta_slice.dat', pro_Theta)

    def reflection_correction_slice(self, array):
        print("Reflecting slice...")
        ref_array = np.zeros(array.shape)
        flipped = np.flip(array, 1)
        ref_array = np.add(array, flipped)
        plt.imshow(ref_array)
        plt.show()
        return ref_array

    def sin_theta_correction_slice(self, array):
        print("Applying sin(theta) correction...")
        for (r_i, t_i), corr in np.ndenumerate(array):
            # print(corr, t_i, r_i)
            # print((t_i * self.angular_bin))
            if 0.0 < (t_i * self.angular_bin) < 180.0:
                factor = np.sin(np.deg2rad(t_i * self.angular_bin))
                # print(factor)
            else:
                factor = 1.0
            array[r_i, t_i] = corr / factor
        plt.imshow(array)
        plt.show()
        return array

    def average_theta_correction_slice(self, array):
        """
        :param array:
        :return:
        """
        print("Subtracting average over theta")
        r_n = np.arange(0, array.shape[0])
        for n in r_n:
            average = np.average(array[n, :])
            array[n, :] = array[n, :] - average
        plt.imshow(array)
        plt.show()
        return array

    def rpower_correction(self, array, power):
        print("Applying 1/r^n correction...")
        for (r_i, t_i), corr in np.ndenumerate(array):
            if r_i > 0:
                real_r = r_i * self.r_dist_bin
            else:
                real_r = 1.0
            array[r_i, t_i] = array[r_i, t_i] / real_r ** power
        plt.imshow(array)
        plt.show()
        return array

    def slice_corrections(self, array):
        """
        Perform corrections to r=r' slice
        :param array:
        :return: corrected slice
        """
        # array = np.load(self.root + self.project + self.project[:-2] + '_slice_total_sum.npy')
        array = self.sin_theta_correction_slice(array)
        array = self.average_theta_correction_slice(array)
        array = self.reflection_correction_slice(array)
        array = self.rpower_correction(array, 1)
        print(array.shape)

    def parallel_pool_npy_accounting(self, j):
        """
        Sums arrays together for each cycle
        :return:
        """
        BigTheta = self.generate_empty_theta(3)
        for i in range(int(self.processor_num)):
            # print(i)
            chunk_Theta = np.load(self.root + self.project + self.project[:-1] + '_Theta_' + str(i) + '.npy')
            BigTheta = np.add(BigTheta, chunk_Theta)
        np.save(self.root + self.project + self.project[:-1] + '_Theta_loop_' + str(j), BigTheta)


    def parallel_pool_slice_accounting(self, j):
        """
        Sums arrays together for each cycle
        :return:
        """
        BigTheta = self.generate_empty_theta(2)
        for i in range(int(self.processor_num)):
            # print(i)
            chunk_Theta = np.load(self.root + self.project + self.project[:-1] + '_Theta_' + str(i) + '.npy')
            BigTheta = np.add(BigTheta, chunk_Theta)
        np.save(self.root + self.project + self.project[:-1] + '_Theta_loop_' + str(j), BigTheta)

    def add_bodies_to_theta_pool(self, k, a_i):
        """
        Calculates all three- and four-body contacts and adds them to Theta
        :return:
        """
        start = time.time()
        Theta = self.generate_empty_theta(3)
        print("Thread ", str(k), ":")
        print("Calculating contacts and adding to Theta...")
        target_atoms = np.array(make_interaction_sphere(self.r_probe, a_i, self.extended_atoms))
        print("Thread ", str(k), ": correlation sphere contains ", len(target_atoms), "atoms")
        for a_j in target_atoms:
            for a_k in target_atoms:
                # Find vectors, differences, and angles:
                r_ij = fast_vec_difmag(a_i[0], a_i[1], a_i[2], a_j[0], a_j[1], a_j[2])
                r_ik = fast_vec_difmag(a_i[0], a_i[1], a_i[2], a_k[0], a_k[1], a_k[2])
                ij = np.array(fast_vec_subtraction(a_i[0], a_i[1], a_i[2], a_j[0], a_j[1], a_j[2]))
                ik = np.array(fast_vec_subtraction(a_i[0], a_i[1], a_i[2], a_k[0], a_k[1], a_k[2]))
                theta = fast_vec_angle(ij[0], ij[1], ij[2], ik[0], ik[1], ik[2])
                if 0.0 <= theta <= m.pi:
                    self.add_cor_vec_to_theta([r_ij, r_ik, theta], Theta)
                else:
                    continue
                if self.fourbody:
                    k_target_atoms = np.array(make_interaction_sphere(self.r_probe, a_k, self.extended_atoms))
                    for a_m in k_target_atoms:
                        r_km = fast_vec_difmag(a_k[0], a_k[1], a_k[2], a_m[0], a_m[1], a_m[2])
                        km = np.array(fast_vec_subtraction(a_k[0], a_k[1], a_k[2], a_m[0], a_m[1], a_m[2]))
                        theta_km = fast_vec_angle(ij[0], ij[1], ij[2], km[0], km[1], km[2])
                        if 0.0 <= theta_km <= m.pi:
                            self.add_cor_vec_to_theta([r_ij, r_km, theta_km], Theta)
                        else:
                            continue
                else:
                    continue
        end = time.time()
        print("Execution time = ", end - start, " seconds")
        # Save the Theta array as is:
        np.save(self.root + self.project + self.project[:-1] + '_Theta_' + str(k), Theta)

    def add_bodies_to_rrprime_pool(self, k, a_i):
        """
        Calculates all three- and four-body contacts and adds them to the Theta slice
        r = r'
        :return:
        """
        start = time.time()
        Theta = self.generate_empty_theta(2)
        print("Calculating contacts and adding to Theta slice...")
        target_atoms = np.array(make_interaction_sphere(self.r_probe, a_i, self.extended_atoms))
        print("Thread ", str(k), ": correlation sphere contains ", len(target_atoms), "atoms")
        for a_j in target_atoms:
            for a_k in target_atoms:
                # Find vectors, differences, and angles:
                r_ij = fast_vec_difmag(a_i[0], a_i[1], a_i[2], a_j[0], a_j[1], a_j[2])
                r_ik = fast_vec_difmag(a_i[0], a_i[1], a_i[2], a_k[0], a_k[1], a_k[2])
                diff = abs(r_ij - r_ik)
                if diff < self.r_dist_bin:
                    ij = np.array(fast_vec_subtraction(a_i[0], a_i[1], a_i[2], a_j[0], a_j[1], a_j[2]))
                    ik = np.array(fast_vec_subtraction(a_i[0], a_i[1], a_i[2], a_k[0], a_k[1], a_k[2]))
                    theta = fast_vec_angle(ij[0], ij[1], ij[2], ik[0], ik[1], ik[2])
                    if 0.0 <= theta <= m.pi:
                        self.add_cor_vec_to_slice([r_ij, theta], Theta)
                    else:
                        continue
                else:
                    continue
                if self.fourbody:
                    k_target_atoms = np.array(make_interaction_sphere(self.r_probe, a_k, self.extended_atoms))
                    for a_m in k_target_atoms:
                        r_km = fast_vec_difmag(a_k[0], a_k[1], a_k[2], a_m[0], a_m[1], a_m[2])
                        diff_k = abs(r_ij - r_km)
                        if diff_k < self.r_dist_bin:
                            km = np.array(fast_vec_subtraction(a_k[0], a_k[1], a_k[2], a_m[0], a_m[1], a_m[2]))
                            theta_km = fast_vec_angle(ij[0], ij[1], ij[2], km[0], km[1], km[2])
                            if 0.0 <= theta_km <= m.pi:
                                self.add_cor_vec_to_slice([r_ij, theta_km], Theta)
                            else:
                                continue
                else:
                    continue
        end = time.time()
        print("Execution time = ", end - start, " seconds")
        # Save the Theta array as is:
        np.save(self.root + self.project + self.project[:-1] + '_Theta_' + str(k), Theta)

    def run_stm(self):
        """
        Runs the Straight-To-Matrix model PADF calculation
        :return:
        """
        start = time.time()
        # Get the asymmetric unit
        self.asymm = cif_edit_reader(self.root + self.project + self.cif_name,
                                     self.ucds)  # Create the asymmetric unit
        # Get the extended atoms
        self.raw_extended_atoms = read_xyz(self.root + self.project + self.xyz_name)
        self.extended_atoms = self.clean_extended_atoms()

        """
        Here we do the chunking for sending to threads
        """
        np.random.shuffle(self.asymm)  # Shuffle the asymetric unit
        self.loops = int(len(self.asymm) / self.processor_num)  # The number of loops for complete calculation

        # chunked add_bodies_to_theta:
        for j in np.arange(1, int(self.loops) + 2, 1):
            print(str(j) + " / " + str(int(self.loops) + 1))
            cluster_asymm = self.asymm[
                            (j - 1) * self.processor_num:j * self.processor_num]  # Create a chunk of the asym
            processes = [mp.Process(target=self.add_bodies_to_theta_pool, args=(i, cl_atom)) for i, cl_atom in
                         enumerate(cluster_asymm)]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            # Crunch the npy arrays together for this loop
            self.parallel_pool_npy_accounting(j)

        # Check to see if the folder should be cleaned

        # Crunch the npy arrays together for all loops:
        BigTheta = self.generate_empty_theta(3)
        for j in np.arange(1, int(self.loops) + 2, 1):
            chunk_Theta = np.load(self.root + self.project + self.project[:-1] + '_Theta_loop_' + str(j) + '.npy')
            BigTheta = np.add(BigTheta, chunk_Theta)
        # print(BigTheta)
        print(BigTheta.shape)

        np.save(self.root + self.project + self.project[:-1] + '_Theta_total_sum', BigTheta)
        end = time.time()
        if self.verbosity == 0:
            self.clean_project_folder()
        print("Total run time = ", end - start, " seconds")
        self.extract_rrprimetheta_slice(BigTheta)

    def run_stm_rrprime(self):
        """
        Runs the Straight-To-Matrix model PADF calculation
        for the special case of r=r', theta
        :return:
        """
        start = time.time()
        # Get the asymmetric unit
        self.asymm = cif_edit_reader(self.root + self.project + self.cif_name,
                                     self.ucds)  # Create the asymmetric unit
        # Get the extended atoms
        self.raw_extended_atoms = read_xyz(self.root + self.project + self.xyz_name)
        self.extended_atoms = self.clean_extended_atoms()

        """
        Here we do the chunking for sending to threads
        """
        np.random.shuffle(self.asymm)  # Shuffle the asymetric unit
        self.loops = int(len(self.asymm) / self.processor_num)  # The number of loops for complete calculation

        # chunked add_bodies_to_theta:
        for j in np.arange(1, int(self.loops) + 2, 1):
            print("Loop : " + str(j) + " / " + str(int(self.loops) + 1))
            cluster_asymm = self.asymm[
                            (j - 1) * self.processor_num:j * self.processor_num]  # Create a chunk of the asym
            processes = [mp.Process(target=self.add_bodies_to_rrprime_pool, args=(i, cl_atom)) for i, cl_atom in
                         enumerate(cluster_asymm)]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            # Crunch the npy arrays together for this loop
            self.parallel_pool_slice_accounting(j)
        # Check to see if the folder should be cleaned

        # Crunch the npy arrays together for all loops:
        BigTheta = self.generate_empty_theta(2)
        for j in np.arange(1, int(self.loops) + 2, 1):
            chunk_Theta = np.load(self.root + self.project + self.project[:-1] + '_Theta_loop_' + str(j) + '.npy')
            BigTheta = np.add(BigTheta, chunk_Theta)
        print(BigTheta.shape)
        np.save(self.root + self.project + self.project[:-1] + '_slice_total_sum', BigTheta)
        end = time.time()
        if self.verbosity == 0:
            self.clean_project_folder()
        print("Total run time = ", end - start, " seconds")
        plt.imshow(BigTheta)
        plt.show()


if __name__ == '__main__':
    modelp = ModelPADF()
    modelp.write_all_params_to_file()
