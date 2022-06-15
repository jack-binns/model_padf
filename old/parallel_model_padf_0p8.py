import numpy as np
import time
import multiprocessing as mp
import os


def append_new_line(fn, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(fn, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def read_xyz(file):
    raw_atoms = []
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
    raw_x = [float(x) for x in raw_x]
    raw_y = [float(y) for y in raw_y]
    raw_z = [float(z) for z in raw_z]
    raw_atoms = np.column_stack((raw_x, raw_y, raw_z))
    print("I found ", len(raw_x), " atoms in " + file)
    return raw_atoms


def clip_atoms(atom_list, ucdims):
    raw_x = []
    raw_y = []
    raw_z = []
    a_lim = float(ucdims[0])
    b_lim = float(ucdims[1])
    c_lim = float(ucdims[2])

    for atom in atom_list:
        # if -a_lim <= atom[0] <= a_lim and -b_lim <= atom[1] <= b_lim and -c_lim <= atom[2] <= c_lim:
        if 0 <= atom[0] <= a_lim and 0 <= atom[1] <= b_lim and 0 <= atom[2] <= c_lim:
            raw_x.append(atom[0])
            raw_y.append(atom[1])
            raw_z.append(atom[2])
    raw_x = [float(x) for x in raw_x]
    raw_y = [float(y) for y in raw_y]
    raw_z = [float(z) for z in raw_z]
    clip = np.column_stack((raw_x, raw_y, raw_z))
    return clip


def read_ucds(file):
    with open(file, "r") as xyz:
        for line in xyz:
            splot = line.split()
            if len(splot) == 6:
                a = splot[0]
                b = splot[1]
                c = splot[2]
                al = splot[3]
                be = splot[4]
                ga = splot[5]
                unit_cell_dims = [a, b, c, al, be, ga]
            else:
                continue
    unit_cell_dims = [float(x) for x in unit_cell_dims]
    return unit_cell_dims


def tally_distances_indep(clip, atoms, probe, a_lim, b_lim, c_lim, path):
    print("Calculating two-body array...")
    distances = []
    for atom in clip:
        for target in atoms:
            if -a_lim <= atom[0] <= a_lim and -b_lim <= atom[1] <= b_lim and -c_lim <= atom[2] <= c_lim:
                # print(atom,target)
                r_at = np.sqrt((atom[0] - target[0]) ** 2 + (atom[1] - target[1]) ** 2 + (atom[2] - target[2]) ** 2)
                # print("r_at", r_at)
                if 0.0 < r_at <= probe:
                    distances.append(r_at)
                    thba_string = str("{:.3f}".format(float(r_at))) + " " + str(
                        "{:.3f}".format(float(0.0))) + " " + str(atom[0]) + " " + str(atom[1]) + " " + str(
                        atom[2]) + " " + str(target[0]) + " " + str(target[1]) + " " + str(target[2]) + " " + str(
                        target[0]) + " " + str(target[1]) + " " + str(target[2])
                    append_new_line(path + "_two_body_array.dat", thba_string)
    return distances


def pdb_cif_trimmer(cif):
    foo = open(cif)
    raw = []
    for line in foo:
        sploot = line.split()
        # print(sploot)
        if 'H' not in sploot[2]:
            raw_atom = [float(sploot[11]), float(sploot[12]), float(sploot[13])]
            raw.append(raw_atom)
    np.array(raw)
    # print(raw)
    foo.close()
    return raw


def pdb_raw_trimmer(raw):
    foo = open(raw)
    raw = []
    for line in foo:
        sploot = line.split()
        # print(sploot)
        if 'H' not in sploot[-1]:
            raw_atom = [float(sploot[6]), float(sploot[7]), float(sploot[8])]
            raw.append(raw_atom)
    np.array(raw)
    # print(raw)
    foo.close()
    return raw


def pdf_histogram(distances, limit, bin_size):
    histogram_array = []
    print("Calculating 2 body histogram...")
    for binrange in np.arange(0, limit, bin_size):
        # print(binrange - bin_size, binrange)
        tally = 0.0
        for r in distances:
            # print("r,", r)
            if binrange - bin_size < r <= binrange:
                tally = tally + 1.0
        histogram_array.append((binrange, tally))
    np.array(histogram_array)
    return histogram_array


def reflection_corr(array):
    output = []
    for corr in array:
        mirror_corr_theta = 180 - corr[1]
        mirror_corr = [corr[0], mirror_corr_theta, corr[2], corr[3], corr[4], corr[5], corr[6], corr[7], corr[8],
                       corr[9], corr[10]]
        output.append(corr)
        output.append(mirror_corr)
    output = np.array(output)
    return output


def tb_histogram(three_body_path, dist_limit, angular_bin_size, dist_bin_size, mirror_flag):
    start = time.time()
    angle_dist_histogram_list = []
    angular_range = np.arange(0, 180 + angular_bin_size, angular_bin_size)
    dist_range = np.arange(0, dist_limit + dist_bin_size, dist_bin_size)
    theta_r_matrix = np.zeros((len(angular_range), len(dist_range)))
    print("theta_r_matrix.shape", theta_r_matrix.shape)
    n = 0
    print("Binning...")
    with open(three_body_path) as fileobject:
        for corr in fileobject:
            sploot = corr.split()
            # print(sploot[0], type(sploot[0]))
            n = n + 1
            # print(n, " / ", len(three_body_list))
            # print(n)
            r_curr = float(sploot[0])
            theta_curr = float(sploot[1])
            for r_bin in dist_range:
                diff = r_curr - r_bin
                if diff <= dist_bin_size:
                    # print("HIT in R")
                    selected_r_bin = r_bin
                    selected_r_bin_i = np.where(dist_range == selected_r_bin)
                    break
                else:
                    continue
            for a_bin in angular_range:
                diff = theta_curr - a_bin
                if diff <= angular_bin_size:
                    selected_th_bin = a_bin
                    selected_th_bin_i = np.where(angular_range == selected_th_bin)
                    selected_th_bin_i_reflection = np.where(angular_range == 180 - selected_th_bin)
                    break
                else:
                    continue
            theta_r_matrix[selected_th_bin_i[0], selected_r_bin_i[0]] = theta_r_matrix[
                                                                            selected_th_bin_i[0], selected_r_bin_i[
                                                                                0]] + 1.0
            if mirror_flag:
                theta_r_matrix[selected_th_bin_i_reflection[0], selected_r_bin_i[0]] = theta_r_matrix[
                                                                                           selected_th_bin_i_reflection[
                                                                                               0], selected_r_bin_i[
                                                                                               0]] + 1.0
    end = time.time()
    print(n, " correlations in three-body array")
    print("Histogram execution time = ", end - start, " s")
    return theta_r_matrix


def fb_histogram(four_body_path, dist_limit, angular_bin_size, dist_bin_size, mirror_flag):
    start = time.time()
    angular_range = np.arange(0, 180 + angular_bin_size, angular_bin_size)
    dist_range = np.arange(0, dist_limit + dist_bin_size, dist_bin_size)
    theta_r_matrix = np.zeros((len(angular_range), len(dist_range)))
    print("theta_r_matrix.shape", theta_r_matrix.shape)
    n = 0
    print("Binning...")
    with open(four_body_path) as fileobject:
        for corr in fileobject:
            sploot = corr.split()
            n = n + 1
            r_curr = float(sploot[0])
            theta_curr = float(sploot[1])
            for r_bin in dist_range:
                diff = r_curr - r_bin
                if diff <= dist_bin_size:
                    selected_r_bin = r_bin
                    selected_r_bin_i = np.where(dist_range == selected_r_bin)
                    break
                else:
                    continue
            for a_bin in angular_range:
                diff = theta_curr - a_bin
                if diff <= angular_bin_size:
                    selected_th_bin = a_bin
                    selected_th_bin_i = np.where(angular_range == selected_th_bin)
                    selected_th_bin_i_reflection = np.where(angular_range == 180 - selected_th_bin)
                    break
                else:
                    continue
            theta_r_matrix[selected_th_bin_i[0], selected_r_bin_i[0]] = theta_r_matrix[
                                                                            selected_th_bin_i[0], selected_r_bin_i[
                                                                                0]] + 1.0
            if mirror_flag:
                theta_r_matrix[selected_th_bin_i_reflection[0], selected_r_bin_i[0]] = theta_r_matrix[
                                                                                           selected_th_bin_i_reflection[
                                                                                               0], selected_r_bin_i[
                                                                                               0]] + 1.0
    end = time.time()
    print(n, " correlations in four-body array")
    print("Histogram execution time = ", end - start, " s")
    return theta_r_matrix


def pdb_reader(raw):
    foo = open(raw)
    raw = []
    for line in foo:
        sploot = line.split()
        if 'ATOM' in (sploot[0]):
            if 'H' not in sploot[-1]:
                raw_atom = [float(sploot[6]), float(sploot[7]), float(sploot[8])]
                raw.append(raw_atom)
    np.array(raw)
    # print(raw)
    foo.close()
    return raw


def cif_edit_reader(raw, ucds):
    foo = open(raw)
    atoms = []
    for line in foo:
        sploot = line.split()
        if len(sploot) == 8:
            if sploot[7] is not 'H':
                # print(len(sploot))
                # print(len(ucds))
                raw_x = float(sploot[2])
                raw_y = float(sploot[3])
                raw_z = float(sploot[4])
                id = str(sploot[7])
                # if raw_x < 0:
                #    raw_x = 1.0 - raw_x
                # if raw_x >= 1.0:
                #    raw_x = raw_x - 1.0
                # if raw_y < 0:
                #    raw_y = 1.0 - raw_y
                # if raw_y >= 1.0:
                #    raw_y = raw_y - 1.0
                # if raw_z < 0:
                #    raw_z = 1.0 - raw_z
                # if raw_z >= 1.0:
                #    raw_z = raw_z - 1.0
                raw_atom = [float(raw_x * ucds[0]), float(raw_y * ucds[1]),
                            float(raw_z * ucds[2])]
                atoms.append(raw_atom)
    np.array(atoms)
    # print(raw)
    foo.close()
    return atoms


def asym_reader(raw):
    foo = open(raw)
    atoms = []
    for line in foo:
        if len(line) == 7:
            sploot = line.split()
            raw_x = float(sploot[1])
            raw_y = float(sploot[2])
            raw_z = float(sploot[3])
            id = str(sploot[0])
            # if raw_x < 0:
            #    raw_x = 1.0 - raw_x
            # if raw_x >= 1.0:
            #    raw_x = raw_x - 1.0
            # if raw_y < 0:
            #    raw_y = 1.0 - raw_y
            # if raw_y >= 1.0:
            #    raw_y = raw_y - 1.0
            # if raw_z < 0:
            #    raw_z = 1.0 - raw_z
            # if raw_z >= 1.0:
            #   raw_z = raw_z - 1.0
            raw_atom = [float(raw_x), float(raw_y), float(raw_z)]
            atoms.append(raw_atom)
    atoms = np.array(atoms)
    # print(raw)
    foo.close()
    return atoms


def make_interaction_sphere(probe, center, atoms):
    sphere = []
    for tar_1 in atoms:
        r_ij = np.sqrt((center[0] - tar_1[0]) ** 2 + (center[1] - tar_1[1]) ** 2 + (center[2] - tar_1[2]) ** 2)
        if r_ij != 0.0 and r_ij <= probe:
            sphere.append(tar_1)
    sphere = np.array(sphere)
    # print(sphere.shape)
    return sphere


def molecular_n_body_problem_detective(clip, atoms, probe, distance_bin, path):
    print("Calculating molecular three body array...")
    start = time.time()
    c = 1
    fourbody = True
    np.random.shuffle(clip)
    print(clip[0])
    for a_i in clip:
        print(c, "/", len(clip))
        target_atoms = make_interaction_sphere(probe, a_i, atoms)
        # print(target_atoms)
        print("Correlation sphere contains ", len(target_atoms), "atoms")
        for a_j in target_atoms:
            for a_k in target_atoms:
                r_ij = np.sqrt((a_i[0] - a_j[0]) ** 2 + (a_i[1] - a_j[1]) ** 2 + (a_i[2] - a_j[2]) ** 2)
                r_ik = np.sqrt((a_i[0] - a_k[0]) ** 2 + (a_i[1] - a_k[1]) ** 2 + (a_i[2] - a_k[2]) ** 2)
                if r_ij != 0.0 and 0.0 < (r_ij - r_ik) < distance_bin and r_ij <= probe:
                    ij = np.array(a_j - a_i)
                    ik = np.array(a_k - a_i)
                    theta = angle_between(ij, ik)
                    thba_string = str("{:.3f}".format(float(r_ij))) + " " + str(
                        "{:.3f}".format(float(np.degrees(theta))))
                    append_new_line(path + "_three_body_array.dat", thba_string)
                    if fourbody:
                        # make the target atoms for k
                        target_atoms_k = make_interaction_sphere(probe, a_k, atoms)
                        for a_m in target_atoms_k:
                            r_km = np.sqrt((a_m[0] - a_k[0]) ** 2 + (a_m[1] - a_k[1]) ** 2 + (a_m[2] - a_k[2]) ** 2)
                            if r_km != 0.0 and 0.0 < (r_km - r_ik) < distance_bin and r_km <= probe:
                                km = np.array(a_m - a_k)
                                theta_k = angle_between(ij, km)
                                foba_string = str("{:.3f}".format(float(r_km))) + " " + str(
                                    "{:.3f}".format(float(np.degrees(theta_k))))
                                append_new_line(path + "_four_body_array.dat", foba_string)
                    else:
                        continue
        c = c + 1
    end = time.time()
    print("Execution time = ", end - start, " seconds")
    # return np.array(thba_list)


def molecular_n_body_problem_pool(i, a_i, atoms, probe, distance_bin, path):
    """
    :param i: chunk number
    :param a_i: target atom in asymmetric unitt
    :param atoms: full set of atoms
    :param probe: max distance of the probe sphere
    :param distance_bin: resolution in r
    :param path: where the array will be written to
    :return:
    """
    print("Calculating 3-, 4-body arrays...")
    start = time.time()
    fourbody = True
    target_atoms = make_interaction_sphere(2 * probe, a_i, atoms)
    print("Correlation sphere contains ", len(target_atoms), "atoms")
    for a_j in target_atoms:
        for a_k in target_atoms:
            r_ij = np.sqrt((a_i[0] - a_j[0]) ** 2 + (a_i[1] - a_j[1]) ** 2 + (a_i[2] - a_j[2]) ** 2)
            r_ik = np.sqrt((a_i[0] - a_k[0]) ** 2 + (a_i[1] - a_k[1]) ** 2 + (a_i[2] - a_k[2]) ** 2)
            if r_ij != 0.0 and 0.0 < (r_ij - r_ik) < distance_bin and r_ij <= probe:
                ij = np.array(a_j - a_i)
                ik = np.array(a_k - a_i)
                theta = angle_between(ij, ik)
                thba_string = str("{:.3f}".format(float(r_ij))) + " " + str("{:.3f}".format(float(np.degrees(theta))))
                append_new_line(path + "_" + str(i) + "_three_body_array.dat", thba_string)
                if fourbody:
                    target_atoms_k = target_atoms  # this is quicker than making a second sphere for k
                    for a_m in target_atoms_k:
                        r_km = np.sqrt((a_m[0] - a_k[0]) ** 2 + (a_m[1] - a_k[1]) ** 2 + (a_m[2] - a_k[2]) ** 2)
                        if r_km != 0.0 and 0.0 < (r_km - r_ik) < distance_bin and r_km <= probe:
                            km = np.array(a_m - a_k)
                            theta_k = angle_between(ij, km)
                            foba_string = str("{:.3f}".format(float(r_km))) + " " + str(
                                "{:.3f}".format(float(np.degrees(theta_k))))
                            append_new_line(path + "_" + str(i) + "_four_body_array.dat", foba_string)
                else:
                    continue
    end = time.time()
    print("Execution time = ", end - start, " seconds")


def three_body_rnr(clip, atoms, probe, distance_bin, path):
    print("Calculating three body array with variable r...")
    start = time.time()
    c = 1
    for a_i in clip:
        print(c, "/", len(clip))
        print(a_i)
        target_atoms = make_interaction_sphere(probe, a_i, atoms)
        # print(target_atoms)
        print("Correlation sphere contains ", len(target_atoms), "atoms")
        for a_j in target_atoms:
            for a_k in target_atoms:
                r_ij = np.sqrt((a_i[0] - a_j[0]) ** 2 + (a_i[1] - a_j[1]) ** 2 + (a_i[2] - a_j[2]) ** 2)
                r_ik = np.sqrt((a_i[0] - a_k[0]) ** 2 + (a_i[1] - a_k[1]) ** 2 + (a_i[2] - a_k[2]) ** 2)
                ij = np.array(a_j - a_i)
                ik = np.array(a_k - a_i)
                theta = angle_between(ij, ik)
                thba_string = str("{:.3f}".format(float(r_ij))) + " " + str("{:.3f}".format(float(r_ik))) + " " + str(
                    "{:.3f}".format(float(np.degrees(theta)))) + " " + str(a_i[0]) + " " + str(a_i[1]) + " " + str(
                    a_i[2]) + " " + str(a_j[0]) + " " + str(a_j[1]) + " " + str(a_j[2]) + " " + str(a_k[0]) + " " + str(
                    a_k[1]) + " " + str(a_k[2])
                append_new_line(path + "_rrth.dat", thba_string)
        c = c + 1
    end = time.time()
    print("Execution time = ", end - start, " seconds")
    # return np.array(thba_list)


def three_body_rnr_theta(clip, atoms, probe, probe_th, path):
    theta_tol = 2.0
    print("Calculating three body array with variable r...")
    start = time.time()
    c = 1
    for a_i in clip:
        print(c, "/", len(clip))
        print(a_i)
        target_atoms = make_interaction_sphere(probe, a_i, atoms)
        # print(target_atoms)
        print("Correlation sphere contains ", len(target_atoms), "atoms")
        for a_j in target_atoms:
            for a_k in target_atoms:
                r_ij = np.sqrt((a_i[0] - a_j[0]) ** 2 + (a_i[1] - a_j[1]) ** 2 + (a_i[2] - a_j[2]) ** 2)
                r_ik = np.sqrt((a_i[0] - a_k[0]) ** 2 + (a_i[1] - a_k[1]) ** 2 + (a_i[2] - a_k[2]) ** 2)
                ij = np.array(a_j - a_i)
                ik = np.array(a_k - a_i)
                theta = angle_between(ij, ik)
                theta = np.degrees(theta)
                if abs(theta - probe_th) < theta_tol:
                    # print(abs(theta - probe_th), theta)
                    thba_string = str("{:.3f}".format(float(r_ij))) + " " + str(
                        "{:.3f}".format(float(r_ik))) + " " + str(
                        "{:.3f}".format(float(theta))) + " " + str(a_i[0]) + " " + str(a_i[1]) + " " + str(
                        a_i[2]) + " " + str(a_j[0]) + " " + str(a_j[1]) + " " + str(a_j[2]) + " " + str(
                        a_k[0]) + " " + str(
                        a_k[1]) + " " + str(a_k[2])
                    append_new_line(path + "_rrth_" + str(probe_th) + ".dat", thba_string)
        c = c + 1
    end = time.time()
    print("Execution time = ", end - start, " seconds")
    # return np.array(thba_list)


def sin_theta_correction(t_r_matrix, th_bin_size):
    for (t_i, r_i), corr in np.ndenumerate(t_r_matrix):
        # print(corr, t_i, r_i )
        # print((t_i * th_bin_size))
        if 0.0 < (t_i * th_bin_size) < 180.0:
            factor = np.sin(np.deg2rad(t_i * th_bin_size))
        else:
            factor = 1.0
        t_r_matrix[t_i, r_i] = corr / factor
    return t_r_matrix


def average_theta_correction(t_r_matrix):
    r_n = np.arange(0, t_r_matrix.shape[1])
    for n in r_n:
        average = np.average(t_r_matrix[30:60, n])
        t_r_matrix[:, n] = t_r_matrix[:, n] - average
    return t_r_matrix


def rpower_correction(t_r_matrix, dist_bin, power):
    for (t_i, r_i), corr in np.ndenumerate(t_r_matrix):
        if r_i > 0:
            real_r = r_i * dist_bin
        else:
            real_r = 1.0
        t_r_matrix[t_i, r_i] = t_r_matrix[t_i, r_i] / real_r ** power
    return t_r_matrix


def parallel_histo(pool_num, parent_loop, root, angular_bin_size, dist_bin_size, dist_limit):
    summed_tba = []  # Empty sums
    summed_fba = []  # Empty sums
    for k in np.arange(0, int(pool_num - 1), 1):
        print("Summing three-body arrays for loop ", parent_loop)
        tba_path = root + "_" + str(k) + "_three_body_array.dat"
        tba_array = np.loadtxt(tba_path)  # load the kth three-body array
        [summed_tba.append(line) for line in tba_array]  # append the contents of the k tba to the sum
        print("Summing four-body arrays for loop ", parent_loop)
        fba_path = root + "_" + str(k) + "_four_body_array.dat"
        fba_array = np.loadtxt(fba_path)
        [summed_fba.append(line) for line in fba_array]
        print("Generating histograms...")
    loop_sum_path_tb = root + "_loop_sum_" + str(parent_loop) + "_three_body_array.dat"
    loop_sum_path_fb = root + "_loop_sum_" + str(parent_loop) + "_four_body_array.dat"
    np.savetxt(loop_sum_path_tb, summed_tba)
    np.savetxt(loop_sum_path_fb, summed_fba)
    tb_hist_unrefcor = tb_histogram(loop_sum_path_tb, r_probe, angular_bin, r_dist_bin,
                                    True)  # calculate the histogram for three-body sum
    fb_hist_unrefcor = fb_histogram(loop_sum_path_fb, r_probe, angular_bin, r_dist_bin,
                                    True)  # calculate the histogram for four-body sum

    # Write out the first stage histograms
    np.savetxt(root + "_unrefcor_loop_" + str(parent_loop) + "_tb.dat", tb_hist_unrefcor)
    np.savetxt(root + "_unrefcor_loop_" + str(parent_loop) + "_fb.dat", fb_hist_unrefcor)
    # Write out the sum:
    sum_hist = np.add(tb_hist_unrefcor, fb_hist_unrefcor)
    np.savetxt(root + "_unrefcor_loop_" + str(parent_loop) + "_sum.dat", sum_hist)
    # Saves the raw arrays - uncomment to get a very large results folder
    # np.savetxt(root + "_unrefcor_loop_" + str(parent_loop) + "_three_body_array.dat", summed_tba)
    # np.savetxt(root + "_unrefcor_loop_" + str(parent_loop) + "_four_body_array.dat", summed_fba)
    #   Do some cleaning of the work folder
    #   Comment out to save it
    for k in np.arange(0, int(pool_num - 1), 1):
        os.remove(root + "_" + str(k) + "_three_body_array.dat")
        os.remove(root + "_" + str(k) + "_four_body_array.dat")


def cossim_test(current_loop, path):
    # first we sum the histograms up to the previous loop:1al1_ex_unrefcor_loop_2_sum
    sumhist_old = np.ndarray.flatten(np.loadtxt(path + "_unrefcor_loop_1_sum.dat"))
    i = 1
    while i <= current_loop - 1:
        i_hist = np.ndarray.flatten(np.loadtxt(path + "_unrefcor_loop_" + str(i) + "_sum.dat"))
        sumhist_old = sumhist_old + i_hist
        i = i + 1

    # now we sum to the current loop:
    sumhist_new = np.ndarray.flatten(np.loadtxt(path + "_unrefcor_loop_1_sum.dat"))
    j = 1
    while j <= current_loop:
        j_hist = np.ndarray.flatten(np.loadtxt(path + "_unrefcor_loop_" + str(j) + "_sum.dat"))
        sumhist_new = sumhist_new + j_hist
        j = j + 1

    cos_sim_it = np.dot(sumhist_old, sumhist_new) / (np.linalg.norm(sumhist_old) * np.linalg.norm(sumhist_new))
    string = str(current_loop) + ' ' + str(cos_sim_it)
    append_new_line(path + "_cossim.dat", string)


def finalise_histograms(path, ang_bin, r_exp, loop):
    sumhist = np.loadtxt(path + "_unrefcor_loop_1_sum.dat")
    a = 1
    while a <= loop:
        j_hist = np.loadtxt(path + "_unrefcor_loop_" + str(a) + "_sum.dat")
        sumhist = sumhist + j_hist
        a = a + 1
    print("Applying sin(theta) correction...")
    sum_hist_st = sin_theta_correction(sumhist, ang_bin)
    np.savetxt(path + "_stcor.dat", sum_hist_st)
    print("Applying average(theta) correction...")
    sum_hist_avth = average_theta_correction(sum_hist_st)
    np.savetxt(path + "_atcor.dat", sum_hist_avth)
    print("Applying 1/r^n correction...")
    sum_hist_avth_r = rpower_correction(sum_hist_st, r_dist_bin, r_exp)
    np.savetxt(path + "_rcor.dat", sum_hist_avth_r)
    print("...done")


if __name__ == '__main__':
    print("Model PADF maker")

    root = "/home/jack/python/model_padf/"

    project = "1al1/test/"
    xyz_name = "1al1_ex.xyz"  # the xyz file contains the cartesian coords of the crystal structure expanded to include r_probe
    cif_name = "1al1_edit.cif"  # the cif containing the asymmetric unit. This often needs to be edited for PDB CIFs hence '_edit' here

    output_path = root + project + xyz_name[:-4]

    pool = mp.Pool(mp.cpu_count())  # change to set the number of processors

    print("Project: ", root + project + xyz_name)
    extended_atoms = read_xyz(root + project + xyz_name)
    print("I found ", len(extended_atoms), " atoms in the extended atom set")

    ucds = [62.35000, 62.35000, 62.35000, 90.0000, 90.0000, 90.0000]
    print("extended atoms are loaded...")

    asymm = cif_edit_reader(root + project + cif_name, ucds)  # Create the asymmetric unit
    print(len(asymm), " atoms in the asymmetric unit")

    # probe radius
    r_probe = 20.0
    angular_bin = 2.0
    r_dist_bin = 0.1
    probe_theta = 90.0
    r_power = 2
    # If you wish to compute the final PADFs from a
    # partial data set use this flag and input the loop
    # at which the data is converged (check the _cosim.dat plot)
    convergence_check_flag = False

    '''
    Calculation begins below 
    '''

    np.random.shuffle(asymm)  # shuffle the atoms in the asymmetric units
    conf_freq = 8  # This should be equal to the number of processors, this chunks the calculations into batches
    loops = int(len(asymm) / conf_freq)  # The number of loops for complete calculation

    for j in np.arange(1, int(loops) + 1, 1):
        print(str(j) + " / " + str(int(loops) + 1))
        cluster_asymm = asymm[(j - 1) * conf_freq:j * conf_freq]
        processes = [mp.Process(target=molecular_n_body_problem_pool,
                                args=(i, atom, extended_atoms, r_probe, r_dist_bin, output_path)) for i, atom in
                     enumerate(cluster_asymm)]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        # Perform binning on the loop from this round
        parallel_histo(conf_freq, j, output_path, angular_bin, r_dist_bin, r_probe)

        # Perform the cosine similarity test for convergence
        if j > 1:
            cossim_test(j, output_path) # calculates the cosine similarity for this loop


    # Now finalie the histograms either by selecting

    ###### NEW TO SUM ALL PREVIOUS LOOPS FIRST

    if convergence_check_flag:
        convergence_loop = input("converged loop: ")
        hist_path = output_path + "_unrefcor_loop_" + str(convergence_loop) + "_sum.dat"
        finalise_histograms(output_path, angular_bin, r_power, convergence_loop)
    else:
        hist_path = output_path + "_unrefcor_loop_" + str(int(loops) + 1) + "_sum.dat"
        finalise_histograms(output_path, angular_bin, r_power, loops)


