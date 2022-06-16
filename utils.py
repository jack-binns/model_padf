import numba
import math as m
import numpy as np
import atomic_z as atoms


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


def subject_atom_reader(raw, ucds=None):
    """
    An exceedingly ungainly function for reading
    in various file types
    """
    print("Finding the subject atoms [subject_atom_reader]...")
    if raw[-3:] == 'cif':
        atom_loop_count = 0  # counts the number of _atom_ labels
        atoms = []
        with open(raw, 'r') as foo:
            for line in foo:
                if '_atom_site_' in line:
                    atom_loop_count += 1
        with open(raw, 'r') as foo:
            if atom_loop_count == 8:
                for line in foo:
                    sploot = line.split()
                    if len(sploot) == atom_loop_count:  # VESTA TYPE
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
            elif atom_loop_count == 5:  # AM TYPE
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
    elif raw[-3:] == 'xyz':
        atoms = read_xyz(raw)
    else:
        print("WARNING: model_padf couldn't understand your subject_atom_name")
        atoms = []
    print("Asymmetric unit contains ", len(atoms), " atoms found in ", raw)
    np.array(atoms)

    return atoms


def read_xyz(file):
    print(f"<utils.read_xyz> Finding atoms in {file}...")
    raw_x = []
    raw_y = []
    raw_z = []
    raw_f = []
    with open(file, "r") as xyz:
        for line in xyz:
            splot = line.split()
            if len(splot) == 4:
                if 'H' != splot[0]:
                    raw_f.append(get_z(splot[0]))
                    raw_x.append(splot[1])
                    raw_y.append(splot[2])
                    raw_z.append(splot[3])
                else:
                    continue
            elif len(splot) == 3:
                raw_x.append(splot[0])
                raw_y.append(splot[1])
                raw_z.append(splot[2])
    raw_x = [float(x) for x in raw_x]
    raw_y = [float(y) for y in raw_y]
    raw_z = [float(z) for z in raw_z]
    raw_atoms = np.column_stack((raw_x, raw_y, raw_z, raw_f))
    print("<utils.read_xyz> Atom set contains ", len(raw_x), " atoms found in " + file)
    return raw_atoms


def cossim_measure(array_a, array_b):
    array_a = np.ndarray.flatten(array_a)
    array_b = np.ndarray.flatten(array_b)
    sim = np.dot(array_a, array_b) / (np.linalg.norm(array_a) * np.linalg.norm(array_b))
    return sim


def calc_rfactor(array_a, array_b):
    delta = 0.0
    yobs = 0.0
    for i, dp in enumerate(array_b):
        delta = delta + (array_b[i, 1] - array_a[i, 1]) ** 2
        yobs = yobs + (array_b[i, 1]) ** 2
    r_p = np.sqrt(delta / yobs)
    # print(f'R_p : {r_p}')
    return r_p


def output_reference_xyz(atom_list, path):
    print(f"<utils.output_reference_xyz> Writing to {path}")
    with open(path, 'w') as foo:
        foo.write(f'{len(atom_list)}\n')
        foo.write(f'{path}\n')
        for k, atom in enumerate(atom_list):
            # print(atom)
            atom_id = get_id(z=atom[3])
            atom_out = f'{atom_id} '
            for frag in atom[:3]:
                atom_out += f'{frag:11.6f} '
            atom_out += '\n'
            foo.write(atom_out)


def get_z(atom_name):
    # print(f'atom_name {atom_name}')
    for element in atoms.ELEMENTS:
        if atoms.ELEMENTS[element].symbol == atom_name:
            # print(f'FOUND {atoms.ELEMENTS[element].symbol}')
            return atoms.ELEMENTS[element].atomic_number


def get_id(z):
    # print(f'atom_name {atom_name}')
    for element in atoms.ELEMENTS:
        if atoms.ELEMENTS[element].atomic_number == z:
            # print(f'FOUND {atoms.ELEMENTS[element].symbol}')
            return atoms.ELEMENTS[element].symbol
