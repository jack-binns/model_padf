import numpy as np
import utils as u
import scipy.stats as sp


class rmcCell:

    def __init__(self):
        self.ucds = [10, 10, 10, 90, 90, 90]
        self.atom_list = []
        self.formula = [()]
        self.atom_number = 0

    def read_xyz(self, file):
        print("Finding extended atom set [read_xyz]...")
        raw_x = []
        raw_y = []
        raw_z = []
        raw_atoms = []
        with open(file, "r") as xyz:
            for line in xyz:
                splot = line.split()
                if len(splot) == 4:
                    raw_x = splot[1]
                    raw_y = splot[2]
                    raw_z = splot[3]
                elif len(splot) == 3:
                    raw_x = splot[0]
                    raw_y = splot[1]
                    raw_z = splot[2]
                else:
                    continue
                raw_atoms.append([splot[0], raw_x, raw_y, raw_z])
        self.atom_list = raw_atoms

    def gen_seed(self):
        print('Generating seed random cell')
        for atomtype in self.formula:
            for nz in range(atomtype[1]):
                print(atomtype, nz)
                no_bump = False  # assume bumps
                while not no_bump:
                    randx = np.random.uniform(0, self.ucds[0])
                    randy = np.random.uniform(0, self.ucds[1])
                    randz = np.random.uniform(0, self.ucds[2])
                    test_atom = [randx, randy, randz]
                    no_bump = self.bump_check(test_atom)  # are there no bumps?
                    # print(no_bump)
                if no_bump:
                    self.atom_list.append([atomtype[0], randx, randy, randz])

    def write_xyz(self, target):
        with open(target, 'w') as foo:
            foo.write(str(len(self.atom_list)) + '\n')
            foo.write("test" + '\n')
            for atom in self.atom_list:
                foo.write(str(atom[0]) + " " + str(atom[1]) + " " + str(atom[2]) + " " + str(atom[3]) + '\n')

    def bump_check(self, test_atom):
        flag = True
        for atom_i in self.atom_list:
            diff = u.fast_vec_difmag(atom_i[1], atom_i[2], atom_i[3], test_atom[0], test_atom[1], test_atom[2])
            if diff < 2.4:
                flag = False
        # print(flag)
        return flag

    def jitter(self, jit_mag=1.0):
        natoms = len(self.atom_list)
        z = np.random.randint(low=0, high=natoms)
        print(f'out of {natoms} jittering {z}')
        randx = np.random.uniform(-1, 1)
        randy = np.random.uniform(-1, 1)
        randz = np.random.uniform(-1, 1)
        print(self.atom_list[z])
        self.atom_list[z][1] = float(self.atom_list[z][1]) + (jit_mag * randx)
        self.atom_list[z][2] = float(self.atom_list[z][2]) + (jit_mag * randy)
        self.atom_list[z][3] = float(self.atom_list[z][3]) + (jit_mag * randz)
        print(self.atom_list[z])