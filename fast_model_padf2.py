"""
Fast Model PADF Calculator

@author: andrewmartin, jack-binns
"""
import shutil
import numpy as np
import time
import math as m
import matplotlib.pyplot as plt
import os
import utils as u
import pyshtools as pysh
from scipy.special import spherical_jn, legendre
import multiprocessing as mp
import copy


def thread_subcell_model_padf(cells, j, modelp, method, return_dict): 
    """
    #    calculation of model padf from subcells of the volume, via multiprocessing
    """
    
    padfsum_evens = np.zeros( (modelp.nr, modelp.nr, modelp.nth))
    padfsum_odds  = np.zeros( (modelp.nr, modelp.nr, modelp.nth)) 
    #sphvol_evens = np.zeros( (modelp.nr, modelp.nthvol, modelp.phivol))
    #sphvol_odds  = np.zeros( (modelp.nr, modelp.nthvol, modelp.phivol)) 

    for icell, cell in enumerate(cells):
        print( f"Processor {j}; cell {icell+1} / {len(cells)}", end='\r', flush=True) 
        ia, ib, ic = cell[0], cell[1], cell[2] 
        modelp.limits = np.array([ia*modelp.subcellsize+modelp.x_min,(ia+1)*modelp.subcellsize+modelp.x_min, \
                    ib*modelp.subcellsize+modelp.y_min,(ib+1)*modelp.subcellsize+modelp.y_min,\
                    ic*modelp.subcellsize+modelp.z_min,(ic+1)*modelp.subcellsize+modelp.z_min])   

        if method == 'serial':
            ModelPadfCalculator.run_fast_serial_calculation(modelp)
        elif method=='matrix':
            ModelPadfCalculator.run_fast_matrix_calculation(modelp)
        elif method == 'spharmonic':
            ModelPadfCalculator.run_spherical_harmonic_calculation(modelp)   
        else:
            print(" <method> parameter is not one of the valid options: serial, matrix, spharmonic")

        padfsum_evens += np.copy(modelp.rolling_Theta_evens)
        padfsum_odds += np.copy(modelp.rolling_Theta_odds)
        #sphvol_evens += np.copy(modelp.sphvol_evens)
        #sphvol_odds += np.copy(modelp.sphvol_odds)

        if j==0 and icell==0: 
            modelp.write_subject_atoms_to_xyz( modelp.root+modelp.project+modelp.tag+'crop_tiled_circ.xyz')

    return_dict[j] = [padfsum_evens, padfsum_odds]





class ModelPadfCalculator:
    """ 
    Tools to calculate the PADF from an atomic model 

    Attributes
    ----------
    outpath : str
        path to save output of the calculation

    tag     : str
        a string to prepend to all output files (change to avoid overwriting previous calculations)

    subject_atoms : str
        name of an .xyz file that contains atomic coordinates of a sample region
        [- overwritten by a list of atoms once the subject atoms are read from file]

    supercell_atoms :  str
        name of an .xyz file that contains the atomic coordinates of the subject_atoms and surrounding atoms.

    rmin : float
        minimum pair distance value (r) to use in the PADF calculation

    rmax : float
        maximum pair distance value (r) to use in the PADF calculation  

    nr   : int
        number of radial samples to use in the PADF

    nth  : int
        number of angular samples used in teh PADF

    nthvol : int
        number of angular samples used (check where and how this is used)

    r_dist_bin : float
        = rmax/nr. Calcualted in the code. Size of one radial bin.

    angular_bin : float
        = 180.0/nth. Calculated in the code. Size of one angular bin. (degrees)

    r_power : float
        multiply the radial  dimensions of the PADF 
    
    convergence_check_flag : Bool
        determine whether to hc==0

    r12_reflection : Bool
        add the reflected pair vectors to the list

    mode : 'str'
        Select which type of PADF simulation to do: 'stm', 'reqr', 'rconst'

    dimension : int
        Number of dimensions in the simulation

    processor_num : int
        Number of processors to use

    chunksize : int
        Number of pairs to process as a chunk

    verbosity : int
        0 - print cycle assessment each 100 steps; 1 print every step

    rolling_Theta : numpy array
        stores the calculated total PADF

    rolling_Theta_evens : numpy array
        stores the PADF calculated from even indexed pairs
     
    rolling_Theta_odd : numpy array
        stores the PADF calculated from odd indexed pairs

    n2_contacts : numpy array of floats
        stores radial distances of pairs

    interatomic_vectors : numpy array (floats)
        stores pair-wise interatomic vectors

    raw_extended_atoms : list
        stores the atomic coordinates of supercell atoms prior to cleaning

    extended_atoms : list
        stores the atomic coordinates of atoms to correlated to subject_atoms
        This is either subject_atoms or to the cleaned supercell atoms depending on the 
        vale of use_supercell. 
    
    loop_similarity_array : numpy array
        stores values of cosine similarity between odds and evens during loop version

    convergence_target : float
        Value between 0 and 1. Calculation will stop if target is exceeded.

    converged_flag : Bool
        Stores the value True when the convergence target has been exceeded

    com_cluster_flag : Bool
        course grain the atom list 

    com_radius : float
        size of the atomic cluster to use for course graining

    total_contribs : int
        stores the total number of contributing pair-pair contacts

    calculation_time : float
        stores the final calculation time

    percent_milestones : array (floats)
        the number of interatomic vectors at each 10% milesteone

    iteration_times : array (floats)
        stores the time of each cycle assessment

    use_supercell : Bool
        If true, use the supercell atoms

    use_atom_weights : Bool
        If True, weight each atom by its atomic number. If False, each atom is weighted by 1.

    nthreads : int
        Number of threads to use (in spherical harmonic calc only at the moment)      
    """

    def __init__(self, outpath="", tag="", supercell_atoms="", subject_atoms="", rmin=0.0, rmax=10.0, nr=100,\
                 nth=180, nthvol=180,nthreads=1, verbosity=1 ):

        self.outpath = outpath
        self.tag = tag
        self.supercellxyz = supercell_atoms  # the xyz file contains the cartesian coords of the crystal structure expanded
        # to include r_probe
        self.subject_atoms = ""  # the cif containing the asymmetric unit
        self.subjectxyz = ""  # the cif containing the asymmetric unit
        # probe radius
        self.rmin = rmin
        self.rmax = rmax
        self.nr = nr
        self.r_dist_bin = rmax/nr
        self.nth = nth
        self.nthvol = nthvol
        self.angular_bin = 180.0/nth
        self.r_power = 2
        self.convergence_check_flag = False
        self.r12_reflection = True
        self.mode = 'stm'
        self.dimension = 3
        self.processor_num = nthreads
        self.chunksize = 50
        #self.Pool = mp.Pool(self.processor_num)
        self.verbosity = 1
        self.rolling_Theta = np.zeros(0)
        self.rolling_Theta_odds = np.zeros(0)
        self.rolling_Theta_evens = np.zeros(0)
        self.n2_contacts = []
        self.interatomic_vectors = []
        self.raw_extended_atoms = []
        self.extended_atoms = []
        self.loop_similarity_array = []
        self.convergence_target = 1.0
        self.converged_flag = False
        self.com_cluster_flag = False
        self.com_radius = 0.0
        self.total_contribs = 0
        self.calculation_time = 0.0
        self.percent_milestones = np.zeros(0)
        self.iteration_times = np.zeros(0)
        self.use_supercell = True
        self.use_atom_weights = True

    # Yes
    def parameter_check(self):
        """
        Check and print calculation parameters and a welcome
        """
        self.r_dist_bin = self.rmax / self.nr
        self.angular_bin = 180 / self.nth

        if self.verbosity >0:
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

    # yes
    def get_dimension(self):
        """
        Sets the dimension of the calculation. Mostly just important if you're calculating
        a slice rather than the full r, r', theta matrix.
        
        Returns
        -------------
        self.dimension : int
            The number of dimensions of the calculated PADF (=3 for whole PADF; =2 for a slice)
        """
        if self.mode == 'rrprime':
            if self.verbosity > 0: print("<get_dimension>: Calculating r = r' slice")
            self.dimension = 2
        elif self.mode == 'rrtheta':
            if self.verbosity > 0: print("<get_dimension>: Calculating r, r', theta slices")
            pass
        elif self.mode == 'stm':
            if self.verbosity > 0: print("<get_dimension>: Calculating Theta(r,r',theta) directly...")
            self.dimension = 3
        return self.dimension

    # yes
    def write_all_params_to_file(self):
        """
        Writes all the input parameters to a log file
        """
        path = self.outpath
        if not os.path.isdir(path):
            print('<write_all_params_to_file>: Moving files...', path)
            os.mkdir(path)

#            src = self.root + self.supercell_atoms
#            dst = self.root + self.project + self.supercell_atoms
#            shutil.copy(src, dst)
#            src = self.root + self.subject_atoms
#            dst = self.root + self.project + self.subject_atoms
#            shutil.copy(src, dst)
#        else:
        f = open(self.outpath + f'{self.tag}_mPADF_param_log.txt', 'w')
        f.write("# log of input parameters for model PADF calculation\n")
        f.write('["model PADF"]\n')
        a = self.__dict__
        for d, e in a.items():
            f.write(d + " = " + str(e) + "\n")
        f.close()

    #yes
    def write_calculation_summary(self):
        """
        Writes out a summary of the calculation
        """
        with open(self.outpath + f'{self.tag}_calculation_log.txt', 'w') as f:
            f.write(f'Calculation time: {self.calculation_time} s\n')
            f.write(f'Total number of interatomic vectors {len(self.interatomic_vectors)}\n')
            f.write(f'Total number of atoms in system {len(self.extended_atoms)}\n')
            f.write(f'Total number of contributing contacts {self.total_contribs}\n')
        np.savetxt(self.outpath + f'{self.tag}_similarity_log.txt', np.array(self.loop_similarity_array))

    def write_subject_atoms_to_xyz(self, fname, element="C"):
        """
        Saves current subject atoms to file (useful if subject atoms have been cropped)
        """
        with open( fname, 'w') as f:
            f.write( str(self.subject_atoms.shape[0])+"\n")
            f.write( "cropped subject atoms\n" )
            txt = "{x:12.6f}{y:12.6f}{z:12.6f}"
            for i, xyz in enumerate(self.subject_atoms):
                f.write(f" {element}"+txt.format(x=xyz[0],y=xyz[1],z=xyz[2]) )
                if i<self.subject_atoms.shape[0]-1: f.write("\n")

    # yes
    def subject_target_setup(self):
        """
        Handlers to read in the subject atoms (a.k.a. asymmetric unit) and the extended atoms (environment)
        
        Returns
        -------------
        self.subject_atoms : list / numpy array
            list of arrays containing the atoms in the volume of interest
        
        self.extended_atoms : list / numpy array
            list of atoms in the extended (supercell) volume
            if supercell is not used, then the extended atoms are equal to the subject atoms
        """
        if self.verbosity>0: print(f'<subject_target_setup> Reading in subject set...')
        self.subject_atoms = u.read_xyz(
            f'{self.subjectxyz}')  # Read the full asymmetric unit
        if self.com_cluster_flag:
            self.subject_atoms = self.clean_subject_atoms()

        if self.verbosity>0:
            print(f'<subject_target_setup> Reading in extended atom set...')
            print(f'DEBUG <subject_target_setup>', self.root)
            print(f'DEBUG <subject_target_setup>', self.project)
            print(f'DEBUG <subject_target_setup>', self.supercellxyz)
        self.raw_extended_atoms = u.read_xyz(
            f'{self.supercellxyz}')  # Take in the raw environment atoms
        if self.use_supercell:
            self.extended_atoms = self.clean_extended_atoms()  # Trim to the atoms probed by the subject set
        else:
            self.extended_atoms = np.copy(self.subject_atoms)
        # if self.com_cluster_flag:
        #     self.output_cluster_xyz()       ## WRITE OUT THE CLUSTER GEOMETRIES
        u.output_reference_xyz(self.subject_atoms, path=f'{self.tag}_clean_subject_atoms.xyz')
        u.output_reference_xyz(self.extended_atoms,
                               path=f'{self.tag}_clean_extended_atoms.xyz')
        return self.subject_atoms, self.extended_atoms

    # only if we keep serial calc 
    def cycle_assessment(self, k, start_time):
        """
        Assesses the similarity between the Odd and Even padf calculations
        Outputs an update to the terminal if verbocity == 1.

        Parameters
        ----------------
        k : integer
            iteration number

        start_time : ?
            The time the began as recorded by time.time()
        """


        # Measure internal convergence
        if k > 1:
            loop_cos = u.cossim_measure(self.rolling_Theta_odds, self.rolling_Theta_evens)
            # print(f"{loop_cos=}")
            self.loop_similarity_array.append([k, loop_cos])
            if self.verbosity > 1:
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
        if self.verbosity > 1:
            if time_remaining > 3600:
                print(
                    f"| {k} / {len(self.interatomic_vectors)} | Estimate {round(time_remaining / 3600, 3)} hr remaining")
            else:
                print(f"| {k} / {len(self.interatomic_vectors)} | Estimate {round(time_remaining, 3)} s remaining")


    # yes (used in clean_setup_atoms)
    def clean_extended_atoms(self):
        """
        Trims the length of the extended atoms to the set probed by
        the r_probe and asymmetric unit
        
        Returns
        --------------
        clean_ex : numpy array
            extended atoms with atoms beyond r_probe removed and, if selected, centre of mass clustering applied.
        """
        if self.verbosity>0: print(f'<fast_model_padf.clean_extended_atoms> Trimming atom sets to rmax')
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

    # yes
    def clean_subject_atoms(self):
        """
        Clusters subject atoms

        Returns
        --------------
        cluster_subject : numpy array
            subject atoms with centre of mass clustering applied.
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

    
    def reduce_subject_atoms_to_subcell(self,limits):
        """Select a subcell from the atom list"""

        xmin, xmax, xcen = limits[0],limits[1], 0.5*(limits[1]+limits[0])
        ymin, ymax, ycen = limits[2],limits[3], 0.5*(limits[3]+limits[2])
        zmin, zmax, zcen = limits[4],limits[5], 0.5*(limits[5]+limits[4])
        #r = np.min( [np.abs(xmax-xcen), np.abs(ymax-ycen), np.abs(zmax-zcen)])
        r = np.min( [np.abs(xmax-xcen), np.abs(ymax-ycen), np.abs(zmax-zcen), np.abs(xmin-xcen),np.abs(ymin-ycen),np.abs(zmin-zcen)])
        selected = []
        for v in self.subject_atoms:
             norm = np.sqrt( (v[0]-xcen)**2 + (v[1]-ycen)**2 + (v[2]-zcen)**2 )
             if (v[0]>xmin)and(v[0]<xmax)and(v[1]>ymin)and(v[1]<ymax)and(v[2]>zmin)and(v[2]<zmax)and(norm<r):
                selected.append(v)
        self.subject_atoms = np.array(selected)

        selected = []
        for v in self.extended_atoms:
             norm = np.sqrt( (v[0]-xcen)**2 + (v[1]-ycen)**2 + (v[2]-zcen)**2 )
             if (v[0]>xmin)and(v[0]<xmax)and(v[1]>ymin)and(v[1]<ymax)and(v[2]>zmin)and(v[2]<zmax)and(norm<r):
                selected.append(v)
        self.extended_atoms = np.array(selected)

    def box_dimensions_from_subject_atoms(self):
            """Identify the size of the box in x, y, and z dimensions"""
            self.x_min = np.min(self.subject_atoms[:, 0])
            self.y_min = np.min(self.subject_atoms[:, 1])
            self.z_min = np.min(self.subject_atoms[:, 2])

            self.x_max = np.max(self.subject_atoms[:, 0])
            self.y_max = np.max(self.subject_atoms[:, 1])
            self.z_max = np.max(self.subject_atoms[:, 2])

            self.x_wid = self.x_max - self.x_min
            self.y_wid = self.y_max - self.y_min
            self.z_wid = self.z_max - self.z_min

    # used in serial calcultion
    def bin_cor_vec_to_theta(self, cor_vec, fz, array):
        """
        Adds a pair to the corresponding location in the PADF volume

        Parameters
        ---------------

        cor_vec : numpy array
            array of the r, r', theta coordinates of the correlated pair

        fz : float
            The scattering factor weighting to add to the PADF volume

        array : numpy array
            The PADF volume that the vector will be added to
    
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
            #if r1_index==r2_index: print( "th_index", th_index)
            array[r1_index, r2_index, th_index] = array[r1_index, r2_index, th_index] + fz
            if self.r12_reflection:
                array[r2_index, r1_index, th_index] = array[r2_index, r1_index, th_index] + fz

    # used in serial calculation
    def calc_padf_frm_iav(self, k, r_ij):
        """
        For a given pair r_ij, bin correlation with all other pairs into the padf volume
        
        Parameters
        ---------------
        k : int
           thread number (not implemented)

        r_ij : numpy array (float)
            pair vector coordinates and atomic weight 
        """
        start = time.time()
        # print(f'<calc_padf_frm_iav>: Starting calculation on thread {k}...')
        fb_hit_count = 0
        for r_xy in self.interatomic_vectors:
            if np.array_equal(r_ij, r_xy):
                continue
            theta = u.fast_vec_angle(r_ij[0], r_ij[1], r_ij[2], r_xy[0], r_xy[1], r_xy[2])
            fprod = r_ij[4] * r_xy[4]
            #if (r_ij[3]==r_xy[3]) and (r_ij[3]>0.5): print( r_ij[3], r_xy[3], theta, theta*180/np.pi) 
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

    # not used in sphharmonic, but yes for other calcs
    def pair_dist_calculation(self,writefreq=1000, outputpairs=False):
        """ 
        Compute all interatomic vectors

        writefreq : int
           update message at this number of iterations

        outputpairs : Bool
             save a list of all interatomic pairs, if True.
        """
        print(f'<pair_dist_calculation> Calculating pairwise interatomic distances...')
        # interatomic_vectors
        iv_list = []
        for k, a_i in enumerate(self.subject_atoms):
            if k % int( writefreq) == 0:
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
            
            """
            # alternative matrix version  - THIS DOESN'T WORK AT THE MOMENT; NP.CONCATENATE LINE IS BUGGY
            diff = self.extended_atoms - a_i
            mags = np.sqrt(np.sum(diff*diff,1))
            igood = np.where( (mags>0.8) )
            print( diff[igood,:].shape, mags[igood].shape, self.extended_atoms[igood,3].shape, "shapes")
            chunk_a_i = np.concatenate( (np.squeeze(diff[igood,:]), np.squeeze(mags[igood]).reshape((mags[igood].shape[0],1)), self.extended_atoms[igood,3].T*a_i[3]),axis=1)
            #iv_list += list(chunk_a_i)
            print( chunk_a_i.shape, len(iv_list) )
            #iv_list.append(chunk_a_i)

            if k==0:
                ivectors = np.copy(chunk_a_i)
            else:
                ivectors = np.concatenate( (ivectors, chunk_a_i), axis=0) 

            

        iv_list = ivectors

        iv_list = np.array(iv_list)
        self.interatomic_vectors = iv_list #np.concatenate( iv_list, axis=0) #ivectors       
        print(self.interatomic_vectors.shape)
        """

        self.interatomic_vectors = np.array(self.interatomic_vectors)

        self.n2_contacts = self.interatomic_vectors[:,2] #ivectors[:,2]

        np.array(self.n2_contacts)
        print(f'<pair_dist_calculation> {len(self.interatomic_vectors)} interatomic vectors')
        if outputpairs: np.savetxt(self.outpath + self.tag + '_atomic_pairs.txt', self.n2_contacts)
        if outputpairs: np.save(self.outpath + self.tag + '_interatomic_vectors.npy', self.interatomic_vectors)
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
        np.savetxt(self.outpath + self.tag + '_PDF.txt', pdf_arr)
        np.savetxt(self.outpath + self.tag + '_APDF.txt', np.column_stack((adfr_r, adfr_int)))
        print(f"{self.outpath + self.tag + '_PDF.txt'}")
        print(f"{self.outpath + self.tag + '_APDF.txt'}")
        return self.interatomic_vectors

    # not used in spherical harmonic calculation
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
        print( "debug ", self.rmin)
        print(f'<trim_interatomic_vectors_to_probe> ..after trimming to < self.rmax : {len(b)} vectors')
        print(f'<trim_interatomic_vectors_to_probe> ..after trimming to > self.rmin : {len(c)} vectors')
        print(f'<trim_interatomic_vectors_to_probe> ..after trimming : {len(c)} vectors')
        self.interatomic_vectors = c
        np.save(self.outpath + self.tag + '_interatomic_vectors_trim.npy', self.interatomic_vectors)


    # the serial calculation
    def run_fast_serial_calculation(self):
        """
        Original model PADF calculation based on histograming every pair-pair combination.
        Although it says 'fast', it is the slowest of the versions in this code.
        """
        global_start = time.time()
        self.parameter_check()
        self.write_all_params_to_file()
        self.subject_atoms, self.extended_atoms = self.subject_target_setup()  # Sets up the atom positions of the subject set and supercell 
        if self.subcellsize > 0.0: self.reduce_subject_atoms_to_subcell(self.limits)
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
            if self.converged_flag and self.convergence_check_flag:
                break

        # Save the rolling PADF arrays
        np.save(self.outpath + self.tag + '_mPADF_total_sum', self.rolling_Theta)
        np.save(self.outpath + self.tag + '_mPADF_odds_sum', self.rolling_Theta_odds)
        np.save(self.outpath + self.tag + '_mPADF_evens_sum', self.rolling_Theta_evens)

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
    #  The core of the matrix correlation routine
    #
# calculate vec dot vec_prime for every pair of vectors
  

    # used in the faster matrix calculation
    def calc_padf_frm_iav_matrix(self,vectors,vectors2,nr=100,nth=100):
          """
          Vectorized form of the pair-pair correlation calcualtion

          vectors : numpy array (floats)
            the atomic coordinates of a set of atoms

          vectors2 : numpy array (floats)
            the atomic coordinates of a different set of atoms
            
          nr : int
            number of radial bins in the padf
        
          nth : int
            number of angular bins in the padf
          """
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
          angles = np.arccos(div)
          #print("min max angles", np.min(angles), np.max(angles))
      
          # the final step would be to use array masking to histogram the calculation...
          nvec  = len(vectors)
          nvec2 = len(vectors2)
          rrth_coords = np.array([np.outer(norms,np.ones(nvec2).T).flatten(),
                      np.outer(np.ones(nvec),norms2).flatten(),
                      angles.flatten() ])
          #print(nr, nth, np.min(rrth_coords), np.max(rrth_coords))
          padf, edges = np.histogramdd( rrth_coords.T, bins=(nr,nr,nth), range=((0,self.rmax),(0,self.rmax), (0,np.pi)))
          #print(edges[0], )
          return padf, edges


    #
    # A matrix implementation of the model calculation   
    #
    #
    def run_fast_matrix_calculation(self,alreadysetup=False):
        """
        Computer pair-pair correlations (padf) using the vectorised calculation

        alreadysetup : Bool
            Skip the set up of the subject & extended atoms, if this has been done prior to calling the function
        """
        global_start = time.time()
        if not alreadysetup:
            self.parameter_check()
            self.write_all_params_to_file()
            self.subject_atoms, self.extended_atoms = self.subject_target_setup()  # Sets up the atom positions of the subject set and supercell
            if self.subcellsize > 0.0: self.reduce_subject_atoms_to_subcell(self.limits)

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
        np.save(self.outpath + self.tag + '_mPADF_total_sum', self.rolling_Theta)
        np.save(self.outpath + self.tag + '_mPADF_odds_sum', self.rolling_Theta_odds)
        np.save(self.outpath + self.tag + '_mPADF_evens_sum', self.rolling_Theta_evens)

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



    # used in spherical harmonic volume...
    # need to add weights to this calculation...
    def calc_sphvol_from_interatomic_vectors(self,vectors,nr=50,nth=90,nphi=90, tol=5e-6, use_atom_weights=True):
          """
          Calculate a 3D pair distribution in spherical coordiantes from a set of interatomic vectors

          vectors : numpy array (float)
             an array of interatomic vectors

          nr : int
             the number of radial bins in the pair distribution

          nth : int
             the number of angular bins in the pair distribution

          nphi : int
             number of phi bins in the pair distribution

          tol : float
             tolerance to remove atom pairs that are too close together (i.e. self pairs)

          use_atom_weights : Bool
             weight the pair distribution by the atomic weights
          """ 
          # get the norm of all vectors and make them into a norm*norm_prime matrix
          norms   = np.sqrt(np.sum(vectors[:,:3]**2,1))
          
          # the final step would be to use array masking to histogram the calculation...
          nvec  = len(vectors)

          sphv = np.zeros( (vectors.shape[0], 3) )
          sphv[:,0] = norms
          inorm = np.where( norms > tol )
          sphv[inorm,1] =  np.arccos(vectors[inorm,2]/norms[inorm])

          ix = np.where( (np.abs(vectors[:,0])>tol)*(vectors[:,0]>0) )   #norm of x above thresh; and x positive
          sphv[ix,2] = np.pi/2   + np.arctan( vectors[ix,1]/vectors[ix,0])

          ix = np.where( (np.abs(vectors[:,0])>tol)*(vectors[:,0]<0) )   #norm of x above thresh; and x negative
          sphv[ix,2] = 3*np.pi/2 + np.arctan( vectors[ix,1]/vectors[ix,0])


          ix = np.where( (np.abs(vectors[:,0])<tol)*(vectors[:,1]>0) )   #norm of x below thresh; and y positive
          sphv[ix,2] = np.pi #3*np.pi/2
          
          ix = np.where( (np.abs(vectors[:,0])<tol)*(vectors[:,1]<0) )   #norm of x below thresh; and y negative
          sphv[ix,2] = 2.0*np.pi #np.pi/2

          #print("vectors")
          #for i in range(nvec): 
          #      if norms[i]>2: print(i, vectors[i], norms[i], sphv[i,:])

          if use_atom_weights:
                outvol, edges = np.histogramdd( sphv, bins=(nr,nth,nphi), range=((0,self.rmax),(0,np.pi),(0,2*np.pi)), weights=vectors[:,3])
          else:
                outvol, edges = np.histogramdd( sphv, bins=(nr,nth,nphi), range=((0,self.rmax),(0,np.pi),(0,2*np.pi)))


          #sphv[ix,2] += np.pi
          #outvol2, edges = np.histogramdd( sphv, bins=(nr,nth,nphi), range=((0,self.rmax),(0,np.pi),(0,2*np.pi)))
          #outvol[:,:,nphi//2:] = outvol[:,:,:nphi//2] 
     
          #NEED TO CHECK THE RANGE OF THE ARCCOS AND ARCSIN FUNCTIONS - WHAT ABOUT AFFECT OF THE ARCSIN RANGE???? 
          return outvol, edges



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
    # function to split subject atom loop over threads
    #
    def thread_subject_atom_loop_sph(self,subject_atoms,j,return_dict):
        """
        To compute the pair distribution in spherical coordinates from a set of atoms (subject atoms) on a thread

        subject_atoms : numpy array (float)
            an array of atomic coordinates for a set of atoms

        j : int
            thread number

        return_dict : multiprocessing return dictionary
            return dictionary for multiprocessing  - stores output pair distribution
        """
        sphvol_oe = np.zeros((self.nr, self.nthvol, self.phivol,2)) 
        for i, a_i in enumerate(subject_atoms):
                print(f"thread {j}, atom {i+1}/{len(subject_atoms)}", end='\r', flush=True)
                all_interatomic_vectors = np.zeros( (self.extended_atoms.shape[0], 4) )
                all_interatomic_vectors[:,:3] = self.extended_atoms[:,:3] - np.outer(np.ones(self.extended_atoms.shape[0]),a_i[:3])
                all_interatomic_vectors[:,3] = self.extended_atoms[:,3]*a_i[3]  # this is the product of the scattering factors                
                norms = np.sqrt(np.sum(all_interatomic_vectors[:,:3]**2,1))
                inorm = np.where( norms<self.rmax)
                interatomic_vectors = all_interatomic_vectors[inorm]
                
                chunksize = np.min([500,len(interatomic_vectors)])
                nchunks = len(interatomic_vectors)//chunksize
                
                for k in range(nchunks):
                    #if k==0: print(f'Creating 3D pair distributions in spherical coordinates; chunk index {k+1}/{nchunks}', np.max(interatomic_vectors), chunksize)
                    vectors = interatomic_vectors[k*chunksize:(k+1)*chunksize,:4]
                    voltmp, edges = self.calc_sphvol_from_interatomic_vectors(vectors,nr=self.nr,nth=self.nthvol,nphi=self.phivol,use_atom_weights=self.use_atom_weights)
                    if i%2==0:
                        sphvol_oe[:,:,:,0] += voltmp
                    else:
                        sphvol_oe[:,:,:,1] += voltmp
                     
                    #if k==0: print("sum sphvol", np.sum(sphvol_oe) )
        
    
        #
        # SINTHETA CORRECTION (ASSUMING THETA SAMPLING)
        #  
        anglegrid = np.mgrid[:self.nr,:self.nthvol,:self.phivol,:2]
        thgrid = anglegrid[1]*np.pi/self.nthvol
        sthgrid = np.sin(thgrid)
        ith = sthgrid>1e-2
        sphvol_oe[ith] = sphvol_oe[ith]/sthgrid[ith]
 
        return_dict[j] = sphvol_oe
            



    #
    # A spherical harmonic version of the model calculation   
    #
    #
    def run_spherical_harmonic_calculation(self):
        """
        The PADF calculation from spherical harmonics via the 3D pair distribution calculation
        """
        manager = mp.Manager()            
        return_dict = manager.dict()

        global_start = time.time()
        self.parameter_check()
        self.write_all_params_to_file() 
        self.subject_atoms, self.extended_atoms = self.subject_target_setup()  # Sets up the atom positions of the subject set and supercell
        ########self.subject_atoms = self.subject_atoms[:4000] #HACK!!!!
        
        if self.subcellsize > 0.0: 
            self.reduce_subject_atoms_to_subcell(self.limits)


        """
        Here we do the chunking for sending to threads
        """
        self.sphvol_evens = np.zeros((self.nr, self.nthvol, self.phivol))
        self.sphvol_odds  = np.zeros((self.nr, self.nthvol, self.phivol))
        if self.verbosity>0: print(f'<fast_model_padf.run_fast_spherical_harmonic_calculation> Working...')
        global_start = time.time()
        """
        # the original version of the spherical harmonic calculation without the threading
        for i, a_i in enumerate(self.subject_atoms):
                #print("a_i", a_i)
                #if i<10000: continue
                #if i>10000+500: break
                all_interatomic_vectors = np.zeros( (self.extended_atoms.shape[0], 4) )
                all_interatomic_vectors[:,:3] = self.extended_atoms[:,:3] - np.outer(np.ones(self.extended_atoms.shape[0]),a_i[:3])
                all_interatomic_vectors[:,3] = self.extended_atoms[:,3]*a_i[3]  # this is the product of the scattering factors
                #print( self.extended_atoms.shape, self.extended_atoms[0], a_i, all_interatomic_vectors[0])

                
                norms = np.sqrt(np.sum(all_interatomic_vectors[:,:3]**2,1))
                inorm = np.where( norms<self.rmax)
                interatomic_vectors = all_interatomic_vectors[inorm]
                
                chunksize = np.min([500,len(interatomic_vectors)])
                nchunks = len(interatomic_vectors)//chunksize
                if (i%500)==0: 
                    print("Subject atoms binned:", i, "/", len(self.subject_atoms), f"time passed {time.time()-global_start}") 
                
                for k in range(nchunks):
                    #print(f'Creating 3D pair distributions in spherical coordinates; chunk index {k+1}/{nchunks}', np.max(interatomic_vectors))
                    vectors = interatomic_vectors[k*chunksize:(k+1)*chunksize,:4]
                    voltmp, edges = self.calc_sphvol_from_interatomic_vectors(vectors,nr=self.nr,nth=self.nthvol,nphi=self.phivol,use_atom_weight=self.use_atom_weights)
                    if (i)%2==0:
                        self.sphvol_evens += voltmp
                    else:
                        self.sphvol_odds += voltmp
    

        """

         
        #
        # The threaded version of the spherical harmonic calculation
        #
        if self.processor_num>1:
            self.nthreads = self.processor_num
            natoms_per_thread = len(self.subject_atoms)//self.nthreads
            processes = []
            print( "nthreads, atoms_per_thread, total atoms", self.nthreads, natoms_per_thread, len(self.subject_atoms))
            for j in np.arange(self.nthreads):
                print("Main threading loop", j)
                subject_atoms = self.subject_atoms[j*natoms_per_thread:(j+1)*natoms_per_thread]
                p = mp.Process(target=self.thread_subject_atom_loop_sph, \
                                args=(subject_atoms,j,return_dict))
                p.start()
                processes.append(p)

            print("joining")
            for p in processes:
                p.join()
            
            for j in np.arange(self.nthreads):
                self.sphvol_evens += return_dict[j][:,:,:,0]
                self.sphvol_odds += return_dict[j][:,:,:,1]
        else:
            sphvol_list = [0]
            print( "Number of subject atoms", len(self.subject_atoms))
            self.thread_subject_atom_loop_sph(self.subject_atoms,0,sphvol_list)
            self.sphvol_evens = sphvol_list[0][:,:,:,0]
            self.sphvol_odds = sphvol_list[0][:,:,:,1]
                                             

        
        np.save( self.outpath+f"{self.tag}_sphvol_evens.npy", self.sphvol_evens)
        #plt.figure()
        #plt.imshow( np.sum(self.sphvol_evens[11:14,:,:],0))
        #plt.show()

        #print(
        #    f'<fast_model_padf.run_spherical_harmonic_calculation> Total interatomic vectors: {len(self.interatomic_vectors)}')
    
        #print( "debug spherical harmonic calc; nl nlmin", self.nl, self.nlmin)

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
        r = coords[0]
        th = coords[2]*2*np.pi/self.nth

        # Set up the rolling PADF arrays
        self.rolling_Theta_odds =  self.Blrr_to_padf( Blrr_odds, (self.nr,self.nr,self.nth)) *np.abs(np.sin(th))
        self.rolling_Theta_evens = self.Blrr_to_padf( Blrr_evens, (self.nr, self.nr, self.nth)) *np.abs(np.sin(th))
        # Here we loop over interatomic vectors


        # Save the rolling PADF arrays
        #np.save(self.outpath + self.tag + '_mPADF_total_sum', self.rolling_Theta_odds + self.rolling_Theta_evens)
        #np.save(self.outpath + self.tag + '_mPADF_odds_sum', self.rolling_Theta_odds)
        #np.save(self.outpath + self.tag + '_mPADF_evens_sum', self.rolling_Theta_evens)

        self.calculation_time = time.time() - global_start
        if self.verbosity>0:
            print(
                f"<fast_model_padf.run_fast_model_calculation> run_fast_model_calculation run time = {self.calculation_time} seconds")
            print(
                f"<fast_model_padf.run_fast_model_calculation> Total contributing contacts (for normalization) = {self.total_contribs}")
        self.write_calculation_summary()




    def calculate_model_padf(self, method='spharmonic', nthreads=1):
        """
        #    Runs a model padf calculation; triages for calcualtion type and use of multiprocessing
        """        

        #
        # Run the calculation!
        #
        if self.subcellsize < 0.0:
            print( "Calculation will NOT be split into subcells")

            if method == 'serial':
                self.run_fast_serial_calculation()
            elif method=='matrix':
                self.run_fast_matrix_calculation()
            elif method == 'spharmonic':
                self.run_spherical_harmonic_calculation()   
            else:
                print(" <method> parameter is not one of the valid options: serial, matrix, histogram, spharmonic")
        else:
            self.tag += "_tiled"
            self.processor_num = 1 #HANDLE THE THREADING IN THIS SCRIPT, ENURE FAST MODEL PADF DOES NOT USE THREADING
            self.parameter_check()
            self.write_all_params_to_file()
            self.verbosity = 0

            self.subject_atoms, self.extended_atoms = self.subject_target_setup()  # Sets up the atom positions of the subject set and supercell
            self.box_dimensions_from_subject_atoms()
            na, nb, nc = int(self.x_wid/self.subcellsize), int(self.x_wid/self.subcellsize), int(self.x_wid/self.subcellsize)
            cellindices = []
            for ia in range(na): 
                for ib in range(nb): 
                    for ic in range(nc): 
                        cellindices.append( [ia,ib,ic])

            nsubcells = len(cellindices)        
            print( "Volume has been tiled into subcells. Number of subcells in x, y, z:",  na, nb, nc)

            if nthreads > 1:
                manager = mp.Manager()            
                return_dict = manager.dict()
        
                cells_per_proc = np.max( [nsubcells // nthreads,1])
                remainder = nsubcells%nthreads
                print( "cells per processor", cells_per_proc)

                processes = []
                for iproc in range(nthreads):
                        if iproc < remainder:
                            cells = cellindices[iproc*cells_per_proc:(iproc+1)*cells_per_proc+1]
                        else:
                            cells = cellindices[iproc*cells_per_proc:(iproc+1)*cells_per_proc]
                        pr = mp.Process(target=thread_subcell_model_padf, \
                                        args=(cells,iproc,copy.deepcopy(self),method,return_dict))
                        pr.start()
                        processes.append(pr)

                for pr in processes:
                    pr.join()

 
                self.rolling_Theta_evens = return_dict[0][0]+ return_dict[0][1]
                self.rolling_Theta_odds = 0.0*return_dict[0][1]
                #self.sphvol_evens = return_dict[0][2]+ return_dict[0][3]
                #self.sphvol_odds = 0.0*return_dict[0][3]

                for iproc in range(1,nthreads):
                    if iproc%2==0:
                        self.rolling_Theta_evens += return_dict[iproc][0]+return_dict[iproc][1]
                    else:
                        self.rolling_Theta_odds += return_dict[iproc][0]+return_dict[iproc][1]
                
                     
            else:
               #single thread
                for ia in range(na+1): 
                    for ib in range(nb+1): 
                        for ic in range(nc+1): 
                            self.limits = np.array([self.x_min+ia*self.subcellsize,self.x_min+(ia+1)*self.subcellsize, \
                                                        self.y_min+ib*self.subcellsize,self.y_min+(ib+1)*self.subcellsize,
                                                        self.z_min+ic*self.subcellsize,self.z_min+(ic+1)*self.subcellsize])   

                            if method == 'serial':
                                self.run_fast_serial_calculation()
                            elif method=='matrix':
                                self.run_fast_matrix_calculation()
                            elif method=='histogram':
                                self.run_histogram_calculation()   
                            elif method == 'spharmonic':
                                self.run_spherical_harmonic_calculation()   
                            else:
                                print(" <method> parameter is not one of the valid options: serial, matrix, histogram, spharmonic")

                            if (ia==0)and(ib==0)and(ic==0):
                                padfsum_evens = np.copy(self.rolling_Theta_evens)
                                padfsum_odds = np.copy(self.rolling_Theta_odds)
                            else:
                                padfsum_evens += np.copy(self.rolling_Theta_evens)
                                padfsum_odds += np.copy(self.rolling_Theta_odds)

                            if ia==0 and ib==0 and ic==0: 
                                self.write_subject_atoms_to_xyz( self.root+self.project+self.tag+'crop_tiled_circ.xyz')
           
                self.rolling_Theta_evens = padfsum_evens
                self.rolling_Theta_odds  = padfsum_odds

     
        np.save(self.root + self.project + self.tag + '_mPADF_total_sum', self.rolling_Theta_odds + self.rolling_Theta_evens)
        np.save(self.root + self.project + self.tag + '_mPADF_odds_sum', self.rolling_Theta_odds)
        np.save(self.root + self.project + self.tag + '_mPADF_evens_sum', self.rolling_Theta_evens)

            
# if __name__ == '__main__':
#     modelp = ModelPadfCalculator()
