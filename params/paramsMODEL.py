
"""
paramsMODEL.py

set up a params class to read/store the input parameters for a modelPADF.py calculation

Classes:
    paramsMODEL - params class instance with MODEL input parameters
"""

from params.params import params
import os


class paramsMODEL(params):
    """
    params derived class with MODEL input parameters  

    Parameters
    ----------
    config : str
        name of the config file with parameter values defined

    subjectxyz : str
        atom coordinates file (.xyz)
        The first column is the element symbol
        The 2nd-4th columns are x, y, z coordinates

    supercellxyz : str
        A second .xyz file with supercell atom positions.
        supercellfile is used if use_supercell == True        

    outpath : str
        path to save output of the padf calculation

    tag : str
        prefix for all output file names
        (used not to overwrite previous calculations)

    method : str
        'serial' for a pair-pair calculation on a single corr
        'matrix' - a faster pair-pair vectorised calculation
        'histogram' - bin the pairs into a 3D volume prior to correlation

    mode : str
        'rrprime' - compute r=r' slice only
        'stm' - compute the full 3D model

    nr : int
        number of radial samples in real space

    nth : int
        number of angular samples
        must be the same in q-space and real-space

    rmax : float
        maximum value of r in the correlation file. Units: metres

    nl : int
        order (l) of the maximum spherical harmonic to use in 'spharmonic' calculation

    nlmin : int
        order (l) of the minimum spherical harmonic to use in 'spharmonic' calculation

    use_supercell : Bool
        Compute the pairs between subject file and supercell atoms if True
        If False, only subject atoms are used

    r_power : int
        Multiply padf by r^{r_power} - CHECK IF IMPLEMENTED!!

    check_convergence : Bool
        Stop model padf calcualtion if convergence target reached
    
    convergence_target : Float
        A number between 0 (no convergence) and 1 (full convergence)

    nthreads : int
        Number of CPU processors to use    

    verbosity : int
        Report progress during calculation           

    use_atom_weights : Bool
        If true, each atom is wieghted by no. of electrons; if false, each atom counts as 1.

    subcellsize : float
        If greater than 0, then use a subcells to tile the volume (default -1.0) 
    """


    def __init__(self):
        """Construct the paramsPADF class instance. 
           Adds all parameters to the class with default values.
        """        


        ch = ["MODELPADF"]

        params.__init__(self,"PADF parameters",configheaders=ch)


        self.add_parameter("config", "None", cmdline="--config",cmdline2="-c", help="Name of file with all input parameters",
                nargs=1,header=["PADF"],pathflag=True)

        self.add_parameter("subjectxyz", "None", cmdline="--subjectxyz",cmdline2="-xyz", help="Input .xyz file with atomic coordinates ",
                        nargs=1,header=ch[0],pathflag=True)
        
        self.add_parameter("supercellxyz", "None", cmdline="--supercellxyz",cmdline2="-sc", help="Input .xyz file with atomic coordinates for supercell atoms, if required ",
                        nargs=1,header=ch[0],pathflag=True)
        
        self.add_parameter("use_supercell", False, cmdline="--use_supercell",cmdline2="-usc", help="Use the supercell coordinates",
                            nargs=1,header=ch[0],pathflag=False)   

        self.add_parameter("outpath", "None", cmdline="--outpath",cmdline2="-o", help="Path where files will be written.",
                        nargs=1,header=ch[0],pathflag=True)
        
        self.add_parameter("tag", "None", cmdline="--tag",cmdline2="-t", help="text string prefix for each output file.",
                        nargs=1,header=ch[0],pathflag=False)

        self.add_parameter("method", "matrix", cmdline="--method",cmdline2="-m", help="method of calculating the model padf 'histogram', 'matrix', 'serial' ",
                        nargs=1,header=ch[0],pathflag=False)
        
        self.add_parameter("mode", "stm", cmdline="--mode",cmdline2="-d", help="method of calculating the model padf 'rrprime', 'stm' ",
                        nargs=1,header=ch[0],pathflag=False)

        self.add_parameter("nr", int(100), cmdline="--nr",cmdline2="-nr", help="Number of real-space radial samples",
                        nargs=1,header=ch[0],pathflag=False)

        self.add_parameter("nth", int(100), cmdline="--nth",cmdline2="-nth", help="Number of theta samples",
                        nargs=1,header=ch[0],pathflag=False)
        
        self.add_parameter("nl", int(32), cmdline="--nl",cmdline2="-nl", help="Order (l) of the largest spherical harmonic to use in the 'spharmonic' calculation",
                        nargs=1,header=ch[0],pathflag=False)
        
        self.add_parameter("nlmin", int(2), cmdline="--nlmin",cmdline2="-nlm", help="Order (l) of the smallest spherical harmonic to use in the 'spharmonic' calculation",
                        nargs=1,header=ch[0],pathflag=False)

        self.add_parameter("nthvol", int(100), cmdline="--nthvol",cmdline2="-nthv", help="Number of theta samples in the 3D volume (histrogram calc only)",
                        nargs=1,header=ch[0],pathflag=False)
        
        self.add_parameter("nphivol", int(100), cmdline="--nphivol",cmdline2="-nphiv", help="Number of phi samples in the 3D volume (histrogram calc only)",
                        nargs=1,header=ch[0],pathflag=False)

        self.add_parameter("rmax", 10.0, cmdline="--rmax",cmdline2="-rx", help="maximum r value (A)",
                        nargs=1,header=ch[0],pathflag=False)

        self.add_parameter("convergence_target", 1.0, cmdline="--convergence_target" ,cmdline2="-ct", help="convergence target value between 0 and 1",
                        nargs=1,header=ch[0],pathflag=False)

        self.add_parameter("check_convergence", False, cmdline="--check_convergence",cmdline2="-cc", help="If true, stop calculation when convergence target is met",
                            nargs=1,header=ch[0],pathflag=False)   
        
        self.add_parameter("r_power", 0, cmdline="--r_power",cmdline2="-p", help="Multiply the PADF by r^{r_power}",
                            nargs=1,header=ch[0],pathflag=False)   

        self.add_parameter("nthreads", 1, cmdline="--nthreads",cmdline2="-n", help="Number of CPU threads to use (NOT IMPLEMENTED)",
                            nargs=1,header=ch[0],pathflag=False)   
        
        self.add_parameter("verbosity", 0, cmdline="--verbosity",cmdline2="-v", help="Report progress during the simulation",
                            nargs=1,header=ch[0],pathflag=False)   
        
        self.add_parameter("use_atom_weights", True, cmdline="--use_atom_weights",cmdline2="-uaw", help="Weight each atom by no. of electrons if true; otherwise weight of all atoms is 1.",
                            nargs=1,header=ch[0],pathflag=False)
 
        self.add_parameter("subcellsize", -1.0, cmdline="--subcellsize" ,cmdline2="-scs", help="if greater than one, tile the volume into subcells and sum the subcell padfs.",
                        nargs=1,header=ch[0],pathflag=False)


    def read_config_file(self):
        """Read the values of the input parameters from a text (config) file.
        
        All parameters are written to a log file in the outpath.
        """

        #cwd = os.getcwd()+"/"
        #self.read_parameters_from_file(cwd+self.parser.parse_args().config[0] )
        abspath = os.path.abspath(self.parser.parse_args().config[0])
        self.parse_config_file( abspath )
        self.parse_commandline_args()
        self.convert_paths()
        self.checkpaths()
        print("config file name:", abspath )
        outname = self.makefname( self.outpath, self.tag, "_modelpadf_parameter_log",".txt" )
        self.write_params_to_file( outname )
