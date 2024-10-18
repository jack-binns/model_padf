"""
params.py

Tools for reading (writing) input parameters from (to) a config (log) files
Based on configparseri, argparse 

Classes:
    parameter - stores information about a single parameter 
    params    - tools for multiple parameters

Functions:
    copy_common_params - copy the dictionaries of two parameters whereever they have common key names

"""

import argparse
import configparser
import pathlib
import os
import glob

# noinspection PyAttributeOutsideInit

#
# stores the basic attibutes of a parameter
#
class parameter:

    """
    Stores information about a single input parameter
    
    Attributes
    ----------
    name : str
        name of the parameter (used in the code)
    
    value : int/float/string/bool
        value of the parameter (int,float,string,bool supported)
    
    type : str
        datatype
    
    cmdline : str
        prefix for command line (e.g. "-c")
    
    help : str
        help text to explain parameter
    
    nargs : int
        number of arguments
    
    header : str
        header for the config file
    
    pathflag : bool
        set to True is parameter is a path
    """

    def __init__(self, name="default",value=0.0,cmdline="-d",cmdline2="--default",help="help!",nargs=1,header="DEFAULT",pathflag=False):
        """
            Constructs parameter class
        """    
        self.name  = name
        self.value = value
        self.type = type(self.value)
        self.cmdline = cmdline
        self.cmdline2 = cmdline2
        self.help = help
        self.nargs = nargs
        self.header = header
        self.pathflag = pathflag


#
# A class to read, store and write many parameters
#
class params:

    """
    Stores information about a single input parameter
    
    Attributes
    ----------
    d : dict
        dictionary of parameter names and objects
    
    parser : argparse.ArgumentParser object
        makes parameter recognised by argparser
    
    config : configparser() object
        allows config to be parsed by configparser (not used)
    
    configheaders : list(str)
        list of headers in the config file    

    """

    def __init__(self,description="parameters",configheaders=[]):

        self.d = {}
        self.parser = argparse.ArgumentParser(description=description)
        self.configp = configparser.ConfigParser()
        self.configheaders = []

        vfile = os.path.join(os.path.dirname(__file__), 'version.txt')
        self.version = open(vfile,'r').readline()

    """
    #
    # Add a parameter to the param dictionary
    #
    """
    def add_parameter(self, name, value, cmdline="-d", cmdline2="--default",help="help!",nargs=1,header="DEFAULT",pathflag=False):
        """
        add parameter to the params class

        Parameters
        ----------
        name : str
            name of the parameter
        
        value : str or float or int or bool
            parameter value

        cmdline :str 
            prefix for command line entry
        
        help : str 
            help string
        
        nargs : int 
            number of arguments

        header : str
            header string for config file (unused)
        
        pathflag : bool 
            True if parameter is a path location

        Returns
        -------
            None
    
            class attributes modified: self.d, self.parser 

        """
    
        p = parameter( name, value, cmdline, cmdline2, help, nargs, header, pathflag=pathflag)
        self.d.update( {p.name: p} )
        self.parser.add_argument( cmdline, cmdline2, nargs=nargs, help=help)
        setattr( self, name, value )

        
    #
    # write all parameters in the param dictionary to file
    #
    def write_params_to_file(self,fname):
        """
        Writes all parameters to a text file.
        
        Attributes
        ----------
        fname : str
            file name for output data
        """
        f = open( fname, 'w')
        for k, v in self.d.items():
            f.write( k+" = "+str(v.value)+"\n" )
        f.close()

    #
    # read parameter from file (an obselete version)
    #
    def read_parameters_from_file(self,fname):

        """
        Read configuration paramneters of a  text file.
        
        Parameters
        ----------
        fname : str
            file name for input data
        """
        
        f = open( fname, 'r')
        
        for line in f:
            if line[0]=="#":
                continue

            bits = line.split()
            if len(bits) < 2:
                continue

            brhs = bits[2]
            if len(bits)>3:
                for b in bits[3:]:
                    brhs+=" "+b

            if bits[0] in self.d.keys():

                if type(self.d[bits[0]].value) is float:
                    self.d[bits[0]].value = float(bits[2])
                elif type(self.d[bits[0]].value) is int:
                    self.d[bits[0]].value = int(bits[2])
                elif type(self.d[bits[0]].value) is str:
                    self.d[bits[0]].value = brhs #bits[2]
                elif type(self.d[bits[0]].value) is bool:
                    if (bits[2]=='True') or (bits[2]==1):
                        self.d[bits[0]].value = True
                    elif (bits[2]=='False') or (bits[2]==0):
                        self.d[bits[0]].value = False

                print(bits[0], self.d[bits[0]].value)
                self.__dict__[bits[0]] = self.d[bits[0]].value

        f.close()

    #
    # read a config file using the configparser package
    #
    def parse_config_file( self, fname ):
        """
        Read configuration paramneters of a  text file (using configparser package).
        
        Parameters
        ----------
        fname : str
            file name for input data

        Returns
        -------
        None

        class attributes modified: self.d
        """
        self.parse_args()
        if self.args.config[0] is not None:        
            self.configp.read( fname )
            self.d['config'].value = fname
        else:
            print("No config file was given at the command line. No parameters have been read from file.")
            return

        for k in self.configp.keys():
            for k2 in self.configp[k].keys():
                if k2 in self.d.keys():
                        #if k2=='pw': print("debug params.py", type(self.d[k2].value))
                        if type(self.d[k2].value) is float:
                            self.d[k2].value = float(self.configp[k][k2])
                        elif type(self.d[k2].value) is int:
                            self.d[k2].value = int(self.configp[k][k2])
                        elif type(self.d[k2].value) is str:
                            self.d[k2].value = self.configp[k][k2]
                        elif type(self.d[k2].value) is bool:
                            if (self.configp[k][k2].lower()=='true') or (self.configp[k][k2]==1):
                                self.d[k2].value = True
                            elif (self.configp[k][k2].lower()=='false') or (self.configp[k][k2]==0):
                                self.d[k2].value = False
                        
                        #if k2=='pw': print("debug params.py", type(self.d[k2].value))
                        #self.d[k2].value = self.configp[k][k2]
                        # print(k2, self.configp[k][k2], self.d[k2].value)
                        self.__dict__[k2] = self.d[k2].value
                else:
                        print( "parameter in config file not required (check the name) :"+k2 )

        #self.convert_paths()

    
    def parse_commandline_args( self ):
        """
        Read configuration paramneters of a  text file (using configparser package).
        
        Parameters
        ----------
        fname : str
            file name for input data

        Returns
        -------
        None

        class attributes modified: self.d
        """
        self.parse_args()

        for k in self.args.__dict__.keys():
            if k in self.d.keys():
                        if self.args.__dict__[k] is not None:
                            v = self.args.__dict__[k][0]
                        else:
                            continue
                        if type(self.d[k].value) is float:
                            self.d[k].value = float(v)
                        elif type(self.d[k].value) is int:
                            self.d[k].value = int(v)
                        elif type(self.d[k].value) is str:
                            self.d[k].value = v
                        elif type(self.d[k].value) is bool:
                            if (v.lower()=='true') or (v==1):
                                self.d[k].value = True
                            elif (v.lower()=='false') or (v==0):
                                self.d[k].value = False
                        
                        #if k2=='pw': print("debug params.py", type(self.d[k2].value))
                        #self.d[k2].value = self.configp[k][k2]
                        # print(k2, self.configp[k][k2], self.d[k2].value)
                        self.__dict__[k] = self.d[k].value
            else:
                        print( "parameter o file not required (check the name) :"+k )

        #self.convert_paths()


    def convert_paths( self):
        """Convert parameter value to path object if pathflag==True
        """        

        for k in self.d.keys():
            if self.d[k].pathflag == True:
                #print("path before:", k, self.d[k].value) #DEBUGGING LINE
                self.d[k].value = pathlib.Path(self.d[k].value)
                #print("path after:", k, self.d[k].value)  #DEBUGGING LINE
                self.__dict__[k] = self.d[k].value

    def set_up_commandline_options(self):
        """Add arguments to parser object for all elements of self.d
        """
        for k, v in self.d.items():
            self.parser.add_argument("--"+k, v.cmdline, nargs=v.nargs, help=v.help)

    def makefname( self, path, tag, suffix, fext):
        """Create a file name from a pathlib.Path() object
        """
        outpath = path / (tag+suffix+fext)
        outname = str(outpath.resolve())
        return outname

    def path_to_string( self, path ):
        """Convert pathlib object to a string
        """
        tmp = path / ""
        return str(tmp.resolve())

    def parse_args(self):
        """parses the command line arguments and stores them in an attribute of the params class
        """
        self.args = self.parser.parse_args()

    def checkpaths(self):
        patherror = False
        for k in self.d.keys():
            #print(k, self.d[k].value)
            if isinstance(self.d[k].value, pathlib.PurePath):
                tmp = self.path_to_string(self.d[k].value.resolve())
                #print(tmp, type(tmp)) 
                if os.path.exists(tmp):
                    continue
                    #print("Found path:", k, tmp)
                elif len(glob.glob(tmp))!=0:
                    continue
                else:
                    print(f'Warning: Input value error. The file/directory "{k}" does not exist: {tmp}')
                    patherror = True

        # sometimes paths might not exist, e.g. if maskflag is not set.
        #if patherror:
        #    print('Exiting... input paths or files given do not exist. Please check config file and/or command line arguments.')
        #    exit()

#
# copy the dictionaries of two parameters whereever they have common key names
#
def copy_common_params( pin, pout ):
    """ copy the dictionaries of two parameters whereever they have common key names
        
    Parameters
    ----------
    pin : dict
        source dictionary (not modified)
    pout : dict
        destination dictionary (modified)

    """
    for k in pin.keys():
        for k2 in pout.keys():
            if k==k2:
                pout[k2] = pin[k]

