

import numpy as np
import matplotlib.pyplot as plt


# shift - a 2D version of numpy's roll
def array_shift(array,xshift=0,yshift=0):
	array = np.roll(array,xshift,0)
	array = np.roll(array,yshift,1)
	return array

## make an array with a circle set to one
def circle(nx, ny, rad=None, cenx=None, ceny=None, invert=0 ): 
	
    # set defaults
    if rad is None: rad = np.min([nx,ny])/2
    if cenx is None: cenx = nx//2
    if ceny is None: ceny = ny//2

    # define the circle
    x = np.outer(np.arange(nx),np.ones(ny)) - nx//2
    y = np.outer(np.ones(nx),np.arange(ny)) - ny//2
   
    dist = np.sqrt(x**2 + y**2)
    a = np.zeros([nx,ny])
    icirc = np.where(dist <= rad)
    a[icirc] = 1.

    if (invert==1):
	    a = abs(1-a)

    out = array_shift(a, -nx//2, -ny//2)
    out = array_shift(out, cenx, ceny)
    return out
#end circle


def write_cif_file( points, fname ):

    f = open( fname, 'w' )
    f.write( "_cell_length_a                  5.0\n")
    f.write( "_cell_length_b                  5.0\n")
    f.write( "_cell_length_c                  5.0\n")
    f.write( "_cell_angle_alpha               90.0\n")
    f.write( "_cell_angle_beta                90.0\n")
    f.write( "_cell_angle_gamma               90.0\n")
    f.write( "_symmetry_space_group_name_H-M 'P 1'\n")
    f.write( "_symmetry_Int_Tables_number    '1'\n")
    f.write( "\n" )
    f.write( "loop_\n")
    f.write( "_atom_site_label\n")
    f.write( "_atom_site_type_symbol\n")
    f.write( "_atom_site_fract_x\n")
    f.write( "_atom_site_fract_y\n")
    f.write( "_atom_site_fract_z\n")


    count =0
#write each line of the file 
    for point in points:
            #previous lines + atom number + atom label +    x                  + y            +z + newline
        f.write( str(count)+' O'+str(count+1)+ ' ' +str(point[0])+' '+str(point[1])+' '+str(point[2])+'\n' )
        count = count +1
    f.close()

def reflection( xyz, nvec, pos, remap=True ):

    c = np.sum( nvec * xyz )
    cneg = pos - (c - pos)
    output = cneg*nvec + (xyz - c*nvec)
    if remap: output = remap_frac_coords( output )
    return output

def remap_frac_coords( xyz ):
    return xyz%1.0

def translation( xyz, nvec, shift, remap=True ):
    nvec *= 1.0/np.sqrt(np.sum(nvec*nvec))
    output = xyz + shift*nvec
    if remap:
        output = remap_frac_coords( output )
    return output

#
#  Rotate a 3D vector with axis of rotation and angle
#  vect and axis are double[3]
#
def rotate_vector( vect, axis, angle, remap=True):
    
    vout = vect* 0.0
  
 
    #
    #  Let's ensure that the axis is unit length
    #
    axis *= 1.0/np.sqrt(np.sum(axis*axis))


    #
    # Define the quaternian and conjugate for rotation
    #
    q = np.zeros(4)
    q[0] = np.cos(angle/2.0);
    q[1] = axis[0]*np.sin(angle/2.0)
    q[2] = axis[1]*np.sin(angle/2.0)
    q[3] = axis[2]*np.sin(angle/2.0)

    qstar = -q
    qstar[0] = q[0]

    qvect = np.zeros(4)
    qvect[0] = 0.0
    qvect[1:] = vect;


    qtemp = quaternion_multiply( qvect, qstar )
    qout = quaternion_multiply( q, qtemp )

    vout = qout[1:]

    if remap:
        vout = remap_frac_coords( vout )
    return vout

#
#  Multiply two quaternions
#
def quaternion_multiply( q1, q2 ):

    qout = np.zeros(4)
    qout[0] = q1[0]*q2[0]

    qout[0] += -np.sum(q1[1:]*q2[1:])
     
    qout[1:] = q1[0]*q2[1:] + q2[0]*q1[1:]

    #
    # A cross product
    #       
    qout[1] +=  -q1[2]*q2[3] + q1[3]*q2[2] 
    qout[2] += +q1[1]*q2[3] - q1[3]*q2[1] 
    qout[3] +=  -q1[1]*q2[2] + q1[2]*q2[1] 
    
    return qout

def check_and_append( list1, list2, tol=0.0005 ):

    lout = []
    for xyz in list2:
        keep = True
        for xyz2 in list1:
            d = np.sum( (xyz-xyz2)**2 )
            #print xyz, xyz2, d
            if d <tol:
                keep = False
                break
        if keep : lout.append( xyz )
    lout += list1
    return lout



class pn3m_model():
    
    def __init__(self):


        self.path = "C:\\Users\\benru\\Documents\\PYTHON\\ONPS2186\\PADF_Project\\pn3m_first_version\\pn3m_cell\\"
        self.tag = "testdd"
        self.logname = self.tag+"_parameter_log.txt"
        self.doublediamondflag = True

        #
        # List of original atoms
        #
        self.atoms = [] # np.array([0,0,0]) ]
        #atoms.append( np.array([0.15,0.02,0.02]) )


        self.alat = []
        self.alat.append(np.array([ 1., 0.0, 0.0]))
        self.alat.append(np.array([ 0., 1., 0.0]))
        self.alat.append(np.array([ 0.0, 0.0, 1.]))


        # channel shape parameters
        self.r = 0.05
        self.rnew = 0.05
        self.nsamp = 3
        self.offset = 0.0
        
        # parameters to display the ring as an image
        self.nx = 256
        self.scale = 100
        
        # make a tube from the ring
        self.nsym = 6
        
        # make a supercell
        self.na = 3
        self.nb = 3
        self.nc = 3


        self.log_exclude_list = ['log_exclude_list', 'atoms', 'alat']

    #
    #  Writes all the input parameters to a log file
    #
    def write_all_params_to_file(self, name="None", script="pn3m_cell.py"):
        
            if name=="None":
                f = open( self.path+self.logname, 'w')
            else:
                f = open( name, 'w' )
            f.write( "# log of input parameters\n")
    
            if script != "None":
                f.write( "# generated by "+script+"\n")
    
            a = self.__dict__
            for d, e in a.items():
                
                #   list of parameters to exclude
                if d in self.log_exclude_list:
                    #print("found one", d)
                    continue
                
                # write the parameter to log file
                f.write( d+" = "+str(e)+"\n" )
    
            f.close()

    #
    # make a trianglular shape with straight or curved sides
    #
    # r - distance to triangle vertices
    # rnew - radius of curvature of edges (when = r it's a circle)
    # nsamp - number of points along each edge
    # offset - rotate the triangle by an angle in radians
    #
    def make_triangle(self) :
        r=self.r
        rnew=self.rnew
        nsamp=self.nsamp
        offset=self.offset

        a = r * np.cos( np.pi/3.0 )
        edgelength = r * np.sin( np.pi/3.0 ) 
        print( "triangle r a edgelength", r, a, edgelength )
        x = (np.arange( nsamp) - nsamp//2)/(float(nsamp)/2.0) * edgelength
        y = np.ones(nsamp) * a #1.0 / 2.0
        theta = np.arcsin( edgelength / (rnew ) )
        print( "theta/pi :", theta/np.pi)

        phi = (np.arange( nsamp) - nsamp//2)/(float(nsamp)/2.0) * theta

        #y = np.ones(nsamp)*rnew*np.cos(theta) + np.sqrt( rnew*rnew - rnew*rnew*np.sin(phi)*np.sin(phi) )
        #phi = (np.arange( nsamp) - nsamp/2)/(float(nsamp)/2.0) * theta
        shift = rnew*np.cos( theta )
        y = +a-shift*np.ones(nsamp) + np.sqrt( rnew*rnew - rnew*rnew*np.sin(phi)*np.sin(phi) )
#print "theta, shift, a", theta, shift, a#
        tmp = []
        for i in np.arange(nsamp):
            tmp.append( np.array([x[i],y[i], 0.0]) )

        #vlist = tmp[:]
        
        vlist = []
        nthsteps = 3
        thstep = 2*np.pi/3.0
        for ith in np.arange( nthsteps ):
            	for v in tmp:
                    vlist.append( rotate_vector( v, np.array([0.,0.,1.]), ith*thstep, False ))
                    
        return vlist



    #
    # display the ring
    #
    def display_the_ring(self, vlist):
        nx=self.nx
        scale=self.scale
        image = np.zeros( (nx,nx) )
        c = circle( nx, nx, rad=3, cenx=0, ceny=0)
        for a in vlist:
            ix = int(a[0]*scale)
            iy = int(a[1]*scale)
            print( ix, iy, a)
            #    image[ix:ix+3,iy:iy+3] = 1.0
            image += np.roll(np.roll( c, ix, 0), iy, 1 )

        image = array_shift( image, nx//2, nx//2 )
        plt.imshow( image )
        plt.draw()
        plt.show()
#exit()


    def rotate_the_ring(self,vlist):
        
        #
        # Rotate the ring
        #
        tmp = vlist[:]
        vlist = []
        for v in tmp:
             t = rotate_vector( v , np.array([1.,-1.,0]), 0.955316618125, False )
             t = translation( t, np.array([1.,1.,1.]), 0.15 )
             vlist.append( t )

        return vlist

#        print( "CHECK:", rotate_vector( np.array([0.,0.,1.]), np.array([1.,-1.,0]), 0.955316618125, False ) )
#        print (len(vlist) )
        
    
    #
    # make a tube
    #
    # nsym = number of rings along the tube
    #
    def make_tube_from_ring(self,vlist):
        nasym = self.nsym
        maxlen = np.sqrt(3)/2.0
        step = maxlen / float(nasym)
        #seed = np.array([0.15,0.02,0.02]) 
        atoms = []
        for i in np.arange(nasym):
            for v in vlist:
                atoms.append( translation( v, np.array([1.,1.,1.]), i*step ) )
        return atoms            
        

    def generate_symmetry_equivalent_channels(self,atoms):



        #
        # list of rotation axes
        #
        threefold = [ np.array([1.,1.,1.])  ,  np.array([1.,1.,-1.]) ,  np.array([1.,-1.,1.]),  np.array([-1.,1.,1.]) ]

        #
        # list of glide planes  (need an n-glide function)
        #
        nglide_trans = [  np.array([1.,1.,0.]) ] #,  np.array([0.,1.,1.]),  np.array([1.,0.,1.]),  np.array([1.,-1.,0.]),  np.array([1.,0.,-1.]),  np.array([0.,1.,-1.]) ]
        nglide_refl = [  np.array([0.,0.,1.]) ] #,  np.array([1.,0.,0.]),  np.array([0.,1.,0.]),  np.array([0.,0.,1.]),  np.array([0.,1.,0.]),  np.array([1.,0.,0.]) ]


 


        ucell = atoms[:]
        for th in threefold:
            for i in np.arange( 3 ):
                ucelltmp = ucell[:]
                for a in ucell:
                    v = rotate_vector( a, th, 2.0*i*np.pi/3.0, True ) 
                    ucelltmp = check_and_append( ucelltmp, [v] )
                ucell = ucelltmp[:]
        #print ucell 
        #ucelltmp = ucell
        
        
        ucelltmp = ucell[:]
        for atom in ucell:
            for i, nglide in enumerate(nglide_refl):
                tmp = reflection( atom, nglide, 0.25 )
                output = translation( tmp, nglide_trans[i], 0.5*np.sqrt(2.0) )
                ucelltmp = check_and_append( ucelltmp, [output] )
                            
        ucelltmp = check_and_append( ucelltmp, ucell )
        ucell = ucelltmp[:]
        
        
        #ucelltmp = []
        #for th in threefold:
        #    for i in np.arange( 3 ):
        #        for a in ucelltmp:
        #           v = rotate_vector( a, th, i*np.pi/3.0, True ) 
        #           ucelltmp = check_and_append( ucelltmp, [v] )
        
        #ucell = check_and_append( ucelltmp, ucell )
        #print ucell
        # Output the unit cell list to cif file to check...
        #header of the cif file
        
        
        
        if self.doublediamondflag:
        	ucell2 = ucell[:]
        	for d in ucell2:
        		ucell.append( translation( d, self.alat[0], 0.5 ) )
                
        return ucell
	
    def tile_supercell(self,ucell):
        na=self.na
        nb=self.nb
        nc=self.nc
        alat = self.alat
    
        # Tile the unit cell list over a number of unit cells
        na, nb, nc = 3, 3, 3
        scell = []
        for i in np.arange( na )-na//2:
            for j in np.arange( nb ) - nb//2:
        	    for k in np.arange( nc ) -nc//2:
        		    for d in ucell: 
        			    scell.append( i*alat[0] + j*alat[1] + k*alat[2] + d )
                        
        return scell

    def output_to_file(self,ucell,scell):

        # Output the final cell list to a file for calculating the angular histogram
        #array = np.zeros( (len(scell),4) )
        #array[:,:3] = np.array( scell )
        #array[:,3] = 1.0
        
        #outname = self.path +self.tag+"_dlattice.txt" #_xscale"+"{:.2f}".format(xscale)+"_yscale{:.2f}".format(yscale)+".txt"
        #np.savetxt( outname, array)
        
        outname = self.path +self.tag+"_dlattice.xyz" #_xscale"+"{:.2f}".format(xscale)+"_yscale{:.2f}".format(yscale)+".txt"
        np.savetxt( outname, scell)
        
        
        #array = np.zeros( (len(ucell),4) )
        #array[:,:3] = np.array( ucell )
        #array[:,3] = 1.0
        
        #outname = self.path+self.tag+"_central_lattice.txt" #_xscale"+"{:.2f}".format(xscale)+"_yscale{:.2f}".format(yscale)+".txt"
        #np.savetxt( outname, array)
        
        outname = self.path+self.tag+"_central_lattice.xyz" #_xscale"+"{:.2f}".format(xscale)+"_yscale{:.2f}".format(yscale)+".txt"
        np.savetxt( outname, ucell)
        print( "File written. Done!")

        
#        fname = self.tag+".cif"
#        write_cif_file( ucell, self.path+fname )


    def make_pn3m_model(self):
        
        vlist = self.make_triangle()
        self.display_the_ring(vlist)
        vlist = self.rotate_the_ring(vlist)
        vlist = self.make_tube_from_ring(vlist)
        ucell = self.generate_symmetry_equivalent_channels(vlist)
        scell = self.tile_supercell(ucell,self.alat,na=3,nb=3,nc=3)
        self.output_to_file(ucell,scell)


if __name__ == '__main__':
    pm = pn3m_model()
    pm.write_all_params_to_file()
    pm.make_pn3m_model()