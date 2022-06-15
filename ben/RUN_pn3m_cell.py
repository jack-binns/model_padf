#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:52:47 2020

@author: andrewmartin
"""

import pn3m_cell as pn3mc
import numpy as np


#
# make an instance of the pn3m model class
#
cell = pn3mc.pn3m_model()

# location for output files
cell.path = "C:\\Users\\benru\\Documents\\PYTHON\\ONPS2186\\PADF_Project\\pn3m_first_version\\para_testing\\"

# string to prefix all output file names
cell.tag = "para_testing"

# generate a second set of interlaced channels
cell.doublediamondflag = False

#
# channel shape parameters
#
# r - distance to triangle vertices
# rnew - radius of curvature of edges (when rnew=r it's close to a circle)
# nsamp - number of points along each edge
# offset - rotate the triangle by an angle in radians
# 
cell.r = 0.2        # 0.2   referred to as "r" [fractional coordinates as a % of the unit cell with ]
cell.rnew = 0.2     # 0.2   referred to as "r'"
cell.nsamp = 5      # 5      referred to as "n"
cell.offset = 0.0   # 0.0


#
# parameters to display the ring as an image
#
# nx - number of pixels along side length of array
# scale - multiply dimension of ring to get to pixel coordinates
#
cell.nx = 256
cell.scale = 100

#
# make a tube from the ring
#
# nsym - number of rings in a channel
cell.nsym = 6

# Supercell parameters. Number of unit cells along each lattice vector.
cell.na = 1 #3
cell.nb = 1 #3 
cell.nc = 1 #3


#
# Lattice vectors (don't need to change this for cubic structures)
#

lattice_new = 100      #create new lattice parameter

cell.alat = []
cell.alat.append(np.array([ 1., 0.0, 0.0]))
cell.alat.append(np.array([ 0., 1., 0.0]))
cell.alat.append(np.array([ 0.0, 0.0, 1.]))

#
# log the parameters and generate the cell
#
print("Write parameters to file:", cell.path+cell.logname)
cell.write_all_params_to_file()


#
# First we make a list of point on a ring. This defines channel shape
#
vlist = cell.make_triangle()

# Display the ring shape
print('len(vlist)', len(vlist))
cell.display_the_ring(vlist)

# Rotates the ring to the orientation of a pn3m channel.
# Don't change this
vlist = cell.rotate_the_ring(vlist)


# Translates the ring shape along the channel.
# In this project, we may replace this function to make different types of channel
#
vlist = cell.make_tube_from_ring(vlist)

# Generate the symmetry equivalent channels
ucell = cell.generate_symmetry_equivalent_channels(vlist)
ucell_b = lattice_new*np.array(ucell)  # edited


# Generate a supercell
scell = cell.tile_supercell(ucell)
scell = lattice_new*np.array(scell) # edited 


# Output the lists of points to xyz/cif files
#
# These files can be viewed in Vesta (https://jp-minerals.org/vesta/en/)
#or CrsytalOgraph (https://www.epfl.ch/schools/sb/research/iphys/teaching/crystallography/crystalograph/)
#
cell.output_to_file(ucell_b,scell)

fname = cell.tag+".cif"
pn3mc.write_cif_file( ucell, cell.path+fname )
