"""
Plotting 3D PADF Model (take 3D data, convert to 2D and plot result) to get the 
2D crosssections of the r = r' plane of the PADF for cubic sample which refers to:
- 

"""

import numpy as np                  #import entire NumPy library
import matplotlib.pyplot as plt     #relative import of matplotlib

path = "C:\\Users\\benru\\Documents\\PYTHON\\ONPS2186\\PADF_Project\\plotting\\dd_lattice\\"
#tag = "dd_lattice\\"

#datain = np.load(path+tag+"dd_lattice_Theta_total_sum")
datain = np.load(path+"dd_lattice_Theta_total_sum.npy")

print(datain.shape)     #display dimensions of the input data file (3D) "(70, 70, 90)"

s = datain.shape        #assign file dimensions to variable, "s" 

data = np.zeros((s[0],s[2]))   #create a 2D NumPy array of zeros (70, 90)

for i in np.arange(s[0]):       #loop over NumPy array to return evenly spaced values within a given interval 
    data[i,:] = datain[i,i,:]



data += data[:,::-1]

print(data.shape)       #display dimensions of the amended input data file (2D) with 3rd dimesion removed "(70, 90)"

rmax = 25.0

#apply geometric corrections 
th = np.outer( np.ones(data.shape[0]), np.arange(data.shape[1]))
ith = np.where(th>0.0)
data[ith] *= 1.0/np.abs(np.sin( np.pi*th[ith]/float(data.shape[1]))+1e-3)
r = np.outer( np.arange( data.shape[0]), np.ones(data.shape[1]) )*rmax/float(data.shape[0])
ir = np.where(r>1.0)
data[ir] *= 1.0/(r[ir]**3) #radial correction (edit this line [default value: **4])

print(np.max(data[50:,5:-5]))   #display... 

# axis labels and plot title 
plt.xlabel("Theta (degrees)")          #refers to the angles, Theta (units: degrees)
plt.ylabel("r = r' (nm)")              #refers to the r = r' plane distance, (units: nm)
plt.title("Theta Total Sum (dd_lattice)")    #refers to the total sum of all the Theta values (the angle between r and r' when r = r')

# x-axis 
x_ticks = np.arange(0, 200, 20)   #(start, stop, step)
plt.xticks(x_ticks)

# y-axis
y_ticks = np.arange(0, 30, 5)    #(start, stop, step)
plt.yticks(y_ticks)

plt.imshow(data, extent=[0,180,0,rmax], origin='lower', aspect=180/rmax)
plt.clim([0,10]) #set max/min to be displayed (edit this line [default values: 0, 1])

plt.draw()     #redraw the current figure (uncomment when viewing in CMD window)
plt.show()     #displays all open figures (uncomment when viewing in CMD window)

plt.savefig("dd_lattice\\data_out_dd_lattice.png", bbox_inches='tight', pad_inches=0.5) #save plot as .png file 
