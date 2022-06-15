"""
Plotting 3D PADF Model (take 3D data, convert to 2D and plot result) to get the 
2D crosssections of the r = r' plane of the PADF for cubic sample which refers to:
- 

"""

import numpy as np                  #import entire NumPy library
import matplotlib.pyplot as plt     #relative import of matplotlib

# path = "C:\\Users\\benru\\Documents\\PYTHON\\ONPS2186\\PADF_Project\\plotting\\para_testing\\"
# path = "G:\\RMIT\\cluster_simulation\\"
path = "G:\\python\\model_padf\\ben\\"
# tag = "n_2\\"
tag = "para_testing\\"

datain = np.load("G:\\python\\model_padf\\ben\\para_testing\\plot_test\\para_testing_slice_total_sum_50.npy")
# datain = np.load(path + tag +"para_testing_slice_total_sum_20.npy")

print(datain.shape)     #display dimensions of the input data file (2D) "(400, 90)"

s = datain.shape 

data = datain #reassign datain to data to avoid changing the below code taken from the 3D plot script
rmax = 50.0

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
plt.title("Theta Slice Total (para_testing)")    #refers to the total sum of all the Theta values (the angle between r and r' when r = r')

# x-axis 
x_ticks = np.arange(0, 200, 20)   #(start, stop, step)
plt.xticks(x_ticks)

# y-axis
y_ticks = np.arange(0, 50, 5)    #(start, stop, step)
plt.yticks(y_ticks)

plt.imshow(data, origin='lower', aspect=180/rmax)
plt.clim([0,10]) #set max/min to be displayed (edit this line [default values: 0, 1])

plt.draw()     #redraw the current figure (uncomment when viewing in CMD window)


plt.savefig("para_testing\\data_out_para_testing.png", bbox_inches='tight', pad_inches=0.5) #save plot as .png file 
plt.show()     #displays all open figures (uncomment when viewing in CMD window)