import numpy as np
import matplotlib.pyplot as plt
from model_padf_plt import ModelPlotTools

fname = "G:\\python\\model_padf\\ben\\para_testing\\plot_test\\para_testing_slice_total_sum_50.npy"
padf = np.load(fname)

print(padf.shape)
fig = plt.figure()
plt.imshow(padf)
plt.draw()
plt.show()
