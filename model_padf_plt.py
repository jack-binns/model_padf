import matplotlib.pyplot as plt
import numpy as np
import math as m
import itertools


class ModelPlotTools:

    def __init__(self, root, project, array_fn):
        self.root = root
        self.project = project
        self.array_fn = array_fn

        # Probe parameters:
        self.r_probe = 10.0
        self.angular_bin = 2.0
        self.r_dist_bin = 0.1
        self.probe_theta_bin = 10.0

        self.array = np.load(self.root + self.project + self.array_fn)
        print("loaded ", self.array_fn)
        print("shape:", self.array.shape)

    def polar_slice(self, pie_slice):
        """
        Plots a 2D slice in polar coords.
        Needs updating with nice labels etc.
        """
        ax1 = plt.subplot(projection="polar")
        ax1.set_thetamin(0)
        ax1.set_thetamax(180)
        x = np.linspace(0, pie_slice.shape[0], pie_slice.shape[0])
        y = np.linspace(0, m.pi, pie_slice.shape[-1])
        ax1.pcolormesh(y, x, pie_slice, shading='auto')
        plt.show()


    def r_scan_rp_theta(self, target_r):
        """
        Create a 2D array of Theta values through the slice
        with a fixed target_r, dims r_prime & theta
        :param target_r:
        :return:
        """
        r_yard_stick = np.arange(self.r_dist_bin, self.r_probe + self.r_dist_bin, self.r_dist_bin)
        target_r_index = (np.abs(r_yard_stick - target_r)).argmin()
        print("target_r_index :", target_r_index)

        # now let's go through Theta:
        print(self.array.shape)

        self.array += self.array[:, :, ::-1]  # slice magic
        pie_slice = self.array[target_r_index, :, :]
        end_sliver = self.array[target_r_index, :, 0]
        pie_slice = np.column_stack((pie_slice, end_sliver))

        print("slice.shape:")
        print(pie_slice.shape)
        plt.show()

        self.polar_slice(pie_slice)


if __name__ == '__main__':
    print("Plotting model PADF slices...")
    # root = '/home/jack/python/model_padf/'
    # project = 'graphite/'
    # array_fn = 'graphite_Theta_total_sum.npy'
    root = "G:\\python\\model_padf\\"
    project = "graphite\\"
    array_fn = "graphite_Theta_total_sum.npy"
    model_plot = ModelPlotTools(root, project, array_fn)
    print(model_plot.root)
    print(model_plot.project)
    # If you want to play around with probes and binning put it here

    # r_of_interest = 5.0
    # for r_of_interest in np.arange(0.5, 7.0, 0.5):
    #     model_plot.r_scan_rp_theta(r_of_interest)

    padf = np.load(root + project + "graphite_slice_total_sum.npy")

    model_plot.polar_slice(padf)

    # plt.imshow(padf)
    # plt.draw()
    # plt.show()
