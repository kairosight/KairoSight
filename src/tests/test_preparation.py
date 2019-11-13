import unittest
from util.preparation import *
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]
gray_light, gray_med, gray_heavy = ['#D0D0D0', '#808080', '#606060']
color_vm, color_ca = ['#FF9999', '#99FF99']


def plot_test():
    fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
    axis = fig.add_subplot(111)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    plt.rc('xtick', labelsize=fontsize2)
    plt.rc('ytick', labelsize=fontsize2)
    return fig, axis


class TestPrepOpenSignal(unittest.TestCase):
    # File paths and files needed for tests
    cwd = Path.cwd()
    tests = str(cwd)
    file_signal = tests + '/data/20190404-rata-12-150_right_signal1.csv'
    file_signal1_wrong = tests + '/data/20190404-rata-12-150_right_signal1'
    print("sys.maxsize : " + str(sys.maxsize) +
          ' \nIs it greater than 32-bit limit? : ' + str(sys.maxsize > 2 ** 32))

    def test_params(self):
        # Make sure type errors are raised when necessary
        # source : str
        self.assertRaises(TypeError, open_signal, source=250)

        # Make valid errors are raised when parameters are invalid
        self.assertRaises(FileNotFoundError, open_signal, source=self.tests)
        self.assertRaises(FileNotFoundError, open_signal, source=self.file_signal1_wrong)

    def test_results(self):
        # Make sure files are opened and read correctly
        time, data = open_signal(source=self.file_signal)
        # signal_time : ndarray
        self.assertIsInstance(time, np.ndarray)
        # signal_data : ndarray
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(len(time), len(data))

    def test_plot_single(self):
        time_vm, data_vm = open_signal(source=self.file_signal, fps=800)

        # Build a figure to plot model data
        fig_single, ax_single = plot_test()
        ax_single.set_title('An Imported Signal (Default FPS=500)')
        ax_single.set_ylabel('Arbitrary Fluorescent Units', color=gray_heavy)
        ax_single.set_xlabel('Time (ms)', color=gray_heavy)

        # Plot aligned model data
        # ax_dual_multi.set_ylim([1500, 2500])
        # data_vm_align = -(data_vm - data_vm.max())
        # data_ca_align = data_ca - data_ca.min()
        plot_vm, = ax_single.plot(time_vm, data_vm, color=color_vm, label='Ca')
        # plot_ca, = ax_single.plot(time_ca, data_ca, marker='+', color=color_ca, label='Ca')
        # plot_baseline = ax_single.axhline(color='gray', linestyle='--', label='baseline')
        ax_single.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_single.show()



class TestPrepOpen(unittest.TestCase):
    # File paths and files needed for tests
    cwd = Path.cwd()
    tests = str(cwd)
    file_single1 = tests + '/data/about1.tif'
    file_single1_wrong = tests + '/data/about1'
    file_single2 = tests + '/data/02-250_Vm.tif'
    file_single2_wrong = tests + '/data/02-250_Vm'
    file_meta = tests + '/data/02-250_Vm.pcoraw.rec'
    file_meta_wrong = tests + '/data/02-250.pcoraw.txt'
    print("sys.maxsize : " + str(sys.maxsize) +
          ' \nIs it greater than 32-bit limit? : ' + str(sys.maxsize > 2 ** 32))

    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, open_single, source=250)
        self.assertRaises(TypeError, open_single, source=self.file_single2, meta=True)
        # Make valid errors are raised when parameters are invalid
        self.assertRaises(FileNotFoundError, open_single, source=self.tests)
        self.assertRaises(FileNotFoundError, open_single, source=self.file_single1_wrong)
        self.assertRaises(FileNotFoundError, open_single, source=self.file_single1, meta=self.file_meta_wrong)

    def test_results(self):
        # Make sure files are opened and read correctly
        self.assertIsInstance(open_single(source=self.file_single1)[0], np.ndarray)
        self.assertIsInstance(open_single(source=self.file_single1)[1], dict)
        self.assertIsInstance(open_single(source=self.file_single2)[1], dict)
        self.assertIsInstance(open_single(source=self.file_single2, meta=self.file_meta)[1], str)


if __name__ == '__main__':
    unittest.main()
