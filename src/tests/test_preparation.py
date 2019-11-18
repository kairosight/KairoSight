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
# File paths  and files needed for tests
dir_cwd = Path.cwd()
dir_tests = str(dir_cwd)


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
    file_name = '2019/04/04 rata-12-Ca, PCL 150ms'
    file_signal = dir_tests + '/data/20190404-rata-12-150_right_signal1.csv'
    file_signal1_wrong = dir_tests + '/data/20190404-rata-12-150_right_signal1'
    print("sys.maxsize : " + str(sys.maxsize) +
          ' \nIs it greater than 32-bit limit? : ' + str(sys.maxsize > 2 ** 32))

    def test_params(self):
        # Make sure type errors are raised when necessary
        # source : str
        self.assertRaises(TypeError, open_signal, source=250)

        # Make valid errors are raised when parameters are invalid
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
        fps = 800
        time_ca, data_ca = open_signal(source=self.file_signal, fps=fps)

        # Build a figure to plot model data
        fig_single, ax_single = plot_test()
        ax_single.set_title('An Imported Signal (fps: {})'.format(fps))
        ax_single.set_ylabel('Arbitrary Fluorescent Units')
        ax_single.set_xlabel('Time (ms)')

        # Plot aligned model data
        # ax_dual_multi.set_ylim([1500, 2500])
        # data_vm_align = -(data_vm - data_vm.max())
        # data_ca_align = data_ca - data_ca.min()
        plot_vm, = ax_single.plot(time_ca, data_ca, color=color_ca)
        # plot_ca, = ax_single.plot(time_ca, data_ca, marker='+', color=color_ca, label='Ca')
        # plot_baseline = ax_single.axhline(color='gray', linestyle='--', label='baseline')
        ax_single.text(0.65, -0.12, self.file_name,
                       color=gray_med, fontsize=fontsize2, transform=ax_single.transAxes)
        # ax_single.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_single.savefig(dir_tests + '/results/prep_OpenSingle.png')
        fig_single.show()


class TestPrepOpen(unittest.TestCase):
    def setUp(self):
        # File paths and files needed for tests
        self.file_single1 = dir_tests + '/data/about1.tif'
        self.file_single1_wrong = dir_tests + '/data/about1'
        self.file_single2 = dir_tests + '/data/02-250_Vm.tif'
        self.file_single2_wrong = dir_tests + '/data/02-250_Vm'
        self.file_meta = dir_tests + '/data/02-250_Vm.pcoraw.rec'
        self.file_meta_wrong = dir_tests + '/data/02-250.pcoraw.txt'
        print("sys.maxsize : " + str(sys.maxsize) +
              ' \nIs it greater than 32-bit limit? : ' + str(sys.maxsize > 2 ** 32))

        self.stack1, self.meta1 = open_stack(source=self.file_single1)
        self.stack2, self.meta_default = open_stack(source=self.file_single2)
        self.stack2, self.meta2 = open_stack(source=self.file_single2, meta=self.file_meta)

    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, open_stack, source=250)
        self.assertRaises(TypeError, open_stack, source=self.file_single2, meta=True)
        # Make valid errors are raised when parameters are invalid
        self.assertRaises(FileNotFoundError, open_stack, source=self.file_single1_wrong)
        self.assertRaises(FileNotFoundError, open_stack, source=self.file_single1, meta=self.file_meta_wrong)

    def test_results(self):
        # Make sure files are opened and read correctly
        # stack : ndarray
        self.assertIsInstance(self.stack1, np.ndarray)  # TODO is it really?! not an Array?!
        # meta : dict
        self.assertIsInstance(self.meta1, dict)
        self.assertIsInstance(self.meta_default, dict)
        self.assertIsInstance(self.meta2, str)


if __name__ == '__main__':
    unittest.main()
