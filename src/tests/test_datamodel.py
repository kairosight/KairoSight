import unittest
from algorithms.datamodel import model_transients, circle_area
from math import pi
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]


class TestModelTransients(unittest.TestCase):
    def test_model_params(self):
        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, model_transients, t=-2)     # no negative total times
        self.assertRaises(ValueError, model_transients, t=99)     # total time at least 100 ms long
        self.assertRaises(ValueError, model_transients, t=150, t0=150)       # start time < than total time
        self.assertRaises(ValueError, model_transients, fps=150)      # no fps < 200
        self.assertRaises(ValueError, model_transients, fps=1001)      # no fps > 1000
        self.assertRaises(ValueError, model_transients, f_0=2 ** 16)     # no baseline > 16-bit max
        self.assertRaises(ValueError, model_transients, f_amp=2 ** 16)  # no amplitude > 16-bit max
        self.assertRaises(ValueError, model_transients, f_amp=-2)   # no amplitude < 0
        self.assertRaises(ValueError, model_transients, f_0=0, f_amp=20)   # no amplitude < 0, especially Vm
        # Multiple transients
        self.assertRaises(ValueError, model_transients, num=-1)  # no negative transients
        self.assertRaises(ValueError, model_transients, t=300, t0=50, num=3)  # not too many transients

        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, model_transients, t=3 + 5j)
        self.assertRaises(TypeError, model_transients, t=True)
        self.assertRaises(TypeError, model_transients, t='radius')
        # Some parameters must be int
        self.assertRaises(TypeError, model_transients, t=250, t0=20, fps=50.3, f_0=1.5, f_amp=10.4)

    def test_model_result(self):
        # Make sure model results are valid
        self.assertEqual(len(model_transients(t=150)), 2)      # time and data arrays returned as a tuple
        self.assertEqual(model_transients(t=1000)[0].size, model_transients(t=1000)[1].size)    # time and data same size
        # Test the returned time array
        self.assertEqual(model_transients(t=150)[0].size, 150)      # length is correct
        self.assertEqual(model_transients(t=1000, t0=10, fps=223)[0].size, 223)      # length is correct with odd fps
        self.assertGreaterEqual(model_transients(t=100)[0].all(), 0)     # no negative times
        self.assertLess(model_transients(t=200)[0].all(), 200)     # no times >= total time parameter
        # Test the returned data array
        self.assertEqual(model_transients(t=150)[1].size, 150)      # length is correct
        self.assertEqual(model_transients(t=1000, t0=10, fps=223)[1].size, 223)     # length is correct with odd fps
        self.assertGreaterEqual(model_transients(t=100)[1].all(), 0)     # no negative values
        self.assertLess(model_transients(t=100)[1].all(), 2 ** 16)     # no values >= 16-bit max
        self.assertGreaterEqual(2000, model_transients(model_type='Vm', f_0=2000)[1].max())  # Vm amplitude handled properly
        self.assertLessEqual(2000, model_transients(model_type='Ca', f_0=2000)[1].min())  # Ca amplitude handled properly
        #

    def test_model_plot_single(self):
        # Test ideal model Vm and Ca single transient data
        time_vm, data_vm = model_transients(f_0=2000, f_amp=250, noise=5)
        time_ca, data_ca = model_transients(model_type='Ca', f_0=1000, f_amp=250, noise=5)
        self.assertEqual(time_vm.size, data_vm.size)     # data and time arrays returned as a tuple

        # Build a figure to plot model data
        fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
        plt.rc('xtick', labelsize=fontsize2)
        plt.rc('ytick', labelsize=fontsize2)
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='x', which='minor', length=3, bottom=True, top=True)
        ax.tick_params(axis='x', which='major', length=8, bottom=True, top=False)
        ax.xaxis.set_major_locator(plticker.MultipleLocator(25))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(5))
        # ax.set_ylim([1500, 2500])

        # Plot aligned model data
        plot_vm, = ax.plot(time_vm, -(data_vm-2000), 'r+', label='Vm')
        plot_ca, = ax.plot(time_ca, data_ca-1000, 'y+', label='Ca')
        plot_baseline = ax.axhline(color='gray', linestyle='--', label='baseline')

        fig.show()

    def test_model_plot_multi(self):
        # Test ideal model Vm and Ca multi transient data
        num = 4
        time_vm, data_vm = model_transients(t=500, t0=25, num=num, f_0=2000, f_amp=250, noise=2)
        time_ca, data_ca = model_transients(model_type='Ca', t=500, t0=25, num=num, f_0=1000, f_amp=250, noise=2)

        # Build a figure to plot model data
        fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
        plt.rc('xtick', labelsize=fontsize2)
        plt.rc('ytick', labelsize=fontsize2)
        ax = fig.add_subplot(111)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='x', which='minor', length=3, bottom=True, top=True)
        ax.tick_params(axis='x', which='major', length=8, bottom=True, top=False)
        ax.xaxis.set_major_locator(plticker.MultipleLocator(25))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(5))
        # ax.set_ylim([1500, 2500])

        # Plot aligned model data
        plot_vm, = ax.plot(time_vm, -(data_vm-2000), 'r+', label='Vm')
        plot_ca, = ax.plot(time_ca, data_ca-1000, 'y+', label='Ca')
        plot_baseline = ax.axhline(color='gray', linestyle='--', label='baseline')

        fig.show()


# Example tests
class TestModelCircle(unittest.TestCase):
    def test_area(self):
        # Test areas when radius >= 0
        self.assertAlmostEqual(circle_area(1), pi)
        self.assertAlmostEqual(circle_area(0), 0)
        self.assertAlmostEqual(circle_area(2.1), pi * 2.1 * 2.1)

    def test_values(self):
        # Make sure valid errors are raised when necessary
        self.assertRaises(ValueError, circle_area, -2)

    def test_type(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, circle_area, 3+5j)
        self.assertRaises(TypeError, circle_area, True)
        self.assertRaises(TypeError, circle_area, 'radius')


if __name__ == '__main__':
    unittest.main()
