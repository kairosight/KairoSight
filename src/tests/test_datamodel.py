import unittest
from util.datamodel import model_transients, model_images, circle_area
from math import pi
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]


class TestModelTransients(unittest.TestCase):
    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, model_transients, t=3 + 5j)
        self.assertRaises(TypeError, model_transients, t=True)
        self.assertRaises(TypeError, model_transients, t='radius')
        # Some parameters must be an int
        self.assertRaises(TypeError, model_transients, t=250, t0=20, fps=50.3, f_0=1.5, f_amp=10.4)

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

    def test_results(self):
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

    plt.rc('xtick', labelsize=fontsize2)
    plt.rc('ytick', labelsize=fontsize2)

    def test_plot_fps(self):
        # Test model Vm and Ca single transient data, at different fps's
        time_ca_1, data_ca_1 = model_transients(model_type='Ca', fps=250, f_0=1000, f_amp=250)
        time_ca_2, data_ca_2 = model_transients(model_type='Ca', fps=500, f_0=1000, f_amp=250)
        time_ca_3, data_ca_3 = model_transients(model_type='Ca', fps=1000, f_0=1000, f_amp=250)

        # Build a figure to plot model data
        fig_dual_fps = plt.figure(figsize=(8, 5))  # _ x _ inch page
        ax_dual_fps = fig_dual_fps.add_subplot(111)
        ax_dual_fps.spines['right'].set_visible(False)
        ax_dual_fps.spines['top'].set_visible(False)
        ax_dual_fps.tick_params(axis='x', which='minor', length=3, bottom=True, top=True)
        ax_dual_fps.tick_params(axis='x', which='major', length=8, bottom=True, top=False)
        ax_dual_fps.xaxis.set_major_locator(plticker.MultipleLocator(25))
        ax_dual_fps.xaxis.set_minor_locator(plticker.MultipleLocator(5))

        # Plot aligned model data
        # ax.set_ylim([1500, 2500])
        plot_ca_1, = ax_dual_fps.plot(time_ca_1, data_ca_1-1000, '#D0D0D0', marker='+', label='Ca')
        plot_ca_2, = ax_dual_fps.plot(time_ca_2, data_ca_2-1000, '#808080', marker='+', label='Ca')
        plot_ca_3, = ax_dual_fps.plot(time_ca_3, data_ca_3-1000, '#606060', marker='+', label='Ca')
        plot_baseline = ax_dual_fps.axhline(color='gray', linestyle='--', label='baseline')

        fig_dual_fps.show()

    def test_plot_dual(self):
        # Test model Vm and Ca single transient data
        time_vm, data_vm = model_transients(f_0=2000, f_amp=250)
        time_ca, data_ca = model_transients(model_type='Ca', f_0=1000, f_amp=250)
        self.assertEqual(time_vm.size, data_vm.size)     # data and time arrays returned as a tuple

        # Build a figure to plot model data
        fig_dual = plt.figure(figsize=(8, 5))  # _ x _ inch page
        ax_dual = fig_dual.add_subplot(111)
        ax_dual.spines['right'].set_visible(False)
        ax_dual.spines['top'].set_visible(False)
        ax_dual.tick_params(axis='x', which='minor', length=3, bottom=True, top=True)
        ax_dual.tick_params(axis='x', which='major', length=8, bottom=True, top=False)
        ax_dual.xaxis.set_major_locator(plticker.MultipleLocator(25))
        ax_dual.xaxis.set_minor_locator(plticker.MultipleLocator(5))

        # Plot aligned model data
        # ax.set_ylim([1500, 2500])
        plot_vm, = ax_dual.plot(time_vm, -(data_vm-2000), 'r+', label='Vm')
        plot_ca, = ax_dual.plot(time_ca, data_ca-1000, 'y+', label='Ca')
        plot_baseline = ax_dual.axhline(color='gray', linestyle='--', label='baseline')

        fig_dual.show()

    def test_plot_dual_multi(self):
        # Test model Vm and Ca multi-transient data, with noise
        num = 4
        time_vm, data_vm = model_transients(t=500, t0=25, num=num, f_0=2000, f_amp=250, noise=2)
        time_ca, data_ca = model_transients(model_type='Ca', t=500, t0=25, num=num, f_0=1000, f_amp=250, noise=2)

        # Build a figure to plot model data
        fig_dual_multi = plt.figure(figsize=(8, 5))  # _ x _ inch page
        ax_dual_multi = fig_dual_multi.add_subplot(111)
        ax_dual_multi.spines['right'].set_visible(False)
        ax_dual_multi.spines['top'].set_visible(False)
        ax_dual_multi.tick_params(axis='x', which='minor', length=3, bottom=True, top=True)
        ax_dual_multi.tick_params(axis='x', which='major', length=8, bottom=True, top=False)
        ax_dual_multi.xaxis.set_major_locator(plticker.MultipleLocator(25))
        ax_dual_multi.xaxis.set_minor_locator(plticker.MultipleLocator(5))

        # Plot aligned model data
        # ax_dual_multi.set_ylim([1500, 2500])
        plot_vm, = ax_dual_multi.plot(time_vm, -(data_vm-2000), 'r+', label='Vm')
        plot_ca, = ax_dual_multi.plot(time_ca, data_ca-1000, 'y+', label='Ca')
        plot_baseline = ax_dual_multi.axhline(color='gray', linestyle='--', label='baseline')

        fig_dual_multi.show()

#
# class TestModelIamges(unittest.TestCase):
#     def test_params(self):
#         # Make sure type errors are raised when necessary
#
#
#     def test_results(self):
#         # Make sure model results are valid


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
