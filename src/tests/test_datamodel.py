import unittest
import time
from util.datamodel import model_transients, model_stack, model_stack_propagation, circle_area
from math import pi
from scipy.signal import find_peaks
from imageio import volwrite
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]


def plot_test():
    fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
    axis = fig.add_subplot(111)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.tick_params(axis='x', which='minor', length=3, bottom=True, top=True)
    axis.tick_params(axis='x', which='major', length=8, bottom=True, top=False)
    axis.xaxis.set_major_locator(plticker.MultipleLocator(25))
    axis.xaxis.set_minor_locator(plticker.MultipleLocator(5))
    plt.rc('xtick', labelsize=fontsize2)
    plt.rc('ytick', labelsize=fontsize2)
    return fig, axis


class TestModelTransients(unittest.TestCase):
    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, model_transients, model_type=True)
        self.assertRaises(TypeError, model_transients, t=3 + 5j)
        self.assertRaises(TypeError, model_transients, t=True)
        self.assertRaises(TypeError, model_transients, t='radius')
        # Some parameters must be an int
        self.assertRaises(TypeError, model_transients, t=250.5, t0=20.5, fps=50.3, f_0=1.5, f_amp=10.4)
        self.assertRaises(TypeError, model_transients, num=True)  # num must be an int or 'full'

        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, model_transients, model_type='voltage')     # proper model type
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
        self.assertRaises(ValueError, model_transients, num='5')  # if a string, must be 'full'
        self.assertRaises(ValueError, model_transients, t=300, t0=50, num=3)  # not too many transients
        self.assertRaises(ValueError, model_transients, t=300, t0=50, num=2, cl=49)  # minimum Cycle Length

    def test_results(self):
        # Make sure model results are valid
        self.assertEqual(len(model_transients()), 2)      # time and data arrays returned as a tuple
        self.assertEqual(model_transients(t=1000)[0].size, model_transients(t=1000)[1].size)    # time and data same size
        # Test the returned time array
        self.assertEqual(model_transients(t=150)[0].size, 150)      # length is correct
        self.assertEqual(model_transients(t=1000, t0=10, fps=223)[0].size, 223)      # length is correct with odd fps
        self.assertGreaterEqual(model_transients(t=100)[0].all(), 0)     # no negative times
        self.assertLess(model_transients(t=200)[0].all(), 200)     # no times >= total time parameter
        # Test the returned data array
        self.assertTrue(model_transients(noise=5)[1].dtype in [int])  # data values returned as ints
        self.assertEqual(model_transients(t=150)[1].size, 150)      # length is correct
        self.assertEqual(model_transients(t=1000, t0=10, fps=223)[1].size, 223)     # length is correct with odd fps
        self.assertGreaterEqual(model_transients(t=100)[1].all(), 0)     # no negative values
        self.assertLess(model_transients(t=100)[1].all(), 2 ** 16)     # no values >= 16-bit max
        self.assertGreaterEqual(2000, model_transients(model_type='Vm', f_0=2000)[1].max())  # Vm amplitude handled properly
        self.assertLessEqual(2000, model_transients(model_type='Ca', f_0=2000)[1].min())  # Ca amplitude handled properly

        # Test multiple transient generation
        num = 4
        peak_min_height = 50

        time_vm, data_vm = model_transients(t=500, f_0=2000, f_amp=250, num=num)
        data_vm = (-(data_vm - 2000)) + 2000
        peaks_vm, _ = find_peaks(data_vm, height=peak_min_height)
        self.assertEqual(peaks_vm.size, num)      # detected peaks matches number of generated transients

        time_ca, data_ca = model_transients(model_type='Ca', t=500, f_0=1000, f_amp=250, num=num)
        peaks_ca, _ = find_peaks(data_ca, height=1000 + peak_min_height)
        self.assertEqual(peaks_ca.size, num)      # detected peaks matches number of generated transients

        # time_ca_full, data_ca_full = model_transients(model_type='Ca', t=500, f_0=1000, f_amp=250, num='full')
        time_ca_full, data_ca_full = model_transients(model_type='Ca', t=5000, f_0=1000, f_amp=250, num='full')
        peaks_ca, _ = find_peaks(data_ca_full, height=1000 + peak_min_height)
        self.assertEqual(peaks_ca.size, int(5000 / 100))      # detected peaks matches calculated transients for 'full'

    def test_plot_fps(self):
        # Test model Ca single transient data, at different fps
        time_ca_1, data_ca_1 = model_transients(model_type='Ca', fps=250, f_0=1000, f_amp=250)
        time_ca_2, data_ca_2 = model_transients(model_type='Ca', fps=500, f_0=1000, f_amp=250)
        time_ca_3, data_ca_3 = model_transients(model_type='Ca', fps=1000, f_0=1000, f_amp=250)

        # Build a figure to plot model data
        fig_dual_fps, ax_dual_fps = plot_test()

        # Plot aligned model data
        # ax.set_ylim([1500, 2500])
        plot_ca_1, = ax_dual_fps.plot(time_ca_1, data_ca_1-1000, '#D0D0D0', marker='1', label='Ca, fps: 250')
        plot_ca_2, = ax_dual_fps.plot(time_ca_2, data_ca_2-1000, '#808080', marker='+', label='Ca, fps: 500')
        plot_ca_3, = ax_dual_fps.plot(time_ca_3, data_ca_3-1000, '#606060', marker='2', label='Ca, fps: 1000')
        plot_baseline = ax_dual_fps.axhline(color='gray', linestyle='--', label='baseline')
        ax_dual_fps.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=False)
        fig_dual_fps.show()

    def test_plot_multi(self):
        # Test model Vm multi transient data, at different Cycle Lengths
        num = 4
        time_vm_1, data_vm_1 = model_transients(t=500, f_0=2000, f_amp=250, num=num, cl=50)
        time_vm_2, data_vm_2 = model_transients(t=500, f_0=2000, f_amp=250, num=num)
        time_vm_3, data_vm_3 = model_transients(t=500, f_0=2000, f_amp=250, num=num, cl=150)

        # Build a figure to plot model data
        fig_dual_fps, ax_dual_fps = plot_test()

        # Plot aligned model data
        # ax.set_ylim([1500, 2500])
        plot_vm_1, = ax_dual_fps.plot(time_vm_1, -(data_vm_1-2000), '#D0D0D0', marker='1', label='Vm, CL: 50')
        plot_vm_2, = ax_dual_fps.plot(time_vm_2, -(data_vm_2-2000), '#808080', marker='+', label='Ca, CL: 100')
        plot_vm_3, = ax_dual_fps.plot(time_vm_3, -(data_vm_3-2000), '#606060', marker='2', label='Ca, CL: 150')
        plot_baseline = ax_dual_fps.axhline(color='gray', linestyle='--', label='baseline')
        ax_dual_fps.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=False)
        fig_dual_fps.show()

    def test_plot_dual(self):
        # Test model Vm and Ca single transient data
        time_vm, data_vm = model_transients(f_0=2000, f_amp=250)
        time_ca, data_ca = model_transients(model_type='Ca', f_0=1000, f_amp=250)
        self.assertEqual(time_vm.size, data_vm.size)     # data and time arrays returned as a tuple

        # Build a figure to plot model data
        fig_dual, ax_dual = plot_test()

        # Plot aligned model data
        # ax.set_ylim([1500, 2500])
        plot_vm, = ax_dual.plot(time_vm, -(data_vm-2000), 'r+', label='Vm')
        plot_ca, = ax_dual.plot(time_ca, data_ca-1000, 'y+', label='Ca')
        plot_baseline = ax_dual.axhline(color='gray', linestyle='--', label='baseline')

        fig_dual.show()

    def test_plot_dual_multi(self):
        # Test model Vm and Ca multi-transient data, with noise
        num = 4
        time_vm, data_vm = model_transients(t=500, t0=25, f_0=2000, f_amp=250, noise=2, num=num)
        time_ca, data_ca = model_transients(model_type='Ca', t=500, t0=25, f_0=1000, f_amp=250, noise=2, num=num)

        # Build a figure to plot model data
        fig_dual_multi, ax_dual_multi = plot_test()

        # Plot aligned model data
        # ax_dual_multi.set_ylim([1500, 2500])
        plot_vm, = ax_dual_multi.plot(time_vm, -(data_vm-2000), 'r+', label='Vm')
        plot_ca, = ax_dual_multi.plot(time_ca, data_ca-1000, 'y+', label='Ca')
        plot_baseline = ax_dual_multi.axhline(color='gray', linestyle='--', label='baseline')

        fig_dual_multi.show()


class TestModelStack(unittest.TestCase):
    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, model_stack, size=20)  # size must be a tuple, e.g. (100, 50)
        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, model_stack, size=(20, 5))     # no size > (10, 10)
        self.assertRaises(ValueError, model_stack, size=(5, 20))     # no size > (10, 10)

    def test_results(self):
        # Make sure model results are valid
        self.assertEqual(len(model_stack(t=1000)), 2)      # time and data arrays returned as a tuple
        stack_time, stack_data = model_stack(t=1000)
        self.assertEqual(stack_time.size, stack_data.shape[0])    # time and data same size

        # Test the returned time array
        self.assertEqual(stack_time.size, 1000)      # length is correct
        self.assertGreaterEqual(stack_time.all(), 0)     # no negative times
        self.assertLess(stack_time.all(), 1000)     # no times >= total time parameter

        # Test the returned data array
        self.assertEqual(stack_data.shape[0], 1000)      # length is correct
        self.assertEqual(stack_data.shape, (1000, 100, 50))    # default dimensions (T, Y, X)
        self.assertGreaterEqual(stack_data.all(), 0)     # no negative values
        self.assertLess(stack_data.all(), 2 ** 16)     # no values >= 16-bit max
        stackSize_time, stackSize_data = model_stack(t=1000, size=(100, 100))
        self.assertEqual(stackSize_data.shape, (1000, 100, 100))    # dimensions (T, Y, X)

    def test_tiff(self):
        # Make sure this stack is similar to a 16-bit .tif/.tiff
        time_vm, data_vm = model_stack(t=1000)
        volwrite('ModelStack_vm.tif', data_vm)

        time_ca, data_ca = model_stack(model_type='Ca', t=1000)
        volwrite('ModelStack_ca.tif', data_ca)


class TestModelStackPropagation(unittest.TestCase):
    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, model_stack_propagation, size=20)  # size must be a tuple, e.g. (100, 50)
        self.assertRaises(TypeError, model_stack_propagation, cv='50')
        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, model_stack_propagation, size=(20, 5))     # no size > (10, 10)
        self.assertRaises(ValueError, model_stack_propagation, size=(5, 20))     # no size > (10, 10)
        self.assertRaises(ValueError, model_stack_propagation, cv=4)     # no cv > 5

    def test_results(self):
        # Make sure model results are valid
        self.assertEqual(len(model_stack_propagation(t=1000)), 2)  # time and data arrays returned as a tuple
        stack_time, stack_data = model_stack_propagation(t=1000)
        self.assertEqual(stack_time.size, stack_data.shape[0])

        # Test the returned time array
        self.assertEqual(stack_time.size, 1000)  # length is correct
        self.assertGreaterEqual(stack_time.all(), 0)  # no negative times
        self.assertLess(stack_time.all(), 1000)  # no times >= total time parameter

        # Test the returned data array
        self.assertEqual(stack_data.shape[0], 1000)  # length is correct
        self.assertEqual(stack_data.shape, (1000, 100, 50))  # default dimensions (T, Y, X)
        self.assertGreaterEqual(stack_data.all(), 0)  # no negative values
        self.assertLess(stack_data.all(), 2 ** 16)  # no values >= 16-bit max
        stackSize_time, stackSize_data = model_stack_propagation(t=1000, size=(100, 100))
        self.assertEqual(stackSize_data.shape, (1000, 100, 100))  # dimensions (T, Y, X)

    def test_tiff(self):
        # Make sure this stack is similar to a 16-bit .tif/.tiff
        start = time.process_time()
        time_vm, data_vm = model_stack_propagation(t=1000)
        end = time.process_time()
        print('Timing, test_tiff, Vm : ', end - start)
        volwrite('ModelStackPropagation_vm.tif', data_vm)

        time_ca, data_ca = model_stack_propagation(model_type='Ca', t=1000)
        volwrite('ModelStackPropagation_ca.tif', data_ca)

    def test_tiff_noise(self):
        # Make sure this stack is similar to a 16-bit .tif/.tiff
        start = time.process_time()
        time_vm, data_vm = model_stack_propagation(t=1000, noise=5, num='full')
        end = time.process_time()
        print('Timing, test_tiff_noise, Vm : ', end - start)
        volwrite('ModelStackPropagation_vm.tif', data_vm)

        time_ca, data_ca = model_stack_propagation(model_type='Ca', t=1000, noise=10, num='full')
        volwrite('ModelStackPropagation_ca.tif', data_ca)


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
