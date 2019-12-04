import unittest

from matplotlib.patches import Circle

from util.datamodel import *
from util.analysis import *
from util.processing import *
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.colors as colors
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import util.ScientificColourMaps5 as SCMaps
import util.vikO as SCMapsViko

# File paths needed for tests
dir_tests = str(Path.cwd().parent)
dir_unit = str(Path.cwd())

fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]

gray_light, gray_med, gray_heavy = ['#D0D0D0', '#808080', '#606060']
color_ideal, color_raw, color_filtered = [gray_light, '#FC0352', '#03A1FC']
color_vm, color_ca = ['#FF9999', '#99FF99']
colors_times = {'Start': '#C07B60',
                'Activation': '#842926',
                'Peak': '#4B133D',
                'Downstroke': '#436894',
                'End': '#94B0C3',
                'Baseline': '#C5C3C2'}  # SCMapsViko, circular colormap
# colors_times = {'Start': '#FFD649',
#                 'Activation': '#FFA253',
#                 'Peak': '#F6756B',
#                 'Downstroke': '#CB587F',
#                 'End': '#8E4B84',
#                 'Baseline': '#4C4076'}  # yellow -> orange -> purple
# colors_times = [SCMapsViko[0], SCMapsViko[0], SCMapsViko[0],
#                 SCMapsViko[0], SCMapsViko[0], SCMapsViko[0]]  # redish -> purple -> blue

# Colormap and normalization range for activation maps
cmap_activation = SCMaps.lajolla
ACT_MAX = 100
cmap_norm_activation = colors.Normalize(vmin=0, vmax=ACT_MAX)


def plot_test():
    fig = plt.figure(figsize=(8, 5))  # _ x _ inch page
    axis = fig.add_subplot(111)
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.tick_params(axis='x', which='minor', length=3, bottom=True)
    axis.tick_params(axis='x', which='major', length=8, bottom=True)
    plt.rc('xtick', labelsize=fontsize2)
    plt.rc('ytick', labelsize=fontsize2)
    return fig, axis


def plot_map():
    # Setup a figure to show a frame and a map generated from that frame
    fig = plt.figure(figsize=(8, 4))  # _ x _ inch page
    axis_img = fig.add_subplot(121)
    axis_map = fig.add_subplot(122)
    # Common between the two
    for ax in [axis_img, axis_map]:
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticklabels([])

    return fig, axis_img, axis_map


class TestStart(unittest.TestCase):
    def setUp(self):
        # Create data to test with
        self.signal_t = 200
        self.signal_t0 = 10
        self.signal_fps = 1000
        self.signal_noise = 3

        self.time_vm, self.signal_vm = model_transients(t=self.signal_t, t0=self.signal_t0,
                                                        fps=self.signal_fps, noise=self.signal_noise)
        self.time_ca, self.signal_ca = model_transients(t=self.signal_t, t0=self.signal_t0,
                                                        fps=self.signal_fps)
        self.time, self.signal = self.time_vm, invert_signal(self.signal_vm)

        self.zoom_t = 40

    def test_parameters(self):
        # Make sure type errors are raised when necessary
        signal_bad_type = np.full(100, True)
        # signal_in : ndarray, dtyoe : uint16 or float
        self.assertRaises(TypeError, find_tran_start, signal_in=True)
        self.assertRaises(TypeError, find_tran_start, signal_in=signal_bad_type)

        # Make sure parameters are valid, and valid errors are raised when necessary

    def test_results(self):
        # Make sure result types are valid
        # i_start : np.int64
        i_start = find_tran_start(self.signal)
        self.assertIsInstance(i_start, np.int64)  # index of start start
        self.assertAlmostEqual(self.time[i_start], self.signal_t0, delta=5)  # start time

    def test_plot(self):
        # Build a figure to plot the signal, it's derivatives, and any analysis points
        # General layout
        fig_points, ax_points = plot_test()
        ax_points.set_title('Analysis Point: Start')
        ax_points.set_ylabel('Arbitrary Fluorescent Units')
        ax_points.set_xlabel('Time (ms)')
        points_lw = 3

        ax_points.plot(self.time, self.signal, color=gray_heavy,
                       linestyle='-', marker='x', label='Vm (Model)')
        ax_points.set_xlim(0, self.zoom_t)

        ax_dfs = ax_points.twinx()  # instantiate a second axes that shares the same x-axis
        ax_dfs.set_ylabel('dF/dt, d2F/dt2')  # we already handled the x-label with ax1

        time_x = np.linspace(0, len(self.signal) - 1, len(self.signal))
        spl = UnivariateSpline(time_x, self.signal)

        df_spline = spl(time_x, nu=1)
        df_smooth = savgol_filter(df_spline, window_length=5, polyorder=3)
        spl_df_smooth = UnivariateSpline(time_x, df_smooth)

        d2f_smooth = spl_df_smooth(time_x, nu=1)

        # df/dt
        ax_dfs.plot(self.time, df_smooth,
                    color=gray_med, linestyle='--', label='dF/dt')
        # d2f/dt2
        ax_dfs.plot(self.time, d2f_smooth,
                    color=gray_med, linestyle=':', label='d2F/dt2')
        df_max = round(max(max(df_smooth, key=abs), max(d2f_smooth, key=abs)) + 5.1, -1)
        ax_dfs.set_ylim([-df_max, df_max])

        # Start
        i_start = find_tran_start(self.signal)  # 1st df2 max, Start
        ax_points.axvline(self.time[i_start], color=colors_times['Start'], linewidth=points_lw,
                          label='Start')

        ax_dfs.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        ax_points.legend(loc='upper left', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_points.show()


class TestActivation(unittest.TestCase):
    def setUp(self):
        # Create data to test with
        self.signal_t = 200
        self.signal_t0 = 10
        self.signal_fps = 1000
        self.signal_noise = 3

        self.time_vm, self.signal_vm = model_transients(t=self.signal_t, t0=self.signal_t0,
                                                        fps=self.signal_fps, noise=self.signal_noise)
        self.time_ca, self.signal_ca = model_transients(t=self.signal_t, t0=self.signal_t0,
                                                        fps=self.signal_fps)
        self.time, self.signal = self.time_vm, invert_signal(self.signal_vm)

        self.zoom_t = 40

    def test_parameters(self):
        # Make sure type errors are raised when necessary
        signal_bad_type = np.full(100, True)
        # signal_in : ndarray, dtyoe : uint16 or float
        self.assertRaises(TypeError, find_tran_act, signal_in=True)
        self.assertRaises(TypeError, find_tran_act, signal_in=signal_bad_type)

        # Make sure parameters are valid, and valid errors are raised when necessary

    def test_results(self):
        # Make sure result types are valid
        # i_activation : np.int64
        i_activation = find_tran_act(self.signal)
        self.assertIsInstance(i_activation, np.int64)  # index of activation time
        self.assertGreater(self.time[i_activation], self.signal_t0)  # activation time
        # self.assertLess(self.time[i_activation], self.signal_t0, delta=5)  # activation time

    def test_plot(self):
        # Build a figure to plot the signal, it's derivatives, and any analysis points
        # General layout
        fig_points, ax_points = plot_test()
        ax_points.set_title('Analysis Point: Activation')
        ax_points.set_ylabel('Arbitrary Fluorescent Units')
        ax_points.set_xlabel('Time (ms)')
        points_lw = 3

        ax_points.plot(self.time, self.signal, color=gray_heavy,
                       linestyle='-', marker='x', label='Vm (Model)')
        ax_points.set_xlim(0, self.zoom_t)

        # show the workings of this analysis
        ax_dfs = ax_points.twinx()  # instantiate a second axes that shares the same x-axis
        ax_dfs.set_ylabel('dF/dt')
        # Limit search to before the peak and after the first non-prominent point (baseline) before the peak
        # Characterize the signal
        signal_bounds = (self.signal.min(), self.signal.max())
        signal_range = signal_bounds[1] - signal_bounds[0]
        # find the peak
        i_peaks, properties = find_peaks(self.signal, prominence=(signal_range / 2))
        i_peak = i_peaks[0]  # the first detected peak
        # use the prominence of the peak to find a baseline
        prominece_floor = self.signal[i_peak] - (properties['prominences'][0] * 0.8)
        i_baslines = np.where(self.signal[:i_peak] <= prominece_floor)
        i_baseline = np.max(i_baslines)

        search_min = i_baseline
        search_max = i_peak

        time_x = np.linspace(0, len(self.signal) - 1, len(self.signal))
        spl = UnivariateSpline(time_x, self.signal)

        df_spline = spl(time_x, nu=1)
        df_smooth = savgol_filter(df_spline, window_length=5, polyorder=3)
        # spl_df_smooth = UnivariateSpline(time_x, df_smooth)

        # d2f_smooth = spl_df_smooth(time_x, nu=1)

        # df/dt
        ax_dfs.plot(self.time, df_smooth,
                    color=gray_med, linestyle='--', label='dF/dt')
        # d2f/dt2
        # ax_dfs.plot(self.time, d2f_smooth,
        #             color=gray_med, linestyle=':', label='d2F/dt2')
        df_max = round(max(df_smooth, key=abs) + 5.1, -1)
        ax_dfs.set_ylim([-df_max, df_max])

        # Activation
        i_start = find_tran_act(self.signal)  # 1st df max, Activation
        ax_points.hlines(self.signal[i_baseline], xmin=search_min, xmax=search_max,
                          color=colors_times['Activation'], linewidth=points_lw/2, label='Search Area')
        ax_points.axvline(self.time[i_start], color=colors_times['Activation'], linewidth=points_lw,
                          label='Activation')

        ax_dfs.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        ax_points.legend(loc='upper left', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_points.show()


class TestPeak(unittest.TestCase):
    def setUp(self):
        # Create data to test with
        self.signal_t = 200
        self.signal_t0 = 10
        self.signal_noise = 5  # as a % of the signal amplitude

        time_vm, signal_vm = model_transients(t=self.signal_t, t0=self.signal_t0,
                                              noise=self.signal_noise)
        time_ca, signal_ca = model_transients(model_type='Ca', t=self.signal_t, t0=self.signal_t0,
                                              noise=self.signal_noise)
        self.time, self.signal = time_vm, invert_signal(signal_vm)

    def test_parameters(self):
        # Make sure type errors are raised when necessary
        signal_bad_type = np.full(100, True)
        # signal_in : ndarray, dtyoe : uint16 or float
        self.assertRaises(TypeError, find_tran_peak, signal_in=True)
        self.assertRaises(TypeError, find_tran_peak, signal_in=signal_bad_type)

        # Make sure parameters are valid, and valid errors are raised when necessary

    def test_results(self):
        # Make sure result types are valid
        # i_peak : np.int64
        i_peak = find_tran_peak(self.signal)
        self.assertIsInstance(i_peak, np.int64)  # index of peak time


# class TestDownstroke(unittest.TestCase):
#     # Setup data to test with
#     signal_F0 = 1000
#     signal_amp = 100
#     signal_t0 = 20
#     signal_time = 500
#     noise = 5  # as a % of the signal amplitude
#     noise_count = 100
#     time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
#                                           f0=signal_F0, famp=signal_amp, noise=noise)
#
#     def test_parameters(self):
#         # Make sure type errors are raised when necessary
#         signal_bad_type = np.full(100, True)
#         # signal_in : ndarray, dtyoe : uint16 or float
#         self.assertRaises(TypeError, find_tran_downstroke, signal_in=True)
#         self.assertRaises(TypeError, find_tran_downstroke, signal_in=signal_bad_type)
#
#         # Make sure parameters are valid, and valid errors are raised when necessary
#
#     def test_results(self):
#         # Make sure result types are valid
#         #  i_downstroke : int
#         i_downstroke = find_tran_downstroke(self.signal_ca)
#         self.assertIsInstance(i_downstroke, np.int64)
#
#         self.assertAlmostEqual(i_downstroke, self.signal_t0 + 10, delta=5)  # time to peak of an OAP/OCT
#
#
# class TestEnd(unittest.TestCase):
#     # Setup data to test with
#     signal_F0 = 1000
#     signal_amp = 100
#     signal_t0 = 20
#     signal_time = 500
#     noise = 5  # as a % of the signal amplitude
#     noise_count = 100
#     time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
#                                           f0=signal_F0, famp=signal_amp, noise=noise)
#
#     def test_parameters(self):
#         # Make sure type errors are raised when necessary
#         signal_bad_type = np.full(100, True)
#         # signal_in : ndarray, dtyoe : uint16 or float
#         self.assertRaises(TypeError, find_tran_end, signal_in=True)
#         self.assertRaises(TypeError, find_tran_end, signal_in=signal_bad_type)
#
#         # Make sure parameters are valid, and valid errors are raised when necessary
#
#     def test_results(self):
#         # Make sure result types are valid
#         #  i_end : int
#         i_end = find_tran_end(self.signal_ca)
#         self.assertIsInstance(i_end, np.int64)


class TestAnalysisPoints(unittest.TestCase):
    def setUp(self):
        # Create data to test with
        self.signal_t = 200
        self.signal_t0 = 10
        self.signal_fps = 1000
        self.signal_noise = 3

        self.time_vm, self.signal_vm = model_transients(t=self.signal_t, t0=self.signal_t0,
                                                        fps=self.signal_fps, noise=self.signal_noise)
        self.time_ca, self.signal_ca = model_transients(t=self.signal_t, t0=self.signal_t0,
                                                        fps=self.signal_fps)
        self.time, self.signal = self.time_vm, invert_signal(self.signal_vm)

        self.zoom_t = 50

    def test_plot(self):
        # Build a figure to plot the signal, it's derivatives, and the analysis points
        # General layout
        fig_points, ax_points = plot_test()
        ax_points.set_title('Analysis Points')
        ax_points.set_ylabel('Arbitrary Fluorescent Units')
        ax_points.set_xlabel('Time (ms)')
        points_lw = 3

        ax_points.plot(self.time, self.signal, color=gray_heavy,
                       linestyle='-', marker='x', label='Vm (Model)')
        ax_points.set_xlim(0, self.zoom_t)

        ax_dfs = ax_points.twinx()  # instantiate a second axes that shares the same x-axis
        ax_dfs.set_ylabel('dF/dt, d2F/dt2')  # we already handled the x-label with ax1

        time_x = np.linspace(0, len(self.signal) - 1, len(self.signal))
        spl = UnivariateSpline(time_x, self.signal)

        df_spline = spl(time_x, nu=1)
        df_smooth = savgol_filter(df_spline, window_length=5, polyorder=3)
        spl_df_smooth = UnivariateSpline(time_x, df_smooth)

        d2f_smooth = spl_df_smooth(time_x, nu=1)

        # df/dt
        ax_dfs.plot(self.time, df_smooth,
                    color=gray_med, linestyle='--', label='dF/dt')
        # d2f/dt2
        ax_dfs.plot(self.time, d2f_smooth,
                    color=gray_med, linestyle=':', label='d2F/dt2')
        df_max = round(max(max(df_smooth, key=abs), max(d2f_smooth, key=abs)) + 5.1, -1)
        ax_dfs.set_ylim([-df_max, df_max])

        # Start
        i_start = find_tran_start(self.signal)  # 1st df2 max, Start
        ax_points.axvline(self.time[i_start], color=colors_times['Start'], linewidth=points_lw,
                          label='Start')
        # Activation
        i_activation = find_tran_act(self.signal)  # 1st df max, Activation
        ax_points.axvline(self.time[i_activation], color=colors_times['Activation'], linewidth=points_lw,
                          label='Activation')
        # Peak
        i_peak = find_tran_peak(self.signal)  # max of signal, Peak
        ax_points.axvline(self.time[i_peak], color=colors_times['Peak'], linewidth=points_lw,
                          label='Peak')
        # Downstroke
        i_downstroke = find_tran_downstroke(self.signal)  # df min, Downstroke
        ax_points.axvline(self.time[i_downstroke], color=colors_times['Downstroke'], linewidth=points_lw,
                          label='Downstroke')
        # End
        i_end = find_tran_end(self.signal)  # 2st df2 max, End
        ax_points.axvline(self.time[i_end], color=colors_times['End'], linewidth=points_lw,
                          label='End')

        ax_dfs.legend(loc='upper right', ncol=1, prop={'size': fontsize3}, numpoints=1, frameon=True)
        ax_points.legend(loc='upper left', ncol=1, prop={'size': fontsize3}, numpoints=1, frameon=True)

        fig_points.savefig(dir_unit + '/results/analysis_AnalysisPoints.png')
        fig_points.show()


# class TestDuration(unittest.TestCase):
#     # Setup data to test with
#     signal_F0 = 1000
#     signal_amp = 100
#     signal_t0 = 20
#     signal_time = 500
#     noise = 5  # as a % of the signal amplitude
#     noise_count = 100
#     time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
#                                           f0=signal_F0, famp=signal_amp, noise=noise)
#
#     def test_parameters(self):
#         # Make sure type errors are raised when necessary
#         signal_bad_type = np.full(100, True)
#         # signal_in : ndarray, dtyoe : int or float
#         #  percent : int
#         self.assertRaises(TypeError, calc_tran_duration, signal_in=True, percent=True)
#         self.assertRaises(TypeError, calc_tran_duration, signal_in=signal_bad_type, percent='500')
#         self.assertRaises(TypeError, calc_tran_duration, signal_in='word', percent=3j + 7)
#         self.assertRaises(TypeError, calc_tran_duration, signal_in=3j + 7)
#
#         # Make sure parameters are valid, and valid errors are raised when necessary
#         # signal_in : >=0
#         # percent : >=0
#         signal_bad_value = np.full(100, 10)
#         signal_bad_value[20] = signal_bad_value[20] - 50
#         percent_bad_value = -1
#         self.assertRaises(ValueError, calc_tran_duration, signal_in=signal_bad_value, percent=percent_bad_value)
#
#     def test_results(self):
#         # Make sure result types are valid
#         #  duration : int
#         duration = calc_tran_duration(self.signal_ca)
#         self.assertIsInstance(duration, np.int32)
#
#         self.assertAlmostEqual(duration, self.signal_t0)


# class TestTau(unittest.TestCase):
#     # Setup data to test with
#     signal_F0 = 1000
#     signal_amp = 100
#     signal_t0 = 20
#     signal_time = 500
#     noise = 5  # as a % of the signal amplitude
#     noise_count = 100
#     time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
#                                           f0=signal_F0, famp=signal_amp, noise=noise)
#
#     def test_parameters(self):
#         # Make sure type errors are raised when necessary
#         signal_bad_type = np.full(100, True)
#         # signal_in : ndarray, dtyoe : int or float
#         self.assertRaises(TypeError, calc_tran_di, signal_in=True)
#         self.assertRaises(TypeError, calc_tran_di, signal_in=signal_bad_type)
#         self.assertRaises(TypeError, calc_tran_di, signal_in='word')
#         self.assertRaises(TypeError, calc_tran_di, signal_in=3j + 7)
#
#         # Make sure parameters are valid, and valid errors are raised when necessary
#         # signal_in : >=0
#         signal_bad_value = np.full(100, 10)
#         signal_bad_value[20] = signal_bad_value[20] - 50
#         self.assertRaises(ValueError, calc_tran_tau, signal_in=signal_bad_value)
#
#         # should not be applied to signal data containing at least one transient
#
#     def test_results(self):
#         # Make sure result types are valid
#         #  di : float
#         di = calc_tran_duration(self.signal_ca)
#         self.assertIsInstance(di, np.float32)
#
#         self.assertAlmostEqual(di, self.signal_t0)


# class TestDI(unittest.TestCase):
#     # Setup data to test with
#     signal_F0 = 1000
#     signal_amp = 100
#     signal_t0 = 20
#     signal_time = 500
#     noise = 5  # as a % of the signal amplitude
#     noise_count = 100
#     time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
#                                           f0=signal_F0, famp=signal_amp, noise=noise)
#
#     def test_parameters(self):
#         # Make sure type errors are raised when necessary
#         signal_bad_type = np.full(100, True)
#         # signal_in : ndarray, dtyoe : int or float
#         self.assertRaises(TypeError, calc_tran_tau, signal_in=True)
#         self.assertRaises(TypeError, calc_tran_tau, signal_in=signal_bad_type)
#         self.assertRaises(TypeError, calc_tran_tau, signal_in='word')
#         self.assertRaises(TypeError, calc_tran_tau, signal_in=3j + 7)
#
#         # Make sure parameters are valid, and valid errors are raised when necessary
#         # signal_in : >=0
#         signal_bad_value = np.full(100, 10)
#         signal_bad_value[20] = signal_bad_value[20] - 50
#         self.assertRaises(ValueError, calc_tran_tau, signal_in=signal_bad_value)
#
#     def test_results(self):
#         # Make sure result types are valid
#         #  tau : float
#         tau = calc_tran_duration(self.signal_ca)
#         self.assertIsInstance(tau, np.float32)
#
#         self.assertAlmostEqual(tau, self.signal_t0)


#  class TestMapTau(unittest.TestCase):
#  class TestMapTau(unittest.TestCase):

#  class TestDFreq(unittest.TestCase):

class TestEnsemble(unittest.TestCase):
    def setUp(self):
        # Create data to test with
        self.signal_t = 800
        self.signal_t0 = 50
        self.signal_f0 = 1000
        self.signal_famp = 100
        self.signal_fps = 500
        self.signal_noise = 5  # as a % of the signal amplitude
        self.signal_num = 5
        self.signal_cl = 150

        self.time_vm, self.signal_vm = \
            model_transients(t=self.signal_t, t0=self.signal_t0, fps=self.signal_fps,
                             f0=self.signal_f0, famp=self.signal_famp, noise=self.signal_noise,
                             num=self.signal_num, cl=self.signal_cl)
        self.time, self.signal = self.time_vm, invert_signal(self.signal_vm)

    def test_params(self):
        time_bad_type = np.full(100, True)
        signal_bad_type = np.full(100, True)
        # Make sure type errors are raised when necessary
        # time_in : ndarray, dtyoe : int or float
        self.assertRaises(TypeError, calc_ensemble, time_in=True, signal_in=self.signal)
        self.assertRaises(TypeError, calc_ensemble, time_in=time_bad_type, signal_in=self.signal)
        # signal_in : ndarray, dtyoe : uint16 or float
        self.assertRaises(TypeError, calc_ensemble, time_in=self.time, signal_in=True)
        self.assertRaises(TypeError, calc_ensemble, time_in=self.time, signal_in=signal_bad_type)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : must have more than one peak
        time_short, signal_shot = model_transients(model_type='Ca')
        self.assertRaises(ValueError, calc_ensemble, time_in=time_short, signal_in=signal_shot)

    def test_results(self):
        # Make sure spatial filter results are correct
        time_out, signal_out, signals, i_peaks, est_cycle = calc_ensemble(self.time, self.signal)
        # time_out : ndarray
        self.assertIsInstance(time_out, np.ndarray)  # ensembled signal
        self.assertAlmostEqual(len(time_out), est_cycle * (self.signal_fps / 1000), delta=10)  #

        # signal_out : ndarray
        self.assertIsInstance(signal_out, np.ndarray)  # ensembled signal
        self.assertEqual(len(signal_out), len(signal_out))  #

        # signals : list
        self.assertIsInstance(signals, list)  # ensembled signal
        self.assertEqual(len(signals), self.signal_num)  #

        # i_peaks : ndarray
        self.assertIsInstance(i_peaks, np.ndarray)  # indicies of peaks
        self.assertEqual(len(i_peaks), self.signal_num)

        # est_cycle : float
        self.assertIsInstance(est_cycle, float)  # estimated cycle length (ms) of ensemble
        self.assertAlmostEqual(est_cycle, self.signal_cl, delta=5)  #

    def test_plot(self):
        # Make sure ensembled transient looks correct
        time_ensemble, signal_ensemble, signals, signal_peaks, est_cycle_length = calc_ensemble(self.time, self.signal)

        # Build a figure to plot SNR results
        # fig_snr, ax_snr = plot_test()
        fig_ensemble = plt.figure(figsize=(12, 8))  # _ x _ inch page
        gs0 = fig_ensemble.add_gridspec(2, 1, height_ratios=[0.2, 0.8])  # 2 rows, 1 column
        ax_signal = fig_ensemble.add_subplot(gs0[0])
        ax_ensemble = fig_ensemble.add_subplot(gs0[1])

        ax_signal.spines['right'].set_visible(False)
        ax_signal.spines['top'].set_visible(False)
        ax_signal.tick_params(axis='x', which='minor', length=3, bottom=True)
        ax_signal.tick_params(axis='x', which='major', length=8, bottom=True)
        plt.rc('xtick', labelsize=fontsize2)
        plt.rc('ytick', labelsize=fontsize2)

        fig_ensemble.suptitle('Ensemble Averaging')
        ax_signal.set_ylabel('Arbitrary Fluorescent Units')
        ax_signal.set_xlabel('Time (ms)')
        # ax_snr.set_ylim([self.signal_F0 - 20, self.signal_F0 + self.signal_amp + 20])

        ax_signal.plot(self.time, self.signal, color=gray_light,
                       linestyle='None', marker='+', label='Ca pixel data')
        ax_signal.plot(self.time[signal_peaks], self.signal[signal_peaks],
                       "x", color=colors_times['Peak'], markersize=10, label='Peaks')

        ax_ensemble.set_ylabel('Arbitrary Fluorescent Units')
        ax_ensemble.text(0.65, 0.5, 'PCL (samples): {}'.format(est_cycle_length),
                         color=gray_heavy, fontsize=fontsize1, transform=ax_ensemble.transAxes)
        ax_ensemble.text(0.65, 0.55, '# Peaks : {}'.format(len(signal_peaks)),
                         color=gray_heavy, fontsize=fontsize1, transform=ax_ensemble.transAxes)
        # spl_ensemble = UnivariateSpline(time_ensemble, signal_ensemble)
        # spline_ensemble = spl_ensemble(time_ensemble)
        # ax_ensemble.plot(time_ensemble, spline_ensemble, color=gray_heavy,
        #                  linestyle='-', label='Ensemble spline')
        ax_ensemble.plot(time_ensemble, signal_ensemble, color=gray_heavy,
                         linestyle='-', marker='+', label='Ensemble signal')
        # Activation
        i_activation = find_tran_act(signal_ensemble)  # 1st df max, Activation
        ax_ensemble.axvline(time_ensemble[i_activation], color=colors_times['Activation'], linewidth=3,
                            label='Activation')
        signals_activations = []

        for signal in signals:
            # spl_signal = UnivariateSpline(time_ensemble, signal)
            # spline_signal = spl_signal(time_ensemble)
            # ax_ensemble.plot(time_ensemble, spline_signal, color=gray_light, linestyle='-')
            ax_ensemble.plot(time_ensemble, signal, color=gray_light, linestyle='-')
            signal_act = find_tran_act(signal)
            signals_activations.append(signal_act)
            ax_ensemble.plot(time_ensemble[signal_act], signal[signal_act],
                             "x", color=colors_times['Activation'], markersize=10, label='Peaks')
        # # Activation error bar
        # error_act = np.mean(signals_activations).astype(int)
        # ax_ensemble.errorbar(time_ensemble[error_act],
        #                      signal_ensemble[error_act],
        #                      xerr=statistics.stdev(signals_activations), fmt="x",
        #                      color=colors_times['Activation'], lw=3,
        #                      capsize=4, capthick=1.0)

        fig_ensemble.savefig(dir_unit + '/results/analysis_Ensemble.png')
        fig_ensemble.show()


class TestMapAnalysis(unittest.TestCase):
    def setUp(self):
        # Create data to test with, a propagating stack
        self.size = (100, 100)
        self.fps = 500
        self.f0 = 1000
        self.famp = 200
        self.noise = 10
        self.velocity = 10

        time_ca, stack_ca = model_stack_propagation(
            size=self.size, model_type='Ca', fps=self.fps, f0=self.f0, famp=self.famp,
            noise=self.noise, velocity=self.velocity)
        self.time, self.stack = time_ca, stack_ca

        self.FRAMES = self.stack.shape[0]
        self.HEIGHT, self.WIDTH = (self.stack.shape[1], self.stack.shape[2])
        self.frame_shape = (self.HEIGHT, self.WIDTH)
        self.origin_x, self.origin_y = self.WIDTH / 2, self.HEIGHT / 2
        self.div_borders = np.linspace(start=int(self.HEIGHT / 2), stop=self.HEIGHT / 2 / 5, num=5)

    def test_params(self):
        # Make sure type errors are raised when necessary
        # stack_in : ndarray, 3-D array
        stack_bad_shape = np.full((100, 100), 100, dtype=np.uint16)
        stack_bad_type = np.full(self.stack.shape, True)
        self.assertRaises(TypeError, map_tran_analysis, time_in=self.time, stack_in=stack_bad_shape)
        self.assertRaises(TypeError, map_tran_analysis, time_in=self.time, stack_in=stack_bad_type)
        # analysis_type : method
        # self.assertRaises(TypeError, map_tran_analysis, stack_in=self.stack_ca, analysis_type=True)
        # time_in : ndarray
        self.assertRaises(TypeError, map_tran_analysis, time_in=True, stack_in=self.stack)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # analysis_type : an analysis method
        # self.assertRaises(ValueError, map_tran_analysis, stack_in=self.stack_ca, analysis_type='activation')

    def test_results(self):
        # Make sure Analysis Map results are correct
        analysis_map = map_tran_analysis(self.stack, find_tran_act)
        self.assertIsInstance(analysis_map, np.ndarray)  # snr map type
        self.assertEqual(analysis_map.shape, self.frame_shape)  # snr map shape

    def test_map(self):
        # Make sure analysis map looks correct
        # Plot a frame from the stack and the SNR map of that frame
        fig_map_snr, ax_frame, ax_map = plot_map()

        # timestamps of analysis results
        analysis_map = map_tran_analysis(self.stack, find_tran_act, self.time)
        analysis_max = np.nanmax(analysis_map)
        # print('Activation Map: ')

        ax_frame.set_title('Model Data\n(noise SD: {}, velocity: {} cm/s)'
                           .format(self.noise, self.velocity))
        ax_map.set_title('Activation Map')
        # Frame from stack
        frame_num = int(analysis_max / 2 * self.time[1])  # interesting frame
        cmap_frame = SCMaps.grayC.reversed()
        img_frame = ax_frame.imshow(self.stack[frame_num, :, :], cmap=cmap_frame)
        # Draw circles showing borders of SNR variance
        for idx, div_border in enumerate(self.div_borders):
            div_circle = Circle((self.origin_x, self.origin_y), radius=div_border,
                                fc=None, fill=None, ec=gray_light, lw=1, linestyle='--')
            ax_frame.add_artist(div_circle)
        ax_frame.set_ylabel('1.0 cm', fontsize=fontsize3)
        ax_frame.set_xlabel('0.5 cm', fontsize=fontsize3)
        ax_map.set_ylabel('1.0 cm', fontsize=fontsize3)
        ax_map.set_xlabel('0.5 cm', fontsize=fontsize3)
        # Add colorbar (lower right of frame)
        ax_ins_img = inset_axes(ax_frame, width="5%", height="80%", loc=5,
                                bbox_to_anchor=(0.2, 0, 1, 1), bbox_transform=ax_frame.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_frame, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('Intensity\n(a.u).', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)
        # Analysis Map
        img_snr = ax_map.imshow(analysis_map, norm=cmap_norm_activation, cmap=cmap_activation)
        # Add colorbar (lower right of map)
        ax_ins_map = inset_axes(ax_map, width="5%", height="80%", loc=5,
                                bbox_to_anchor=(0.2, 0, 1, 1), bbox_transform=ax_map.transAxes,
                                borderpad=0)
        cb1_map = plt.colorbar(img_snr, cax=ax_ins_map, orientation="vertical")
        cb1_map.ax.set_xlabel('Activation\ntime (ms)', fontsize=fontsize3)
        cb1_map.ax.yaxis.set_major_locator(plticker.LinearLocator(5))
        cb1_map.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb1_map.ax.tick_params(labelsize=fontsize3)

        fig_map_snr.savefig(dir_unit + '/results/processing_MapActivation.png')
        fig_map_snr.show()

    def test_map_signal(self):
        # Make sure analysis map and analysis points look correct
        # Plot a frame from the stack, the map of that stack, and a signals
        # fig_map_snr, ax_frame, ax_map_snr = plot_map()
        fig_map_snr = plt.figure(figsize=(8, 6))  # _ x _ inch page
        gs0 = fig_map_snr.add_gridspec(2, 1, height_ratios=[0.6, 0.4])  # 2 rows, 1 column
        ax_signal = fig_map_snr.add_subplot(gs0[1])
        ax_signal.set_xlabel('Time (ms)')
        ax_signal.set_xlim(0, ACT_MAX)
        ax_signal.set_ylim(self.f0 - 10, self.f0 + self.famp + 10)
        ax_signal.spines['right'].set_visible(False)
        ax_signal.spines['top'].set_visible(False)
        ax_signal.tick_params(axis='x', labelsize=fontsize3, which='minor', length=3)
        ax_signal.tick_params(axis='x', labelsize=fontsize3, which='major', length=8)
        ax_signal.tick_params(axis='y', labelsize=fontsize3)
        ax_signal.xaxis.set_major_locator(plticker.MultipleLocator(25))
        ax_signal.xaxis.set_minor_locator(plticker.MultipleLocator(5))

        gs_frame_map = gs0[0].subgridspec(1, 3, width_ratios=[0.475, 0.475, 0.05], wspace=0.4)  # 1 row, 3 columns
        ax_frame = fig_map_snr.add_subplot(gs_frame_map[0])
        ax_frame.set_title('Model Data\n(noise SD: {}, velocity: {} cm/s)'
                           .format(self.noise, self.velocity))
        ax_map = fig_map_snr.add_subplot(gs_frame_map[1])
        ax_map.set_title('Activation Map')
        for ax in [ax_frame, ax_map]:
            ax.tick_params(axis='x', labelsize=fontsize4)
            ax.tick_params(axis='y', labelsize=fontsize4)

        # Calculate the activation map, returns timestamps
        analysis_map = map_tran_analysis(self.stack, find_tran_act, self.time)
        analysis_max_time = np.nanmax(analysis_map)

        # Frame from stack
        frame_num = int(self.stack.shape[0] / 4)  # interesting frame
        cmap_frame = SCMaps.grayC.reversed()
        cmap_norm_frame = colors.Normalize(vmin=self.stack.min(), vmax=self.stack.max())
        img_frame = ax_frame.imshow(self.stack[frame_num, :, :], norm=cmap_norm_frame, cmap=cmap_frame)

        # Add colorbar (right of frame)
        ax_ins_img = inset_axes(ax_frame, width="3%", height="80%", loc=5,
                                bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax_frame.transAxes,
                                borderpad=0)
        cb_img = plt.colorbar(img_frame, cax=ax_ins_img, orientation="vertical")
        cb_img.ax.set_xlabel('arb. u.', fontsize=fontsize3)
        cb_img.ax.yaxis.set_major_locator(plticker.LinearLocator(2))
        cb_img.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb_img.ax.tick_params(labelsize=fontsize3)

        # Analysis Map
        # Create normalization range for map (0 and max rounded up to the nearest 10)
        # cmap_activation = SCMaps.tokyo
        # cmap_norm_activation = colors.Normalize(vmin=0, vmax=round(analysis_max + 5.1, -1))
        img_map = ax_map.imshow(analysis_map, norm=cmap_norm_activation, cmap=cmap_activation)
        # Add colorbar (right of map)
        ax_ins_map = inset_axes(ax_map, width="3%", height="80%", loc=5,
                                bbox_to_anchor=(0.1, 0, 1, 1), bbox_transform=ax_map.transAxes,
                                borderpad=0)
        cb1_map = plt.colorbar(img_map, cax=ax_ins_map, orientation="vertical")
        cb1_map.ax.set_xlabel('ms', fontsize=fontsize3)
        cb1_map.ax.yaxis.set_major_locator(plticker.LinearLocator(5))
        cb1_map.ax.yaxis.set_minor_locator(plticker.LinearLocator(10))
        cb1_map.ax.tick_params(labelsize=fontsize3)
        # Map histogram
        ax_act_hist = fig_map_snr.add_subplot(gs_frame_map[2], xticklabels=[], sharey=ax_ins_map)
        ax_act_hist.hist(analysis_map.flatten(), ACT_MAX, histtype='stepfilled',
                         orientation='horizontal', color='gray')
        ax_act_hist.tick_params(axis='y', labelsize=fontsize3)
        ax_act_hist.yaxis.set_major_locator(plticker.LinearLocator(2))
        ax_act_hist.yaxis.set_minor_locator(plticker.LinearLocator(10))

        # Plot signal and/or points of interest
        for iy, ix in np.ndindex(self.frame_shape):
            # Signal
            # signal = self.stack[:, iy, ix]
            # ax_signal.plot(self.time, signal, color=gray_light, linestyle='-')
            # Activation
            time_act = analysis_map[iy, ix]  # 1st df max, Activation
            # index of that activation time
            i_act = np.where(self.time == time_act)
            ax_signal.plot(time_act, self.stack[i_act, iy, ix], "|",
                           color=colors_times['Activation'], markersize=3, label='Activation')
            # ax_signal.axvline(i_act, color=colors_times['Activation'], linewidth=1,
            #                   label='Activation')
            # signal_act = find_tran_act(signal)
            # signals_activations.append(signal_act)
            # ax_signal.plot(self.time[signal_act], signal[signal_act],
            #                  "x", color=colors_times['Activation'], markersize=10, label='Peaks')

        # fig_map_snr.savefig(dir_integration + '/results/integration_MapActivation.png')
        fig_map_snr.show()


# class TestPhase(unittest.TestCase):
#     # Setup data to test with
#     signal_F0 = 1000
#     signal_amp = 100
#     signal_t0 = 50
#     signal_time = 1000
#     signal_num = 5
#     noise = 2  # as a % of the signal amplitude
#     noise_count = 100
#     time_vm, signal_vm = model_transients(t0=signal_t0, t=signal_time,
#                                           f0=signal_F0, famp=signal_amp,
#                                           noise=noise, num=signal_num)
#     time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0 + 15, t=signal_time,
#                                           f0=signal_F0, famp=signal_amp,
#                                           noise=noise, num=signal_num)
#
#     def test_parameters(self):
#         # Make sure type errors are raised when necessary
#         signal_bad_type = np.full(100, True)
#         # signal_in : ndarray, dtyoe : int or float
#         self.assertRaises(TypeError, calc_phase, signal_in=True)
#         self.assertRaises(TypeError, calc_phase, signal_in=signal_bad_type)
#         self.assertRaises(TypeError, calc_phase, signal_in='word')
#         self.assertRaises(TypeError, calc_phase, signal_in=3j + 7)
#
#         # Make sure parameters are valid, and valid errors are raised when necessary
#         # signal_in : >=0
#         signal_bad_value = np.full(100, 10)
#         signal_bad_value[20] = signal_bad_value[20] - 50
#         self.assertRaises(ValueError, calc_phase, signal_in=signal_bad_value)
#
#     def test_results(self):
#         # Make sure result types are valid
#         signal_vm_phase = calc_phase(self.signal_vm)
#         signal_ca_phase = calc_phase(self.signal_ca)
#         # signal_FF0 : ndarray, dtyoe : float
#         self.assertIsInstance(signal_ca_phase, np.ndarray)  # The array of phase
#         self.assertIsInstance(signal_ca_phase[0], float)  # dtyoe : float
#
#         # Make sure result values are valid
#         self.assertAlmostEqual(signal_ca_phase.min(), signal_vm_phase.max(), delta=0.01)


if __name__ == '__main__':
    unittest.main()
