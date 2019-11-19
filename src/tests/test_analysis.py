import unittest

from util.analysis import *
from util.processing import *
from util.datamodel import model_transients
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]
gray_light, gray_med, gray_heavy = ['#D0D0D0', '#808080', '#606060']
color_ideal, color_raw, color_filtered = [gray_light, '#FC0352', '#03A1FC']
color_vm, color_ca = ['#FF9999', '#99FF99']
# File paths  and files needed for tests
dir_cwd = Path.cwd()
dir_tests = str(dir_cwd)
# colors_times = ['#FFD649', '#FFA253', '#F6756B', '#CB587F', '#8E4B84', '#4C4076']  # yellow -> orange -> purple
colors_times = {'Start': '#FFD649',
                'Activation': '#FFA253',
                'Peak': '#F6756B',
                'Downstroke': '#CB587F',
                'End': '#8E4B84',
                'Baseline': '#4C4076'}  # yellow -> orange -> purple


# colors_times = ['#FFD649', '#FFA253', '#F6756B', '#CB587F', '#8E4B84', '#4C4076']  # redish -> purple -> blue


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


class TestStart(unittest.TestCase):
    # Setup data to test with
    signal_F0 = 1000
    signal_amp = 100
    signal_t0 = 20
    signal_time = 500
    noise = 5  # as a % of the signal amplitude
    noise_count = 100
    time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
                                          f_0=signal_F0, f_amp=signal_amp, noise=noise)

    def test_parameters(self):
        # Make sure type errors are raised when necessary
        signal_bad_type = np.full(100, True)
        # signal_in : ndarray, dtyoe : int or float
        self.assertRaises(TypeError, find_tran_start, signal_in=True)
        self.assertRaises(TypeError, find_tran_start, signal_in=signal_bad_type)
        self.assertRaises(TypeError, find_tran_start, signal_in='word')
        self.assertRaises(TypeError, find_tran_start, signal_in=3j + 7)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        signal_bad_value = np.full(100, 10)
        signal_bad_value[20] = signal_bad_value[20] - 50
        self.assertRaises(ValueError, find_tran_start, signal_in=signal_bad_value)

    def test_results(self):
        # Make sure result types are valid
        # i_start : int
        i_start = find_tran_start(self.signal_ca)
        self.assertIsInstance(i_start, np.int32)  # index of start

        self.assertAlmostEqual(i_start, self.signal_t0 + 10, delta=5)  # time to peak of an OAP/OCT


class TestActivation(unittest.TestCase):
    # Setup data to test with
    signal_F0 = 1000
    signal_amp = 100
    signal_t0 = 20
    signal_time = 500
    noise = 5  # as a % of the signal amplitude
    noise_count = 100
    time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
                                          f_0=signal_F0, f_amp=signal_amp, noise=noise)

    def test_parameters(self):
        # Make sure type errors are raised when necessary
        signal_bad_type = np.full(100, True)
        # signal_in : ndarray, dtyoe : int or float
        self.assertRaises(TypeError, calc_tran_activation, signal_in=True)
        self.assertRaises(TypeError, calc_tran_activation, signal_in=signal_bad_type)
        self.assertRaises(TypeError, calc_tran_activation, signal_in='word')
        self.assertRaises(TypeError, calc_tran_activation, signal_in=3j + 7)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        signal_bad_value = np.full(100, 10)
        signal_bad_value[20] = signal_bad_value[20] - 50
        self.assertRaises(ValueError, calc_tran_activation, signal_in=signal_bad_value)

    def test_results(self):
        # Make sure result types are valid
        # i_activation : int
        i_activation = calc_tran_activation(self.signal_ca)
        self.assertIsInstance(i_activation, np.int32)

        self.assertAlmostEqual(i_activation, self.signal_t0 + 10, delta=5)  # time to peak of an OAP/OCT


class TestPeak(unittest.TestCase):
    # Setup data to test with
    signal_F0 = 1000
    signal_amp = 100
    signal_t0 = 20
    signal_time = 500
    noise = 5  # as a % of the signal amplitude
    noise_count = 100
    time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
                                          f_0=signal_F0, f_amp=signal_amp, noise=noise)

    def test_parameters(self):
        # Make sure type errors are raised when necessary
        signal_bad_type = np.full(100, True)
        # signal_in : ndarray, dtyoe : uint16 or float
        self.assertRaises(TypeError, find_tran_peak, signal_in=True)
        self.assertRaises(TypeError, find_tran_peak, signal_in=signal_bad_type)

        # Make sure parameters are valid, and valid errors are raised when necessary

    def test_results(self):
        # Make sure result types are valid
        i_peak = find_tran_peak(self.signal_ca)
        # i_peak : int
        self.assertIsInstance(i_peak, np.int32)  # index of peak

        #  Make sure result values are valid
        self.assertAlmostEqual(i_peak, self.signal_t0 + 10, delta=5)  # time to peak of an OAP/OCT


class TestDownstroke(unittest.TestCase):
    # Setup data to test with
    signal_F0 = 1000
    signal_amp = 100
    signal_t0 = 20
    signal_time = 500
    noise = 5  # as a % of the signal amplitude
    noise_count = 100
    time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
                                          f_0=signal_F0, f_amp=signal_amp, noise=noise)

    def test_parameters(self):
        # Make sure type errors are raised when necessary
        signal_bad_type = np.full(100, True)
        # signal_in : ndarray, dtyoe : int or float
        self.assertRaises(TypeError, find_tran_downstroke, signal_in=True)
        self.assertRaises(TypeError, find_tran_downstroke, signal_in=signal_bad_type)
        self.assertRaises(TypeError, find_tran_downstroke, signal_in='word')
        self.assertRaises(TypeError, find_tran_downstroke, signal_in=3j + 7)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        signal_bad_value = np.full(100, 10)
        signal_bad_value[20] = signal_bad_value[20] - 50
        self.assertRaises(ValueError, find_tran_downstroke, signal_in=signal_bad_value)

    def test_results(self):
        # Make sure result types are valid
        #  i_downstroke : int
        i_downstroke = find_tran_end(self.signal_ca)
        self.assertIsInstance(i_downstroke, np.int32)

        self.assertAlmostEqual(i_downstroke, self.signal_t0 + 10, delta=5)  # time to peak of an OAP/OCT


class TestEnd(unittest.TestCase):
    # Setup data to test with
    signal_F0 = 1000
    signal_amp = 100
    signal_t0 = 20
    signal_time = 500
    noise = 5  # as a % of the signal amplitude
    noise_count = 100
    time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
                                          f_0=signal_F0, f_amp=signal_amp, noise=noise)

    def test_parameters(self):
        # Make sure type errors are raised when necessary
        signal_bad_type = np.full(100, True)
        # signal_in : ndarray, dtyoe : int or float
        self.assertRaises(TypeError, find_tran_end, signal_in=True)
        self.assertRaises(TypeError, find_tran_end, signal_in=signal_bad_type)
        self.assertRaises(TypeError, find_tran_end, signal_in='word')
        self.assertRaises(TypeError, find_tran_end, signal_in=3j + 7)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        signal_bad_value = np.full(100, 10)
        signal_bad_value[20] = signal_bad_value[20] - 50
        self.assertRaises(ValueError, find_tran_end, signal_in=signal_bad_value)

    def test_results(self):
        # Make sure result types are valid
        #  i_end : int
        i_end = find_tran_end(self.signal_ca)
        self.assertIsInstance(i_end, np.int32)

        self.assertAlmostEqual(i_end, self.signal_t0 + 10, delta=5)  # time to peak of an OAP/OCT


class TestAnalysisPoints(unittest.TestCase):
    # TODO move to 'integration' tests?
    def setUp(self):
        # Setup data to test with
        self.signal_t = 200
        self.zoom_t = 40
        self.signal_t0 = 10
        self.fps = 1000
        self.time_vm, self.signal_vm = model_transients(t=self.signal_t, t0=self.signal_t0, fps=self.fps)
        self.time_ca, self.signal_ca = model_transients(t=self.signal_t, t0=self.signal_t0, fps=self.fps)
        self.time, self.signal = self.time_vm, invert_signal(self.signal_vm)
        # self.time_ca, self.signal_ca = model_transients(model_type='Ca', t=self.signal_t, t0=self.signal_t0, fps=self.fps)
        # self.time, self.signal = self.time_ca[0:self.zoom_t], self.signal_ca[0:self.zoom_t]
        self.sample_rate = float(self.fps)
        # signal_ca = filter_temporal(signal_ca, sample_rate)
        # dt = int(time_ca[1])
        # TODO try derivative using spline and nu=1, nu=2
        # spl = UnivariateSpline(time, signal)
        # signal_df = spl(time, nu=1)
        self.signal_df = np.diff(self.signal, n=1, prepend=int(self.signal[0])).astype(float)
        self.signal_d2f = np.diff(self.signal, n=2, prepend=[int(self.signal[0]), int(self.signal[0])]).astype(float)
        # signal_df = np.gradient(signal_noisy_ca, dt) / np.gradient(time_ca, dt)
        # signal_d2f = np.gradient(signal_df, dt) / np.gradient(time_ca, dt)

    def test_plot(self):
        # Build a figure to plot the signal, it's derivatives, and the analysis points
        # fig_points = plt.figure(figsize=(8, 8))  # _ x _ inch page
        # General layout
        fig_points, ax_points = plot_test()
        ax_points.set_title('Analysis Points\n1st and 2nd derivatives')
        ax_points.set_ylabel('Arbitrary Fluorescent Units')
        ax_points.set_xlabel('Time (ms)')
        points_lw = 3

        ax_points.plot(self.time[0:self.zoom_t], self.signal[0:self.zoom_t], color=gray_heavy, linestyle='-', marker='x',
                       label='Vm (Model)')

        ax_dfs = ax_points.twinx()  # instantiate a second axes that shares the same x-axis
        # ax_df2 = ax_points.twinx()  # instantiate a second axes that shares the same x-axis
        # ax_dfs.baseline = ax_dfs.axhline(color=gray_light, linestyle='-.')
        ax_dfs.set_ylabel('dF/dt, d2F/dt2')  # we already handled the x-label with ax1
        df_smooth = filter_temporal(self.signal_df, self.sample_rate, filter_order=5)
        d2f_smooth = filter_temporal(self.signal_d2f, self.sample_rate, filter_order=5)

        # self.time, self.signal = self.time[0:self.zoom_t], invert_signal(self.signal[0:self.zoom_t])
        # df_smooth, d2f_smooth = df_smooth[0:self.zoom_t], d2f_smooth[0:self.zoom_t]

        ax_dfs.set_ylim([-25, 25])
        # ax_df2.set_ylim([-20, 20])

        # df/dt
        ax_dfs.plot(self.time[0:self.zoom_t], df_smooth[0:self.zoom_t],
                    color=gray_med, linestyle='--', label='dF/dt')
        # d2f/dt2
        ax_dfs.plot(self.time[0:self.zoom_t], d2f_smooth[0:self.zoom_t],
                    color=gray_med, linestyle=':', label='d2F/dt2')

        # Start
        df2_max1 = np.argmax(d2f_smooth)
        ax_points.axvline(self.time[df2_max1], color=colors_times['Start'], linewidth=points_lw,
                          label='Start')  # 1st df2 max, Upstroke
        # Activation
        df_max1 = np.argmax(df_smooth)  # 1st df max, Activation
        ax_points.axvline(self.time[df_max1], color=colors_times['Activation'], linewidth=points_lw,
                          label='Activation')
        # Peak
        ax_points.axvline(self.time[np.argmax(self.signal)], color=colors_times['Peak'], linewidth=points_lw,
                          label='Peak')  # max of signal, Peak
        # Downstroke
        df_min = np.argmin(df_smooth)  # df min, Downstroke
        ax_points.axvline(self.time[df_min], color=colors_times['Downstroke'], linewidth=points_lw,
                          label='Downstroke')
        # End
        df2_max2 = np.argmax(d2f_smooth[df2_max1 + 3:])  # 2st df2 max, Downstroke
        ax_points.axvline(self.time[df2_max2 + df2_max1 + 3], color=colors_times['End'], linewidth=points_lw,
                          label='End')

        ax_dfs.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        ax_points.legend(loc='upper left', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_points.savefig(dir_tests + '/results/analysis_AnalysisPoints.png')
        fig_points.show()


class TestDuration(unittest.TestCase):
    # Setup data to test with
    signal_F0 = 1000
    signal_amp = 100
    signal_t0 = 20
    signal_time = 500
    noise = 5  # as a % of the signal amplitude
    noise_count = 100
    time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
                                          f_0=signal_F0, f_amp=signal_amp, noise=noise)

    def test_parameters(self):
        # Make sure type errors are raised when necessary
        signal_bad_type = np.full(100, True)
        # signal_in : ndarray, dtyoe : int or float
        #  percent : int
        self.assertRaises(TypeError, calc_tran_duration, signal_in=True, percent=True)
        self.assertRaises(TypeError, calc_tran_duration, signal_in=signal_bad_type, percent='500')
        self.assertRaises(TypeError, calc_tran_duration, signal_in='word', percent=3j + 7)
        self.assertRaises(TypeError, calc_tran_duration, signal_in=3j + 7)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        # percent : >=0
        signal_bad_value = np.full(100, 10)
        signal_bad_value[20] = signal_bad_value[20] - 50
        percent_bad_value = -1
        self.assertRaises(ValueError, calc_tran_duration, signal_in=signal_bad_value, percent=percent_bad_value)

    def test_results(self):
        # Make sure result types are valid
        #  duration : int
        duration = calc_tran_duration(self.signal_ca)
        self.assertIsInstance(duration, np.int32)

        self.assertAlmostEqual(duration, self.signal_t0)


class TestTau(unittest.TestCase):
    # Setup data to test with
    signal_F0 = 1000
    signal_amp = 100
    signal_t0 = 20
    signal_time = 500
    noise = 5  # as a % of the signal amplitude
    noise_count = 100
    time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
                                          f_0=signal_F0, f_amp=signal_amp, noise=noise)

    def test_parameters(self):
        # Make sure type errors are raised when necessary
        signal_bad_type = np.full(100, True)
        # signal_in : ndarray, dtyoe : int or float
        self.assertRaises(TypeError, calc_tran_di, signal_in=True)
        self.assertRaises(TypeError, calc_tran_di, signal_in=signal_bad_type)
        self.assertRaises(TypeError, calc_tran_di, signal_in='word')
        self.assertRaises(TypeError, calc_tran_di, signal_in=3j + 7)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        signal_bad_value = np.full(100, 10)
        signal_bad_value[20] = signal_bad_value[20] - 50
        self.assertRaises(ValueError, calc_tran_tau, signal_in=signal_bad_value)

        # should not be applied to signal data containing at least one transient

    def test_results(self):
        # Make sure result types are valid
        #  di : float
        di = calc_tran_duration(self.signal_ca)
        self.assertIsInstance(di, np.float32)

        self.assertAlmostEqual(di, self.signal_t0)


class TestDI(unittest.TestCase):
    # Setup data to test with
    signal_F0 = 1000
    signal_amp = 100
    signal_t0 = 20
    signal_time = 500
    noise = 5  # as a % of the signal amplitude
    noise_count = 100
    time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0, t=signal_time,
                                          f_0=signal_F0, f_amp=signal_amp, noise=noise)

    def test_parameters(self):
        # Make sure type errors are raised when necessary
        signal_bad_type = np.full(100, True)
        # signal_in : ndarray, dtyoe : int or float
        self.assertRaises(TypeError, calc_tran_tau, signal_in=True)
        self.assertRaises(TypeError, calc_tran_tau, signal_in=signal_bad_type)
        self.assertRaises(TypeError, calc_tran_tau, signal_in='word')
        self.assertRaises(TypeError, calc_tran_tau, signal_in=3j + 7)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        signal_bad_value = np.full(100, 10)
        signal_bad_value[20] = signal_bad_value[20] - 50
        self.assertRaises(ValueError, calc_tran_tau, signal_in=signal_bad_value)

    def test_results(self):
        # Make sure result types are valid
        #  tau : float
        tau = calc_tran_duration(self.signal_ca)
        self.assertIsInstance(tau, np.float32)

        self.assertAlmostEqual(tau, self.signal_t0)


#  class TestMapTau(unittest.TestCase):

#  class TestDFreq(unittest.TestCase):

class TestEnsemble(unittest.TestCase):
    def setUp(self):
        self.signal_F0 = 1000
        self.signal_amp = 100
        self.signal_t0 = 50
        self.signal_t = 800
        self.fps = 500
        self.signal_num = 5
        self.cycle_length = 150
        self.noise = 5  # as a % of the signal amplitude
        self.noise_count = 100
        self.time_vm, self.signal_vm = model_transients(t0=self.signal_t0, t=self.signal_t, fps=self.fps,
                                                        f_0=self.signal_F0, f_amp=self.signal_amp,
                                                        noise=self.noise, num=self.signal_num,
                                                        cl=self.cycle_length)
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
        self.assertAlmostEqual(len(time_out), est_cycle, delta=10)  #

        # signal_out : ndarray
        self.assertIsInstance(signal_out, np.ndarray)  # ensembled signal
        self.assertEqual(len(signal_out), len(signal_out)) #

        # signals : list
        self.assertIsInstance(signals, list)  # ensembled signal
        self.assertEqual(len(signals), self.signal_num) #

        # i_peaks : ndarray
        self.assertIsInstance(i_peaks, np.ndarray)  # indicies of peaks
        self.assertEqual(len(i_peaks), self.signal_num)

        # est_cycle : float
        self.assertIsInstance(est_cycle, float)  # estimated cycle length of ensemble
        self.assertAlmostEqual(est_cycle, self.cycle_length * (self.fps / 1000), delta=10)  #

    def test_plot(self):
        # Make sure ensembled transient looks correct
        time_ensemble, signal_ensemble, signals, signal_peaks, est_cycle_length = calc_ensemble(self.time, self.signal)

        # Build a figure to plot SNR results
        # fig_snr, ax_snr = plot_test()
        fig_ensemble = plt.figure(figsize=(8, 10))  # _ x _ inch page
        gs0 = fig_ensemble.add_gridspec(2, 1, height_ratios=[0.5, 0.5])  # 2 rows, 1 column
        ax_signal = fig_ensemble.add_subplot(gs0[0])
        ax_ensemble = fig_ensemble.add_subplot(gs0[1])

        ax_signal.spines['right'].set_visible(False)
        ax_signal.spines['top'].set_visible(False)
        ax_signal.tick_params(axis='x', which='minor', length=3, bottom=True)
        ax_signal.tick_params(axis='x', which='major', length=8, bottom=True)
        plt.rc('xtick', labelsize=fontsize2)
        plt.rc('ytick', labelsize=fontsize2)

        ax_signal.set_title('Ensemble Averaging')
        ax_signal.set_ylabel('Arbitrary Fluorescent Units')
        ax_signal.set_xlabel('Time (ms)')
        # ax_snr.set_ylim([self.signal_F0 - 20, self.signal_F0 + self.signal_amp + 20])

        ax_signal.plot(self.time, self.signal, color=gray_light, linestyle='None', marker='+', label='Ca pixel data')
        ax_signal.plot(self.time[signal_peaks], self.signal[signal_peaks], "x", color='g', markersize=10, label='Peaks')

        for signal in signals:
            ax_ensemble.plot(time_ensemble, signal, color=gray_light, linestyle='None', marker='+')
        ax_ensemble.plot(time_ensemble, signal_ensemble, color=gray_heavy, linestyle='None', marker='+', label='Ensembled signal')
        # ax_snr.plot(ir_noise, self.signal[ir_noise], "x", color='r', markersize=2, label='Noise')
        # ax_snr.plot_real_noise = ax_snr.axhline(y=self.signal_F0,
        #                                         color=gray_light, linestyle='--', label='Noise (actual)')
        # ax_snr.plot_rms_noise = ax_snr.axhline(y=rms_bounds[0],
        #                                        color=gray_med, linestyle='-.', label='Noise RMS')
        #
        # ax_snr.plot_real_peak = ax_snr.axhline(y=np.nanmax(self.signal),
        #                                        color=gray_light, linestyle='--', label='Peak (actual)')
        # ax_snr.plot_rms_peak = ax_snr.axhline(y=rms_bounds[1],
        #                                       color=gray_med, linestyle='-.', label='Peak RMS')

        # ax_signal.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        ax_signal.text(0.65, 0.5, '# Peaks : {}'.format(len(signal_peaks)),
                       color=gray_med, fontsize=fontsize2, transform=ax_signal.transAxes)
        ax_signal.text(0.65, 0.6, 'PCL (samples): {}'.format(est_cycle_length),
                       color=gray_med, fontsize=fontsize2, transform=ax_signal.transAxes)
        # ax_snr.text(0.65, 0.5, 'SNR (Noise SD, Actual) : {}'.format(self.noise),
        #             color=gray_med, fontsize=fontsize2, transform=ax_snr.transAxes)
        # ax_snr.text(0.65, 0.45, 'SNR (Noise SD, Calculated) : {}'.format(round(sd_noise, 3)),
        #             color=gray_med, fontsize=fontsize2, transform=ax_snr.transAxes)
        # ax_snr.text(0.65, 0.4, 'SNR : {}'.format(round(snr, 5)),
        #             color=gray_heavy, fontsize=fontsize2, transform=ax_snr.transAxes)
        # ax_snr.text(-1, .18, r'Omega: $\Omega$', {'color': 'b', 'fontsize': 20})

        # fig_ensemble.savefig(dir_tests + '/results/processing_SNRDetection.png')
        fig_ensemble.show()


class TestPhase(unittest.TestCase):
    # Setup data to test with
    signal_F0 = 1000
    signal_amp = 100
    signal_t0 = 50
    signal_time = 1000
    signal_num = 5
    noise = 2  # as a % of the signal amplitude
    noise_count = 100
    time_vm, signal_vm = model_transients(t0=signal_t0, t=signal_time,
                                          f_0=signal_F0, f_amp=signal_amp,
                                          noise=noise, num=signal_num)
    time_ca, signal_ca = model_transients(model_type='Ca', t0=signal_t0 + 15, t=signal_time,
                                          f_0=signal_F0, f_amp=signal_amp,
                                          noise=noise, num=signal_num)

    def test_parameters(self):
        # Make sure type errors are raised when necessary
        signal_bad_type = np.full(100, True)
        # signal_in : ndarray, dtyoe : int or float
        self.assertRaises(TypeError, calc_phase, signal_in=True)
        self.assertRaises(TypeError, calc_phase, signal_in=signal_bad_type)
        self.assertRaises(TypeError, calc_phase, signal_in='word')
        self.assertRaises(TypeError, calc_phase, signal_in=3j + 7)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        signal_bad_value = np.full(100, 10)
        signal_bad_value[20] = signal_bad_value[20] - 50
        self.assertRaises(ValueError, calc_phase, signal_in=signal_bad_value)

    def test_results(self):
        # Make sure result types are valid
        signal_vm_phase = calc_phase(self.signal_vm)
        signal_ca_phase = calc_phase(self.signal_ca)
        # signal_FF0 : ndarray, dtyoe : float
        self.assertIsInstance(signal_ca_phase, np.ndarray)  # The array of phase
        self.assertIsInstance(signal_ca_phase[0], float)  # dtyoe : float

        # Make sure result values are valid
        self.assertAlmostEqual(signal_ca_phase.min(), signal_vm_phase.max(), delta=0.01)


if __name__ == '__main__':
    unittest.main()
