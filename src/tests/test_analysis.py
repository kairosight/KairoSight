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
colors_times = {'Upstroke': '#FFD649',
                'Activation': '#FFA253',
                'Peak': '#F6756B',
                'Downstroke': '#CB587F',
                'Restoration': '#8E4B84',
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
    def setUp(self):
        # Setup data to test with
        self.signal_t = 200
        self.zoom_t = 60
        self.signal_t0 = 20
        self.fps = 1000
        self.time_vm, self.signal_vm = model_transients(t=self.signal_t, t0=self.signal_t0, fps=self.fps)
        time_ca, signal_ca = model_transients(model_type='Ca', t=self.signal_t, t0=self.signal_t0, fps=self.fps)
        self.time, self.signal = self.time_vm[0:self.zoom_t], invert_signal(self.signal_vm[0:self.zoom_t])
        self.sample_rate = float(self.fps)
        # signal_ca = filter_temporal(signal_ca, sample_rate)
        # dt = int(time_ca[1])
        # TODO try derivative using spline and nu=1, nu=2
        # spl = UnivariateSpline(time, signal)
        # df_ca = spl(time, nu=1)
        self.df_ca = np.diff(self.signal, n=1, prepend=int(self.signal[0])).astype(float)
        self.d2f_ca = np.diff(self.signal, n=2, prepend=[int(self.signal[0]), int(self.signal[0])]).astype(float)
        # df_ca = np.gradient(signal_noisy_ca, dt) / np.gradient(time_ca, dt)
        # d2f_ca = np.gradient(df_ca, dt) / np.gradient(time_ca, dt)

    def test_plot(self):
        # Build a figure to plot the signal, it's derivatives, and the analysis points
        # fig_points = plt.figure(figsize=(8, 8))  # _ x _ inch page
        # General layout
        fig_points, ax_points = plot_test()
        ax_points.set_title('Analysis Points\n1st and 2nd derivatives')
        ax_points.set_ylabel('Arbitrary Fluorescent Units')
        ax_points.set_xlabel('Time (ms)')

        ax_points.plot(self.time, self.signal, color=gray_med, linestyle='-', marker='x',
                       label='Ca (Model)')

        ax_dfs = ax_points.twinx()  # instantiate a second axes that shares the same x-axis
        # ax_df2 = ax_points.twinx()  # instantiate a second axes that shares the same x-axis
        ax_dfs.baseline = ax_dfs.axhline(color=gray_light, linestyle='-.')
        ax_dfs.set_ylabel('df/dt and d2F/dt2')  # we already handled the x-label with ax1
        df_ca_smooth = filter_temporal(self.df_ca, self.sample_rate, filter_order=5)
        d2f_ca_smooth = filter_temporal(self.d2f_ca, self.sample_rate, filter_order=5)
        ax_dfs.set_ylim([-20, 20])
        # ax_df2.set_ylim([-20, 20])

        # df/dt
        ax_dfs.plot(self.time, df_ca_smooth,
                   color=gray_light, linestyle='--', label='df/dt')
        # d2f/dt2
        ax_dfs.plot(self.time, d2f_ca_smooth,
                    color=gray_light, linestyle=':', label='d2F/dt2')

        # Upstroke
        df2_max1 = np.argmax(d2f_ca_smooth)
        ax_points.axvline(self.time[df2_max1], color=colors_times['Upstroke'],
                          label='Upstroke')  # 1st df2 max, Upstroke
        # Activation
        df_max1 = np.argmax(df_ca_smooth)  # 1st df max, Activation
        ax_points.axvline(self.time[df_max1], color=colors_times['Activation'],
                          label='Activation')
        # Peak
        ax_points.axvline(self.time[np.argmax(self.signal)], color=colors_times['Peak'],
                          label='Peak')  # max of signal, Peak
        # Downstroke
        df2_max2 = np.argmax(d2f_ca_smooth[df2_max1 + 3:])  # 2st df2 max, Downstroke
        ax_points.axvline(self.time[df2_max2 + df2_max1 + 3], color=colors_times['Downstroke'],
                          label='Downstroke')

        ax_dfs.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        ax_points.legend(loc='upper left', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)

        fig_points.savefig(dir_tests + '/results/analysis_AnalysisPoints_ca.png')
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


class TestAnalysisFF0(unittest.TestCase):
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
        # signal_in : ndarray
        self.assertRaises(TypeError, find_tran_peak, signal_in=True)
        self.assertRaises(TypeError, find_tran_peak, signal_in=signal_bad_type)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        signal_bad_value = np.full(100, 10)
        signal_bad_value[20] = signal_bad_value[20] - 50
        self.assertRaises(ValueError, calc_ff0, signal_in=signal_bad_value)

    def test_results(self):
        # Make sure result types are valid
        signal_vm_ff0 = calc_ff0(self.signal_vm, invert=True)
        signal_ca_ff0 = calc_ff0(self.signal_ca)
        # signal_FF0 : ndarray, dtyoe : float
        self.assertIsInstance(signal_ca_ff0, np.ndarray)  # The array of F/F0 fluorescence data
        self.assertIsInstance(signal_ca_ff0[0], float)  # dtyoe : float

        # Make sure result values are valid
        self.assertAlmostEqual(signal_ca_ff0.min(), signal_vm_ff0.max(), delta=0.01)  # Vm is a downward deflection

    def test_plot_dual(self):
        # Make sure F/F0 looks correct
        signal_vm_ff0 = calc_ff0(self.signal_vm, invert=True)
        signal_ca_ff0 = calc_ff0(self.signal_ca)

        # Build a figure to plot F/F0 results
        fig_ff0, ax_ff0 = plot_test()
        ax_ff0.set_ylabel('Arbitrary Fluorescent Units')
        ax_ff0.set_xlabel('Time (ms)')

        ax_ff0.plot(self.time_vm, signal_vm_ff0, color=color_vm, linestyle='None', marker='+', label='Vm, F/F0')
        ax_ff0.plot(self.time_ca, signal_ca_ff0, color=color_ca, linestyle='None', marker='+', label='Ca, F/F0')

        ax_ff0.legend(loc='upper right', ncol=1, prop={'size': fontsize2}, numpoints=1, frameon=True)
        fig_ff0.show()


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
