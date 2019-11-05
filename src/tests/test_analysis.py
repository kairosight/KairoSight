import unittest
from util.analysis import find_tran_peak, calc_ff0, find_tran_start, find_tran_upstroke, calc_tran_activation, \
    find_tran_downstroke, find_tran_end, find_tran_restoration, calc_tran_duration, calc_tran_tau, calc_tran_di, \
    calc_phase
from util.datamodel import model_transients
import numpy as np
import matplotlib.pyplot as plt
fontsize1, fontsize2, fontsize3, fontsize4 = [14, 10, 8, 6]
gray_light, gray_med, gray_heavy = ['#D0D0D0', '#808080', '#606060']
color_vm, color_ca = ['#FF9999', '#99FF99']


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


class TestUpstroke(unittest.TestCase):
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
        self.assertRaises(TypeError, find_tran_upstroke, signal_in=True)
        self.assertRaises(TypeError, find_tran_upstroke, signal_in=signal_bad_type)
        self.assertRaises(TypeError, find_tran_upstroke, signal_in='word')
        self.assertRaises(TypeError, find_tran_upstroke, signal_in=3j + 7)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        signal_bad_value = np.full(100, 10)
        signal_bad_value[20] = signal_bad_value[20] - 50
        self.assertRaises(ValueError, find_tran_upstroke, signal_in=signal_bad_value)

    def test_results(self):
        # Make sure result types are valid
        # i_upstroke : int
        i_upstroke = find_tran_upstroke(self.signal_ca)
        self.assertIsInstance(i_upstroke, np.int32)

        self.assertAlmostEqual(i_upstroke, self.signal_t0 + 10, delta=5)  # time to peak of an OAP/OCT


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
        # signal_in : ndarray, dtyoe : int or float
        self.assertRaises(TypeError, find_tran_peak, signal_in=True)
        self.assertRaises(TypeError, find_tran_peak, signal_in=5)
        self.assertRaises(TypeError, find_tran_peak, signal_in=signal_bad_type)
        self.assertRaises(TypeError, find_tran_peak, signal_in='word')
        self.assertRaises(TypeError, find_tran_peak, signal_in=3j+7)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        signal_bad_value = np.full(100, 10)
        signal_bad_value[20] = signal_bad_value[20] - 50

        self.assertRaises(ValueError, find_tran_peak, signal_in=signal_bad_value)

    def test_results(self):
        # Make sure result types are valid
        i_peak = find_tran_peak(self.signal_ca)
        # i_peak : int
        self.assertIsInstance(i_peak, np.int32)  # index of peak

        #  Make sure result values are valid
        self.assertAlmostEqual(i_peak, self.signal_t0 + 10, delta=5)    # time to peak of an OAP/OCT


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


class TestRestoration(unittest.TestCase):
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
        self.assertRaises(TypeError, find_tran_restoration, signal_in=True)
        self.assertRaises(TypeError, find_tran_restoration, signal_in=signal_bad_type)
        self.assertRaises(TypeError, find_tran_restoration, signal_in='word')
        self.assertRaises(TypeError, find_tran_restoration, signal_in=3j + 7)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        signal_bad_value = np.full(100, 10)
        signal_bad_value[20] = signal_bad_value[20] - 50
        self.assertRaises(ValueError, find_tran_restoration, signal_in=signal_bad_value)

    def test_results(self):
        # Make sure result types are valid
        #  i_restoration : int
        i_restoration = find_tran_restoration(self.signal_ca)
        self.assertIsInstance(i_restoration, np.int32)

        self.assertAlmostEqual(i_restoration, self.signal_t0 + 10, delta=5)  # time to peak of an OAP/OCT


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
        self.assertRaises(TypeError, calc_tran_duration, signal_in=True, percent = True)
        self.assertRaises(TypeError, calc_tran_duration, signal_in=signal_bad_type, percent = '500')
        self.assertRaises(TypeError, calc_tran_duration, signal_in='word', percent = 3j+7)
        self.assertRaises(TypeError, calc_tran_duration, signal_in=3j + 7)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        # percent : >=0
        signal_bad_value = np.full(100, 10)
        signal_bad_value[20] = signal_bad_value[20] - 50
        percent_bad_value = -1
        self.assertRaises(ValueError, find_tran_restoration, signal_in=signal_bad_value, percent = percent_bad_value)

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
        # signal_in : ndarray, dtyoe : int
        self.assertRaises(TypeError, calc_ff0, signal_in=5)
        self.assertRaises(TypeError, calc_ff0, signal_in=signal_bad_type)
        self.assertRaises(TypeError, calc_ff0, signal_in=True)
        self.assertRaises(TypeError, calc_ff0, signal_in='word')
        self.assertRaises(TypeError, calc_ff0, signal_in=3j + 7)

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
        self.assertAlmostEqual(signal_ca_ff0.min(), signal_vm_ff0.max(), delta=0.01)    # Vm is a downward deflection
        
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
        self.assertRaises(TypeError,  calc_phase, signal_in=signal_bad_type)
        self.assertRaises(TypeError, calc_phase, signal_in='word')
        self.assertRaises(TypeError,  calc_phase, signal_in=3j + 7)

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
