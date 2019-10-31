import unittest
from util.analysis import find_tran_peak, calc_FF0
from util.datamodel import model_transients
import numpy as np


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
        signal_bad_type = np.full(100, True)
        # Make sure type errors are raised when necessary
        # signal_in : ndarray, dtyoe : int or float
        self.assertRaises(TypeError, find_tran_peak, signal_in=True)
        self.assertRaises(TypeError, find_tran_peak, signal_in=signal_bad_type)

        # Make sure parameters are valid, and valid errors are raised when necessary
        # signal_in : >=0
        signal_bad_value = np.full(100, 10)
        signal_bad_value[20] = signal_bad_value[20] - 50
        self.assertRaises(ValueError, find_tran_peak, signal_in=signal_bad_value)

    def test_results(self):
        # Make sure results are correct
        i_peak = find_tran_peak(self.signal_ca)

        # i_peak : ndarray, dtyoe : int
        self.assertIsInstance(i_peak, np.ndarray)  # normalized signal


class TestAnalysisFF0(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, calc_FF0, signal=True, i_F0=(0, 10), invert=10)
        self.assertRaises(TypeError, calc_FF0, signal=3 + 5j, i_F0=True, invert='radius')
        self.assertRaises(TypeError, calc_FF0, signal='radius', i_F0='radius', invert=3 + 5j)

        time_ca, signal = model_transients(model_type='Ca', f_0=1000, f_amp=250, noise=5)
        # Make sure parameters are valid and valid errors are raised when necessary
        self.assertRaises(ValueError, calc_FF0, signal, i_F0=(1, 5), invert=False)

    def test_results(self):
        time_ca, signal = model_transients(model_type='Ca', f_0=1000, f_amp=250, noise=5)
        self.assertIsInstance(calc_FF0(signal, i_F0=(1, 10), invert=True)[0], np.ndarray)  # array of normalized data
        self.assertIsInstance(calc_FF0(signal, i_F0=(1, 10), invert=True)[1], int)  # min value of input signal
        self.assertIsInstance(calc_FF0(signal, i_F0=(1, 10), invert=True)[2], int)     # max value of input signal

        # self.assertGreaterEqual(calc_FF0(signal, i_F0=(0, 10), invert=True)[1].max()) # max found properly
        # self.assertLesserEqual(calc_FF0(signal, i_F0=(0, 10), invert=True)[2].min())  # min found properly


if __name__ == '__main__':
    unittest.main()
