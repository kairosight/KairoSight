import unittest
from util.evaluate_signal import calculate_snr, calculate_error
from util.datamodel import model_transients, circle_area
import numpy as np


class TestEvaluateSNR(unittest.TestCase):
    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, calculate_snr, signal=True, i_noise=(0, 10), i_peak=(40, 50))
        self.assertRaises(TypeError, calculate_snr, signal=True, i_noise=(0, 10), i_peak=(60, 70))

        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, calculate_snr, signal=True, i_noise=(0, 10), i_peak=(40, 50))
        self.assertRaises(ValueError, calculate_snr, signal=True, i_noise=(0, 10), i_peak=(60, 70))

        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, calculate_snr, signal=True, t_noise=(0, 10), t_peak=(50, 60))
        self.assertRaises(TypeError, calculate_snr, signal=True)
        self.assertRaises(TypeError, calculate_snr, t='radius')
        self.assertRaises(TypeError, calculate_snr, t=True)
        self.assertRaises(TypeError, calculate_snr, t=3+5j)

    def test_results(self):
        time_ca, signal = model_transients(model_type='Ca', f_0=1000, f_amp=250, noise=5)
        # Make sure files are opened and read correctly
        self.assertIsInstance(calculate_snr(signal, i_noise=(1, 10), i_peak=(50, 60))[0], float)
        self.assertIsInstance(calculate_snr(signal, i_noise=(1, 10), i_peak=(50, 60))[1], float)
        self.assertIsInstance(calculate_snr(signal, i_noise=(1, 10), i_peak=(50, 60))[2], np.ndarray)
        self.assertIsInstance(calculate_snr(signal, i_noise=(1, 10), i_peak=(50, 60))[3], np.ndarray)
        self.assertIsInstance(calculate_snr(signal, i_noise=(1, 10), i_peak=(50, 60))[4], np.ndarray)

        # self.assertIsInstance(calculate_snr(source=self.file_single1)[0], np.ndarray)
        # self.assertIsInstance(calculate_snr(source=self.file_single1)[1], dict)


class TestEvaluateError(unittest.TestCase):
    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, calculate_error, ideal=True, modified=3+5j)
        self.assertRaises(TypeError, calculate_error, ideal='radius', modified=True)
        self.assertRaises(TypeError, calculate_error, ideal=3+5j, modified='radius')

        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, calculate_error, ideal=-2, modified=-2)
        self.assertRaises(ValueError, calculate_error, ideal=1.3, modified=1.3)

    def test_results(self):
        time_ca, ideal, modified = model_transients(model_type='Ca', f_0=1000, f_amp=250, noise=5)
        # Make sure files are opened and read correctly
        self.assertIsInstance(calculate_error(ideal, modified, [0]), np.ndarray)  # array of error
        self.assertIsInstance(calculate_error(ideal, modified, [1]), float)  # mean value of error array
        self.assertIsInstance(calculate_error(ideal, modified, [2]), float)  # sd of error array

if __name__ == '__main__':
    unittest.main()
