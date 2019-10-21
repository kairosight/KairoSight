import unittest
from util.processing import calculate_snr, calculate_error
from util.datamodel import model_transients, circle_area
import numpy as np


class TestEvaluateSNR(unittest.TestCase):
    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, calculate_snr, signal=True, i_noise=(0, 10), i_peak=(40, 50))
        self.assertRaises(TypeError, calculate_snr, signal=3+5j, i_noise=(0, 10), i_peak=(40, 50))
        self.assertRaises(TypeError, calculate_snr, signal='radius', i_noise=(0, 10), i_peak=(40, 50))
        self.assertRaises(TypeError, calculate_snr, i_noise='radius')
        self.assertRaises(TypeError, calculate_snr, i_noise=True)
        self.assertRaises(TypeError, calculate_snr, i_noise=3 + 5j)
        self.assertRaises(TypeError, calculate_snr, i_peak='radius')
        self.assertRaises(TypeError, calculate_snr, i_peak=True)
        self.assertRaises(TypeError, calculate_snr, i_peak=3+5j)

        time_ca, signal = model_transients(model_type='Ca', f_0=1000, f_amp=250, noise=5)
        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, calculate_snr, signal=True, i_noise=(0, 10), i_peak=(40, 50))
        self.assertRaises(ValueError, calculate_snr, signal, i_noise=(0, 15), i_peak=(40, 45))  # range used should be 10

    def test_results(self):
        time_ca, signal = model_transients(model_type='Ca', f_0=1000, f_amp=250, noise=5)
        # Make sure files are opened and read correctly
        self.assertIsInstance(calculate_snr(signal, i_noise=(1, 10), i_peak=(50, 60))[0], float)  # snr
        self.assertIsInstance(calculate_snr(signal, i_noise=(1, 10), i_peak=(50, 60))[1], float)  # sd of noise
        self.assertIsInstance(calculate_snr(signal, i_noise=(1, 10), i_peak=(50, 60))[2], np.ndarray)  # peak values
        self.assertIsInstance(calculate_snr(signal, i_noise=(1, 10), i_peak=(50, 60))[3], np.ndarray)  # peak values

        # self.assertIsInstance(calculate_snr(source=self.file_single1)[0], float)
        # self.assertIsInstance(calculate_snr(source=self.file_single1)[1], float)
        # self.assertIsInstance(calculate_snr(source=self.file_single1)[2], np.ndarray)
        # self.assertIsInstance(calculate_snr(source=self.file_single1)[3], np.ndarray)


class TestEvaluateError(unittest.TestCase):
    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, calculate_error, ideal=True, modified=3+5j)
        self.assertRaises(TypeError, calculate_error, ideal='radius', modified=True)
        self.assertRaises(TypeError, calculate_error, ideal=3+5j, modified='radius')

        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, calculate_error, ideal=-2, modified=-2)  # values should be positive
        self.assertRaises(ValueError, calculate_error, ideal=1.3, modified=1.3)  # ideal and modified should be ints

    def test_results(self):
        time_ca, ideal = model_transients(model_type='Ca', f_0=1000, f_amp=250, noise=5)
        time_ca, modified = model_transients(model_type='Ca', f_0=1000, f_amp=250, noise=5)
        # Make sure files are opened and read correctly
        self.assertIsInstance(calculate_error(ideal, modified)[0], np.ndarray)  # array of error
        self.assertIsInstance(calculate_error(ideal, modified)[1], float)  # mean value of error array
        self.assertIsInstance(calculate_error(ideal, modified)[2], float)  # sd of error array

        # Test returned error array
        self.assertGreaterEqual(calculate_error(ideal, modified)[0].all, 0)  # no negative values
        self.assertEqual(calculate_error(ideal, modified)[1].size, calculate_error(ideal, modified)[2].size) # ideal and modified the same size

        # self.assertEqual(calculate_error(ideal, modified)[1].mean())  # mean of percent array found properly
        # self.assertIsInstance(calculate_error(source=self.file_single1)[0], np.ndarray)
        # self.assertIsInstance(calculate_error(source=self.file_single1)[1], float)
        # self.assertIsInstance(calculate_error(source=self.file_single1)[2], float)


if __name__ == '__main__':
    unittest.main()
