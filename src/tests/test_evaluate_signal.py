import unittest
from util.evaluate_signal import calculate_snr, calculate_error
from util.datamodel import model_transients, circle_area
import numpy as np


class TestEvaluateSNR(unittest.TestCase):
    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, calculate_snr, signal=True, i_noise=(0, 10), i_peak=(40, 50))

        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, calculate_snr, signal=True, i_noise=(0, 10), i_peak=(40, 50))

    def test_results(self):
        time_ca, signal = model_transients(model_type='Ca', f_0=1000, f_amp=250, noise=5)
        # Make sure files are opened and read correctly
        self.assertIsInstance(calculate_snr(signal, i_noise=(1, 10), i_peak=(50, 60)[0]), float)
        # self.assertIsInstance(calculate_snr(source=self.file_single1)[0], np.ndarray)
        # self.assertIsInstance(calculate_snr(source=self.file_single1)[1], dict)


class TestEvaluateError(unittest.TestCase):
    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, calculate_error, ideal=True, modified=3+5j)

        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, calculate_error, ideal=-2, modified=-2)


if __name__ == '__main__':
    unittest.main()
