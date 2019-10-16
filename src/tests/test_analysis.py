import unittest

from util.analysis import calculate_FF0
from util.datamodel import model_transients
import numpy as np

class TestAnalysisFF0(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, calculate_FF0, signal=True, i_F0=(0,10), invert=10)
        self.assertRaises(TypeError, calculate_FF0,signal=3+5j, i_F0=True, invert='radius')
        self.assertRaises(TypeError, calculate_FF0, signal='radius', i_F0='radius', invert=3+5j)

        time_ca, signal = model_transients(model_type='Ca', f_0=1000, f_amp=250, noise=5)
        # Make sure parameters are valid and valid errors are raised when necessary
        self.assertRaises(ValueError, calculate_FF0, signal, i_F0=(1,5), invert=False)

    def test_results(self):
        time_ca, signal = model_transients(model_type='Ca', f_0=1000, f_amp=250, noise=5)
        self.assertIsInstance(calculate_FF0(signal, i_F0=(1, 10), invert=True)[0], np.ndarray)  # array of normalized data
        self.assertIsInstance(calculate_FF0(signal, i_F0=(1, 10), invert=True)[1], dict)  # min value of input signal
        self.assertIsInstance(calculate_FF0(signal, i_F0=(1, 10), invert=True)[2], dict) # max value of input signal

       # self.assertGreaterEqual(calculate_FF0(signal, i_F0=(0, 10), invert=True)[1].max()) # mean of input signal found properly
       # self.assertLesserEqual(calculate_FF0(signal, i_F0=(0, 10), invert=True)[2].min())  # mean of input signal found properly


if __name__ == '__main__':
    unittest.main()
