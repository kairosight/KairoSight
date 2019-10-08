import unittest
from algorithms.evaluate_signal import calculate_snr, calculate_error

class TestEvaluateSNR(unittest.TestCase):
    def test_params(self):
        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, calculate_snr, t=-2)     # no negative total times

        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, calculate_snr, signal=True, t_noise=(0, 10), )
        self.assertRaises(TypeError, calculate_snr, signal=True)
        self.assertRaises(TypeError, calculate_snr, t='radius')


class TestEvaluateError(unittest.TestCase):
    def test_params(self):
        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, calculate_error, t=-2)     # no negative total times

        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, calculate_error, t=3+5j)
        self.assertRaises(TypeError, calculate_error, t=True)
        self.assertRaises(TypeError, calculate_error, t='radius')


if __name__ == '__main__':
    unittest.main()
