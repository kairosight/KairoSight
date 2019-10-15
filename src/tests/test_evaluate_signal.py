import unittest
from util.evaluate_signal import calculate_snr, calculate_error


class TestEvaluateSNR(unittest.TestCase):
    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, calculate_snr, signal=True, i_noise=(0, 10), i_peak=(40, 50))

        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, calculate_snr, signal=True, i_noise=(0, 10), i_peak=(40, 50))


class TestEvaluateError(unittest.TestCase):
    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, calculate_error, ideal=True, modified=3+5j)

        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, calculate_error, ideal=-2, modified=-2)


if __name__ == '__main__':
    unittest.main()
