import unittest
from algorithms.datamodel import model_vm, circle_area
from math import pi


class TestModelVm(unittest.TestCase):
    def test_model_params(self):
        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, model_vm, t=-2)     # no negative total times
        self.assertRaises(ValueError, model_vm, t=99)     # at least 100 ms long
        self.assertRaises(ValueError, model_vm, t=150, t0=150)       # start time must be lower than the total time
        self.assertRaises(ValueError, model_vm, t=100, t0=0, fps=150)      # no fps < 200
        self.assertRaises(ValueError, model_vm, t=100, t0=0, fps=1001)      # no fps > 1000
        self.assertRaises(ValueError, model_vm, t=100, t0=0, fps=500, f0=2**16)     # no baseline > 16-bit max
        self.assertRaises(ValueError, model_vm, t=100, t0=0, fps=500,  f0=0, f_peak=2**16)  # no amplitude > 16-bit max

        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, model_vm, t=3+5j)
        self.assertRaises(TypeError, model_vm, t=True)
        self.assertRaises(TypeError, model_vm, t='radius')
        # Some parameters must be int
        self.assertRaises(TypeError, model_vm, t=250, t0=20, fps=50.3, f0=1.5, f_peak=10.4)

    def test_model_result(self):
        # Make sure model results are valid
        self.assertEqual(len(model_vm(t=150)), 2)      # time and data arrays returned as a tuple
        self.assertEqual(model_vm(1000)[0].size, model_vm(1000)[1].size)    # time and data arrays are same length
        # Test the returned time array
        self.assertEqual(model_vm(t=150)[0].size, 150)      # length is correct
        self.assertEqual(model_vm(t=1000, t0=10, fps=223)[0].size, 223)      # length is correct with odd fps
        self.assertGreaterEqual(model_vm(100)[0].all(), 0)     # no negative times
        self.assertLess(model_vm(200)[0].all(), 200)     # no times >= total time parameter
        # Test the returned data array
        self.assertEqual(model_vm(t=150)[1].size, 150)      # length is correct
        self.assertEqual(model_vm(t=1000, t0=10, fps=223)[1].size, 223)     # length is correct with odd fps
        self.assertGreaterEqual(model_vm(t=100)[1].all(), 0)     # no negative values
        self.assertLess(model_vm(t=100)[1].all(), 2**16)     # no values >= 16-bit max

    def test_model_ideal(self):
        # Test ideal model Vm data
        time, data = model_vm(t=500, t0=50, fps=1000, f0=2000, f_peak=250)
        self.assertEqual(time.size, data.size)     # data and time arrays returned as a tuple


# Example tests
class TestModelCircle(unittest.TestCase):
    def test_area(self):
        # Test areas when radius >= 0
        self.assertAlmostEqual(circle_area(1), pi)
        self.assertAlmostEqual(circle_area(0), 0)
        self.assertAlmostEqual(circle_area(2.1), pi * 2.1 * 2.1)

    def test_values(self):
        # Make sure valid errors are raised when necessary
        self.assertRaises(ValueError, circle_area, -2)

    def test_type(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, circle_area, 3+5j)
        self.assertRaises(TypeError, circle_area, True)
        self.assertRaises(TypeError, circle_area, 'radius')
