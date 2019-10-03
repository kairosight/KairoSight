import unittest
from algorithms.datamodel import model_vm, circle_area
from math import pi


class TestDataModel_Vm(unittest.TestCase):
    def test_model_params(self):
        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(ValueError, model_vm, -2)     # no negative total times
        self.assertRaises(ValueError, model_vm, 99)     # at least 100 ms long
        self.assertRaises(ValueError, model_vm, 150, 150)       # start time must be lower than the total time
        self.assertRaises(ValueError, model_vm, 100, 0, -500)      # no fps < 0
        self.assertRaises(ValueError, model_vm, 100, 0, 1001)      # no fps > 1000
        self.assertRaises(ValueError, model_vm, 100, 0, 500, 2**16)     # no baseline higher than 16-bit max
        self.assertRaises(ValueError, model_vm, 100, 0, 500,  0, 2**16)     # no amplitude higher than 16-bit max

    def test_model_type(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, model_vm, 3+5j)
        self.assertRaises(TypeError, model_vm, True)
        self.assertRaises(TypeError, model_vm, 'radius')
        self.assertRaises(TypeError, model_vm, 250, 20, 'F0=500', 90)

    def test_model_result(self):
        # Make sure model results are valid
        self.assertEqual(len(model_vm(150)), 2)      # data and time arrays returned as a tuple
        self.assertEqual(model_vm(1000)[0].size, model_vm(1000)[1].size)    # data and time arrays are same length
        # Test the returned data array
        self.assertGreaterEqual(model_vm(150)[0].size, 100)  # at least 100 ms long
        self.assertEqual(model_vm(150)[0].size, 150)      # length is correct
        self.assertEqual(model_vm(1000, 10, 500)[0].size, 500)      # length is correct with odd fps
        self.assertGreaterEqual(model_vm(100)[0].all(), 0)     # no negative values
        self.assertLess(model_vm(100)[0].all(), 2**16)     # no values higher than 16-bit max
        # Test the returned time array
        self.assertEqual(model_vm(150)[1].size, 150)      # length is correct
        self.assertEqual(model_vm(1000, 10, 500)[1].size, 500)      # length is correct with odd fps
        self.assertGreaterEqual(model_vm(100)[1].all(), 0)     # no negative values
        self.assertLess(model_vm(200)[1].all(), 200)     # no values higher than total time parameter

    def test_model_ideal(self):
        # Test ideal model Vm data
        data, time = model_vm(150, 50, 1000, 100, 200)
        self.assertEqual(data.size, time.size)     # data and time arrays returned as a tuple


class TestDataModel_Circle(unittest.TestCase):
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
