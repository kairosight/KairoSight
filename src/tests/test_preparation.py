import unittest
from util.preparation import open_single
import sys
import numpy as np
from pathlib import Path


class TestPrepOpen(unittest.TestCase):
    # File paths  and files needed for tests
    cwd = Path.cwd()
    tests = str(cwd)
    file_single1 = tests + '/data/about1.tif'
    file_single1_wrong = tests + '/data/about1'
    file_single2 = tests + '/data/02-250_Vm.tif'
    file_single2_wrong = tests + '/data/02-250_Vm'
    file_meta = tests + '/data/02-250_Vm.pcoraw.rec'
    file_meta_wrong = tests + '/data/02-250.pcoraw.txt'
    print("sys.maxsize : " + str(sys.maxsize) +
          ' \nIs it greater than 32-bit limit? : ' + str(sys.maxsize > 2 ** 32))

    def test_params(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, open_single, source=250)
        self.assertRaises(TypeError, open_single, source=self.file_single2, meta=True)
        # Make valid errors are raised when parameters are invalid
        self.assertRaises(FileNotFoundError, open_single, source=self.tests)
        self.assertRaises(FileNotFoundError, open_single, source=self.file_single1_wrong)
        self.assertRaises(FileNotFoundError, open_single, source=self.file_single1, meta=self.file_meta_wrong)

    def test_results(self):
        # Make sure files are opened and read correctly
        self.assertIsInstance(open_single(source=self.file_single1)[0], np.ndarray)
        self.assertIsInstance(open_single(source=self.file_single1)[1], dict)
        self.assertIsInstance(open_single(source=self.file_single2)[1], dict)
        self.assertIsInstance(open_single(source=self.file_single2, meta=self.file_meta)[1], str)


if __name__ == '__main__':
    unittest.main()
