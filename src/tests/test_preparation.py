import unittest
from util.preparation import load_single
import sys
import numpy as np
from pathlib import Path


class TestPrepLoad(unittest.TestCase):
    def test_load_single(self):
        cwd = Path.cwd()
        tests = str(cwd)

        file_single1 = tests + '/data/about1.tif'
        file_single2 = tests + '/data/02-250.tif'
        file_meta = tests + '/data/02-250.pcoraw.rec'
        file_single1_wrong = tests + '/data/02-250_Vm'
        file_meta_wrong = tests + '/data/02-250_Vm.pcoraw'

        print("%x" % sys.maxsize, sys.maxsize > 2 ** 32)
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, load_single, source=250)
        self.assertRaises(TypeError, load_single, source='data/02-250_Vm', meta=True)
        # Make sure parameters are valid, and valid errors are raised when necessary
        self.assertRaises(FileNotFoundError, load_single, source=tests)
        self.assertRaises(FileNotFoundError, load_single, source=file_single1_wrong)
        self.assertRaises(FileNotFoundError, load_single, source=file_single1, meta=file_meta_wrong)

        # Make sure files are opened and read correctly
        self.assertIsInstance(load_single(source=file_single1)[0], np.ndarray)
        self.assertIsInstance(load_single(source=file_single1)[1], dict)
        self.assertIsInstance(load_single(source=file_single2)[1], dict)
        self.assertIsInstance(load_single(source=file_single2, meta=file_meta)[1], str)


if __name__ == '__main__':
    unittest.main()
