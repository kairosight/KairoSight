import numpy as np


def calculate_F_F0(signal, i_F0, invert=False):
    """Normalize a fluorescence signal against a resting fluorescence,
    i.e. F / F0

       Parameters
       ----------
       signal : ndarray
           The array of fluorescent data to be normalized
       i_F0 : tuple
           The range of indexes for resting fluorescence data elements to be used in the calculation, e.g. (0, 10)
       invert : bool
            If True, expecting a resting fluorescence greater than the signal, default is False

       Returns
       -------
       F_F0 : ndarray
           An array of normalized fluorescence data
       data_min : int
           The minimum value of the input signal
       data_max : int
           The maximum value of the input signal
       """
    # Check parameters

    # F / F0: (F_t - F0) / F0
    F_t = signal
    # Get max and min
    data_min, data_max = np.nanmin(F_t), np.nanmax(F_t)
    if invert:
        F_0 = signal.max()
    else:
        F_0 = signal.min()
    F_F0 = (F_t - F_0) / F_0

    return F_F0
