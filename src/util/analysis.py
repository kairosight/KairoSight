import numpy as np
from scipy.signal import find_peaks


def find_tran_peak(signal_in):
    """Find the time of the peak of a transient,
    defined as the maximum value

       Parameters
       ----------
       signal_in : ndarray
            The array of data to be evaluated

       Returns
       -------
       i_peak : int
            The index of the signal array corresponding to it's peak
       """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [int, float]:
        raise TypeError('Signal values must either be "int" or "float"')

    if any(v < 0 for v in signal_in):
        raise ValueError('All signal values must be >= 0')

    # Characterize the signal
    signal_bounds = (signal_in.min(), signal_in.max())
    signal_range = signal_bounds[1] - signal_bounds[0]

    i_peaks, _ = find_peaks(signal_in, prominence=(signal_range/2))

    if len(i_peaks) > 1:
        raise ArithmeticError('{} peaks detected for a single given transient'.format(len(i_peaks)))

    i_peak = i_peaks[0].astype(int)

    return i_peak


def find_tran_start(time, signal_in):
    """Find the time of the start of a transient,
    defined as the 1st maximum of the 2nd derivative

       Parameters
       ----------
       time : ndarray
            The array of timestamps corresponding to the model_data
       signal_in : ndarray
            The array of data to be evaluated

       Returns
       -------
       i_time : ndarray
            The index within the time array corresponding to start time
       """


def find_tran_upstroke(time, signal_in):
    """Find the time of the upstroke of a transient,
    defined as the 1st maximum of the 1st derivative

       Parameters
       ----------
       time : ndarray
            A 3-D array (T, Y, X) of optical data
       signal_in : ndarray
            The array of data to be evaluated

       Returns
       -------
       i_time : int
            The index within the time array corresponding to start time
       """
    pass


def calc_tran_activation(time, signal_in):
    """Calculate the time of the upstroke activation of a transient,
    defined as the midpoint (not limited by sampling rate) between the start and peak times

       Parameters
       ----------
       time : ndarray
            A 3-D array (T, Y, X) of optical data
       signal_in : ndarray
            The array of data to be evaluated

       Returns
       -------
       time : float
            The value of the activation time
       """
    pass


def calc_ff0(signal_in, invert=False):
    """Normalize a fluorescence signal against a resting fluorescence,
    i.e. F_t / F0

       Parameters
       ----------
       signal_in : ndarray
            The array of fluorescent data (F_t) to be normalized
       invert : bool
            If True, expecting a resting fluorescence value greater than the signal, default is False

       Returns
       -------
       signal_ff0 : ndarray
            The array of F/F0 fluorescence data, dtype : float

        Notes
        -----
        Should not be applied to normalized or drift-removed data.
       """
    # Check parameters

    # F / F0: (F_t - F0) / F0
    f_t = signal_in
    if invert:
        f_0 = signal_in.max()
    else:
        f_0 = signal_in.min()
    signal_ff0 = (f_t - f_0) / f_0

    return signal_ff0
