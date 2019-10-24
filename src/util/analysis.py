import numpy as np


def isolate_transient(signal, i_start, i_end):
    """Isolate a single transient from a signal array of optical data.

       Parameters
       ----------
       signal : ndarray
            The array of data to be evaluated
       i_start : int
            Index or frame to start transient isolation
       i_end : int
            Index or frame to end transient isolation

       Returns
       -------
       transient_out : ndarray
            The isolated array of transient data
       """
    pass


def find_tran_peak(time, signal):
    """Find the time of the peak of a transient,
    defined as the maximum value

       Parameters
       ----------
       time : ndarray
            The array of timestamps corresponding to the model_data
       signal : ndarray
            The array of data to be evaluated

       Returns
       -------
       i_peak : ndarray
            The index within the time array corresponding to peak time
       """


def find_tran_start(time, signal):
    """Find the time of the start of a transient,
    defined as the 1st maximum of the 2nd derivative

       Parameters
       ----------
       time : ndarray
            The array of timestamps corresponding to the model_data
       signal : ndarray
            The array of data to be evaluated

       Returns
       -------
       i_time : ndarray
            The index within the time array corresponding to start time
       """


def find_tran_upstroke(time, signal):
    """Find the time of the upstroke of a transient,
    defined as the 1st maximum of the 1st derivative

       Parameters
       ----------
       time : ndarray
            A 3-D array (T, Y, X) of optical data
       signal : ndarray
            The array of data to be evaluated

       Returns
       -------
       i_time : int
            The index within the time array corresponding to start time
       """
    pass


def calc_tran_activation(time, signal):
    """Calculate the time of the upstroke activation of a transient,
    defined as the midpoint (not limited by sampling rate) between the start and peak times

       Parameters
       ----------
       time : ndarray
            A 3-D array (T, Y, X) of optical data
       signal : ndarray
            The array of data to be evaluated

       Returns
       -------
       time : float
            The value of the activation time
       """
    pass


def calc_FF0(signal, i_F0, invert=False):
    """Normalize a fluorescence signal against a resting fluorescence,
    i.e. F_t / F0

       Parameters
       ----------
       signal : ndarray
            The array of fluorescent data (F_t) to be normalized
       i_F0 : tuple
            The range of indexes for resting fluorescence data elements to be used in the calculation, e.g. (0, 10)
       invert : bool
            If True, expecting a resting fluorescence value greater than the signal, default is False

       Returns
       -------
       F_F0 : ndarray
            The array of normalized fluorescence data
       data_min : int
            The minimum value of the input signal
       data_max : int
            The maximum value of the input signal

        Notes
        -----
        Should not be performed on normalized or drift-removed data.
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

    return F_F0, data_min, data_max
