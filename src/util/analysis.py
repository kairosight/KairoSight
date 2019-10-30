import numpy as np


def find_tran_peak(time, signal_in):
    """Find the time of the peak of a transient,
    defined as the maximum value

       Parameters
       ----------
       time : ndarray
            The array of timestamps corresponding to the model_data
       signal_in : ndarray
            The array of data to be evaluated

       Returns
       -------
       i_peak : ndarray
            The index within the time array corresponding to peak time
       """


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


def calc_FF0(signal_in, i_F0, invert=False):
    """Normalize a fluorescence signal against a resting fluorescence,
    i.e. F_t / F0

       Parameters
       ----------
       signal_in : ndarray
            The array of fluorescent data (F_t) to be normalized
       i_F0 : tuple
            The range of indexes for resting fluorescence data elements to be used in the calculation, e.g. (0, 10)
       invert : bool
            If True, expecting a resting fluorescence value greater than the signal, default is False

       Returns
       -------
       signal_FF0 : ndarray
            The array of normalized fluorescence data
       data_min : int
            The minimum value of the input signal
       data_max : int
            The maximum value of the input signal

        Notes
        -----
        Should not be applied to normalized or drift-removed data.
       """
    # Check parameters

    # F / F0: (F_t - F0) / F0
    F_t = signal_in
    # Get max and min
    data_min, data_max = np.nanmin(F_t), np.nanmax(F_t)
    if invert:
        F_0 = signal_in.max()
    else:
        F_0 = signal_in.min()
    signal_FF0 = (F_t - F_0) / F_0

    return signal_FF0, data_min, data_max
