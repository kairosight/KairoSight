import numpy as np
from scipy.signal import find_peaks


def find_tran_start(signal_in):
    """Find the time of the start of a transient,
    defined as the 1st maximum of the 2nd derivative

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated

        Returns
        -------
        i_start : int
            The index of the signal array corresponding to the start of the transient
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [int, float]:
        raise TypeError('Signal values must either be "int" or "float"')

    if any(v < 0 for v in signal_in):
        raise ValueError('All signal values must be >= 0')

def find_tran_upstroke(signal_in):
    """Find the time of the upstroke of a transient,
    defined as the maximum of the 1st derivative

        Parameters
        ----------
        signal_in : ndarray
             The array of data to be evaluated

        Returns
        -------
        i_upstroke : int
            The index of the signal array corresponding to the upstroke of the transient
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [int, float]:
        raise TypeError('Signal values must either be "int" or "float"')

    if any(v < 0 for v in signal_in):
        raise ValueError('All signal values must be >= 0')

    pass


def calc_tran_activation(signal_in):
    """Calculate the time of the activation of a transient,
    defined as the midpoint (not limited by sampling rate) between the start and peak times

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated

        Returns
        -------
        i_activation : int
            The index of the signal array corresponding to the activation of the transient
        """

    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [int, float]:
        raise TypeError('Signal values must either be "int" or "float"')

    if any(v < 0 for v in signal_in):
        raise ValueError('All signal values must be >= 0')

    pass


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
            The index of the signal array corresponding to the peak of the transient
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


def find_tran_downstroke(signal_in):
    """Find the time of the downstroke of a transient,
    defined as the minimum of the 1st derivative

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated

        Returns
        -------
        i_downstroke : int
            The index of the signal array corresponding to the downstroke of the transient
        """
    pass


def find_tran_end(signal_in):
    """Find the time of the end of a transient,
    defined as the 2nd maximum of the 2nd derivative

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated

        Returns
        -------
        i_end : int
            The index of signal array corresponding to the end of the transient
        """


def find_tran_restoration(signal_in):
    """Find the time of the restoration of a transient,
    defined as the index of the return to a start value

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated

        Returns
        -------
        i_restoration : int
            The index of signal array corresponding to the restoration of the transient
        """


def calc_tran_duration(signal_in, percent=50):
    """Calculate the duration of a transient,
    defined as the number of indices between the start time and a time
    nearest a percentage of the start-to-peak value (e.g. APD-80, CAD-90)

        Parameters
        ----------
        signal_in : ndarray
              The array of data to be evaluated
        percent : int
              Percentage of the value from start to peak to use as the

        Returns
        -------
        duration : int
            The % duration of the transient in number of indices
        """


def calc_tran_tau(signal_in):
    """Calculate the decay time constant (tau) of a transient,
    defined as the time between 30 and 90% decay from peak

        Parameters
        ----------
         signal_in : ndarray
              The array of data to be evaluated

        Returns
        -------
        tau : float
            The decay time constant (tau) of the signal array corresponding to it's peak
        """


def calc_tran_di(signal_in):
    """Calculate the diastolic interval (DI) of a transient,
    defined as the number of indices between restoration and the next transient's activation

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated

        Returns
        -------
        di : float
            The  diastolic interval (DI) of the first transient in a signal array

        Notes
        -----
            Should not be applied to signal data containing at least one transient.
        """


def map_tran_tau(stack_in):
    """Map the decay constant (tau) values for a stack of transient fluorescent data
    i.e.

        Parameters
        ----------
        stack_in : ndarray
            A 3-D array (T, Y, X) of an optical transient

        Returns
        -------
        map_dfreq : ndarray
            A 2-D array of tau values
        """


def map_tran_dfreq(stack_in):
    """Map the dominant frequency values for a stack of transient fluorescent data
    i.e.

        Parameters
        ----------
        stack_in : ndarray
            A 3-D array (T, Y, X) of an optical transient

        Returns
        -------
        map_dfreq : ndarray
            A 2-D array of dominant frequency values
       """


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
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')

    # F / F0: (F_t - F0) / F0
    f_t = signal_in
    if invert:
        f_0 = signal_in.max()
    else:
        f_0 = signal_in.min()
    signal_ff0 = (f_t - f_0) / f_0

    return signal_ff0

def calc_phase(signal_in):
    """Convert a signal from its fluorescent value to its phase,
    i.e.

        Parameters
        ----------
        signal_in : ndarray
            The array of fluorescent data to be converted

        Returns
        -------
        signal_phase : ndarray
            The array of phase data (degrees radians), dtype : float
        """