import numpy as np


def isolate_spatial(stack_in, roi):
    """Isolate a spatial region of a stack (3-D array, TYX) of grayscale optical data.

       Parameters
       ----------
       stack_in : ndarray
            A 3-D array (T, Y, X) of optical data
       roi : `GraphicsItem <pyqtgraph.graphicsItems.ROI>`
            Generic region-of-interest widget.

       Returns
       -------
       stack_out : ndarray
            A spatially isolated 3-D array (T, Y, X) of optical data
       """
    pass


def isolate_temporal(stack_in, i_start, i_end):
    """Isolate a spatial region of a stack (3-D array, TYX) of grayscale optical data.

       Parameters
       ----------
       stack_in : ndarray
            A 3-D array (T, Y, X) of optical data
       i_start : int
            Index or frame to start temporal isolation
       i_end : int
            Index or frame to end temporal isolation

       Returns
       -------
       stack_out : ndarray
            A temporally isolated 3-D array (T, Y, X) of optical data
       """
    pass


def filter_spatial(stack_in, filter_type):
    """Spatially filter a stack (3-D array, TYX) of grayscale optical data.

       Parameters
       ----------
       stack_in : ndarray
            A 3-D array (T, Y, X) of optical data
       filter_type : str
           The type of filter algorithm to use

       Returns
       -------
       stack_out : ndarray
            A spatially filtered 3-D array (T, Y, X) of optical data
       """
    pass


def filter_temporal(stack_in, filter_type):
    """Temporally filter a stack (3-D array, TYX) of grayscale optical data.

       Parameters
       ----------
       stack_in : ndarray
            A 3-D array (T, Y, X) of optical data
       filter_type : str
           The type of filter algorithm to use

       Returns
       -------
       stack_out : ndarray
            A temporally filtered 3-D array (T, Y, X) of optical data
       """
    pass


def drift_remove(signal_in, poly_order):
    """Remove drift from a signal array of grayscale optical data
    using a polynomial fit.

       Parameters
       ----------
       signal_in : ndarray
            The array of data to be processed
       poly_order : int
            The order of the polynomial to fit to

       Returns
       -------
       signal_out : ndarray
            A signal array with drift removed
       """
    pass


def normalize_signal(signal_in):
    """Normalize the values of a signal array to range from 0 to 1.

       Parameters
       ----------
       signal_in : ndarray
            The array of data to be processed

       Returns
       -------
       signal_out : ndarray
            A normalized signal array
       """
    pass


def snr_signal(signal, i_noise, i_peak):
    """Calculate the Signal-to-Noise ratio of a signal array,
    defined as the ratio of the Peak-Peak amplitude to the population standard deviation of the noise.

       Parameters
       ----------
       signal : ndarray
            The array of data to be evaluated
       i_noise : tuple
            The range of indexes for noise data elements to be used in the calculation, e.g. (0, 10)
       i_peak : tuple
            The range of indexes for peak data elements to be used in the calculation, e.g. (50, 60)

       Returns
       -------
       snr : float
            The Signal-to-Noise ratio of the given data
       sd_peak : float
            The standard deviation of the peak values
       sd_noise : float
            The standard deviation of the noise values
       data_peak : ndarray
            The array of peak values used in the calculation
       data_noise : ndarray
            The array of noise values used in the calculation
       """
    # Check parameters
    if type(signal) not in [int, float]:
        raise TypeError("Signal value type must either be 'int' or 'float'")
    # return snr, noise_sd, data_peak, data_noise


def snr_map(stack_in):
    """Generate a map of Signal-to-Noise ratios for signal arrays within a stack,
    defined as the ratio of the Peak-Peak amplitude to the population standard deviation of the noise.

       Parameters
       ----------
       stack_in : ndarray
           A 3-D array (T, Y, X) of optical data

       Returns
       -------
       map : ndarray
            A 2-D array of Signal-to-Noise ratios
       """
    # Check parameters
    pass


def calculate_error(ideal, modified):
    """Calculate the amount of error created by signal modulation or filtering,
    defined as (Ideal - Modified) / Ideal X 100%.

       Parameters
       ----------
       ideal : ndarray
            An array of ideal data
       modified : ndarray
            An array of modified data

       Returns
       -------
       error : ndarray
            An array of percent error
       error_mean : float
            The mean value of the percent error array
       error_sd : float
            The standard deviation of the percent error array
       """
    # Check parameters
