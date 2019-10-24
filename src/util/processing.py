import numpy as np
import statistics
from scipy.signal import find_peaks


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


def filter_temporal(signal, filter_type):
    """Temporally filter an array of optical data.

       Parameters
       ----------
       signal : ndarray
            The array of data to be evaluated
       filter_type : str
           The type of filter algorithm to use

       Returns
       -------
       stack_out : ndarray
            A temporally filtered 3-D array (T, Y, X) of optical data
       """
    pass


def drift_remove(signal, poly_order):
    """Remove drift from an array of optical data using a polynomial fit.

       Parameters
       ----------
       signal : ndarray
            The array of data to be evaluated
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


def snr_signal(signal):
    """Calculate the Signal-to-Noise ratio of a signal array,
    defined as the ratio of the Peak-Peak amplitude to the population standard deviation of the noise.

       Parameters
       ----------
       signal : ndarray
            The array of data to be evaluated

       Returns
       -------
       snr : float
            The Signal-to-Noise ratio of the given data
       peak_peak : float
            The absolute difference between the RMSs of the peak and noise arrays
       ratio_noise : float
            The ratio between the ranges of noise and peak value(s)
       sd_peak : float
            The standard deviation of the peak values
       # data_noise : ndarray
       #      The array of noise values used in the calculation
       # data_peak : ndarray
       #      The array of peak values used in the calculation
       ir_noise : ndarray
            The indexes for noise data elements to be used in the calculation
       ir_peak : ndarray
            The indexes for signal data elements to be used in the calculation

        Notes
        -----
        Must be applied to signals with upward deflections (Peak > noise).
        Assumes max noise value < (peak / 5)
       """
    # Check parameters
    if type(signal) is not np.ndarray:
        raise TypeError('Signal values must either be "int" or "float"')
    if signal.dtype not in [int, float]:
        raise TypeError('Signal values must either be "int" or "float"')

    if any(v < 0 for v in signal):
        raise ValueError('All signal values must be >= 0')

    # Characterize the signal
    signal_bounds = (signal.min(), signal.max())
    signal_range = signal_bounds[1] - signal_bounds[0]

    # Calculate noise values
    i_noise_count = 100  # number of samples to use for noise data
    i_noise_peaks, _ = find_peaks(signal, height=(None, (signal_bounds[0] + signal_range/5)))
    i_noise_calc = np.linspace(start=i_noise_peaks.max() - i_noise_count, stop=i_noise_peaks.max(),
                               num=i_noise_count, endpoint=False).astype(int)
    data_noise = signal[i_noise_peaks.max() - 10: i_noise_peaks.max()]
    noise_bounds = (data_noise.min(), data_noise.max())
    noise_range = noise_bounds[1] - noise_bounds[0]
    noise_rms = np.sqrt(np.mean(data_noise.astype(np.dtype(float)) ** 2))
    noise_mean = data_noise.mean()
    if signal.mean() < noise_rms:
        raise ValueError('Signal peaks seem to be < noise')

    # Calculate peak values
    i_peaks, _ = find_peaks(signal, prominence=(signal_range/2))
    data_peak = signal[i_peaks[0] - 1: i_peaks[0] + 3]
    peak_rms = np.sqrt(np.mean(data_peak.astype(np.dtype(float)) ** 2))
    # Calculate Peak-Peak value
    peak_peak = abs(peak_rms - noise_rms)

    # Calculate SNR
    noise_ratio = peak_peak / noise_range
    noise_sd_pop = statistics.stdev(data_noise)
    snr = peak_peak / noise_sd_pop

    ratio_noise = noise_ratio
    sd_noise = noise_sd_pop
    ir_noise = i_noise_calc
    ir_peak = i_peaks
    return snr, peak_peak, ratio_noise, sd_noise, ir_noise, ir_peak


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
