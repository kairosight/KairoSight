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


def isolate_transient(signal_in, i_start, i_end):
    """Isolate a single transient from a signal array of optical data.

       Parameters
       ----------
       signal_in : ndarray
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


def filter_temporal(signal_in, filter_type):
    """Temporally filter an array of optical data.

       Parameters
       ----------
       signal_in : ndarray
            The array of data to be evaluated
       filter_type : str
           The type of filter algorithm to use

       Returns
       -------
       stack_out : ndarray
            A temporally filtered 3-D array (T, Y, X) of optical data
       """
    pass


def drift_remove(signal_in, poly_order):
    """Remove drift from an array of optical data using a polynomial fit.

       Parameters
       ----------
       signal_in : ndarray
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


def snr_signal(signal_in, noise_count=10):
    """Calculate the Signal-to-Noise ratio of a signal array,
    defined as the ratio of the Peak-Peak amplitude to the population standard deviation of the noise.

       Parameters
       ----------
       signal_in : ndarray
            The array of data to be evaluated, dtyoe : int or float
       noise_count : int
            The number of noise values to be used in the calculation, default is 10

       Returns
       -------
       snr : float
            The Signal-to-Noise ratio of the given data
       rms_bounds : tuple
            The RMSs of the peak and noise arrays, (noise_rms, peak_rms)
       peak_peak : float
            The absolute difference between the RMSs of the peak and noise arrays
       ratio_noise : float
            The ratio between the ranges of noise and peak value(s)
       sd_noise : float
            The standard deviation of the noise values
       sd_peak : float
            The standard deviation of the peak values
       ir_noise : ndarray
            The indexes of noise values used in the calculation
       ir_peak : ndarray
            The indexes of peak values used in the calculation

        Notes
        -----
        Must be applied to signals with upward deflections (Peak > noise).
        Assumes max noise value < (peak / 5)
        Auto-detects noise section as the last noise_count values before the final noisy peak.
       """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [int, float]:
        raise TypeError('Signal values must either be "int" or "float"')
    if type(noise_count) is not int:
        raise TypeError('Number of noise values to use must be an "int"')

    if any(v < 0 for v in signal_in):
        raise ValueError('All signal values must be >= 0')
    if noise_count >= len(signal_in):
        raise ValueError('Number of noise values to use must ne < length of signal array')

    # Characterize the signal
    signal_bounds = (signal_in.min(), signal_in.max())
    signal_range = signal_bounds[1] - signal_bounds[0]

    # Calculate noise values
    noise_height = signal_bounds[0] + signal_range/3    # assumes max noise/signal amps. of 1 / 3
    i_noise_peaks, _ = find_peaks(signal_in, height=(None, noise_height))
    i_noise_calc = np.linspace(start=i_noise_peaks.max() - noise_count, stop=i_noise_peaks.max(),
                               num=noise_count).astype(int)
    if any(i < 0 for i in i_noise_calc):
        raise ValueError('Number of noise values too large for this signal of length {}'.format(len(signal_in)))
    data_noise = signal_in[i_noise_calc[0]: i_noise_calc[-1]]
    noise_rms = np.sqrt(np.mean(data_noise.astype(np.dtype(float)) ** 2))
    noise_sd_pop = statistics.pstdev(data_noise)
    noise_bounds = (noise_rms - noise_sd_pop/2, noise_rms + noise_sd_pop/2)
    noise_range = noise_bounds[1] - noise_bounds[0]
    if signal_in.mean() < noise_rms:
        raise ValueError('Signal peaks seem to be < noise')

    # Calculate peak values
    i_peak_count = 3
    i_peaks, _ = find_peaks(signal_in, prominence=(signal_range/2))
    # use the 3 values centered around the peak (1 before and 1 after)
    i_peaks_calc = np.linspace(start=i_peaks[0] - 1, stop=i_peaks[0] + 1, num=i_peak_count).astype(int)
    data_peak = signal_in[i_peaks_calc]
    peak_sd_pop = statistics.pstdev(data_peak)
    peak_rms = np.sqrt(np.mean(data_peak.astype(np.dtype(float)) ** 2))
    # Calculate Peak-Peak value
    peak_peak = abs(peak_rms - noise_rms)

    # Calculate SNR
    snr = peak_peak / noise_sd_pop

    rms_bounds = (noise_rms, peak_rms)
    sd_noise = noise_sd_pop
    sd_peak = peak_sd_pop
    ir_noise = i_noise_calc
    ir_peak = i_peaks_calc
    return snr, rms_bounds, peak_peak, sd_noise, sd_peak, ir_noise, ir_peak


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
