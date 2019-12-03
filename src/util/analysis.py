from util.processing import *
import time
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.misc import derivative
from scipy.interpolate import UnivariateSpline


def find_tran_start(signal_in):
    """Find the time of the start of a transient,
    defined as the 1st maximum of the 2nd derivative

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float

        Returns
        -------
        i_start : np.int64
            The index of the signal array corresponding to the start of the transient
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "int" or "float"')

    if any(v < 0 for v in signal_in):
        raise ValueError('All signal values must be >= 0')

    # Limit search to ((i_peak - i_act) * 3) before the activation
    i_peak = find_tran_peak(signal_in)
    i_act = find_tran_act(signal_in)
    search_min = i_act - int((i_peak - i_act) * 3)
    search_max = i_act

    # smooth the 1st with a Savitzky Golay filter and, from that, calculate the 2nd derivative
    # https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    time_x = np.linspace(0, len(signal_in) - 1, len(signal_in))
    spl = UnivariateSpline(time_x, signal_in)
    df_spline = spl(time_x, nu=1)
    df_smooth = savgol_filter(df_spline, window_length=5, polyorder=3)

    spl_df_smooth = UnivariateSpline(time_x, df_smooth)
    d2f_smooth = spl_df_smooth(time_x, nu=1)

    # find the 2nd derivative max within the search area
    signal_search_d2f_smooth = d2f_smooth[search_min:search_max]
    i_start_search = np.argmax(signal_search_d2f_smooth)
    i_start = search_min + i_start_search

    return i_start


def find_tran_act(signal_in):
    """Find the time of the activation of a transient,
    defined as the the maximum of the 1st derivative OR

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float

        Returns
        -------
        i_activation : np.int64
            The index of the signal array corresponding to the activation of the transient
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "int" or "float"')

    if any(v < 0 for v in signal_in):
        raise ValueError('All signal values must be >= 0')

    time_x = np.linspace(0, len(signal_in) - 1, len(signal_in))
    # signal_d2f = np.diff(signal_in, n=2, prepend=[int(signal_in[0]), int(signal_in[0])]).astype(float)
    # d2f_smooth = filter_temporal(signal_d2f, sample_rate, filter_order=5)
    # print('Starting analysis splines')
    # print('** Starting UnivariateSpline')
    # start = time.process_time()
    spl = UnivariateSpline(time_x, signal_in)
    # end = time.process_time()
    # print('** Finished UnivariateSpline', end - start)
    # print('** Starting spl')
    # start = time.process_time()
    signal_spline_df = spl(time_x, nu=1)
    # end = time.process_time()
    # print('** Finished spl', end - start)
    # print('Done with analysis splines')
    # print('Timing, test_tiff, Vm : ', end - start)

    i_activation = np.argmax(signal_spline_df)  # 1st df max, Activation

    return i_activation


def find_tran_peak(signal_in):
    """Find the time of the peak of a transient,
    defined as the maximum value

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float

        Returns
        -------
        i_peak : np.int64
            The index of the signal array corresponding to the peak of the transient
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "int" or "float"')

    if any(v < 0 for v in signal_in):
        raise ValueError('All signal values must be >= 0')

    # Characterize the signal
    signal_bounds = (signal_in.min(), signal_in.max())
    signal_range = signal_bounds[1] - signal_bounds[0]

    i_peaks, _ = find_peaks(signal_in, prominence=(signal_range / 2))

    if len(i_peaks) > 1:
        raise ArithmeticError('{} peaks detected for a single given transient'.format(len(i_peaks)))

    i_peak = i_peaks[0]

    return i_peak


def find_tran_downstroke(signal_in):
    """Find the time of the downstroke of a transient,
    defined as the minimum of the 1st derivative

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float

        Returns
        -------
        i_downstroke : np.int64
            The index of the signal array corresponding to the downstroke of the transient
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "int" or "float"')

    time_x = np.linspace(0, len(signal_in) - 1, len(signal_in))
    signal_d2f = np.diff(signal_in, n=2, prepend=[int(signal_in[0]), int(signal_in[0])]).astype(float)
    # d2f_smooth = filter_temporal(signal_d2f, sample_rate, filter_order=5)
    spl = UnivariateSpline(time_x, signal_in)
    signal_spline_df = spl(time_x, nu=1)
    signal_spline_d2f = spl(time_x, nu=2)

    i_downstroke = np.argmin(signal_spline_df)  # df min, Downstroke

    return i_downstroke


def find_tran_end(signal_in):
    """Find the time of the end of a transient,
    defined as the 2nd maximum of the 2nd derivative

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float

        Returns
        -------
        i_end : np.int64
            The index of signal array corresponding to the end of the transient
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "int" or "float"')

    time_x = np.linspace(0, len(signal_in) - 1, len(signal_in))
    signal_d2f = np.diff(signal_in, n=2, prepend=[int(signal_in[0]), int(signal_in[0])]).astype(float)
    # d2f_smooth = filter_temporal(signal_d2f, sample_rate, filter_order=5)
    spl = UnivariateSpline(time_x, signal_in)
    signal_spline_df = spl(time_x, nu=1)
    signal_spline_d2f = spl(time_x, nu=2)

    i_max1 = np.argmax(signal_spline_d2f)
    i_end = np.argmax(signal_spline_d2f[i_max1 + 1:])  # 2st df2 max, End

    return i_end


def calc_tran_activation(signal_in):
    """Calculate the time of the activation of a transient,
    defined as the midpoint (not limited by sampling rate) between the start and peak times

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float

        Returns
        -------
        i_activation : np.int64
            The index of the signal array corresponding to the activation of the transient
        """

    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "int" or "float"')

    if any(v < 0 for v in signal_in):
        raise ValueError('All signal values must be >= 0')

    pass


def calc_tran_duration(signal_in, percent=50):
    """Calculate the duration of a transient,
    defined as the number of indices between the start time and a time
    nearest a percentage of the start-to-peak value (e.g. APD-80, CAD-90)

        Parameters
        ----------
        signal_in : ndarray
              The array of data to be evaluated, dtype : uint16 or float
        percent : int
              Percentage of the value from start to peak to use as the

        Returns
        -------
        duration : np.int64
            The % duration of the transient in number of indices
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "int" or "float"')
    if type(percent) is not int:
        raise TypeError('Percent data type must be an "int"')

    if any(v < 0 for v in signal_in):
        raise ValueError('All signal values must be >= 0')
    if any(x < 0 or x >= 100 for x in signal_in):
        raise ValueError('All signal values must be between 0-99%')


def calc_tran_tau(signal_in):
    """Calculate the decay time constant (tau) of a transient,
    defined as the time between 30 and 90% decay from peak

        Parameters
        ----------
         signal_in : ndarray
              The array of data to be evaluated, dtype : uint16 or float

        Returns
        -------
        tau : float
            The decay time constant (tau) of the signal array corresponding to it's peak
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "int" or "float"')

    if any(v < 0 for v in signal_in):
        raise ValueError('All signal values must be >= 0')


def calc_tran_di(signal_in):
    """Calculate the diastolic interval (DI) of a transient,
    defined as the number of indices between restoration and the next transient's activation

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float

        Returns
        -------
        di : float
            The  diastolic interval (DI) of the first transient in a signal array

        Notes
        -----
            Should not be applied to signal data containing at least one transient.
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "int" or "float"')

    if any(v < 0 for v in signal_in):
        raise ValueError('All signal values must be >= 0')


def map_tran_analysis(time_in, stack_in, analysis_type):
    """Map an analysis point's values for a stack of transient fluorescent data
        i.e.
        Parameters
        ----------
        time_in : ndarray
            The array of timestamps (ms) corresponding to signal_in, dtyoe : int or float
        stack_in : ndarray
            A 3-D array (T, Y, X) of an optical transient, dtype : uint16 or float
        analysis_type : method
            The type of analysis to be mapped

        Returns
        -------
        map_analysis : ndarray
            A 2-D array of analysis values
        """
    # Check parameters
    if type(stack_in) is not np.ndarray:
        raise TypeError('Stack type must be an "ndarray"')
    if len(stack_in.shape) is not 3:
        raise TypeError('Stack must be a 3-D ndarray (T, Y, X)')
    if stack_in.dtype not in [np.uint16, float]:
        raise TypeError('Stack values must either be "np.uint16" or "float"')

    # if type(analysis_type) is not classmethod:
    #     raise TypeError('Analysis type must be a "classmethod"')

    map_shape = stack_in.shape[1:]
    map_out = np.empty(map_shape)
    # Assign a value to each pixel
    print('Generating map with {} ...'.format(analysis_type))
    for iy, ix in np.ndindex(map_shape):
        print('\rRow:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy, map_shape[0], ix, map_shape[1]), end='', flush=True)
        pixel_data = stack_in[:, iy, ix]
        # pixel_ensemble = calc_ensemble(time_in, pixel_data)
        # snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak = calculate_snr(pixel_data, noise_count)
        # snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak = calculate_snr(pixel_data, noise_count)
        # Set every pixel's values to the analysis value of the signal at that pixel
        # map_out[iy, ix] = analysis_type(pixel_ensemble[1])
        map_out[iy, ix] = analysis_type(pixel_data)

    return map_out


def map_tran_tau(stack_in):
    """Map the decay constant (tau) values for a stack of transient fluorescent data
    i.e.

        Parameters
        ----------
        stack_in : ndarray
            A 3-D array (T, Y, X) of an optical transient, dtype : uint16 or float

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
            A 3-D array (T, Y, X) of an optical transient, dtype : uint16 or float

        Returns
        -------
        map_dfreq : ndarray
            A 2-D array of dominant frequency values
       """


def calc_phase(signal_in):
    """Convert a signal from its fluorescent value to its phase,
    i.e.

        Parameters
        ----------
        signal_in : ndarray
            The array of fluorescent data to be converted, dtype : uint16 or float

        Returns
        -------
        signal_phase : ndarray
            The array of phase data (degrees radians), dtype : float
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')


def calc_ensemble(time_in, signal_in):
    """Convert a signal from multiple transients to an averaged signal,
    segmented by activation times

        Parameters
        ----------
        time_in : ndarray
            The array of timestamps (ms) corresponding to signal_in, dtyoe : int or float
        signal_in : ndarray
            The array of fluorescent data to be converted

        Returns
        -------
        signal_time : ndarray
            An array of timestamps (ms) corresponding to signal_out
        signal_out : ndarray
            The array of an ensembled transient signal, dtype : float
        signals : list
            The list of signal arrays used to create the ensemble
        i_peaks : ndarray
            The idecies of peaks from signal_in used
        est_cycle : float
            Estimated cycle length (ms) of transients in signal_in
        """
    # Check parameters
    if type(time_in) is not np.ndarray:
        raise TypeError('Time data type must be an "ndarray"')
    if time_in.dtype not in [int, float]:
        raise TypeError('Time values must either be "int" or "float"')
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "uint16" or "float"')

    # Calculate the number of transients in the signal
    # Characterize the signal
    signal_bounds = (signal_in.min(), signal_in.max())
    signal_range = signal_bounds[1] - signal_bounds[0]

    # Calculate noise values, detecting the last noise peak and using indexes [peak - noise_count, peak]
    noise_height = signal_bounds[0] + signal_range / 2  # assumes max noise/signal amps. of 1 / 2
    i_noise_peaks, _ = find_peaks(signal_in, height=(None, noise_height))
    if len(i_noise_peaks) == 0:
        i_noise_peaks = [len(signal_in) - 1]

    # Find indices of peak values, between peaks_window tall and with a prominence of half the signal range
    peaks_window = (noise_height, signal_bounds[1] + signal_range / 2)
    i_peaks, _ = find_peaks(signal_in, height=peaks_window,
                            prominence=signal_range / 2)
    if len(i_peaks) == 0:
        raise ArithmeticError('No peaks detected'.format(len(i_peaks), i_peaks))
    if len(i_peaks) > 1:
        # i_peak_calc = i_peaks[0].astype(int)
        # raise ArithmeticError('{} peaks detected at {} for a single given transient'.format(len(i_peaks), i_peaks))
        # ir_peak = i_peaks
        print('{} peaks detected at {} for a single given transient'.format(len(i_peaks), i_peaks))

    if len(i_peaks) > 1:
        # raise ArithmeticError('{} peaks detected at {} for a single given transient'.format(len(i_peaks), i_peaks))
        print('{} peaks detected at {} in signal_in'.format(len(i_peaks), i_peaks))
    else:
        raise ValueError('Only {} peak detected at {} in signal_in'.format(len(i_peaks), i_peaks))

    i_peaks_df = np.diff(i_peaks, n=1).astype(float)
    est_cycle_i = np.nanmean(i_peaks_df)
    est_cycle = est_cycle_i * (time_in[1] - time_in[0])
    est_cycle_int = np.floor(est_cycle_i).astype(int)
    cycle_shift = min(i_peaks[0], np.floor(est_cycle_int / 2).astype(int))

    signal_time = time_in[0: est_cycle_int]
    signals_trans = []
    signals_trans_act = []

    for peak_num, peak in enumerate(i_peaks):
        signals_trans.append(signal_in[i_peaks[peak_num] - cycle_shift:
                                       i_peaks[peak_num] + est_cycle_int - cycle_shift])

    # With that peak detection, find activation times
    for act_num, signal in enumerate(signals_trans):
        i_start = find_tran_start(signal)
        i_act = find_tran_act(signal)
        signals_trans_act.append(signal_in[i_act - 3:
                                           i_act + est_cycle_int - 3])

    signal_out = np.nanmean(signals_trans, axis=0)
    signals = signals_trans
    # signal_out = np.nanmean(signals_trans_act, axis=0)
    # signals = signals_trans_act

    return signal_time, signal_out, signals, i_peaks, est_cycle
