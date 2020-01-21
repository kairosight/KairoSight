from util.processing import *
import time
import numpy as np
from scipy.signal import savgol_filter
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

    # Limit search to
    # before the peak and
    # after the "middle" non-prominent point (baseline) before the peak
    i_baselines = find_tran_baselines(signal_in)
    i_act = find_tran_act(signal_in)
    # search_min = max(0, i_act - int((i_peak - i_act) * 2))
    d_search = int(np.max((i_baselines) - np.min(i_baselines)) / 2)
    search_min = np.min(i_baselines) + d_search
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

    # if any(v < 0 for v in signal_in):
    #     raise ValueError('All signal values must be >= 0')

    # Limit search to just after the peak and after the median baseline point before the peak
    i_peak = find_tran_peak(signal_in)
    if i_peak is np.nan:
        return np.nan
    i_baselines = find_tran_baselines(signal_in, peak_side='left')
    i_baseline = int(np.median(i_baselines))

    search_min = i_baseline
    search_max = i_peak + (i_peak - i_baseline)

    # use a spline
    time_search = np.linspace(search_min, search_max - 1,
                              search_max - search_min)
    signal_search = signal_in[search_min:search_max]

    # spl = UnivariateSpline(time_baseline, signal_baseline)
    # spl = InterpolatedUnivariateSpline(time_baseline, signal_baseline)
    # # df/dt (with x20 as many time samples)
    # spline_fidelity = 20    # TODO optimize here
    # time_spline = np.linspace(search_min, search_max - 1,
    #                           (search_max - search_min)*spline_fidelity)
    # spl.set_smoothing_factor(200)    # TODO optimize here
    # df_spline = spl(time_spline, nu=1, ext='extrapolate')

    time_spline, df_spline, spline_fidelity = spline_signal(time_search, signal_search)

    # find the 1st derivative max within the search area
    i_act_search = np.argmax(df_spline)
    i_act_search = int(i_act_search / spline_fidelity)

    # time_x = np.linspace(0, len(signal_in) - 1, len(signal_in))
    # # print('Starting analysis splines')
    # # print('** Starting UnivariateSpline')
    # # start = time.process_time()
    # spl = UnivariateSpline(time_x, signal_in)
    # # spl.set_smoothing_factor(0.7)
    # # end = time.process_time()
    # # print('** Finished UnivariateSpline', end - start)
    # # print('** Starting spl')
    # # start = time.process_time()
    # df_spline = spl(time_x, nu=1)
    # # smooth the 1st with a Savitzky Golay filter
    # # https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    # df_smooth = savgol_filter(df_spline, window_length=5, polyorder=3)
    # # end = time.process_time()
    # # print('** Finished spl', end - start)
    # # print('Done with analysis splines')
    # # print('Timing, test_tiff, Vm : ', end - start)
    #
    # # find the 1st derivative max within the search area
    # signal_search_df_smooth = df_smooth[search_min:search_max]
    # i_act_search = np.argmax(signal_search_df_smooth)  # 1st df max, Activation

    i_activation = search_min + i_act_search

    # if i_act_search < 20:
    #     print('\tLow rel. activation time: {}'.format(i_act_search),
    #           end='', flush=True)

    return i_activation


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

    # Limit search to after the peak and before the end
    i_peak = find_tran_peak(signal_in)
    search_min = i_peak
    i_baseline = find_tran_baselines(signal_in, peak_side='right')
    search_max = max(i_baseline)

    time_x = np.linspace(0, len(signal_in) - 1, len(signal_in))
    spl = UnivariateSpline(time_x, signal_in)
    df_spline = spl(time_x, nu=1)
    df_smooth = savgol_filter(df_spline, window_length=5, polyorder=3)

    # find the 2nd derivative max within the search area
    search_df_smooth = df_smooth[search_min:search_max]
    i_start_search = np.argmin(search_df_smooth)  # df min, Downstroke
    i_downstroke = search_min + i_start_search

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

    # Limit search to after the Downstroke and before a the end of the ending baseline
    i_downstroke = find_tran_downstroke(signal_in)
    search_min = i_downstroke
    # first index after peak where the signal < before its start value,
    # e.g. at signal[((i_peak - i_act) * 3)] before the activation
    i_baseline = find_tran_baselines(signal_in, peak_side='right')
    search_max = max(i_baseline)

    # smooth the 1st with a Savitzky Golay filter and, from that, calculate the 2nd derivative
    # https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    time_x = np.linspace(0, len(signal_in) - 1, len(signal_in))
    spl = UnivariateSpline(time_x, signal_in)
    df_spline = spl(time_x, nu=1)
    df_smooth = savgol_filter(df_spline, window_length=5, polyorder=3)

    spl_df_smooth = UnivariateSpline(time_x, df_smooth)
    d2f_smooth = spl_df_smooth(time_x, nu=1)

    search_d2f_smooth = d2f_smooth[search_min: search_max]

    i_end_search = np.argmax(search_d2f_smooth)  # 2st df2 max, End
    i_end = search_min + i_end_search

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


def calc_tran_duration(signal_in, percent=80):
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
    # if type(signal_in) is not np.ndarray:
    #     raise TypeError('Signal data type must be an "ndarray"')
    # if signal_in.dtype not in [np.uint16, float]:
    #     raise TypeError('Signal values must either be "int" or "float"')
    if type(percent) is not int:
        raise TypeError('Percent data type must be an "int"')

    # if any(v < 0 for v in signal_in):
    #     raise ValueError('All signal values must be >= 0')
    # if any(x < 0 or x >= 100 for x in signal_in):
    #     raise ValueError('All signal values must be between 0-99%')

    snr, rms_bounds, peak_peak, sd_noise, ir_noise, i_peak = calculate_snr(signal_in)
    if snr is np.nan:
        return np.nan
    if type(i_peak) not in [np.int64, float]:
        i_peak = i_peak[0]  # use the first detected peak

    noise_rms = rms_bounds[0]
    i_activation = find_tran_act(signal_in)

    cutoff = noise_rms + (float(peak_peak) * float(((100 - percent) / 100)))

    i_search = np.where(signal_in[i_peak:] <= cutoff)
    if len(i_search) == 0:
        return np.nan
    try:
        i_duration = i_peak + i_search[0][0]
        i_relative_duration = i_duration - i_activation
    except Exception:
        return np.nan

    # Exclusions
    return i_relative_duration


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
            Should not be applied to signal data containing only 1 transient.
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "int" or "float"')

    if any(v < 0 for v in signal_in):
        raise ValueError('All signal values must be >= 0')


# def calc_pixel_analysis(x, y, map):
#     print('\r\tRow:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix, map_shape[1]), end='',
#           flush=True)
#     pixel_data = stack_in[:, iy, ix]
#     # Check if pixel has been masked (0 at every frame)
#     # # or was masked and spatially filtered (constant at every frame)
#     # peak = find_tran_peak(pixel_data)
#     # if peak is np.nan:  # if there's no detectable peak
#     #     pixel_analysis_value = np.NaN
#
#     unique, counts = np.unique(pixel_data, return_counts=True)
#     if len(unique) < 5:  # signal is too flat to have a valid peak
#         pixel_analysis_value = np.NaN
#     else:
#         analysis_result = analysis_type(pixel_data)
#         if time_in is not None:
#             pixel_analysis_value = time_in[analysis_result]
#         else:
#             pixel_analysis_value = analysis_result
#
#     map_out[iy, ix] = pixel_analysis_value

def map_tran_analysis(stack_in, analysis_type, time_in=None):
    """Map an analysis point's values for a stack of transient fluorescent data
        i.e.

        Parameters
        ----------
        stack_in : ndarray
            A 3-D array (T, Y, X) of an optical transient, dtype : uint16 or float
        analysis_type : method
            The type of analysis to be mapped
        time_in : ndarray, optional
            The array of timestamps (ms) corresponding to signal_in, dtyoe : int or float
            If used, map values are timestamps

        Returns
        -------
        map_analysis : ndarray
            A 2-D array of analysis values
            dtype : analysis_type().dtype or float if time_in provided
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

    print('Generating map with {} ...'.format(analysis_type))
    map_shape = stack_in.shape[1:]
    map_out = np.empty(map_shape)

    # # Calculate with parallel processing
    # with Pool(5) as p:
    #     p.map()

    # Assign a value to each pixel
    for iy, ix in np.ndindex(map_shape):
        print('\r\tRow:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix, map_shape[1]), end='',
              flush=True)
        pixel_data = stack_in[:, iy, ix]
        # Check if pixel has been masked (0 at every frame)
        # # or was masked and spatially filtered (constant at every frame)
        # peak = find_tran_peak(pixel_data)
        # if peak is np.nan:  # if there's no detectable peak
        #     pixel_analysis_value = np.NaN

        unique, counts = np.unique(pixel_data, return_counts=True)
        if len(unique) < 10:  # signal is too flat to have a valid peak
            pixel_analysis_value = np.NaN
            map_out[iy, ix] = pixel_analysis_value
        else:
            analysis_result = analysis_type(pixel_data)
            if analysis_result is np.nan:
                map_out[iy, ix] = np.nan
            else:
                if time_in is not None:
                    pixel_analysis_value = time_in[analysis_result]
                else:
                    pixel_analysis_value = analysis_result
                map_out[iy, ix] = pixel_analysis_value

    # If mapping activation, align times with the "first" aka lowest activation time
    if analysis_type is find_tran_act:
        map_out = map_out - np.nanmin(map_out)
    print('\nDONE Generating map')

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
    segmented by activation times. Discards the first and last transients.

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
            or zeroes if no peak was detected
        signals : list
            The list of signal arrays used to create the ensemble
        i_peaks : ndarray
            The indexes of peaks from signal_in used
        i_acts  : ndarray
            The indexes of activations from signal_in used
        est_cycle : float
            Estimated cycle length (ms) of transients in signal_in

        Notes
        -----
            # Normalizes signal from 0-1 in the process
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
    unique, counts = np.unique(signal_in, return_counts=True)

    if len(unique) < 10:  # signal is too flat to have a valid peak
        return np.zeros_like(signal_in)

    # Find the peaks
    i_peaks, _ = find_peaks(signal_in, prominence=signal_range / 4,
                            distance=20)
    if len(i_peaks) == 0:
        raise ArithmeticError('No peaks detected'.format(len(i_peaks), i_peaks))
    if len(i_peaks) > 3:
        print('* {} peaks detected at {} in signal_in'.format(len(i_peaks), i_peaks))
    else:
        raise ValueError('Only {} peak detected at {} in signal_in'.format(len(i_peaks), i_peaks))

    # do not use the first and last peaks
    i_peaks = i_peaks[1:-1]
    # Split up the signal using peaks and estimated cycle length
    est_cycle = np.diff(i_peaks).astype(float)
    est_cycle_i = np.nanmean(est_cycle)
    est_cycle = est_cycle_i * np.nanmean(np.diff(time_in))
    est_cycle_i = np.floor(est_cycle_i).astype(int)
    cycle_shift = np.floor(est_cycle_i / 2).astype(int)

    signal_time = time_in[0: est_cycle_i]
    signals_trans_peak = []
    i_baselines_full = []
    i_acts_full = []
    signals_trans_act = []

    for peak_num, peak in enumerate(i_peaks):
        signal = signal_in[i_peaks[peak_num] - cycle_shift:
                           i_peaks[peak_num] + est_cycle_i - cycle_shift]
        signals_trans_peak.append(signal)

        i_baselines = find_tran_baselines(signal)
        i_baselines_full.append((i_peaks[peak_num] - cycle_shift) + i_baselines)
        i_act_signal = find_tran_act(signal)
        i_act_full = (i_peaks[peak_num] - cycle_shift) + i_act_signal
        i_acts_full.append(i_act_full)

    # With that peak detection, find activation times and align transient
    for act_num, signal in enumerate(signals_trans_peak):
        # i_start_signal = find_tran_start(signal)
        # signal = normalize_signal(signal)
        # i_act_signal = find_tran_act(signal)
        # i_act_full = (i_peaks[act_num] - cycle_shift) + i_act_signal
        # i_act_full = i_act_signal + (est_cycle_i * act_num)

        # # align along activation times, and crop using the last signal's left-most baseline
        # # i_align = i_act_full - int((est_cycle_i/8))
        # # align along activation times, and crop using start time and est_cycle_i
        # last_baselines = np.min(i_baselines_full[-1])
        # last_act = i_acts_full[-1]
        # d_last_crop = last_act - last_baselines
        # i_align = i_act_full - d_last_crop
        # # cycle_shift = min(i_peaks)
        # signal_cycle = normalize_signal(signal_in[i_align:i_align + est_cycle_i])

        # # align along activation times, and crop using the first signal's left-most baseline
        # i_align = i_acts_full[act_num] - 40
        # i_align = i_acts_full[act_num] - (i_acts_full[0] - (i_peaks[0] - cycle_shift))
        i_align = i_acts_full[act_num] - (i_acts_full[0] - i_baselines_full[0][0])

        # signal_align = normalize_signal(signal_in[i_align:i_align + est_cycle_i])
        signal_align = signal_in[i_align:i_align + est_cycle_i]
        signals_trans_act.append(signal_align)

    # use the lowest activation time
    # cycle_shift = min(min(i_acts), cycle_shift)
    # for act_num, act in enumerate(i_acts):
    #     cycle_shift = max(cycle_shift, act)
    #     signals_trans_act.append(signal_in[i_acts[act_num] - cycle_shift:
    #                                        i_acts[act_num] + est_cycle_i - cycle_shift])

    # use the mean of all signals (except the last)
    # TODO try a rms calculation instead of a mean
    signal_out = np.nanmean(signals_trans_act, axis=0)
    signals = signals_trans_act
    i_acts = i_acts_full
    # signal_out = np.nanmean(signals_trans_act, axis=0)
    # signals = signals_trans_act

    return signal_time, signal_out, signals, i_peaks, i_acts, est_cycle


def calc_ensemble_stack(time_in, stack_in):
    """Convert a stack from pixels with multiple transients to those with an averaged signal,
    segmented by activation times. Discards the first and last transients.

        Parameters
        ----------
        time_in : ndarray
            The array of timestamps (ms) corresponding to signal_in, dtyoe : int or float
        stack_in : ndarray
            A 3-D array (T, Y, X) of an optical transient, dtype : uint16 or float

        Returns
        -------
        stack_out : ndarray
             A spatially isolated 3-D array (T, Y, X) of optical data, dtype : stack_in.dtype

        Notes
        -----
            Should not be applied to signal data containing at least one transient.
            Pixels with incalculable ensembles are assigned an array of zeros
        """

    print('Ensembling a stack ...')
    stack_out = np.empty_like(stack_in)
    map_shape = stack_in.shape[1:]
    i_peak_0_min = stack_in.shape[0]
    # i_peak_1_min = len(stack_in)

    # for each pixel ...
    for iy, ix in np.ndindex(map_shape):
        print('\r\tPeak Search of Row:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(
            iy + 1, map_shape[0], ix + 1, map_shape[1]), end='', flush=True)
        # get signal
        pixel_data = stack_in[:, iy, ix]
        # Characterize the signal
        signal_bounds = (pixel_data.min(), pixel_data.max())
        signal_range = signal_bounds[1] - signal_bounds[0]
        unique, counts = np.unique(pixel_data, return_counts=True)

        if len(unique) < 10:  # signal is too flat to have a valid peak
            continue

        # Find the peaks
        # i_peaks, _ = find_peaks(pixel_data, prominence=signal_range / 4,
        #                         distance=20)
        i_peaks, _ = find_tran_peak(pixel_data, props=True)

        if len(i_peaks) < 4:
            # raise ArithmeticError('No peaks detected'.format(len(i_peaks), i_peaks))
            # np.zeros_like(pixel_data)
            continue
        # if len(i_peaks) > 3:
        #     print('* {} peaks detected at {} in signal_in'.format(len(i_peaks), i_peaks))
        # else:
        #     raise ValueError('Only {} peak detected at {} in signal_in'.format(len(i_peaks), i_peaks))

        # find the first peak and preserve the minimum among all pixels
        if i_peaks[0] < i_peak_0_min:
            i_peak_0_min = i_peaks[0]
        # i_peak_1_min = max(i_peaks[1], i_peak_1_min)

    # truncate start of stack to avoid improper ensemble by
    # calculating padding needed to preserve activation propagation
    # * Cut at the minimum peak_0
    stack_in = stack_in[i_peak_0_min:]

    # for each pixel ...
    for iy, ix in np.ndindex(map_shape):
        print('\r\tEnsemble of Row:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix + 1, map_shape[1]),
              end='', flush=True)
        # get signal
        pixel_data = stack_in[:, iy, ix]
        if len(unique) < 10:  # signal is too flat to have a valid peak
            signal_ensemble = np.zeros_like(pixel_data)
        else:
            # calculate the ensemble of it
            time_ensemble, signal_ensemble, signals, signal_peaks, signal_acts, est_cycle_length \
                = calc_ensemble(time_in, pixel_data)

        # TODO ValueError: could not broadcast input array from shape (295) into shape (325)
        stack_out[:, iy, ix] = signal_ensemble

    print('\nDONE Ensembling stack')

    return stack_out
