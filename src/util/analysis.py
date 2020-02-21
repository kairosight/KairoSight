from util.processing import *
import time
import numpy as np
from scipy.signal import savgol_filter
from scipy.misc import derivative
from scipy.interpolate import UnivariateSpline

TRAN_MAX = 100
# Colormap and normalization range for Activation maps
ACT_MAX = 150
# Colormap and normalization range for Duration maps
DUR_MIN = 20  # ms
DUR_MAX = 200  # ms


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

    # Limit the search to be well before
    # and well after the peak (depends on which side of the peak the baselines are)
    i_peak = find_tran_peak(signal_in)
    if i_peak is np.nan:
        return np.nan
    i_baselines = find_tran_baselines(signal_in, peak_side='left')
    # i_baseline = int(np.median(i_baselines))
    i_baseline = i_baselines[-2]

    if i_baseline < i_peak:
        search_min = i_baseline
    else:
        search_min = 0  # not enough baselines before the peak, use everything before the peak
        # search_min = np.argmin(signal_in[:i_peak])  # not enough baselines before the peak, use ____

    search_max_calc = i_peak + ((i_peak - search_min) * 2)
    search_max = np.min((search_max_calc, len(signal_in)-1))
    # search_max = len(signal_in) - 1

    xx_search = np.linspace(search_min, search_max - 1,
                            search_max - search_min)
    signal_search = signal_in[search_min:search_max]

    # use a spline
    df_spline, spline_fidelity = spline_signal(xx_search, signal_search)

    # find the 1st derivative max within the search area (first likely to be extreme)
    i_act_search_df = np.argmax(df_spline[1:]) + 1
    i_act_search = int(np.floor(i_act_search_df / spline_fidelity))

    # # https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    # df_smooth = savgol_filter(df_spline, window_length=5, polyorder=3)

    i_activation = search_min + i_act_search

    if i_activation == i_peak:
        print('\tWarning! Activation time same as Peak: {}'.format(i_activation))

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
    if (i_relative_duration < DUR_MIN) or (i_relative_duration > DUR_MAX):
        return np.nan

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
        print('\r\tRow:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix + 1, map_shape[1]), end='',
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


def calc_ensemble(time_in, signal_in, crop='center'):
    """Convert a signal from multiple transients to an averaged signal,
    segmented by activation times. Discards the first and last transients.

        Parameters
        ----------
        time_in : ndarray
            The array of timestamps (ms) corresponding to signal_in, dtyoe : int or float
        signal_in : ndarray
            The array of fluorescent data to be converted
        crop : str or tuple
            The type of cropping applied, default is center
            If a tuple, begin aligned crop at crop[0] time index and end at crop[1]

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
    i_peaks, _ = find_tran_peak(signal_in, props=True)

    if len(i_peaks) == 0:
        raise ArithmeticError('No peaks detected'.format(len(i_peaks), i_peaks))
    if len(i_peaks) > 3:
        print('* {} peaks detected at {} in signal_in'.format(len(i_peaks), i_peaks))
    else:
        raise ValueError('Only {} peak detected at {} in signal_in'.format(len(i_peaks), i_peaks))

    # do not use the first and last peaks
    i_peaks = i_peaks[1:-2]
    # Split up the signal using peaks and estimated cycle length
    est_cycle_array = np.diff(i_peaks).astype(float)
    est_cycle_i = np.nanmean(est_cycle_array)
    est_cycle = est_cycle_i * np.nanmean(np.diff(time_in))
    est_cycle_i = np.floor(est_cycle_i).astype(int)
    cycle_shift = np.floor(est_cycle_i / 2).astype(int)

    signal_time = time_in[0: est_cycle_i]
    signals_trans_peak = []
    i_baselines_full = []
    i_acts_full = []
    signals_trans_act = []

    # roughly isolate all transients centered on their peaks
    # and cropped with a cycle-length-wide window
    # TODO ensembles are too wide due to bad activation times
    # TODO ensembles distorted by early peaks (late activation times?)
    for peak_num, peak in enumerate(i_peaks):
        signal = signal_in[i_peaks[peak_num] - cycle_shift:
                           i_peaks[peak_num] + cycle_shift]
        signals_trans_peak.append(signal)
        # signal = normalize_signal(signal)

        i_baselines = find_tran_baselines(signal)
        i_baselines_full.append((i_peaks[peak_num] - cycle_shift) + i_baselines)
        i_act_signal = find_tran_act(signal)
        i_act_full = (i_peaks[peak_num] - cycle_shift) + i_act_signal
        i_acts_full.append(i_act_full)

    # TODO exclude those with abnormal rise times?

    # With that peak detection, find activation times and align transient
    for act_num, i_act_full in enumerate(i_acts_full):
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

        # signal_align = normalize_signal(signal_in[i_align:i_align + est_cycle_i])
        if crop is 'center':
            # center : crop transients using the cycle length
            # cropped to center at the alignment points

            # align along activation times, and crop using the first signal's baseline
            # i_baseline = i_baselines_full[0][-1]
            i_baseline = int(np.median(i_baselines_full[0]))
            i_align = i_act_full - (i_acts_full[0] - i_baseline)

            i_baseline = int(np.median(i_baselines_full[0]))
            if i_baseline < i_act_full:
                crop_l = i_baseline
            else:
                crop_l = 0  # not enough baselines before the act time, use everything before the peak

            # i_align = i_act_full - (i_acts_full[0] - crop_l)
            signal_align = signal_in[i_align:i_align + est_cycle_i]
        elif type(crop) is tuple:
            # stack : crop transients using the cycle length
            # cropped to allow for an ensemble stack with propagating transients

            # Use the earliest end of SNR in the frame

            # stacked to capture the second full transient
            # at the edge of a propogating wave and avoid sliced transients
            # align starting with provided crop times,
            i_align = i_act_full - (i_acts_full[0] - crop[0])
            signal_align = signal_in[i_align:i_align + (crop[1] - crop[0])]

        signal_align = normalize_signal(signal_align)
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

        # 1) Confirm the brightest pixel has enough peaks
        # 2) Find pixel(s) with earliest second peak
        # 3) Use the cycle time and end time of that peak's left baseline to align all ensembled signals

        Parameters
        ----------
        time_in : ndarray
            The array of timestamps (ms) corresponding to signal_in, dtyoe : int or float
        stack_in : ndarray
            A 3-D array (T, Y, X) of an optical transient, dtype : uint16 or float

        Returns
        -------
        stack_out : ndarray
             A spatially isolated 3-D array (T, Y, X) of optical data, dtype : float

        Notes
        -----
            Should not be applied to signal data containing at least one transient.
            Pixels with incalculable ensembles are assigned an array of zeros
        """

    print('Ensembling a stack ...')
    map_shape = stack_in.shape[1:]
    i_peak_0_min = stack_in.shape[0]
    yx_peak_1_min = (0, 0)
    i_peak_1_min = stack_in.shape[0]

    # for each pixel ...
    for iy, ix in np.ndindex(map_shape):
        print('\r\tPeak Search of Row:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(
            iy + 1, map_shape[0], ix + 1, map_shape[1]), end='', flush=True)
        # Get first half of signal to save time
        pixel_data = stack_in[:int(stack_in.shape[0]), iy, ix]
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

        if i_peaks is np.nan:
            continue
        if len(i_peaks) < 4:
            # raise ArithmeticError('No peaks detected'.format(len(i_peaks), i_peaks))
            # np.zeros_like(pixel_data)
            continue
        # if len(i_peaks) > 3:
        #     print('* {} peaks detected at {} in signal_in'.format(len(i_peaks), i_peaks))
        # else:
        #     raise ValueError('Only {} peak detected at {} in signal_in'.format(len(i_peaks), i_peaks))

        # 2) Find pixel(s) with earliest second peak
        # find the first peak and preserve the minimum among all pixels
        if i_peaks[0] < i_peak_0_min:
            i_peak_0_min = i_peaks[0]
            i_peak_1_min = i_peaks[1]
            yx_peak_1_min = (iy, ix)
        # i_peak_1_min = max(i_peaks[1], i_peak_1_min)

    # calculating alignment crop needed to preserve activation propagation
    pixel_data = stack_in[:, yx_peak_1_min[0], yx_peak_1_min[1]]

    i_peaks, _ = find_tran_peak(pixel_data, props=True)

    # Split up the signal using peaks and estimated cycle length
    est_cycle = np.diff(i_peaks).astype(float)
    est_cycle_i = np.nanmean(est_cycle)
    est_cycle_i = np.floor(est_cycle_i).astype(int)
    # est_cycle = est_cycle_i * np.nanmean(np.diff(time_in))
    # cycle_shift = np.floor(est_cycle_i / 2).astype(int)

    peak_1_min_crop = (i_peak_1_min - est_cycle_i, i_peak_1_min + est_cycle_i)
    pixel_data_peak_1_min = pixel_data[peak_1_min_crop[0]: peak_1_min_crop[1]]

    i_peak_1_min_baselines_l = find_tran_baselines(pixel_data_peak_1_min, peak_side='left')
    i_peak_1_min_baselines_r = find_tran_baselines(pixel_data_peak_1_min, peak_side='right')

    # ensemble_crop = (i_peak_1_min_baselines_l[-1], i_peak_1_min_baselines_r[1])
    ensemble_crop = (i_peak_1_min_baselines_l[1] + peak_1_min_crop[0],
                     i_peak_1_min_baselines_r[-1] + peak_1_min_crop[0])
    ensemble_crop_len = ensemble_crop[1] - ensemble_crop[0]

    # 3) Use the cycle time and time of that peak to align all ensembled signals
    # for each pixel ...
    stack_out = np.empty_like(stack_in[:ensemble_crop_len, :, :], dtype=float)

    for iy, ix in np.ndindex(map_shape):
        print('\r\tEnsemble of Row:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix + 1, map_shape[1]),
              end='', flush=True)
        # get signal
        pixel_data = stack_in[:, iy, ix]
        unique, counts = np.unique(pixel_data, return_counts=True)
        if len(unique) < 10:  # signal is too flat to have a valid peak
            signal_ensemble = np.zeros_like(pixel_data[:ensemble_crop_len])
        else:
            # calculate the ensemble of it
            time_ensemble, signal_ensemble, signals, signal_peaks, signal_acts, est_cycle_length \
                = calc_ensemble(time_in, pixel_data, crop=ensemble_crop)

        stack_out[:, iy, ix] = signal_ensemble

    ensemble_yx = yx_peak_1_min
    print('\nDONE Ensembling stack')

    return stack_out, ensemble_crop, ensemble_yx
