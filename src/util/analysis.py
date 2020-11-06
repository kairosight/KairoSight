from util.processing import *
import time
import numpy as np
from scipy.signal import savgol_filter
from scipy.misc import derivative
from scipy.interpolate import UnivariateSpline

# Constants
# Transient feature limits (ms)
TRAN_MAX = 500
RISE_MAX = 150
# Colormap and normalization limits for Activation maps (ms)
ACT_MAX = 150
# Colormap and normalization limits for Duration maps (ms)
DUR_MIN = 20
DUR_MAX = 300
# Colormap and normalization limits for EC Coupling maps (ms)
EC_MAX = 50


# TODO finish remaining analysis point algorithms
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
    if signal_in.dtype not in [np.uint16, float, np.float32]:
        raise TypeError('Signal values must either be "int" or "float"')

    # if any(v < 0 for v in signal_in):
    #     raise ValueError('All signal values must be >= 0')

    # Limit search to
    # before the peak and after the mid-baseline began
    baselines = find_tran_baselines(signal_in)
    i_search_l = baselines[int(len(baselines) / 2)]
    i_search_r = find_tran_peak(signal_in)

    xdf, df_spline = spline_deriv(signal_in)
    xdf2, df2_spline = spline_deriv(df_spline)
    # find the 2nd derivative max within the search area
    i_df_start = np.argmax(df2_spline[i_search_l * SPLINE_FIDELITY * SPLINE_FIDELITY:
                                      i_search_r * SPLINE_FIDELITY * SPLINE_FIDELITY])
    i_start = i_search_l + int(i_df_start / SPLINE_FIDELITY / SPLINE_FIDELITY)

    return i_start


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
    if signal_in.dtype not in [np.uint16, np.float32]:
        raise TypeError('Signal values must either be "int" or "float"')

    # Limit search to after the peak and before the end
    i_peak = find_tran_peak(signal_in)
    search_min = i_peak
    search_max = len(signal_in) - SPLINE_FIDELITY
    xdf, df_spline = spline_deriv(signal_in)
    # find the 2nd derivative max within the search area
    i_start_search = int(np.argmin(
        df_spline[search_min * SPLINE_FIDELITY:search_max * SPLINE_FIDELITY]) / SPLINE_FIDELITY)  # df min, Downstroke
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
    if signal_in.dtype not in [np.uint16, np.float32]:
        raise TypeError('Signal values must either be "int" or "float"')

    i_downstroke = find_tran_downstroke(signal_in)

    # Limit search to after the Downstroke and before a the end of the ending baseline
    i_search_l = i_downstroke

    snr, rms_bounds, peak_peak, sd_noise, ir_noise, i_peak = calculate_snr(signal_in)
    cutoff = rms_bounds[0]  # TODO use baseline LSQ spline to the RIGHT of the peak

    i_search = i_peak + np.where(signal_in[i_peak:] <= cutoff)
    if len(i_search) == 0:
        return np.nan  # exclusion criteria: transient does not return to cutoff value
    if len(i_search[0]) == 0:
        return np.nan  # exclusion criteria: transient does not return to cutoff value
    i_search_r = i_search[0][-1]
    # search_min = i_peak
    # search_max = len(signal_in) - SPLINE_FIDELITY

    xdf, df_spline = spline_deriv(signal_in)
    xdf2, df2_spline = spline_deriv(df_spline)
    # find the 2nd derivative max within the search area
    i_df_start = np.argmax(df2_spline[i_search_l * SPLINE_FIDELITY * SPLINE_FIDELITY:
                                      i_search_r * SPLINE_FIDELITY * SPLINE_FIDELITY])
    i_end = i_search_l + int(i_df_start / SPLINE_FIDELITY / SPLINE_FIDELITY)

    return i_end


def calc_tran_activation(signal_in, start_time, end_time):
    """Calculate the time of the activation of a transient,
    defined as the midpoint (not limited by sampling rate) between the start
    and peak times

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float
        start_time: np.float
            The user defined start time index for analysis of activation times
        end_time: np.float
            The user defined end time index for analysis of activation times

        Returns
        -------
        i_activation : np.int64
            The index of the signal array corresponding to the activation of
            the transient
        """
    # Calculate the derivative
    dVdt = signal_in[1:-1, :, :]-signal_in[0:-2, :, :]
    # Find the indices of the maximums within a given window
    act_ind = np.argmax(dVdt[start_time:end_time, :, :], axis=0)
    # Return the solution
    return act_ind
    '''# Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "int" or "float"')
    if any(v < 0 for v in signal_in):
        raise ValueError('All signal values must be >= 0')
    # Grab the user specified segment of the signal (i.e., data of interest)
    doi = signal_in[start_time:end_time]
    # Calculate the first derivative of the signal
    dVdt = doi[start_time:end_time]
    pass'''



def calc_tran_duration(signal_in, percent=80):
    """Calculate the duration of a transient,
    defined as the number of indices between the activation time
    and the time nearest a percentage of the peak-to-peak range (e.g. APD-80, APD-90, CAD-80 ...)

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
    if signal_in.dtype not in [np.uint16, np.float32, np.float64]:
        raise TypeError('Signal values must either be "int" or "float"')
    if type(percent) is not int:
        raise TypeError('Percent data type must be an "int"')
    if percent < 0 or percent >= 100:
        raise ValueError('Percent must be between 0-99%')

    xs, signal_sql = spline_signal(signal_in)
    signal_spline = signal_sql(xs)

    i_peak = find_tran_peak(signal_in)
    i_baselines = find_tran_baselines(signal_in)
    if i_baselines is np.nan:
        return np.nan
    baselines_rms = np.sqrt(np.mean(signal_in[i_baselines]) ** 2)
    peak_peak = signal_in[i_peak] - baselines_rms
    cutoff = baselines_rms + (float(peak_peak) * float(((100 - percent) / 100)))

    i_spline_search = np.where(signal_spline[i_peak * SPLINE_FIDELITY:] <= cutoff)

    if len(i_spline_search) == 0 or len(i_spline_search[0]) == 0:
        return np.nan  # exclusion criteria: transient does not return to cutoff value
    i_cutoff = i_peak + int(i_spline_search[0][0] / SPLINE_FIDELITY)

    i_activation = find_tran_act(signal_in)
    if i_activation > i_peak or i_activation is np.nan:
        return np.nan  # exclusion criteria: peak seems to before activation
    duration = i_cutoff - i_activation

    if duration < DUR_MIN:
        return np.nan  # exclusion criteria: transient does not return to cutoff value

    return duration


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
    if signal_in.dtype not in [np.uint16, np.float32]:
        raise TypeError('Signal values must either be "int" or "float"')


def calc_tran_di(signal_in):
    """Calculate the diastolic interval (DI) of a transient,
    defined as the number of indices between this transient's
    and the next transient's activation

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
    if signal_in.dtype not in [np.uint16, np.float32]:
        raise TypeError('Signal values must either be "int" or "float"')


def map_tran_analysis(stack_in, analysis_type, time_in=None, raw_data=False, **kwargs):
    """Map an analysis point's values for a stack of transient fluorescent data
        i.e.

        Parameters
        ----------
        stack_in : ndarray
            A 3-D array (T, Y, X) of an optical transient, dtype : uint16 or float
        analysis_type : function
            The type of analysis to be mapped
        time_in : ndarray, optional
            The array of timestamps (ms) corresponding to signal_in, dtyoe : int or float
            If used, map values are timestamps
        raw_data : ndarray, optional
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
    if len(stack_in.shape) != 3:
        raise TypeError('Stack must be a 3-D ndarray (T, Y, X)')
    if stack_in.dtype not in [np.uint16, float]:
        raise TypeError('Stack values must either be "np.uint16" or "float"')

    # if type(analysis_type) is not classmethod:
    #     raise TypeError('Analysis type must be a "classmethod"')

    # print('Generating map with {} ...'.format(analysis_type))
    map_shape = stack_in.shape[1:]
    map_out = np.empty(map_shape)

    # # Calculate with parallel processing
    # with Pool(5) as p:
    #     p.map()

    # Assign a value to each pixel
    for iy, ix in np.ndindex(map_shape):
        # print('\r\tRow:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix + 1, map_shape[1]), end='',
        #       flush=True)
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
            analysis_result = analysis_type(pixel_data, **kwargs)
            if analysis_result is np.nan:
                map_out[iy, ix] = np.nan
            else:
                if time_in is not None:
                    pixel_analysis_value = time_in[analysis_result]  # TODO catch issue w/ duration when t[0] != 0
                else:
                    pixel_analysis_value = analysis_result
                map_out[iy, ix] = pixel_analysis_value

    # If mapping activation, align times with the "first" aka lowest activation time
    if analysis_type is find_tran_act and raw_data is False:
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
    ranging from -pi to +pi

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


def calc_coupling(signal_vm, signal_ca):
    """Find the Excitation-Contraction (EC) coupling time,
    defined as the difference between voltage and calcium activation times

        Parameters
        ----------
        signal_vm : ndarray
            The array of optical voltage data to be evaluated, dtype : uint16 or float
        signal_ca : ndarray
            The array of optical calcium data to be evaluated, dtype : uint16 or float

        Returns
        -------
        coupling : np.int64
            The length of EC coupling time in number of indices
        """
    # Check parameters
    if type(signal_vm) is not np.ndarray or type(signal_ca) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_vm.dtype not in [np.uint16, np.float32, np.float64]\
            or signal_ca.dtype not in [np.uint16, np.float32, np.float64]:
        raise TypeError('Signal values must either be "int" or "float"')

    signal_vm = normalize_signal(signal_vm)
    signal_ca = normalize_signal(signal_ca)

    vm_activation = find_tran_act(signal_vm)
    ca_activation = find_tran_act(signal_ca)

    if vm_activation is np.nan or ca_activation is np.nan:
        return np.nan

    coupling = ca_activation - vm_activation

    return coupling


def map_coupling(map_vm, map_ca):
    """Find the Excitation-Contraction (EC) coupling time,
    defined as the difference between voltage and calcium activation times

        Parameters
        ----------
        map_vm : ndarray
            The array of optical voltage data to be evaluated, dtype : uint16 or float
        map_ca : ndarray
            The array of optical calcium data to be evaluated, dtype : uint16 or float

        Returns
        -------
        map_coupling : ndarray
            A 2-D array of analysis values, dtype : uint16 or float
        """
    # Check parameters

    if type(map_vm) is not np.ndarray or type(map_ca) is not np.ndarray:
        raise TypeError('Map data type must be an "ndarray"')
    if len(map_vm.shape) != 2 or len(map_ca.shape) != 2:
        raise TypeError('Maps must be a 2-D ndarray (Y, X)')
    if map_vm.shape != map_ca.shape:
        raise ValueError('Maps must be a 2-D ndarray (Y, X)')
    if map_vm.dtype not in [np.uint16, np.float32, np.float64]\
            or map_ca.dtype not in [np.uint16, np.float32, np.float64]:
        raise TypeError('Map values must either be "int" or "float"')

    map_ec = np.empty_like(map_ca)

    for x, row in enumerate(map_vm):
        for y, val_vm in enumerate(row):
            val_ca = map_ca[x][y]

            if val_vm == np.nan or val_ca == np.nan or val_vm > val_ca:
                map_ec[x][y] = np.nan
                continue

            val_ec = val_ca - val_vm
            if val_ec < 0 or val_ec > EC_MAX:
                map_ec[x][y] = np.nan
                continue

            map_ec[x][y] = val_ec

    return map_ec
