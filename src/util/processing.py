import statistics
import sys

import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import find_peaks, find_peaks_cwt, filtfilt, kaiserord, firwin, butter
from scipy.optimize import curve_fit
from skimage.morphology import square
# from skimage.restoration import denoise_tv_chambolle, estimate_sigma
from skimage.filters import gaussian
from skimage.filters.rank import median, mean, mean_bilateral, entropy

FILTERS_SPATIAL = ['median', 'mean', 'bilateral', 'gaussian', 'best_ever']
SPLINE_FIDELITY = 3
BASELINES_MIN = 10
BASELINES_MAX = 20
SNR_MAX = 100


# TODO add TV, a non-local, and a weird filter


def spline_signal(signal_in):
    xx_signal = np.arange(0, (len(signal_in)))
    # Lease Square approximation
    # Computing the inner knots and using them:
    x_spline = np.linspace(xx_signal[0], xx_signal[-1], len(xx_signal) * SPLINE_FIDELITY)
    n_knots = 35  # number of knots to use in LSQ spline
    t_knots = np.linspace(xx_signal[0], xx_signal[-1], n_knots)  # equally spaced knots in the interval
    t_knots = t_knots[2:-2]  # discard edge knots
    # t_knots = [0, 1, 2, 3]
    bspline_degree = 3
    # sql = make_lsq_spline(xx_signal, signal_in, t_knots)
    spline = LSQUnivariateSpline(xx_signal, signal_in, t_knots, k=bspline_degree)
    # x_signal = np.arange(len(signal_in))
    # x_spline = np.linspace(x_signal[0], x_signal[-1], len(x_signal) * SPLINE_FIDELITY)
    # # n_segments = int(len(x_signal) / 5)
    # n_segments = 25
    # n_knots = n_segments
    # knots = np.linspace(x_signal[0], x_signal[-1], n_knots + 2)[1:-2]
    # bspline_degree = 3
    # spline = LSQUnivariateSpline(x_signal, signal_in, knots, k=bspline_degree)
    return x_spline, spline

    # return xs, sql


def spline_deriv(signal_in):
    xs, sql = spline_signal(signal_in)

    x_df = xs
    df_spline = sql.derivative()(xs)

    return x_df, df_spline


def find_tran_peak(signal_in, props=False):
    """Find the index of the peak of a transient,
    defined as the maximum value

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float

        Returns
        -------
        i_peak : np.int64
            The index of the signal array corresponding to the peak of the transient
            or NaN if no peak was detected
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    # if signal_in.dtype not in [np.uint16, float]:
    #     raise TypeError('Signal values must either be "int" or "float"')

    # if any(v < 0 for v in signal_in):
    #     raise ValueError('All signal values must be >= 0')

    # Characterize the signal
    unique, counts = np.unique(signal_in, return_counts=True)
    if len(unique) < 10:  # signal is too flat to have a valid peak
        if props:
            return np.nan, np.nan
        else:
            return np.nan

    # Replace NaNs with 0
    # signal_in = np.nan_to_num(signal_in, copy=False, nan=0)

    signal_bounds = (signal_in.min(), signal_in.max())
    signal_mean = np.nanmean(signal_in)
    if signal_in.dtype is np.uint16:
        signal_mean = int(np.floor(np.nanmean(signal_in)))
    signal_range = signal_bounds[1] - signal_mean

    # TODO detect dual peaks, alternans, etc.

    # # Roughly find the peaks using a smoothing wavelet transformation
    # distance = int(len(signal_in) / 2)
    # i_peaks = find_peaks_cwt(signal_in, widths=np.arange(10, distance))
    # if len(i_peaks) is 0:  # no peak detected
    #     return np.nan
    # if len(i_peaks) > 1:
    #     return i_peaks[0]
    # return i_peaks

    # Roughly find the "prominent" peaks a minimum distance from eachother
    prominence = signal_range * 0.8
    distance = int(len(signal_in) / 2)
    i_peaks, properties = find_peaks(signal_in,
                                     height=signal_mean, prominence=prominence,
                                     distance=distance)
    if len(i_peaks) is 0:  # no peak detected
        if props:
            return np.nan, np.nan
        else:
            return np.nan

    # Use the peak with the max prominence (in case of a tie, first is chosen)
    i_peak = i_peaks[np.argmax(properties['prominences'])]
    if props:
        return i_peak, properties
    else:
        return i_peak


def find_tran_baselines(signal_in, peak_side='left'):
    # Characterize the signal
    # signal_bounds = (signal_in.min(), signal_in.max())
    # signal_range = signal_bounds[1] - signal_bounds[0]
    # find the peak (roughly)
    signal_range = signal_in.max() - signal_in.min()
    i_peak = find_tran_peak(signal_in)
    if i_peak is np.nan:
        return np.nan
    signal_cutoff = signal_in.min() + (signal_range / 2)
    # i_signal_cutoff_left = np.where(signal_in[:i_peak] <= signal_cutoff)[0][0]
    i_signal_cutoff_right = np.where(signal_in[:i_peak] <= signal_cutoff)[0][-1]

    # Exclude signals without a prominent peak

    # use the derivative spline to find relatively quiescent baseline period
    xdf, df_spline = spline_deriv(signal_in)
    # df_range = df_spline.max() - df_spline.min()

    # TODO catch atrial-type signals and limit to the plataea before the peak
    # find the df max before the signal's peak (~ large rise time)
    # df_max_search_left = int((i_peak * SPLINE_FIDELITY) * (1 / 2))
    df_search_left = SPLINE_FIDELITY * SPLINE_FIDELITY

    # include indexes within the standard deviation of the local area of the derivative
    df_sd = statistics.stdev(df_spline[df_search_left:-df_search_left])
    df_prominence_cutoff = df_sd * 2

    df_max_search_right = i_signal_cutoff_right * SPLINE_FIDELITY

    i_peak_df = df_search_left + np.argmax(df_spline[df_search_left:df_max_search_right])
    df_search_start_right = i_peak_df

    # i_min_df = df_search_left + np.argmin(df_spline[df_search_left:df_search_start_right])
    i_start_df = i_peak_df

    # find first value within cutoff
    df_spline_search = df_spline[:i_peak_df+1]
    for idx_flip, value in enumerate(np.flip(df_spline_search)):
        if abs(value) < df_prominence_cutoff:
            i_start_df = i_peak_df - idx_flip
            break

    i_left_df = i_start_df
    i_right_df = i_start_df
    # look left TODO allow to go further (higher cutoff?) to not overestimate noisy SNRs
    for value in np.flip(df_spline[df_search_left:i_start_df]):
        if abs(value) < df_prominence_cutoff:
            i_left_df = i_left_df - 1
        else:
            break
    # look right
    for value in df_spline[i_start_df:i_peak_df]:
        if abs(value) < df_prominence_cutoff:
            i_right_df = i_right_df + 1
        else:
            break
    # combine
    i_baselines_search = np.arange(i_left_df, i_right_df)

    if (i_right_df > i_peak_df) or (len(i_baselines_search) < (BASELINES_MIN * SPLINE_FIDELITY)):
        print('\n\t\t* df_cutoff: {} gives [{}:{}]\ti_start_df[{}]: {}\tfrom i_peak_df[{}]: {}'
              .format(round(df_prominence_cutoff, 3), i_left_df, i_right_df,
                      i_start_df, round(df_spline[i_start_df], 3),
                      i_peak_df, round(df_spline[i_peak_df], 3)))

        if i_right_df > i_peak_df:
            return np.nan

        # use arbitrary backup baselines: the 10 signal samples before the df search start (non-inclusive)
        i_right_df = int(i_right_df / SPLINE_FIDELITY)
        if i_right_df > BASELINES_MIN:
            i_baselines_backup = np.arange(i_right_df - BASELINES_MIN, i_right_df)
        else:
            i_baselines_backup = np.arange(0, BASELINES_MIN)
        return i_baselines_backup

    # use all detected indexes
    i_baselines_left = int(i_baselines_search[0] / SPLINE_FIDELITY)
    i_baselines_right = int(i_baselines_search[-1] / SPLINE_FIDELITY)

    i_baselines = np.arange(i_baselines_left, i_baselines_right)
    if len(i_baselines) > BASELINES_MAX:
        i_baselines = i_baselines[-BASELINES_MAX:]

    return i_baselines


def isolate_spatial(stack_in, roi):
    """Isolate a spatial region of a stack (3-D array, TYX) of grayscale optical data.

        Parameters
        ----------
        stack_in : ndarray, dtype : uint16 or float
             A 3-D array (T, Y, X) of optical data
        roi : `GraphicsItem <pyqtgraph.graphicsItems.ROI>`
             Generic region-of-interest widget.

        Returns
        -------
        stack_out : ndarray
             A spatially isolated 3-D array (T, Y, X) of optical data, dtype : stack_in.dtype
       """
    pass


def isolate_temporal(stack_in, i_start, i_end):
    """Isolate a temporal region of a stack (3-D array, TYX) of grayscale optical data.

        Parameters
        ----------
        stack_in : ndarray
             A 3-D array (T, Y, X) of optical data, dtype : uint16 or float
        i_start : int
             Index or frame to start temporal isolation
        i_end : int
             Index or frame to end temporal isolation

        Returns
        -------
        stack_out : ndarray
             A temporally isolated 3-D array (T, Y, X) of optical data, dtype : stack_in.dtype
        """
    pass


def isolate_transient(signal_in, i_start, i_end):
    """Isolate a single transient from a signal array of optical data.

        Parameters
        ----------
        signal_in : ndarray
             The array of data to be evaluated, dtype : uint16 or float
        i_start : int
             Index or frame to start transient isolation
        i_end : int
             Index or frame to end transient isolation

        Returns
        -------
        transient_out : ndarray
             The isolated array of transient data, dtype : signal_in.dtype
        """
    pass


def filter_spatial(frame_in, filter_type='gaussian', kernel=3):
    """Spatially filter a frame (2-D array, YX) of grayscale optical data.

        Parameters
        ----------
        frame_in : ndarray
             A 2-D array (Y, X) of optical data, dtype : uint16 or float
        filter_type : str
            The type of filter algorithm to use, default is gaussian
        kernel : int
            The width and height of the kernel used, must be positive and odd, default is 3

        Returns
        -------
        frame_out : ndarray
             A spatially filtered 2-D array (Y, X) of optical data, dtype : frame_in.dtype
        """
    # Check parameters
    if type(frame_in) is not np.ndarray:
        raise TypeError('Frame type must be an "ndarray"')
    if len(frame_in.shape) is not 2:
        raise TypeError('Frame must be a 2-D ndarray (Y, X)')
    if frame_in.dtype not in [np.uint16, float]:
        raise TypeError('Frame values must either be "np.uint16" or "float"')
    if type(filter_type) is not str:
        raise TypeError('Filter type must be a "str"')
    if type(kernel) is not int:
        raise TypeError('Kernel size must be an "int"')

    if filter_type not in FILTERS_SPATIAL:
        raise ValueError('Filter type must be one of the following: {}'.format(FILTERS_SPATIAL))
    if kernel < 3 or (kernel % 2) == 0:
        raise ValueError('Kernel size {} px must be >= 3 and odd'.format(kernel))

    if filter_type is 'median':
        # Good for ___, but ___
        # k = np.full([kernel, kernel], 1)
        frame_out = median(frame_in, square(kernel))
    elif filter_type is 'mean':
        # Good for ___, but over-smooths?
        # k = np.full([kernel, kernel], 1)
        frame_out = mean(frame_in, square(kernel))
    elif filter_type is 'bilateral':
        # Good for edge preservation, but slow
        # sigma_color = 50  # standard deviation of the intensity gaussian kernel
        # sigma_space = 10  # standard deviation of the spatial gaussian kernel
        frame_out = mean_bilateral(frame_in, square(kernel))
    elif filter_type is 'gaussian':
        # Good for ___, but ___
        sigma = kernel  # standard deviation of the gaussian kernel
        frame_out = gaussian(frame_in, sigma=sigma, mode='mirror', preserve_range=True)
    else:
        raise NotImplementedError('Filter type "{}" not implemented'.format(filter_type))

    return frame_out.astype(frame_in.dtype)


def filter_temporal(signal_in, sample_rate, freq_cutoff=100.0, filter_order='auto'):
    """Apply a lowpass filter to an array of optical data.

        Parameters
        ----------
        signal_in : ndarray
             The array of data to be evaluated, dtype : uint16 or float
        sample_rate : float
            Sample rate (Hz) of signal_in
        freq_cutoff : float
            Cutoff frequency (Hz) of the lowpass filter, default is 100
        filter_order : int or str
            The order of the filter, default is 'auto'
            If 'auto', order is calculated using scipy.signal.kaiserord

        Returns
        -------
        signal_out : ndarray
             A temporally filtered signal array, dtype : signal_in.dtype
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "uint16" or "float"')
    if type(sample_rate) is not float:
        raise TypeError('Sample rate must be a "float"')
    if type(freq_cutoff) is not float:
        raise TypeError('Cutoff frequency must be a "float"')
    if type(filter_order) not in [int, str]:
        raise TypeError('Filter type must be an int or str')

    nyq_rate = sample_rate / 2.0
    n_order = 0

    if type(filter_order) is int:
        # Good for ___, but ___
        # Butterworth (from old code)
        Wn = freq_cutoff / nyq_rate
        n_order = filter_order
        [b, a] = butter(n_order, Wn)
        signal_out = filtfilt(b, a, signal_in)

        # # FIR design arguements
        # Fs = sample_rate           # sample-rate, down-sampled
        # Norder = filter_order
        # Ntaps = Norder + 1   # The desired number of taps in the filter
        # Fpass = 95       # passband edge
        # Fstop = 105     # stopband edge, transition band 100kHz
        # Wp = Fpass/Fs    # pass normalized frequency
        # Ws = Fstop/Fs    # stop normalized frequency
        # taps = ffd.remez(Ntaps, [0, Wp, Ws, .5], [1, 0], maxiter=10000)
        # # FIR design arguements
        # Fpass = 100  # passband edge
        # Fstop = 105  # stopband edge, transition band __ Hz
        # R = 25  # how much to down sample by
        # Fsr = Fs / 25.  # down-sampled sample rate
        # xs = resample(signal_in, int(len(signal_in) / 25.))
        # # Down sampled version, create new filter and plot spectrum
        # R = 4.             # how much to down sample by
        # Fsr = Fs/R          # down-sampled sample rate
        # Wp = Fpass / Fsr  # pass normalized frequency
        # Ws = Fstop / Fsr  # stop normalized frequency

        # signal_filt = lfilter(taps, 1, signal_in)
        # # signal_filt = filtfilt(taps, 1, signal_in, method="gust")
        # # signal_filt = minimum_phase(signal_filt, method='hilbert')
        # signal_out = signal_filt[Norder-1:]

        # ############

    elif filter_order is 'auto':
        # # FIR 4 design  -
        # https://www.programcreek.com/python/example/100540/scipy.signal.firwin
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
        # Compute the order and Kaiser parameter for the FIR filter.
        ripple_db = 30.0
        width = 20  # The desired width of the transition from pass to stop, Hz
        window = 'kaiser'
        n_order, beta = kaiserord(ripple_db, width / nyq_rate)
        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        taps = firwin(numtaps=n_order + 1, cutoff=freq_cutoff, window=(window, beta), fs=sample_rate)
        # signal_out = lfilter(taps, 1.0, signal_in)   # for FIR, a=1
        signal_out = filtfilt(taps, 1, signal_in, method="gust")  # for FIR, a=1
        # # Savitzky Golay
        # window_coef = int(nyq_rate / 50)
        #
        # if window_coef % 2 > 0:
        #     window = window_coef
        # else:
        #     window = window_coef + 1
        #
        # signal_out = savgol_filter(signal_in, window, 3)
    else:
        raise ValueError('Filter order "{}" not implemented'.format(filter_order))

    # # Calculate the phase delay of the filtered signal
    # phase_delay = 0.5 * (filter_order - 1) / sample_rate
    # delay = phase_delay * 1000

    return signal_out.astype(signal_in.dtype)


def filter_drift(signal_in, drift_order=2):
    """Remove drift from an array of optical data using the subtraction of a polynomial fit.

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float
        drift_order : int or str
            The order of the polynomial drift to fit to, default is 'exp'
            If 'exp', drift is calculated using scipy.optimize.curve_fit

        Returns
        -------
        signal_out : ndarray
            A signal array with drift removed, dtype : signal_in.dtype
        drift : ndarray
            The values of the calculated polynomial drift used
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an ndarray')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values type must either be uint16 or float')
    if type(drift_order) not in [int, str]:
        raise TypeError('Drift order must be a "exp" or an int')

    if type(drift_order) is int:
        if (drift_order < 1) or (drift_order > 5):
            raise ValueError('Drift order must be "exp" or an "int" >= 1 and <= 5')
    if type(drift_order) is str:
        if drift_order is not 'exp':
            raise ValueError('Drift order "{}" not implemented'.format(drift_order))

    def func_exp(x, a, b, c):
        return a * np.exp(-b * x) + c  # a decaying exponential curve to fit to

    # TODO drift model must be fit to baseline data (pre & post transient)
    drift_range = signal_in.max() - signal_in.min()
    drift_out = np.zeros_like(signal_in)

    if drift_range < 5:  # signal is too flat to remove drift
        return signal_in, drift_out

    drift_x = np.arange(start=0, stop=len(signal_in))
    exp_b_estimates = (0.01, 0.1)  # assumed bounds of the B constant for a decaying exponential fit

    if type(drift_order) is int:
        # np.polyfit : Least squares polynomial fit
        poly = np.poly1d(np.polyfit(drift_x, signal_in, drift_order))
        poly_y = poly(drift_x)
    else:
        # scipy.optimize.curve_fit : Use non-linear least squares to fit a function, f, to data
        exp_bounds_lower = [0, exp_b_estimates[0], signal_in.min()]
        exp_bounds_upper = [drift_range * 2, exp_b_estimates[1], signal_in.max()]
        try:
            # popt, pcov = curve_fit(func_exp, drift_x, signal_in)
            popt, pcov = curve_fit(func_exp, drift_x, signal_in, bounds=(exp_bounds_lower, exp_bounds_upper))
        except Exception:
            exctype, exvalue, traceback = sys.exc_info()
            print("\t* Failed to calculate signal drift:\n\t" + str(exctype) + ' : ' + str(exvalue) +
                  '\n\t\t' + str(traceback))
            return signal_in, drift_out

        poly_y = func_exp(drift_x, *popt)

    # # linalg.lstsq : Computes a least-squares fit
    # A = np.vstack([drift_x, np.ones(len(drift_x))]).T
    # poly = np.poly1d(linalg.lstsq(A, signal_in)[0])
    # poly_y = poly(drift_x)

    # # scipy.interpolate.UnivariateSpline : Computes spline fits
    # spl = UnivariateSpline(drift_x, signal_in)
    # spl.set_smoothing_factor(2)
    # poly_y = spl(drift_x)

    signal_out = signal_in - poly_y + poly_y.min()
    drift_out = poly_y

    return signal_out.astype(signal_in.dtype), drift_out


def invert_signal(signal_in):
    """Invert the values of a signal array.

        Parameters
        ----------
        signal_in : ndarray
             The array of data to be processed, dtype : uint16 or float

        Returns
        -------
        signal_out : ndarray
             The inverted signal array, dtype : signal_in.dtype
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "uint16" or "float"')

    unique, counts = np.unique(signal_in, return_counts=True)
    if len(unique) is 1:
        return signal_in

    # calculate axis to rotate data around (middle value int or float)
    axis = signal_in.min() + ((signal_in.max() - signal_in.min()) / 2)
    if signal_in.dtype in [np.int32]:
        axis = np.floor(axis).astype(int)

    # rotate the data around it's central value
    signal_out = (axis + (axis - signal_in)).astype(signal_in.dtype)

    return signal_out


def invert_stack(stack_in):
    """Invert the values of an image stack (3-D array).

        Parameters
        ----------
        stack_in : ndarray
            Image stack with shape (T, Y, X)

        Returns
        -------
        stack_out : ndarray
            A cropped 3-D array (T, Y, X) of optical data, dtype : float
        """
    # Check parameters
    if type(stack_in) is not np.ndarray:
        raise TypeError('Stack type must be an "ndarray"')
    if len(stack_in.shape) is not 3:
        raise TypeError('Stack must be a 3-D ndarray (T, Y, X)')
    if stack_in.dtype not in [np.uint16, float]:
        raise TypeError('Stack values must either be "np.uint16" or "float"')

    stack_out = np.empty_like(stack_in)
    map_shape = stack_in.shape[1:]
    # Assign a value to each pixel
    for iy, ix in np.ndindex(map_shape):
        print('\r\tInversion of Row:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix + 1, map_shape[1]),
              end='', flush=True)
        pixel_data = stack_in[:, iy, ix]
        pixel_data_inv = invert_signal(pixel_data)
        stack_out[:, iy, ix] = pixel_data_inv

    return stack_out


def normalize_signal(signal_in):
    """Normalize the values of a signal array to range from 0 to 1.

        Parameters
        ----------
        signal_in : ndarray
             The array of data to be processed, dtype : uint16 or float

        Returns
        -------
        signal_out : ndarray
             The normalized signal array, dtype : float
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    # if signal_in.dtype not in [np.uint16, float]:
    #     raise TypeError('Signal values must either be "uint16" or "float"')

    # if any(v < 0 for v in signal_in):
    #     raise ValueError('All signal values must be >= 0')

    unique, counts = np.unique(signal_in, return_counts=True)

    if len(unique) < 10:  # signal is too flat to have a valid peak
        return np.zeros_like(signal_in)

    xp = [signal_in.min(), signal_in.max()]
    fp = [0, 1]
    signal_out = np.interp(signal_in, xp, fp)

    return signal_out


def normalize_stack(stack_in):
    """Normalize the values of an image stack (3-D array) to range from 0 to 1..

        Parameters
        ----------
        stack_in : ndarray
            Image stack with shape (T, Y, X), dtype : uint16 or float

        Returns
        -------
        stack_out : ndarray
            A normalized image stack (T, Y, X), dtype : float
        """
    # Check parameters
    if type(stack_in) is not np.ndarray:
        raise TypeError('Stack type must be an "ndarray"')
    if len(stack_in.shape) is not 3:
        raise TypeError('Stack must be a 3-D ndarray (T, Y, X)')
    if stack_in.dtype not in [np.uint16, float]:
        raise TypeError('Stack values must either be "np.uint16" or "float"')

    stack_out = np.empty_like(stack_in, dtype=float)
    map_shape = stack_in.shape[1:]
    # Assign a value to each pixel
    for iy, ix in np.ndindex(map_shape):
        print('\r\tNormalizing Row:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix, map_shape[1]),
              end='', flush=True)
        pixel_data = stack_in[:, iy, ix]
        pixel_data_norm = normalize_signal(pixel_data)
        stack_out[:, iy, ix] = pixel_data_norm

    return stack_out


def calc_ff0(signal_in):
    """Normalize a fluorescence signal against a resting fluorescence,
    i.e. F_t / F0

        Parameters
        ----------
        signal_in : ndarray
            The array of fluorescent data (F_t) to be normalized, dtype : uint16 or float
        Returns
        -------
        signal_out : ndarray
            The array of F/F0 fluorescence data, dtype : float

        Notes
        -----
            Should not be applied to normalized or drift-removed data.
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "int" or "float"')

    # F / F0: (F_t - F0) / F0
    f_t = signal_in
    f_0 = signal_in.min()

    signal_out = (f_t - f_0) / f_0

    return signal_out


def calculate_snr(signal_in, noise_count=10):
    """Calculate the Signal-to-Noise ratio of a signal array,
    defined as the ratio of the Peak-Peak amplitude to the sample standard deviation of the noise.

        Parameters
        ----------
        signal_in : ndarray
             The array of data to be evaluated, dtyoe : int or float
        noise_count : int
             The number of noise values to be used in the calculation, default is 10

        Returns
        -------
        snr : float
             The Signal-to-Noise ratio of the given data, recommend using round(snr, 5)
        rms_bounds : tuple
             The RMSs of the peak and noise arrays, (noise_rms, peak_rms)
        peak_peak : float
             The absolute difference between the RMSs of the peak and noise arrays
        ratio_noise : float
             The ratio between the ranges of noise and peak value(s)
        sd_noise : float
             The standard deviation of the noise values
        ir_noise : ndarray
             The indexes of noise values used in the calculation
        ir_peak : int
             The index of peak values used in the calculation

        Notes
        -----
            Must be applied to signals with upward deflections (Peak > noise).  TODO expand to use on negative signals
            Assumes noise SD > 1, otherwise set to 0.5
            Assumes max noise value < (peak / 5)
            Auto-detects noise section as the last noise_count values before the final noisy peak.
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    # if signal_in.dtype not in [np.uint16, float]:
    #     raise TypeError('Signal values must either be "uint16" or "float"')
    if type(noise_count) is not int:
        raise TypeError('Number of noise values to use must be an "int"')

    # if any(v < 0 for v in signal_in):
    #     raise ValueError('All signal values must be >= 0')
    if noise_count < 0:
        raise ValueError('Noise count must be >= 0')
    if noise_count >= len(signal_in):
        raise ValueError('Number of noise values to use must be < length of signal array')

    # Find peak values
    i_peak, properties = find_tran_peak(signal_in, props=True)
    if i_peak is np.nan:
        # raise ArithmeticError('No peaks detected'.format(len(i_peaks), i_peaks))
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Characterize the signal
    signal_bounds = (signal_in.min(), signal_in.max())

    i_peak_calc = i_peak
    ir_peak = i_peak_calc

    # Use the peak value
    peak_value = signal_in[i_peak_calc]

    # Find noise values
    i_noise_calc = find_tran_baselines(signal_in)

    if i_noise_calc is np.nan or len(i_noise_calc) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    data_noise = signal_in[i_noise_calc]

    # Use noise values and their RMS
    noise_rms = np.sqrt(np.mean(data_noise) ** 2)
    noise_sd = statistics.stdev(data_noise.astype(float))  # standard deviation

    # Calculate Peak-Peak value
    peak_peak = abs(peak_value - noise_rms)

    # Exclusions
    if noise_sd == 0:
        noise_sd = peak_peak / 200  # Noise data too flat to detect SD
        print('\tFound noise with SD of 0! Used {} to give max SNR of 200'.format(noise_sd))

    if signal_bounds[1] < noise_rms:
        raise ValueError('Signal max {} seems to be < noise rms {}'.format(signal_bounds[1], noise_rms))

    # Calculate SNR
    snr = peak_peak / noise_sd

    rms_bounds = (noise_rms.astype(signal_in.dtype), peak_value.astype(signal_in.dtype))
    sd_noise = noise_sd
    ir_noise = i_noise_calc
    return snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak


def map_snr(stack_in, noise_count=10):
    """Generate a map_out of Signal-to-Noise ratios for signal arrays within a stack,
    defined as the ratio of the Peak-Peak amplitude to the population standard deviation of the noise.

        Parameters
        ----------
        stack_in : ndarray
            A 3-D array (T, Y, X) of optical data, dtype : uint16 or float
        noise_count : int
             The number of noise values to be used in the calculation, default is 10

        Returns
        -------
        map : ndarray
             A 2-D array of Signal-to-Noise ratios, dtype : float

        Notes
        -----
            Pixels with incalculable SNRs assigned a value of NaN
        """
    # Check parameters
    if type(stack_in) is not np.ndarray:
        raise TypeError('Stack type must be an "ndarray"')
    if len(stack_in.shape) is not 3:
        raise TypeError('Stack must be a 3-D ndarray (T, Y, X)')
    if stack_in.dtype not in [np.uint16, float]:
        raise TypeError('Stack values must either be "np.uint16" or "float"')

    if type(noise_count) is not int:
        raise TypeError('Noise count must be an "int"')

    print('Generating SNR map ...')
    map_shape = stack_in.shape[1:]
    map_out = np.empty(map_shape)
    # Assign an SNR to each pixel
    for iy, ix in np.ndindex(map_shape):
        print('\r\tRow:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix + 1, map_shape[1]),
              end='', flush=True)
        pixel_data = stack_in[:, iy, ix]
        # # Characterize the signal
        # signal_bounds = (pixel_data.min(), pixel_data.max())
        # signal_range = signal_bounds[1] - signal_bounds[0]

        snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak = calculate_snr(pixel_data, noise_count)
        # Set every pixel's values to the SNR of the signal at that pixel
        map_out[iy, ix] = snr

    print('\nDONE Mapping SNR')
    return map_out


def calculate_error(ideal, modified):
    """Calculate the amount of error created by signal modulation or filtering,
    defined as (Modified - Ideal) / Ideal X 100%.
    # defined as (Ideal - Modified) / Ideal X 100%.

        Parameters
        ----------
        ideal : ndarray
             An array of ideal data
        modified : ndarray
             An array of modified data

        Returns
        -------
        error : ndarray
             An array of percent errors
        error_mean : float
             The mean value of the percent error array
        error_sd : float
             The standard deviation of the percent error array
        """
    # Check parameters
    if type(ideal) is not np.ndarray:
        raise TypeError('Ideal data type must be an "ndarray"')
    if ideal.dtype not in [int, np.uint16, float]:
        raise TypeError('Ideal values must either be "int", "uint16" or "float"')
    if type(modified) is not np.ndarray:
        raise TypeError('Modified data type must be an "ndarray"')
    if modified.dtype not in [int, np.uint16, float]:
        raise TypeError('Modified values must either be "int", "uint16" or "float"')

    # MIN = 1  # Min to avoid division by 0

    error = ((modified.astype(float) - ideal.astype(float)) / (ideal.astype(float)) * 100)
    error_mean = error.mean()
    error_sd = statistics.stdev(error)

    return error, error_mean, error_sd
