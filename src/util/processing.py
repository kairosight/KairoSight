import statistics
import numpy as np
from numpy import linalg
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.signal import find_peaks, resample, filtfilt, kaiserord, firwin, firwin2, \
    lfilter, butter, freqs, freqz, minimum_phase, savgol_filter
from scipy.optimize import curve_fit
from skimage.morphology import square
from skimage.restoration import denoise_tv_chambolle, estimate_sigma
from skimage.filters import gaussian
from skimage.filters.rank import median, mean, mean_bilateral
FILTERS_SPATIAL = ['median', 'mean', 'bilateral', 'gaussian', 'best_ever']
# TODO add TV, a non-local, and a weird filter


def spline_signal(xx, signal_in, smoothing=3):
    d_xx = xx[2] - xx[1]
    xx_signal = np.arange(0, (len(xx)))
    spl = InterpolatedUnivariateSpline(xx_signal, signal_in, ext='extrapolate')
    # df/dt (with X__ as many samples)
    spline_fidelity = 200    # TODO optimize here
    # time_spline = np.linspace(xx[0], xx[-1] - d_xx,
    #                           (len(xx))*spline_fidelity)
    # xx_spline = np.linspace(0, 1, (len(xx)) * spline_fidelity)
    xx_spline = np.arange(0, len(xx_signal) - d_xx, d_xx / spline_fidelity)
    spl_array = spl(xx_spline)
    # spl.set_smoothing_factor(smoothing)    # TODO optimize here
    df_spline = spl.derivative()(xx_spline)
    # df_spline = spl(xx_spline, nu=1, ext='extrapolate')

    return df_spline, spline_fidelity


def find_tran_peak(signal_in, props=False):
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
            or NaN if no peak was detected
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "int" or "float"')

    # if any(v < 0 for v in signal_in):
    #     raise ValueError('All signal values must be >= 0')

    # Characterize the signal
    signal_bounds = (signal_in.min(), signal_in.max())
    signal_range = signal_bounds[1] - signal_bounds[0]
    unique, counts = np.unique(signal_in, return_counts=True)

    if len(unique) < 10:    # signal is too flat to have a valid peak
        if props:
            return np.nan, np.nan
        else:
            return np.nan

    # Roughly find the peaks
    prominence = signal_range * 0.4
    i_peaks, properties = find_peaks(signal_in, prominence=prominence, distance=20)

    if len(i_peaks) is 0:   # no peak detected
        if props:
            return np.nan, np.nan
        else:
            return np.nan

    if props:
        return i_peaks, properties
    else:
        # Use the peak with the max prominence (in case of a tie, first is chosen)
        i_peak = i_peaks[np.argmax(properties['prominences'])]
        return i_peak


def find_tran_baselines(signal_in, peak_side='left'):
    # Characterize the signal
    signal_bounds = (signal_in.min(), signal_in.max())
    signal_range = signal_bounds[1] - signal_bounds[0]
    # find the peak (roughly)
    i_peaks, properties = find_tran_peak(signal_in, props=True)
    if i_peaks is np.nan:
        return np.nan
    # if type(i_peaks) is float:
    #     raise ArithmeticError('float i_peaks is: \'{}\' for signal with range: '
    #                           .format(i_peaks, signal_range))
    i_peak = i_peaks[np.argmax(properties['prominences'])]

    # use the prominence of the peak to find a "rough" baseline
    # assumes SNR > 2.85
    prominence_floor = signal_in[i_peak] - (properties['prominences'][0] * 0.65)
    i_baselines_far_l, i_baselines_far_r = 0, 0

    if peak_side is 'left':
        # Find first index below the prominence floor
        i_baselines_far_l = np.where(signal_in[:i_peak] <= prominence_floor)[0][0]
        i_baselines_far_r = np.where(signal_in[:i_peak] <= prominence_floor)[0][-1]

    if peak_side is 'right':
        # Find first index below the prominence floor
        i_baselines_far_l = np.where(signal_in[i_peak:] <= prominence_floor)[0][0] + i_peak
        i_baselines_far_r = np.where(signal_in[i_peak:] <= prominence_floor)[0][-1] + i_peak

    i_baselines_all = np.arange(i_baselines_far_l, i_baselines_far_r+1)

    if len(i_baselines_all) < 20:
        if peak_side is 'left':
            # attempt on right side
            i_baselines_all_r = find_tran_baselines(signal_in, peak_side='right')
            if len(i_baselines_all_r) < len(i_baselines_all):
                i_baselines_all = i_baselines_all
            else:
                i_baselines_all = i_baselines_all_r

    signal_baseline = signal_in[i_baselines_all]
    xx_baseline = np.linspace(0, len(signal_baseline) - 1, len(signal_baseline))

    # # use a spline of the rough baseline and find the "flattest" section
    df_spline, spline_fidelity = spline_signal(xx_baseline, signal_baseline)

    d1f_sd = statistics.stdev(df_spline)    # TODO optimize here
    d1f_prominence_floor = d1f_sd * 1.5
    # if d1f_sd < d1f_prominence_floor:
    #     # where the derivative is less than (min * 2)
    #     i_baselines_search = np.where(df_spline <= d1f_prominence_floor)[0]
    # else:
    #     # where the derivative is less than its standard deviation
    i_baselines_search = np.where(abs(df_spline) <= d1f_prominence_floor)[0]

    if len(i_baselines_search) < 10:
        return i_baselines_all

    i_baselines_d1f_left = int(i_baselines_search[0] / spline_fidelity)
    i_baselines_d1f_right = int(i_baselines_search[-1] / spline_fidelity)

    # use a subset of the flattest baselines
    # search_buffer = int((len(i_baselines_search) / spline_fidelity) / 2)
    # search_buffer = int((i_baselines_d1f_right - i_baselines_d1f_left) / 4)
    # i_baselines_left = i_baselines_d1f_left + i_baselines_all[0] + search_buffer
    # i_baselines_right = i_baselines_d1f_right + i_baselines_all[0] - search_buffer

    # use all detected indexes
    i_baselines_left = i_baselines_d1f_left + i_baselines_all[0]
    i_baselines_right = i_baselines_d1f_right + i_baselines_all[0]

    i_baselines = np.arange(i_baselines_left, i_baselines_right)

    if len(i_baselines) < 10:
        # i_baselines_left = i_baselines_d1f_left + i_baselines_all[0]
        # i_baselines_right = i_baselines_d1f_right + i_baselines_all[0]
        # i_baselines = np.arange(i_baselines_left, i_baselines_right)
        return i_baselines_all
    # if len(i_baselines) < 10:
    #     return i_baselines_all

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
        raise ValueError('Kernel size must be >= 3 and odd')

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
        sigma_color = 50  # standard deviation of the intensity gaussian kernel
        sigma_space = 10  # standard deviation of the spatial gaussian kernel
        frame_out = mean_bilateral(frame_in, square(kernel))
    elif filter_type is 'gaussian':
        # Good for ___, but ___
        sigma = kernel # standard deviation of the gaussian kernel
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
        width = 20   # The desired width of the transition from pass to stop, Hz
        window = 'kaiser'
        n_order, beta = kaiserord(ripple_db, width / nyq_rate)
        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        taps = firwin(numtaps=n_order + 1, cutoff=freq_cutoff, window=(window, beta), fs=sample_rate)
        # signal_out = lfilter(taps, 1.0, signal_in)   # for FIR, a=1
        signal_out = filtfilt(taps, 1, signal_in, method="gust")   # for FIR, a=1
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
        return a * np.exp(-b * x) + c   # a decaying exponential fit to fit to

    drift_range = signal_in.max() - signal_in.min()
    drift_x = np.arange(start=0, stop=len(signal_in))
    exp_b_estimates = (0.01, 0.1)   # assumed bounds of the B constant for a decaying exponential fit

    if type(drift_order) is int:
        # np.polyfit : Least squares polynomial fit
        poly = np.poly1d(np.polyfit(drift_x, signal_in, drift_order))
        poly_y = poly(drift_x)
    else:
        # scipy.optimize.curve_fit : Use non-linear least squares to fit a function, f, to data
        exp_bounds_lower = [0, exp_b_estimates[0], signal_in.min()]
        exp_bounds_upper = [drift_range*2, exp_b_estimates[1], signal_in.max()]
        try:
            # popt, pcov = curve_fit(func_exp, drift_x, signal_in)
            popt, pcov = curve_fit(func_exp, drift_x, signal_in, bounds=(exp_bounds_lower, exp_bounds_upper))
        except RuntimeError as e:
            raise ArithmeticError('Could not fit an exponential curve to the signal')

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
    drift = poly_y

    return signal_out.astype(signal_in.dtype), drift


# TODO rename functions (like calculate_snr to signal_snr and signal_error)
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
    # Check parameters
    if type(stack_in) is not np.ndarray:
        raise TypeError('Stack type must be an "ndarray"')
    if len(stack_in.shape) is not 3:
        raise TypeError('Stack must be a 3-D ndarray (T, Y, X)')
    if stack_in.dtype not in [np.uint16, float]:
        raise TypeError('Stack values must either be "np.uint16" or "float"')

    print('Inverting a stack ...')
    stack_out = np.empty_like(stack_in)
    map_shape = stack_in.shape[1:]
    # Assign a value to each pixel
    for iy, ix in np.ndindex(map_shape):
        print('\r\tInversion of Row:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix + 1, map_shape[1]),
              end='', flush=True)
        pixel_data = stack_in[:, iy, ix]
        # pixel_ensemble = calc_ensemble(time_in, pixel_data)
        # snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak = calculate_snr(pixel_data, noise_count)
        # snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak = calculate_snr(pixel_data, noise_count)
        # Set every pixel's values to the analysis value of the signal at that pixel
        # map_out[iy, ix] = analysis_type(pixel_ensemble[1])
        pixel_data_inv = invert_signal(pixel_data)
        stack_out[:, iy, ix] = pixel_data_inv

    print('\nDONE Inverting stack')
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
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "uint16" or "float"')

    if any(v < 0 for v in signal_in):
        raise ValueError('All signal values must be >= 0')

    xp = [signal_in.min(), signal_in.max()]
    fp = [0, 1]
    signal_out = np.interp(signal_in, xp, fp)

    return signal_out


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
            Must be applied to signals with upward deflections (Peak > noise).  TODO use on negative signals
            Assumes noise SD > 1, otherwise set to 0.5
            Assumes max noise value < (peak / 5)
            Auto-detects noise section as the last noise_count values before the final noisy peak.
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "uint16" or "float"')
    if type(noise_count) is not int:
        raise TypeError('Number of noise values to use must be an "int"')

    # if any(v < 0 for v in signal_in):
    #     raise ValueError('All signal values must be >= 0')
    if noise_count < 0:
        raise ValueError('Noise count must be >= 0')
    if noise_count >= len(signal_in):
        raise ValueError('Number of noise values to use must be < length of signal array')

    # Characterize the signal
    signal_bounds = (signal_in.min(), signal_in.max())
    signal_range = signal_bounds[1] - signal_bounds[0]
    noise_height = signal_bounds[0] + signal_range/2    # assumes max noise/signal amps. of 1 / 2

    # Find peak values, at least (noise_height + signal_range/2) tall and (len(signal_in)/2) samples apart
    i_peaks, properties = find_tran_peak(signal_in, props=True)
    if (i_peaks is np.nan) or (len(i_peaks) == 0):
        # raise ArithmeticError('No peaks detected'.format(len(i_peaks), i_peaks))
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    if len(i_peaks) > 1:
        # Use the peak with the max prominence
        i_peak_calc = i_peaks[np.argmax(properties['prominences'])]
        ir_peak = i_peaks
    else:
        # Use the peak with the max prominence
        i_peak_calc = i_peaks[np.argmax(properties['prominences'])]
        ir_peak = i_peak_calc

    # Use the peak value and peak_extent*2 number of neighboring values
    peak_extent = 1
    data_peak = signal_in[i_peak_calc - peak_extent: i_peak_calc + peak_extent]
    peak_rms = np.sqrt(np.mean(data_peak.astype(np.dtype(float)) ** 2))
    peak_sd = statistics.pstdev(data_peak.astype(float))
    # Use the rms of the peak's extent + their  SD/2, to be weighted towards the peak
    peak_value = signal_in[i_peak_calc]

    # Find noise values
    # TODO find_tran_baselines may be slow with interpolation
    i_noise_calc = find_tran_baselines(signal_in)
    # i_noise_calc = range(np.argmin(signal_in), np.argmin(signal_in) + 10)

    if len(i_noise_calc) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    data_noise = signal_in[i_noise_calc]

    # Use noise values
    noise_rms = np.sqrt(np.mean(data_noise) ** 2)
    noise_sd = statistics.stdev(data_noise.astype(float)) # standard deviation

    # Calculate Peak-Peak value
    peak_peak = abs(peak_value - noise_rms).astype(signal_in.dtype)

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

    MIN = 1    # Min to avoid division by 0

    error = ((modified.astype(float) - ideal.astype(float)) / (ideal.astype(float)) * 100)
    error_mean = error.mean()
    error_sd = statistics.stdev(error)

    return error, error_mean, error_sd
