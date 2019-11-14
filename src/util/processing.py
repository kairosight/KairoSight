import statistics
import numpy as np
from numpy import linalg
from scipy.signal import find_peaks, resample, filtfilt, kaiserord, firwin, firwin2,\
    lfilter, butter, freqs, freqz, minimum_phase
from scipy.optimize import curve_fit
from skimage.morphology import square
from skimage.filters.rank import median, mean, mean_bilateral
from skimage.filters import gaussian
FILTERS_SPATIAL = ['median', 'mean', 'bilateral', 'gaussian']
# TODO add TV, a non-local, and a weird filter


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


def filter_spatial(frame_in, filter_type='median', kernel=3):
    """Spatially filter a frame (2-D array, YX) of grayscale optical data.

        Parameters
        ----------
        frame_in : ndarray
             A 2-D array (Y, X) of optical data
        filter_type : str
            The type of filter algorithm to use: convolve (default), gaussian, bilateral
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
        raise TypeError('Frame must be a 3-D ndarray (T, X, Y)')
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
        sigma = kernel / 3   # standard deviation of the gaussian kernel
        frame_out = gaussian(frame_in, sigma=sigma, preserve_range=True)
    else:
        raise NotImplementedError('Filter type "{}" not implemented'.format(filter_type))

    return frame_out.astype(frame_in.dtype)


def filter_temporal(signal_in, sample_rate, filter_order='auto'):
    """Apply a lowpass filter to array of optical data.

        Parameters
        ----------
        signal_in : ndarray
             The array of data to be evaluated
        sample_rate : float
            Sample rate (Hz) of signal_in
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
    if type(filter_order) not in [int, str]:
        raise TypeError('Filter type must be an int or str')

    f_cutoff = 100  # Cutoff frequency of the lowpass filter
    nyq_rate = sample_rate / 2.0
    n_order = 0

    if type(filter_order) is int:
        # Good for ___, but ___
        # Butterworth (from old code)
        Wn = f_cutoff / nyq_rate
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
        taps = firwin(numtaps=n_order + 1, cutoff=f_cutoff, window=(window, beta), fs=sample_rate)
        # signal_out = lfilter(taps, 1.0, signal_in)   # for FIR, a=1
        signal_out = filtfilt(taps, 1, signal_in, method="gust")   # for FIR, a=1
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
        return a * np.exp(-b * x) + c

    drift_range = signal_in.max() - signal_in.min()
    drift_x = np.arange(start=0, stop=len(signal_in))
    exp_B_estimages = (0.03, 0.1)   # assumed bounds of the B constant for an exponential fit

    if type(drift_order) is int:
        # np.polyfit : Least squares polynomial fit
        poly = np.poly1d(np.polyfit(drift_x, signal_in, drift_order))
        poly_y = poly(drift_x)
    else:
        # # scipy.optimize.curve_fit : Use non-linear least squares to fit a function, f, to data
        exp_bounds_lower = [0, exp_B_estimages[0], signal_in.min()]
        exp_bounds_upper = [drift_range*2, exp_B_estimages[1], signal_in.max()]
        popt, pcov = curve_fit(func_exp, drift_x, signal_in, bounds=(exp_bounds_lower, exp_bounds_upper))
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


def invert_signal(signal_in):
    """Invert the values of a signal array.

        Parameters
        ----------
        signal_in : ndarray
             The array of data to be processed

        Returns
        -------
        signal_inv : ndarray
             The inverted signal array
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "uint16" or "float"')

    # calculate axis to rotate data around (middle value int or float)
    axis = signal_in.min() + ((signal_in.max() - signal_in.min()) / 2)
    if signal_in.dtype in [np.int32]:
        axis = np.floor(axis).astype(int)

    # rotate the data around it's central value
    signal_inv = axis + (axis - signal_in)

    return signal_inv


def normalize_signal(signal_in):
    """Normalize the values of a signal array to range from 0 to 1.

        Parameters
        ----------
        signal_in : ndarray
             The array of data to be processed

        Returns
        -------
        signal_norm : ndarray
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
             The Signal-to-Noise ratio of the given data
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
            Must be applied to signals with upward deflections (Peak > noise).
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

    if any(v < 0 for v in signal_in):
        raise ValueError('All signal values must be >= 0')
    if noise_count < 0:
        raise ValueError('Noise count must be >= 0')
    if noise_count >= len(signal_in):
        raise ValueError('Number of noise values to use must be < length of signal array')

    # Characterize the signal
    signal_bounds = (signal_in.min(), signal_in.max())
    signal_range = signal_bounds[1] - signal_bounds[0]

    # Calculate noise values, detecting the last noise peak and using indexes [peak - noise_count, peak]
    noise_height = signal_bounds[0] + signal_range/2    # assumes max noise/signal amps. of 1 / 2
    i_noise_peaks, _ = find_peaks(signal_in, height=(None, noise_height))
    if len(i_noise_peaks) == 0:
        i_noise_peaks = [len(signal_in)-1]
    i_noise_calc = np.linspace(start=i_noise_peaks[-1] - noise_count, stop=i_noise_peaks[-1],
                               num=noise_count).astype(int)
    if any(i < 0 for i in i_noise_calc):
        raise ValueError('Number of noise values too large for this signal of length {}'.format(len(signal_in)))
    data_noise = signal_in[i_noise_calc[0]: i_noise_calc[-1]].astype(np.dtype(float))
    noise_rms = np.sqrt(np.mean(data_noise ** 2))
    noise_sd = statistics.stdev(data_noise)
    if noise_sd == 0:
        noise_sd = 0.5  # Noise data too flat to detect SD
    noise_bounds = (noise_rms - noise_sd/2, noise_rms + noise_sd/2)
    noise_range = noise_bounds[1] - noise_bounds[0]
    if signal_bounds[1] < noise_rms:
        raise ValueError('Signal max {} seems to be < noise rms {}'.format(signal_bounds[1], noise_rms))

    # Find indices of peak values, at least (noise_height + signal_range/2) tall and (len(signal_in)/2) samples apart
    # i_peaks, _ = find_peaks(signal_in, prominence=(signal_range * 0.8, signal_range), distance=10)
    # i_peaks, _ = find_peaks(signal_in, height=noise_height + signal_range/2, distance=len(signal_in)/2)
    i_peaks, _ = find_peaks(signal_in, height=noise_height + signal_range/2,
                            prominence=signal_range/2,
                            distance=len(signal_in)/2)
    if len(i_peaks) == 0:
        raise ArithmeticError('No peaks detected'.format(len(i_peaks), i_peaks))
    if len(i_peaks) > 1:
        raise ArithmeticError('{} peaks detected at {} for a single given transient'.format(len(i_peaks), i_peaks))
    i_peak_calc = i_peaks[0].astype(int)
    data_peak = signal_in[i_peak_calc].astype(int)
    peak_rms = np.sqrt(np.mean(data_peak.astype(np.dtype(float)) ** 2))

    # Calculate Peak-Peak value
    peak_peak = abs(peak_rms - noise_rms)

    # Calculate SNR
    snr = peak_peak / noise_sd

    rms_bounds = (noise_rms, peak_rms)
    sd_noise = noise_sd
    ir_noise = i_noise_calc
    ir_peak = i_peak_calc
    return snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak


def map_snr(stack_in, noise_count=10):
    """Generate a map_out of Signal-to-Noise ratios for signal arrays within a stack,
    defined as the ratio of the Peak-Peak amplitude to the population standard deviation of the noise.

        Parameters
        ----------
        stack_in : ndarray
            A 3-D array (T, Y, X) of optical data
        noise_count : int
             The number of noise values to be used in the calculation, default is 10

        Returns
        -------
        map : ndarray
             A 2-D array of Signal-to-Noise ratios, dtype : float
        """
    # Check parameters
    if type(stack_in) is not np.ndarray:
        raise TypeError('Stack type must be an "ndarray"')
    if len(stack_in.shape) is not 3:
        raise TypeError('Stack must be a 3-D ndarray (T, X, Y)')
    if stack_in.dtype not in [np.uint16, float]:
        raise TypeError('Stack values must either be "np.uint16" or "float"')

    if type(noise_count) is not int:
        raise TypeError('Noise count must be an "int"')

    map_shape = stack_in.shape[1:]
    map_out = np.empty(map_shape)
    # Assign an SNR to each pixel
    for iy, ix in np.ndindex(map_shape):
        pixel_data = stack_in[:, iy, ix]
        snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak = calculate_snr(pixel_data, noise_count)
        # Set every pixel's values to the SNR of the signal at that pixel
        map_out[iy, ix] = snr

    return map_out


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

    error = ((ideal.astype(float) - modified.astype(float)) / (ideal.astype(float)) * 100)
    error_mean = error.mean()
    error_sd = statistics.stdev(error)

    return error, error_mean, error_sd
    pass
