from math import pi, floor
import numpy as np


def model_transients(model_type='Vm', t=100, t0=0, fps=1000, f_0=100, f_amp=10, noise=0, num=1, cl=100):
    """Create a 2-D array of model 16-bit optical data of either a
    murine action potential (OAP) or a murine calcium transient (OCT).

       Parameters
       ----------
       model_type : str
           The type of transient: 'Vm' or 'Ca', default is 'Vm'
       t : int, float
           Length of array in milliseconds (ms), default is 100
       t0 : int, float
           Start time (ms) of first OAP, default is 0
       fps : int
           Frame rate (frames per second) of optical data acquisition, default is 1000, min is 200
       f_0 : int
           Baseline fluorescence value in counts, default is 100
       f_amp : int
           Amplitude of the OAP in counts, default is 10.
           Can be negative, e.g. cell depolarization with fast voltage dyes
       noise : int
           Magnitude of gaussian noise, as a percentage of f_peak, default is 0
       num : int
           Number of transients to generate, default is 1
       cl : int
           Time (ms) between transients aka Cycle Length, default is 100

       Returns
       -------
       model_time : ndarray
           An array of timestamps corresponding to model_data
       model_data : ndarray
           An array of model 16-bit OAP data
       """
    # Constants
    MIN_TOTAL_T = 100   # Minimum transient length (ms)
    # Check parameters
    if model_type not in ['Vm', 'Ca']:
        raise ValueError("The model type must either be 'Vm' or 'Ca'")
    if (type(t) or type(t0)) not in [int, float]:
        raise TypeError('All time parameters must be a non-negative real number')
    if (type(fps) or type(f_0) or type(f_amp)) not in [int]:
        raise TypeError('All fps and fluorescent parameters must be ints')

    if t < MIN_TOTAL_T:
        raise ValueError('The time length (t) must be longer than {} ms '.format(MIN_TOTAL_T))
    if t0 >= t:
        raise ValueError('The start time (t0) must be less than the time length (t)')
    if fps <= 200 or fps > 1000:
        raise ValueError('The fps must be > 200 or <= 1000')
    if f_amp < 0:
        raise ValueError('The amplitude must >=0')
    if model_type is 'Vm' and (f_0 - f_amp < 0):
        raise ValueError('Effective Vm amplitude is too negative')

    if num <= 0:
        raise ValueError('The number of transients must be > 0')
    if num * MIN_TOTAL_T > t - t0:
        raise ValueError('Too many transients, {}, for the total time, {} ms with start time {} ms'
                         .format(num, t, t0))
    if cl < 50:
        raise ValueError('The Cycle Length must be > 50 ms')

    # Calculate important constants
    FPMS = fps / 1000
    FRAMES = floor(FPMS * t)
    FRAME_T = 1 / FPMS
    FRAME_T0 = round(t0 / FRAME_T)
    FINAL_T = t - FRAME_T

    FL_COUNT_MAX = 2**16 - 1
    if f_0 > FL_COUNT_MAX:
        raise ValueError('The baseline fluorescence (f0) must be less than 2^16 - 1 (65535)')
    if abs(f_amp) > FL_COUNT_MAX:
        raise ValueError('The OAP amplitude (f_peak) must be less than 2^16 - 1 (65535)')

    # Initialize full model arrays
    model_time = np.linspace(start=0, stop=FINAL_T, num=FRAMES)     # time array
    model_data = np.full(int(FPMS * t), f_0, dtype=np.int)      # data array, default value is f_0
    if not np.equal(model_time.size, model_data.size):
        raise ArithmeticError('Lengths of time and data arrays not equal!')

    if model_type is 'Vm':
        # With voltage dyes, depolarization transients have a negative deflection and return to baseline
        # Initialize a single OAP array (50 ms) + 50 ms to sync with Ca
        vm_amp = -f_amp
        # Depolarization phase
        model_dep_period = 5  # XX ms long
        model_dep_frames = floor(model_dep_period / FRAME_T)
        # Generate high-fidelity data
        model_dep_full = np.full(model_dep_period, f_0)
        for i in range(0, model_dep_period):
            model_dep_full[i] = f_0 + (vm_amp * np.exp(-(((i - model_dep_period) / 3) ** 2)))  # a simplified Gaussian
        # Under-sample the high-fidelity data
        model_dep = model_dep_full[::floor(model_dep_period/model_dep_frames)][:model_dep_frames]

        # Early repolarization phase (from peak to APD 20, aka 80% of peak)
        model_rep1_period = 5  # XX ms long
        model_rep1_frames = floor(model_rep1_period / FRAME_T)
        apd_ratio = 0.8
        m_rep1 = -(vm_amp - (vm_amp * apd_ratio)) / model_rep1_period     # slope of this phase
        model_rep1 = np.full(model_rep1_frames, f_0)
        for i in range(0, model_rep1_frames):
            model_rep1[i] = ((m_rep1 * i) + vm_amp + f_0)    # linear

        # Late repolarization phase
        model_rep2_period = 50 - model_dep_period - model_rep1_period  # remaining OAP time
        model_rep2_frames = floor(model_rep2_period / FRAME_T)
        model_rep2_t = np.linspace(0, 50, model_rep2_frames)
        A, B, C = vm_amp * 0.8, (5 / m_rep1), f_0     # exponential decay parameters
        # model_rep2 = A * np.exp(-B * model_rep2_t) + C    # exponential decay, concave down
        tauFall = 10
        model_rep2 = A * np.exp(-model_rep2_t / tauFall) + C    # exponential decay, concave down, using tauFall
        model_rep2 = model_rep2.astype(int, copy=False)
        # Pad the end with 50 ms of baseline
        model_rep2Pad_frames = floor(50 / FRAME_T)
        model_rep2Pad = np.full(model_rep2Pad_frames, f_0, dtype=np.int)
        model_rep2 = np.concatenate((model_rep2, model_rep2Pad), axis=None)

    else:
        # With calcium dyes, depolarization transients have a positive deflection and return to baseline
        # Initialize a single OCT array (100 ms)
        # Depolarization phase
        model_dep_period = 10  # XX ms long
        model_dep_frames = floor(model_dep_period / FRAME_T)
        # Generate high-fidelity data
        model_dep_full = np.full(model_dep_period, f_0)
        for i in range(0, model_dep_period):
            model_dep_full[i] = f_0 + (f_amp * np.exp(-(((i - model_dep_period) / 6) ** 2)))  # a simplified Gaussian
        # Under-sample the high-fidelity data
        model_dep = model_dep_full[::floor(model_dep_period/model_dep_frames)][:model_dep_frames]

        # Early repolarization phase (from peak to CAD 40, aka 60% of peak)
        model_rep1_period = 15  # XX ms long
        model_rep1_frames = floor(model_rep1_period / FRAME_T)
        cad_ratio = 0.6
        m_rep1 = -(f_amp - (f_amp * cad_ratio)) / model_rep1_period     # slope of this phase
        model_rep1 = np.full(model_rep1_frames, f_0)
        model_rep1_full = np.full(model_rep1_period, f_0)
        # Generate high-fidelity data
        for i in range(0, model_rep1_period):
            model_rep1_full[i] = ((m_rep1 * i) + f_amp + f_0)    # linear
        # Under-sample the high-fidelity data
        model_rep1 = model_rep1_full[::floor(model_rep1_period/model_rep1_frames)][:model_rep1_frames]

        # Late repolarization phase
        model_rep2_period = 100 - model_dep_period - model_rep1_period  # remaining OAP time
        model_rep2_frames = floor(model_rep2_period / FRAME_T)
        model_rep2_t = np.linspace(0, 100, model_rep2_frames)
        A, B, C = f_amp * cad_ratio, (0.8 / m_rep1), f_0     # exponential decay parameters
        # model_rep2 = A * np.exp(B * model_rep2_t) + C    # exponential decay, concave up
        tauFall = 30
        model_rep2 = A * np.exp(-model_rep2_t / tauFall) + C    # exponential decay, concave up, using tauFall
        model_rep2 = model_rep2.astype(int, copy=False)

    # Assemble the transient
    model_tran = np.concatenate((model_dep, model_rep1, model_rep2), axis=None)

    # Assemble the start time and OAP(s) into the full array
    cl_frames = floor(cl / FRAME_T)
    if cl_frames < floor(100 / FRAME_T):
        # Shorten the transient array
        model_tran = model_tran[:cl]
    else:
        # Pad the transient array
        tranPad_frames = floor((cl - 100) / FRAME_T)
        tranPad = np.full(tranPad_frames, f_0, dtype=np.int)
        model_tran = np.concatenate((model_tran, tranPad), axis=None)

    # Assemble the train of transients
    model_tran_train = np.tile(model_tran, num)
    if model_tran_train.size > model_data.size - FRAME_T0:
        # Shorten train array to fit into final data array
        model_tran_train = model_tran_train[:model_data.size - FRAME_T0]

    model_data[FRAME_T0:FRAME_T0 + model_tran_train.size] = model_tran_train

    # Add gaussian noise, mean: 0, standard deviation: 10% of peak, length
    model_noise = np.random.normal(0, (noise/100) * f_amp, model_data.size)
    model_data = model_data + model_noise

    return model_time, model_data


def model_images(model_type='Vm', t=100, noise=0, num=1):
    """Create a 3-D array of model 16-bit optical data of either a
    murine action potential (OAP) or a murine calcium transient (OCT).

       Parameters
       ----------
       model_type : str
           The type of transient: 'Vm' or 'Ca', default is 'Vm'
       t : int, float
           Length of array in milliseconds (ms), default is 100
       noise : int
           Magnitude of gaussian noise, as a percentage of f_peak, default is 0
       num : int
           Number of transients to generate, default is 1

       Returns
       -------
       model_time : ndarray
           An array of timestamps corresponding to model_data
       model_data : ndarray
           An array of model 16-bit OAP data
       """
    # Constants


# Code for example tests
def circle_area(r):
    if r < 0:
        raise ValueError('The radius cannot be negative')

    if type(r) not in [int, float]:
        raise TypeError('The radius must be a non-negative real number')

    return pi * (r**2)
