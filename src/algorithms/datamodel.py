from math import pi
import numpy as np


def model_vm(t, t0=0, fps=1000, F0=0, F_peak=10):
    """Create an array of model murine 16-bit optical action potential (OAP) data.

       Parameters
       ----------
       t : int
           Length of array in milliseconds (ms)
       t0 : int
           Time (ms) of first OAP
       fps : int
           Frame rate (frames per second) of optical data acquisition
       F0 : int
           Baseline fluorescence value in counts, default is 0
       F_peak : int
           Amplitude of the OAP in counts, default is 10.
           Can be negative, e.g. cell depolarization with fast voltage dyes

       Returns
       -------
       model_data : ndarray
           An array of model 16-bit OAP data
       model_time : ndarray
           An array of timestamps corresponding to model_data
       """
    # Check parameters
    if (type(t) or type(t0) or type(F0) or type(F_peak)) not in [int, float]:
        raise TypeError('All parameters must be a non-negative real number')
    if t < 100:
        raise ValueError("The time length (t) must be longer than 100 ms ")
    if t0 >= t:
        raise ValueError("The start time (t0) must be less than the time length (t)")

    if fps <= 0 or fps > 1000:
        raise ValueError("The fps must be >0 and <1000")
    FPMS = fps / 1000
    FRAME_T = int(1 / FPMS)

    FL_COUNT_MAX = 2**16 - 1
    if F0 > FL_COUNT_MAX:
        raise ValueError("The baseline fluorescence (F0) must be less than 2^16 - 1 (65535)")
    if F_peak > FL_COUNT_MAX:
        raise ValueError("The OAP amplitude (F_peak) must be less than 2^16 - 1 (65535)")

    # Initialize full model arrays
    model_data = np.full(int(FPMS * t), F0, dtype=np.int)
    model_time = np.arange(0, t, FRAME_T)

    # Initialize a single OAP array
    # Depolarization phase
    model_dep_t = FRAME_T * 20   # 20 ms long
    model_dep = np.full(model_dep_t, F0 + F_peak)
    # Early repolarization phase
    model_rep1_t = FRAME_T * 5   # 5 ms long
    model_rep1 = np.full(model_rep1_t, F0 + int((F_peak/2)))
    # Late repolarization phase
    model_rep2_t = (FRAME_T * 25)   # 25 ms long
    model_rep2 = np.full(model_rep2_t, F0 + int((F_peak/3)))
    model_OAP = np.concatenate((model_dep, model_rep1, model_rep2), axis=None)

    # Assemble the single OAP into the full array
    model_data[(FRAME_T * t0):(FRAME_T * t0) + model_OAP.size] = model_OAP

    return model_data, model_time


def circle_area(r):
    if r < 0:
        raise ValueError("The radius cannot be negative")

    if type(r) not in [int, float]:
        raise TypeError('The radius must be a non-negative real number')

    return pi * (r**2)
