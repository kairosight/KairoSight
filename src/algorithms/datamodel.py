from math import pi, floor
import numpy as np


def model_vm(t=500, t0=50, fps=1000, f0=0, f_peak=10):
    """Create an array of model 16-bit optical data of the murine action potential (OAP).

       Parameters
       ----------
       t : int, float
           Length of array in milliseconds (ms)
       t0 : int, float
           Start time (ms) of first OAP
       fps : int
           Frame rate (frames per second) of optical data acquisition
       f0 : int
           Baseline fluorescence value in counts, default is 0
       f_peak : int
           Amplitude of the OAP in counts, default is 10.
           Can be negative, e.g. cell depolarization with fast voltage dyes

       Returns
       -------
       model_time : ndarray
           An array of timestamps corresponding to model_data
       model_data : ndarray
           An array of model 16-bit OAP data
       """
    # Check parameters
    if (type(t) or type(t0)) not in [int, float]:
        raise TypeError('All time parameters must be a non-negative real number')
    if (type(fps) or type(f0) or type(f_peak)) not in [int]:
        raise TypeError('All fps and fluorescent parameters must be ints')

    if t < 100:
        raise ValueError("The time length (t) must be longer than 100 ms ")
    if t0 >= t:
        raise ValueError("The start time (t0) must be less than the time length (t)")

    if fps <= 200 or fps > 1000:
        raise ValueError("The fps must be >200 and <1000")

    # Calculate important constants
    FPMS = fps / 1000
    FRAMES = floor(FPMS * t)
    FRAME_T = 1 / FPMS
    FRAME_T0 = round(t0 / FRAME_T)
    FINAL_T = t - FRAME_T

    FL_COUNT_MAX = 2**16 - 1
    if f0 > FL_COUNT_MAX:
        raise ValueError("The baseline fluorescence (f0) must be less than 2^16 - 1 (65535)")
    if f_peak > FL_COUNT_MAX:
        raise ValueError("The OAP amplitude (f_peak) must be less than 2^16 - 1 (65535)")

    # Initialize full model arrays
    model_time = np.linspace(start=0, stop=FINAL_T, num=FRAMES)
    model_data = np.full(int(FPMS * t), f0, dtype=np.int)
    if not np.equal(model_time.size, model_data.size):
        raise ArithmeticError("Lengths of time and data arrays not equal!")

    # Initialize a single OAP array
    # Depolarization phase
    model_dep_t = round(20 / FRAME_T)   # 20 ms long
    model_dep = np.full(model_dep_t, f0 + f_peak)
    # Early repolarization phase
    model_rep1_t = round(5 / FRAME_T)   # 5 ms long
    model_rep1 = np.full(model_rep1_t, f0 + int((f_peak / 2)))
    # Late repolarization phase
    model_rep2_t = round(25 / FRAME_T)   # 25 ms long
    model_rep2 = np.full(model_rep2_t, f0 + int((f_peak / 3)))
    model_oap = np.concatenate((model_dep, model_rep1, model_rep2), axis=None)

    # Assemble the start time and single OAP into the full array

    model_data[FRAME_T0:FRAME_T0 + model_oap.size] = model_oap

    return model_time, model_data


# Code for example tests
def circle_area(r):
    if r < 0:
        raise ValueError("The radius cannot be negative")

    if type(r) not in [int, float]:
        raise TypeError('The radius must be a non-negative real number')

    return pi * (r**2)
