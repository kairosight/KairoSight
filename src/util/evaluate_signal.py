import numpy as np


def calculate_snr(signal, i_noise, i_peak):
    """Calculate the Signal-to-Noise ratio of an array,
    defined as the ratio of the Peak-Peak amplitude to the population standard deviation of the noise.

       Parameters
       ----------
       signal : ndarray
           The array of data to be evaluated
       i_noise : tuple
           The range of indexes for noise data elements to be used in the calculation, e.g. (0, 10)
       i_peak : tuple
           The range of indexes for peak data elements to be used in the calculation, e.g. (50, 60)

       Returns
       -------
       snr : float
           The Signal-to-Noise ratio of the given data
       noise_sd : float
           The standard deviation of the noise
       data_peak : ndarray
           The array of peak values used in the calculation
       data_noise : ndarray
           The array of peak values used in the calculation
       """
    # Check parameters


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
           An array of percent error
       error_mean : float
           The mean value of the percent error array
       error_sd : float
           The standard deviation of the percent error array
       """
    # Check parameters
