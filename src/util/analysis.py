from util.processing import *
import time
import numpy as np
import scipy.optimize as opt
import xlsxwriter
from scipy.signal import savgol_filter
from scipy.misc import derivative
from scipy.interpolate import UnivariateSpline
from datetime import datetime
from PyQt5.QtWidgets import QFileDialog

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


def calc_tran_duration(signal_in, mask, act_ind, max_apd_ind, percent_apd=80):
    """Calculate the duration of a transient,
    defined as the number of indices between the activation time
    and the time nearest a percentage of the peak-to-peak range
    (e.g. APD-80, APD-90, CAD-80 ...)

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
    # Find the maximum amplitude of the action potential
    max_amp_ind = np.argmax(
        signal_in[
            act_ind:act_ind+max_apd_ind, :, :], axis=0
        )+act_ind
    # Preallocate variable for percent apd index and value
    apd_ind = np.zeros(max_amp_ind.shape)
    apd_val = apd_ind
    # Step through the data
    for n in np.arange(0, signal_in.shape[1]):
        for m in np.arange(0, signal_in.shape[2]):
            # Ignore pixels that have been masked out
            if not mask[n, m]:
                # Grab the data segment between max amp and end
                tmp = signal_in[
                    max_amp_ind[n, m]:act_ind +
                    max_apd_ind, n, m]
                # Find the minimum to find the index closest to
                # desired apd percent
                apd_ind[n, m] = np.argmin(tmp
                                          - max_amp_ind[n, m]
                                          * percent_apd)
                +max_amp_ind[n, m]
                # Subtract activation time to get apd
                apd_val[n, m] = apd_ind[n, m]-act_ind[n, m]
    return apd_val


def oap_peak_calc(signal_in, start_ind, end_ind, amp_thresh, fps):
    """Calculates the indices for the peaks of each optical action potential

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float
        start_ind : int
            Index that denotes the beginning of the window
        end_ind : int
            Index that denotes the end of the window
        amp_thresh : float
            The numerical threshold above which peaks will be identified
        fps: int
            The frames per second acquisition rate of the given file

        Returns
        -------
        peak_ind : np.int
            The index of each peak within the specified window
    """
    # Start with grabbing all signal indices above the threshold
    thresh_ind = signal_in[start_ind:end_ind] > amp_thresh
    # Create indices for the oap segment
    sig_ind = np.arange(0, end_ind-start_ind)
    # Grab the indices of all points above the amplitude threshold
    sig_ind = sig_ind[thresh_ind]
    # Find the indices that separate the above threshold segments
    chop_sig = sig_ind[1:]-sig_ind[:-1]
    tmp = np.arange(0, len(chop_sig))
    chop_ind = np.append(0, tmp[chop_sig != 1]+1)
    # Create an empty variable for the peak index values
    peak_ind = np.empty(len(chop_ind))
    # Step throught each OAP
    for n in np.arange(0, len(chop_ind)):
        # If it's the last one, go to the end of the segment
        if n == len(chop_ind)-1:
            # Grab the window of indices with the peak
            peak_win = sig_ind[chop_ind[n]:]+start_ind
        # If it's not, go to the next peak
        else:
            # Grab the window of indices with the peak
            peak_win = sig_ind[chop_ind[n]:chop_ind[n+1]]+start_ind
        # Calculate the index of the peak
        peak_ind[n] = np.argmax(signal_in[peak_win])+peak_win[0]
    # Check peak separation, remove any peaks at rates > 500 bpm (i.e., cycle
    # length < 120 ms)
    peak_sep = peak_ind[1:]-peak_ind[:-1]
    rm = [n for n in np.arange(0, len(peak_sep)) if peak_sep[n]*1/fps < 0.120]
    peak_ind = np.delete(peak_ind, rm)
    # Output the indices of the peaks
    return peak_ind.astype(int)


def diast_ind_calc(signal_in, peak_ind):
    """Calculates the end-diastole indices for each optical action potential
    (OAP)

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float
        peak_ind : int
            Index that denotes the location of peaks in signal_in

        Returns
        -------
        diast_ind : np.int
            The index of end_diastole for each OAP
    """
    # Grab the average cycle length
    aveCL = np.average(peak_ind[1:]-peak_ind[:-1])
    # Grab a percentage of the average cycle length
    perCL = aveCL*0.1
    # Use percentage of cycle length create search window for max dV2dt
    start_ind_act = np.around(peak_ind-perCL, decimals=0).astype(int)
    # Calculate first and second derivatives
    dVdt = signal_in[1:]-signal_in[0:-1]
    dV2dt = dVdt[1:]-dVdt[0:-1]
    # Create an empty variable for the end-diastole index
    diast_ind = np.empty(len(peak_ind))
    for n in np.arange(len(peak_ind)):
        # Calculate the index of the max dV2dt
        diast_ind[n] = np.argmax(
            dV2dt[start_ind_act[n]:peak_ind[n]])+start_ind_act[n]
    # Output the end-diastole index
    return diast_ind.astype(int)


def apd_ind_calc(signal_in, end_ind, start_ind_act, peak_ind, apd_thresh):
    """Calculates the indices for the action potential duration (APD) of each
    optical action potential (OAP)

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float
        end_ind : int
            Index that denotes the end of the window
        start_ind_act : int
            Index that denotes the beginning of activation for each OAP
        peak_ind : int
            Index of each peak within signal_in
        apd_thresh : float
            The numerical threshold for action potential duration

        Returns
        -------
        apd_ind : np.int
            The index of at which each OAP passes the APD threshold
    """
    apd_ind = np.empty(len(peak_ind))
    # Calculate APD #
    for n in np.arange(0, len(peak_ind)):
        if n == len(peak_ind)-1:
            seg = signal_in[peak_ind[n]:end_ind]
        else:
            # Grab the segment of OAP between the peak and the next OAP
            seg = signal_in[peak_ind[n]:start_ind_act[n+1]]
        # Establish a threshold
        # apd_thresh = 0.85
        # Create an index variable for the OAP segment in question
        seg_apd_ind = np.arange(0, len(seg))
        # Find all of the index values where the segment value is greater than
        # the threshold
        seg_apd_bool = seg > signal_in[peak_ind[n]]*(1-apd_thresh)
        # Remove index values less than the threshold
        seg_apd_ind = seg_apd_ind[seg_apd_bool]
        # Check for data incontinuity and remove all data after first
        # discontinuity, if any
        dis_bool = seg_apd_ind[1:]-seg_apd_ind[:-1] != 1
        if sum(dis_bool):
            # Find the index values of the discontinuities
            dis_ind = [i for i, x in enumerate(dis_bool) if x]
            # Remove all of the index values after the first discontinuity
            seg_apd_ind = seg_apd_ind[:dis_ind[0]+1]
        # Calculate index value in the context of overall signal
        apd_ind[n] = peak_ind[n]+seg_apd_ind[-1]+1
    # Convert to integer values
    return apd_ind.astype(int)


# Function that calculates activation using end-diastole and peak indices
def act_ind_calc(signal_in, start_ind, end_ind):
    """Calculates the index for activation of each optical action potential
    (OAP)

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float
        start_ind : int
            Index that denotes the start of the window
        end_ind : int
            Index that denotes the end of the window

        Returns
        -------
        act_ind : np.int
            The index of at which each OAP is activated (i.e., at the maximum
            of its first derivative)
    """
    # Calculate ensemble activation of a single OAP
    if len(signal_in.shape) == 1:
        # Calculate the derivative
        dVdt = signal_in[1:]-signal_in[0:-1]
        # Find the indices of the maximums within a given window
        act_ind = np.empty(end_ind.shape)
        for idx, sia in enumerate(start_ind):
            act_ind[idx] = np.argmax(dVdt[sia:end_ind[idx]], axis=0)+sia-1
    # Calculate the image wide activation for a single cycle length
    else:
        # Calculate the derivative
        dVdt = signal_in[1:-1, :, :]-signal_in[0:-2, :, :]
        # Find the indices of the maximums within a given window
        act_ind = np.argmax(dVdt[start_ind:end_ind, :, :], axis=0)
    # Output the activation indices as an integer
    return act_ind.astype(int)


def tau_calc(signal_in, fps, peak_ind, diast_ind, end_ind):
    """Calculates the tau of a the repolorization curve of each optical action
    potential (OAP) using a fitted exponential decay

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float
        fps : int
            The frames per second of the acquired map
        peak_ind : int
            Index of each peak within signal_in
        diast_ind : int
            Index that denotes end-diastole, used in this context as the
            beginning of each OAP
        end_ind : int
            Index that denotes the end of the window

        Returns
        -------
        tau_val : np.float
            The tau value of the fitted exponential decay
    """
    # Calculate time step
    dt = 1/fps
    # Create an empty variable for the results
    tau_val = np.empty(len(peak_ind))
    for n in np.arange(0, len(peak_ind)):
        if n == len(peak_ind)-1:
            decay = signal_in[peak_ind[n]:end_ind]
        else:
            # Grab the segment of OAP between the peak and the next OAP
            decay = signal_in[peak_ind[n]:diast_ind[n+1]]
        # Create a lambda function of the type of curve to fit to the decay
        decay_func = lambda t, a, b, c: a * np.exp(-b * t) + c
        # Create a complimentary time variable
        decay_time = np.linspace(0, len(decay) * dt, len(decay))
        # Fit the lambda function to the decay
        popt, pcov = opt.curve_fit(decay_func, decay_time, decay,
                                   p0=[0, 0.01, 1])
        tau_val[n] = 1/popt[1]
    # Output the tau value of the fitted curve
    return tau_val


def ensemble_xlsx_print(file_name, signal_time, ind_analyze, data_oap, act_ind,
                        peak_ind, tau_fall, apd_val_30, apd_val_80,
                        apd_val_tri, d_f0, f1_f0):
    """Prints ensemble apd analysis results to a spreadsheet

        Parameters
        ----------
        signal_time : ndarray
            The array of data representing time
        ind_analyze : ndarray
            The pixel coordinate being analyzed
        data_oap : list
            List of action potential transients being analyzed
        act_ind : int
            Index that denotes activation
        peak_ind : int
            Index that denotes the peaks of the oaps in each signal
        tau_fall : float
            Rate of exponential decay that fits repolarization
        apd_val_30 : float
            Time to reach 30% peak OAP amplitude
        apd_val_80 : float
            Time to reach 80% peak OAP amplitude
        apd_val_tri : float
            Triangulation of APD30 and APD80
        d_f0 : float
            The ratio of the diatolic fluorescence to baseline fluorescence
        f1_f0 : float
            The ratio of the peak fluorescence to baseline fluorescence

        Returns
        -------
        Nothing. Check directory for new *.xlsx file.
    """
    # Create a workbook and add a worksheet
    # now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # workbook = xlsxwriter.Workbook('EnsembleAPD_' + now + '.xlsx')
    workbook = xlsxwriter.Workbook(file_name)
    bold = workbook.add_format({'bold': True})
    worksheet = workbook.add_worksheet('APD_Data')
    # Grab the greatest number of peaks in the selected signals
    peak_num = max([len(k) for k in peak_ind])
    # Create a loop for writing each data point
    for idx, coor in enumerate(ind_analyze):
        # Calculate a row value
        row = idx*peak_num+idx*7
        # Write coordinate
        worksheet.write(row, 0, '(Row, Column):', bold)
        worksheet.write(row, 1, '({0}, {1})'.format(coor[1], coor[0]))
        # Write the header
        worksheet.write(row+1, 0, 'ActTime (s)', bold)
        worksheet.write(row+1, 1, 'Vmax', bold)
        worksheet.write(row+1, 2, 'TauFall', bold)
        worksheet.write(row+1, 3, 'APD30 (s)', bold)
        worksheet.write(row+1, 4, 'APD80 (s)', bold)
        worksheet.write(row+1, 5, 'APDTri (s)', bold)
        worksheet.write(row+1, 6, 'D_F0', bold)
        worksheet.write(row+1, 7, 'F1_F0', bold)
        worksheet.write(row+1, 8, 'CL (s)', bold)
        # Write out the results as a spreadsheet
        for n in np.arange(0, len(peak_ind[idx])):
            worksheet.write(row+n+2, 0, signal_time[peak_ind[idx][n].item()])
            worksheet.write(row+n+2, 1, data_oap[idx][peak_ind[idx][n]].item())
            worksheet.write(row+n+2, 2, tau_fall[idx][n].item())
            worksheet.write(row+n+2, 3, apd_val_30[idx][n].item())
            worksheet.write(row+n+2, 4, apd_val_80[idx][n].item())
            worksheet.write(row+n+2, 5, apd_val_tri[idx][n].item())
            worksheet.write(row+n+2, 6, d_f0[idx][n].item())
            worksheet.write(row+n+2, 7, f1_f0[idx][n].item())
            if n < len(peak_ind[idx])-1:
                worksheet.write(row+n+2, 8, signal_time[act_ind[idx][n+1]] -
                                signal_time[act_ind[idx][n]].item())
        # Write out the averages
        ave_row = row+len(peak_ind[idx])+3
        worksheet.write(ave_row, 0, 'Mean', bold)
        worksheet.write(ave_row, 1, np.average(
            data_oap[idx][peak_ind[idx]]).item())
        worksheet.write(ave_row, 2, np.average(tau_fall[idx]).item())
        worksheet.write(ave_row, 3, np.average(apd_val_30[idx]).item())
        worksheet.write(ave_row, 4, np.average(apd_val_80[idx]).item())
        worksheet.write(ave_row, 5, np.average(apd_val_tri[idx]).item())
        worksheet.write(ave_row, 6, np.average(d_f0[idx]).item())
        worksheet.write(ave_row, 7, np.average(f1_f0[idx]).item())
        worksheet.write(ave_row, 8, np.average(
            signal_time[act_ind[idx][1:]] -
            signal_time[act_ind[idx][:-1]]).item())
        # Write out the standard deviations
        std_row = row+len(peak_ind[idx])+4
        worksheet.write(std_row, 0, 'Std', bold)
        worksheet.write(std_row, 1, np.std(
            data_oap[idx][peak_ind[idx]]).item())
        worksheet.write(std_row, 2, np.std(tau_fall[idx]).item())
        worksheet.write(std_row, 3, np.std(apd_val_30[idx]).item())
        worksheet.write(std_row, 4, np.std(apd_val_80[idx]).item())
        worksheet.write(std_row, 5, np.std(apd_val_tri[idx]).item())
        worksheet.write(std_row, 6, np.std(d_f0[idx]).item())
        worksheet.write(std_row, 7, np.std(f1_f0[idx]).item())
        worksheet.write(std_row, 8, np.std(
            signal_time[act_ind[idx][1:]] -
            signal_time[act_ind[idx][:-1]]).item())
    # Set the column widths
    worksheet.set_column(0, 8, 15)
    # Close the excel file
    workbook.close()


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
