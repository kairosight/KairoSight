#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implements scipy.signal.find_peaks to find and organize peaks across time from transients and cyclical signals.

@author: Rafael Jaimes
raf@cardiacmap.com
v1: 2019-02-20 
"""

import scipy.signal as sig
import numpy as np

def peak_detect(f, thresh, LOT): # Let the user change lockout time (LOT) and thresh with the GUI and input it here    
    f_grad=np.gradient(f) # 1st derivative of the signal to find max dF/dt (up or upstroke)
    f2_grad=np.gradient(f_grad) # 2nd derivative of the signal to find max d2F/dt2 (t0)
    
    # sequence: t0 -> up -> peak
    t0_thresh=thresh*np.max(f2_grad)
    up_thresh=thresh*np.max(f_grad)
    peak_thresh=thresh*np.max(f)
    base_thresh=thresh*np.max(f)
        
    t0_locs = sig.find_peaks(f2_grad, height=t0_thresh, distance=LOT)
    up_locs, up_props=sig.find_peaks(f_grad, height=up_thresh, distance=LOT) # save properties, max dF/dt (max_vel)
    peak_locs = sig.find_peaks(f, height=peak_thresh, distance=LOT)
    base_locs = sig.find_peaks(-f+np.max(f), height=base_thresh, distance=LOT)

        
    max_vel = up_props["peak_heights"]
    num_peaks = np.min([len(t0_locs[0]), len(up_locs), len(peak_locs[0])])
    return num_peaks, t0_locs[0], up_locs, peak_locs[0], base_locs[0], max_vel, peak_thresh