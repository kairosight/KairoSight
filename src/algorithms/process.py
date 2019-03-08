#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processes a series of transient signals across time.

@author: Rafael Jaimes
raf@cardiacmap.com
v1: 2019-02-20 
"""

import scipy.optimize as opt
import numpy as np
import pandas as pd


# You must run peak_detect first to feed all the peak locations.
# Specify the fluorescent probe you are analyzing:
# probe = 0 for calcium
# probe = 1 for voltage
# The probe specification will alter the column labels in the data frame and change how the activation time is computed


def process(f, dt, t0_locs, up_locs, peak_locs, base_locs, max_vel, per_base, F0, probe):
    if base_locs[0] < t0_locs[0]:  # If there is a baseline point before the first transient, that's a problem
        base_locs = base_locs[1:]  # Chop the first baseline point

    if t0_locs[-1] > base_locs[-1]:  # If the last t0 points occurs after the last baseline point, that's a problem
        t0_locs = t0_locs[0:-1]  # Chop the last t0 point to ensure the new last t0 point occurs before the last base

    num_transients = len(t0_locs)

    results = pd.DataFrame(np.zeros(shape=(num_transients, 13)),
                           columns=['ActTime', 'Vmax', 'RiseTime', 'TimeToPeak', 'D30', 'DXX', 'D90',
                                    'D30_DXX', 'Tri', 'TauFall', 'Dias', 'Sys', 'CL'])

    for trans in range(0, num_transients):  # Don't skip any transients

        if t0_locs[trans] < up_locs[trans] < peak_locs[trans] < base_locs[trans]:
            t0 = t0_locs[trans]
            up = up_locs[trans]
            peak = peak_locs[trans]
            base = base_locs[trans]
        else:
            print('Detection Error')

        # Finding points on the decay portion back to baseline
        decay = (f[peak:base] - f[base]) / (f[peak] - f[base])  # Normalize the decay portion from 1 to 0
        ret30 = np.nonzero(decay < 0.7)[0][0]  # return to 30%, for CaD30/APD30, fitting Tau, and triangulation
        retXX = np.nonzero(decay < (100 - per_base) / 100)[0][0]  # return to XX% for CaDXX/APDXX
        ret90 = np.nonzero(decay < 0.1)[0][0]  # return to 90% for CaD90/APD90 and triangulation

        # Finding point on the upstroke portion up to peak
        upstroke = (f[t0:peak] - f[t0]) / (f[peak] - f[t0])  # Normalize the upstroke portion from 0 to 1
        up90 = np.nonzero(upstroke > 0.9)[0][0]  # return the first point that breaks 90% upstroke

        # Fitting routine using an anonymous function
        decay_func = lambda t, a, b, c: a * np.exp(-b * t) + c
        time = np.linspace(0, len(decay) * dt, len(decay))
        popt, pcov = opt.curve_fit(decay_func, time, decay, p0=[0, 0.01, 1])
        # decay_fit = decay_func(time, *popt)

        if probe == 0:
            start = t0  # for calcium, use max d2F/dt2 as the start
        else:
            start = up  # for voltage, use max dF/dt as the start

        results.ActTime[trans] = t0 * dt
        results.Vmax[trans] = max_vel[trans]
        results.RiseTime[trans] = (up90) * dt
        results.TimeToPeak[trans] = (peak - t0) * dt
        results.D30[trans] = (ret30 + peak - start) * dt
        results.DXX[trans] = (retXX + peak - start) * dt
        results.D90[trans] = (ret90 + peak - start) * dt
        results.D30_DXX[trans] = (ret30 + peak - start) / (retXX + peak - start)
        results.Tri[trans] = (ret90 + peak - start) * dt - (ret30 + peak - start) * dt
        results.TauFall[trans] = 1 / (popt[1])
        results.Sys[trans] = f[peak]

        if trans == 0:
            results.Dias[trans] = F0
        else:
            results.Dias[trans] = f[base]

        if trans < num_transients - 1:
            results.CL[trans] = (peak_locs[trans + 1] - peak) * dt
        else:
            results.CL[trans] = np.nan  # last transient, cannot calculate CL anymore

    if probe == 1:
        APDXX = 'APD' + str(per_base)
        results.rename(columns={'D30': 'APD30', 'DXX': APDXX,
                                'D90': 'APD90', 'D30_DXX': 'APD30/' + APDXX}, inplace=True)
    else:
        CaDXX = 'CaD' + str(per_base)
        results.rename(columns={'D30': 'CaD30', 'DXX': CaDXX,
                                'D90': 'CaD90', 'D30_DXX': 'CaD30/' + CaDXX}, inplace=True)

    return results
