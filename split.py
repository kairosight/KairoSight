#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:25:50 2019
Pulls in an image stack (T, X, Y). Splits in the horizontal direction while preserving Y and T
Typically used to split dual wavelength images with a single sensor.

@author: Rafael Jaimes
raf@cardiacmap.com
v1: 2019-02-28
"""

def split(img, horz):
    [num_frames, h , w] = img.shape   
    left = img[:,:,0:int(w*horz)]
    right = img[:,:,int(w*horz):]
    return left, right