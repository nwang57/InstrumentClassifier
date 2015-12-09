from __future__ import division
import os
import sys
import glob
import numpy as np
import scipy
from utils import *

def norm_shape(shape):
    """
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.
    Parameters
        shape - an int, or a tuple of ints
    Returns
        a shape tuple
    """
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass
    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass
    raise TypeError('shape must be an int, or a tuple')

def sliding_window(data, window_size, step_size = None):
    """
        this is the one dimension implementation of sliding window
        window_size : an int value specifies the window size
        step_size : the hop size
        return an list of each frame
        reference : http://www.johnvinyard.com/blog/?p=268
    """
    if not step_size:
        step_size = window_size
    ws = np.array(norm_shape(window_size))
    ss = np.array(norm_shape(step_size))
    newshape = norm_shape(((np.array(data.shape) - ws) // ss) + 1)
    newshape += norm_shape(ws)
    newstrides = norm_shape(np.array(data.strides * ss)) + data.strides
    strided = np.lib.stride_tricks.as_strided(data,shape = newshape,strides = newstrides)
    return strided

def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.

    f is a vector and x is an index for that vector.

    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.

    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.

    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]

    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)

    source: https://github.com/endolith/waveform-analyzer/blob/master/common.py
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

def estimated_f0(wavedata, window_size):
    """using the peak of the spectrum to find the estimated fundamental frequency
    sometime the harmonics may be stronger than the f0
    """
    spec, f = spectrum(wavedata, window_size)
    ipeak = np.argmax(spec)
    if ipeak:
        ipeak2 = np.argmax(spec[:ipeak])
        if spec[ipeak] - spec[ipeak2] < 0.2 * abs(spec[ipeak]):
            ipeak = ipeak2
    i_interp = parabolic(spec, ipeak)[0]
    return i_interp * 44100.0 / window_size

def harmonics(wavedata,window_size,c=0.1):
    f0 = estimated_f0(wavedata,window_size)
    spec, f = spectrum(wavedata, window_size)
    num_h = int(0.5*44100/f0)
    harmo = np.array([i*f0 for i in xrange(1,num_h+1)])
    harmonics = []

    for h in xrange(num_h):
        possible_hz = (max(0, harmo[h]-c*f0), min(harmo[h]+c*f0, 44100.0/2-44100.0/window_size))
        start = int(possible_hz[0]/44100*window_size)
        end = int(possible_hz[1]/44100*window_size)+1
        if start < end:
            local_max_pos = np.argmax(spec[start:end]) + start
            fq = parabolic(spec, local_max_pos)[0] * 44100.0 / window_size
            mag = spec[local_max_pos]
            harmonics.append((fq, mag))
    return np.array(harmonics)

def spectralEnvelope(h_ampl):
    """
        given the harmonic amplitude, estimate the spectral envelope, moving average of window_size=3
    """
    env = np.zeros(len(h_ampl))
    kk = 0
    env[kk] = (h_ampl[kk] + h_ampl[kk+1]) / 2
    for kk in xrange(1,len(h_ampl)-1):
        env[kk] = (h_ampl[kk-1] + h_ampl[kk] + h_ampl[kk+1])/3
    kk = len(h_ampl)
    env[kk-1] = (env[kk-2] + env[kk-1]) / 2
    return env
