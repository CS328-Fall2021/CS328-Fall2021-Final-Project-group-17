# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import math
import numpy as np

def _compute_mean_features(window):
    """
    Computes the mean x, y, z, mag acceleration over the given window. 
    """
    return np.mean(window, axis=0)

def _compute_std_features(window):
    """
    Computes the standard deviation of x, y, z, mag acceleration over the given window.
    """
    return np.std(window, axis=0)

def _compute_median_features(window):
    """
    Computes the median of x, y, z, mag acceleration over the given window.
    """
    return np.median(window, axis=0)

def _compute_min_features(window):
    """
    Computes the min of x, y, z, mag acceleration over the given window.
    """
    return np.amin(window, axis=0)

def _compute_max_features(window):
    """
    Computes the max of x, y, z, mag acceleration over the given window.
    """
    return np.amax(window, axis=0)

def _compute_25p_features(window):
    """
    Computes the 25th percentile of x, y, z, mag acceleration over the given window.
    """
    return np.percentile(window, 25, axis=0)

def _compute_75p_features(window):
    """
    Computes the 75th percentile of x, y, z, mag acceleration over the given window.
    """
    return np.percentile(window, 75, axis=0)

def _compute_rfft_max(window):
    """
    Computes the frequency that has highest magnitude in the FFT for x, y, z, mag
    acceleration over the given window.
    """
    return np.argmax(np.abs(np.real(np.fft.rfft(window, axis=0))), axis=0)

def _compute_entropy(window):
    """
    Computes the entropy of the distribution of acceleration.
    """
    rval = []
    for col in range(window.shape[1]):
        hist, _ = np.histogram(window[:, col], bins=5)
        entropy = 0
        for i in hist:
            if i > 0:
                entropy -= i/sum(hist) * math.log(i/sum(hist))
        rval.append(entropy)
    return rval


def _compute_peaks(window, N=2):
    """
    Computes the number of peaks in each column.
    """
    rval = []
    for col in range(window.shape[1]):
        peaks = 0
        for row in range(N,window.shape[0]-N):
            if window[row,col] == max(window[row-N:row+N,col]):
                peaks += 1
        rval.append(peaks)
    return rval

def _compute_valleys(window, N=2):
    """
    Computes the number of valleys in each column.
    """
    rval = []
    for col in range(window.shape[1]):
        peaks = 0
        for row in range(N,window.shape[0]-N):
            if window[row,col] == min(window[row-N:row+N,col]):
                peaks += 1
        rval.append(peaks)
    return rval

def extract_features(window_orig):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """

    # to start with, the data has 3 dimensions (x, y, z)
    # we'll add the magnitude as a 4th dimension

    mag = np.linalg.norm(window_orig, axis=1)
    window =np.concatenate([window_orig, np.array([mag]).T], axis=1)
    
    x = []
    feature_names = []

    for name, func in [["mean",      _compute_mean_features],    # Category 1: Statistical
                       ["std",       _compute_std_features],     # Category 1: Statistical
                       ["median",    _compute_median_features],  # Category 1: Statistical
                       ["min",       _compute_min_features],     # Category 1: Statistical
                       ["max",       _compute_max_features],     # Category 1: Statistical
                       ["25p",       _compute_25p_features],     # Category 1: Statistical
                       ["75p",       _compute_75p_features],     # Category 1: Statistical
                       ["rfft_max",  _compute_rfft_max],         # Category 2: FFT
                       ["entropy",   _compute_entropy],          # Category 3: Other
                       ["peaks",     _compute_peaks],            # Category 4: Peak
                       ["valleys",   _compute_valleys]]:         # Category 4: Peak
        x.append(func(window))
        feature_names.append("x_" + name)
        feature_names.append("y_" + name)
        feature_names.append("z_" + name)
        feature_names.append("mag_" + name)

    feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector
    return feature_names, feature_vector
