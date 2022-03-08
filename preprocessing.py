import numpy as np
from scipy import signal
from sklearn import preprocessing

def normalize_data(X):
    '''
    Utilizes sklearn's preprocessing library to normalize and scale data.
    
    Input: Data with shape (# trials, # EEG signals, # time bins)
    
    Returns: Normalized version of input parameter
    '''
    
    num_trials = X.shape[0]
    num_signals = X.shape[1]
    num_bins = X.shape[2]
    
    reshaped_data = np.reshape(X, (num_trials*num_signals, num_bins))
    normalized_data = preprocessing.scale(reshaped_data,axis=1)
    
    return np.reshape(normalized_data, (num_trials, num_signals, num_bins))

def smooth_data(X, num_points=5):
    '''
    Smooths data by using Hanning window. See reference below for choice of window.
    
    Input: 
    X: Data of any arbritary size
    num_points = Number of points in the output window.
    
    Returns: Smooth version of input data
    
    Reference: https://www.wjir.org/download_data/WJIR0804006.pdf
    States Kaiser and Hanning are the best window choices for EEG Data.
    '''
    
    window = signal.windows.hann(sym=True, M=num_points)
    window_reshaped = window.reshape((1, 1, num_points))
    window_normalized = window_reshaped/np.sum(window_reshaped)
    
    return signal.convolve(X, window_normalized, mode='same', method='fft')
