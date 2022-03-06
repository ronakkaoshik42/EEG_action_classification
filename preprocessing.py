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


def augment_data(X_train, y_train, person_train, subsamples=5):
    '''
    Performs data augmentation to return data of a larger sample size
    
    Input: 
    X_train: (# trials, # EEG signals, # time bins)
    y_train: (# trials,)
    person_train: (# trials, 1)
    subsamples: Multiplication factor for output data size.
    
    Returns: Returns a larger sample size of X, y, and persons data.
    Dimensions of return data depends on the number of subsamples specified.
    '''    

    num_trials = X_train.shape[0]
    num_signals = X_train.shape[1]
    num_bins = X_train.shape[2]
    
    augmented_num_trials = subsamples * num_trials

    X_train_augmented = np.zeros((augmented_num_trials, num_signals, num_bins//subsamples))
    y_train_augmented = np.zeros(augmented_num_trials,)
    person_train_augmented = np.zeros((augmented_num_trials, 1))

    for i in range(sample_every):
        start = i * num_trials
        end = start + num_trials
        X_train_augmented[start:end] = X_train[:,:,i::subsamples]
        y_train_augmented[start:end] = y_train[:,]
        person_train_augmented[start:end] = person_train[:,:]

    return X_train_augmented, y_train_augmented, person_train_augmented

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