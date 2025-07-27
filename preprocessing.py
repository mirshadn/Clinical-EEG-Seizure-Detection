"""
EEG Signal Preprocessing Module
- Channel selection (16 bipolar channels)
- Notch filtering (60Hz), Bandpass filtering (0.5-45Hz)
- Robust normalization, Multi-scale windowing
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import RobustScaler

class EEGPreprocessor:
    def __init__(self, sampling_rate=128, n_channels=16):
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        
    def preprocess_signal(self, eeg_data):
        """Complete preprocessing pipeline"""
        # Notch filter (60Hz)
        eeg_filtered = self.notch_filter(eeg_data)
        
        # Bandpass filter (0.5-45Hz)
        eeg_filtered = self.bandpass_filter(eeg_filtered)
        
        # Robust normalization
        eeg_normalized = self.robust_normalize(eeg_filtered)
        
        return eeg_normalized
        
    def notch_filter(self, data, notch_freq=60):
        """Apply notch filter"""
        nyquist = self.sampling_rate / 2
        low = (notch_freq - 1) / nyquist
        high = (notch_freq + 1) / nyquist
        b, a = signal.butter(4, [low, high], btype='bandstop')
        return signal.filtfilt(b, a, data)
        
    def bandpass_filter(self, data, low_freq=0.5, high_freq=45):
        """Apply bandpass filter"""
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
        
    def robust_normalize(self, data):
        """Robust normalization using median and IQR"""
        median_val = np.median(data, axis=1, keepdims=True)
        q75, q25 = np.percentile(data, [75, 25], axis=1, keepdims=True)
        iqr = q75 - q25 + 1e-8
        return (data - median_val) / iqr
