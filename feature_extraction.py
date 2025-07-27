"""
Enhanced Multi-Domain Feature Extraction
Extracts 2,138-dimensional feature vectors from EEG epochs
"""

import numpy as np
import pywt
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import RobustScaler

class EnhancedFeatureExtractor:
    def __init__(self, n_channels=16, sampling_rate=128):
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.n_features = 2138
        
    def extract_features(self, eeg_epoch):
        """Extract comprehensive 2,138-dimensional features"""
        features = []
        
        # Time-frequency features per channel (110 features × 16 channels)
        for ch in range(self.n_channels):
            channel_data = eeg_epoch[ch, :]
            
            # DWT features (50 per channel)
            dwt_features = self.extract_dwt_features(channel_data)
            features.extend(dwt_features)
            
            # STFT features (60 per channel)  
            stft_features = self.extract_stft_features(channel_data)
            features.extend(stft_features)
            
            # Spectral features (15 per channel)
            spectral_features = self.extract_spectral_features(channel_data)
            features.extend(spectral_features)
            
            # Statistical features (8 per channel)
            statistical_features = self.extract_statistical_features(channel_data)
            features.extend(statistical_features)
            
        # Global connectivity features (10 features)
        connectivity_features = self.extract_connectivity_features(eeg_epoch)
        features.extend(connectivity_features)
        
        return np.array(features)
        
    def extract_dwt_features(self, signal_data):
        """Extract DWT features using Daubechies-4 wavelet"""
        coeffs = pywt.wavedec(signal_data, 'db4', level=4)
        features = []
        
        # Statistical measures of wavelet coefficients
        for coeff in coeffs:
            features.extend([
                np.mean(coeff), np.std(coeff), np.var(coeff),
                skew(coeff), kurtosis(coeff),
                np.percentile(coeff, 25), np.percentile(coeff, 75),
                np.max(coeff), np.min(coeff), np.median(coeff)
            ])
            
        return features[:50]  # Ensure exactly 50 features
        
    def extract_stft_features(self, signal_data):
        """Extract STFT features"""
        f, t, Zxx = signal.stft(signal_data, self.sampling_rate, 
                               window='hann', nperseg=128, noverlap=64)
        
        # Power spectral density
        psd = np.abs(Zxx)**2
        
        # Statistical measures across time
        features = []
        for freq_bin in range(min(60, psd.shape[0])):
            features.append(np.mean(psd[freq_bin, :]))
            
        return features[:60]  # Ensure exactly 60 features
        
    def extract_spectral_features(self, signal_data):
        """Extract spectral domain features"""
        # Power spectral density
        freqs, psd = signal.welch(signal_data, self.sampling_rate)
        
        # Frequency bands
        delta = (0.5, 4)    # Delta band
        theta = (4, 8)      # Theta band  
        alpha = (8, 13)     # Alpha band
        beta = (13, 30)     # Beta band
        gamma = (30, 45)    # Gamma band
        
        bands = [delta, theta, alpha, beta, gamma]
        features = []
        
        # Relative band powers
        total_power = np.sum(psd)
        for low, high in bands:
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(psd[band_mask]) / total_power
            features.append(band_power)
            
        # Spectral edge frequencies
        cumsum_psd = np.cumsum(psd)
        features.append(freqs[np.where(cumsum_psd >= 0.5 * total_power)[0][0]])  # 50% edge
        features.append(freqs[np.where(cumsum_psd >= 0.95 * total_power)[0][0]]) # 95% edge
        
        # Spectral entropy
        psd_norm = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        features.append(spectral_entropy)
        
        # Additional spectral statistics
        features.extend([
            np.mean(psd), np.std(psd), skew(psd), kurtosis(psd),
            np.max(psd), np.argmax(psd)
        ])
        
        return features[:15]  # Ensure exactly 15 features
        
    def extract_statistical_features(self, signal_data):
        """Extract statistical domain features"""
        features = [
            np.mean(signal_data),           # Mean
            np.median(signal_data),         # Median  
            np.std(signal_data),            # Standard deviation
            skew(signal_data),              # Skewness
            kurtosis(signal_data),          # Kurtosis
            np.max(signal_data) - np.min(signal_data),  # Range
            np.std(signal_data) / np.mean(np.abs(signal_data)),  # Coefficient of variation
            np.mean(np.abs(np.diff(signal_data)))  # Mean absolute derivative
        ]
        
        return features
        
    def extract_connectivity_features(self, eeg_epoch):
        """Extract connectivity features across channels"""
        # Inter-channel correlation matrix
        corr_matrix = np.corrcoef(eeg_epoch)
        
        # Remove diagonal elements
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        correlations = corr_matrix[mask]
        
        features = [
            np.mean(correlations),      # Mean correlation
            np.std(correlations),       # Std of correlations
            np.max(correlations),       # Max correlation
            np.min(correlations),       # Min correlation
            np.median(correlations),    # Median correlation
            np.percentile(correlations, 25),  # 25th percentile
            np.percentile(correlations, 75),  # 75th percentile
            skew(correlations),         # Skewness
            kurtosis(correlations),     # Kurtosis
            np.sum(np.abs(correlations) > 0.5) / len(correlations)  # High correlation ratio
        ]
        
        return features
