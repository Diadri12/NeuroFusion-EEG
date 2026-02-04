"""
Feature Extraction for EEG Classification
Auto-generated - DO NOT MODIFY
"""

import numpy as np
from scipy.stats import entropy

def extract_features(signal, fs=173.61):
    """
    Extract 30 handcrafted features from EEG signal
    
    Parameters:
    -----------
    signal : np.ndarray
        1D EEG signal (256 samples)
    fs : float
        Sampling frequency (default: 173.61 Hz)
    
    Returns:
    --------
    features : np.ndarray
        30-dimensional feature vector
    """
    signal = np.array(signal).flatten().astype(np.float64)
    features = []
    
    # FFT-based features
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    psd = np.abs(fft)**2
    
    # Band power features (5 bands × 4 features = 20)
    bands = [
        ('delta', 0.5, 4),
        ('theta', 4, 8),
        ('alpha', 8, 13),
        ('beta', 13, 30),
        ('gamma', 30, 50)
    ]
    
    for name, low, high in bands:
        mask = (freqs >= low) & (freqs <= high)
        if np.any(mask):
            band_power = float(np.sum(psd[mask]))
            band_mean = float(np.mean(psd[mask]))
            band_std = float(np.std(psd[mask]))
            peak_freq = float(freqs[mask][np.argmax(psd[mask])])
        else:
            band_power = band_mean = band_std = peak_freq = 0.0
        
        features.extend([band_power, band_mean, band_std, peak_freq])
    
    # Band ratios (3)
    delta_power = features[0]
    theta_power = features[4]
    alpha_power = features[8]
    beta_power = features[12]
    
    features.append(float(theta_power / (alpha_power + 1e-10)))
    features.append(float(alpha_power / (beta_power + 1e-10)))
    features.append(float(delta_power / (theta_power + 1e-10)))
    
    # Entropy measures (3)
    spectral_entropy = float(entropy(np.abs(fft[:len(fft)//2]) + 1e-10))
    sample_entropy_proxy = float(np.std(signal))
    approx_entropy_proxy = float(np.std(np.diff(signal)))
    
    features.extend([spectral_entropy, sample_entropy_proxy, approx_entropy_proxy])
    
    # Hjorth parameters (3)
    activity = np.var(signal)
    diff1 = np.diff(signal)
    mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
    diff2 = np.diff(diff1)
    complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
    
    features.extend([float(activity), float(mobility), float(complexity)])
    
    # Zero-crossing rate (1)
    zero_crossings = float(np.sum(np.diff(np.sign(signal)) != 0))
    features.append(zero_crossings)
    
    return np.array(features, dtype=np.float64)
