"""
Advanced Feature Engineering
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import pandas as pd
from scipy import signal, stats
import pywt
from pathlib import Path
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import time

print("ADVANCED EEG FEATURE EXTRACTION")

BASE_DIR = '/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG'
PROCESSED_DIR = f'{BASE_DIR}/outputs/final_processed'
OUTPUT_DIR = f'{BASE_DIR}/outputs/advanced_features'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

DATASETS = {
    'bonn_eeg': {'fs': 173.61, 'name': 'Bonn EEG'},
    'epilepsy_122mb': {'fs': 178, 'name': 'Epilepsy 122MB'},
    'epileptic_seizure': {'fs': 173.61, 'name': 'Epileptic Seizure Dataset'}
}

def extract_band_powers(sig, fs):
    """Extract EEG band powers - FAST"""
    features = {}
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(256, len(sig)))
    
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 
             'beta': (13, 30), 'gamma': (30, 50)}
    
    total_power = np.sum(psd) + 1e-10
    
    for band_name, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs < high)
        band_power = np.sum(psd[idx])
        features[f'{band_name}_power'] = band_power
        features[f'{band_name}_relative'] = band_power / total_power
    
    features['theta_alpha_ratio'] = features['theta_power'] / (features['alpha_power'] + 1e-10)
    features['delta_beta_ratio'] = features['delta_power'] / (features['beta_power'] + 1e-10)
    features['peak_frequency'] = freqs[np.argmax(psd)]
    
    return features

def extract_stft_features(sig, fs):
    """STFT features"""
    features = {}
    nperseg = min(128, len(sig) // 4)  # Reduced for speed
    if nperseg < 16:
        return {f'stft_{k}': 0.0 for k in ['mean', 'std', 'max', 'energy']}
    
    f, t, Zxx = signal.stft(sig, fs=fs, nperseg=nperseg)
    magnitude = np.abs(Zxx)
    
    features['stft_mean'] = np.mean(magnitude)
    features['stft_std'] = np.std(magnitude)
    features['stft_max'] = np.max(magnitude)
    features['stft_energy'] = np.sum(magnitude**2)
    
    return features

def extract_wavelet_features(sig, wavelet='db4', level=4):  # Reduced level
    """Wavelet features"""
    features = {}
    
    try:
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        
        for i, coeff in enumerate(coeffs):
            prefix = f'wavelet_L{i}'
            features[f'{prefix}_energy'] = np.sum(coeff**2)
            features[f'{prefix}_mean'] = np.mean(coeff)
            features[f'{prefix}_std'] = np.std(coeff)
        
        energies = [np.sum(c**2) for c in coeffs]
        total_energy = sum(energies) + 1e-10
        
        for i, energy in enumerate(energies):
            features[f'wavelet_L{i}_ratio'] = energy / total_energy
        
        probs = np.array(energies) / total_energy
        features['wavelet_entropy'] = -np.sum(probs * np.log2(probs + 1e-10))
        
    except:
        # Fill with zeros if wavelet fails
        for i in range(level + 1):
            features[f'wavelet_L{i}_energy'] = 0.0
            features[f'wavelet_L{i}_mean'] = 0.0
            features[f'wavelet_L{i}_std'] = 0.0
            features[f'wavelet_L{i}_ratio'] = 0.0
        features['wavelet_entropy'] = 0.0
    
    return features

def extract_entropy_features_fast(sig):
    """Entropy"""
    features = {}
    
    # Shannon Entropy
    freqs, psd = signal.welch(sig, nperseg=min(128, len(sig)))
    psd_norm = psd / (np.sum(psd) + 1e-10)
    features['shannon_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    features['spectral_entropy'] = features['shannon_entropy']
    
    # SVD Entropy
    try:
        N = min(len(sig), 1000)  # Limit size
        mat = np.array([sig[i:i+2] for i in range(N-2)])
        _, S, _ = np.linalg.svd(mat[:100, :])  # Use subset
        S_norm = S / (np.sum(S) + 1e-10)
        features['svd_entropy'] = -np.sum(S_norm * np.log2(S_norm + 1e-10))
    except:
        features['svd_entropy'] = 0.0
    
    return features

def extract_hjorth_parameters(sig):
    """Hjorth parameters"""
    dx = np.diff(sig)
    ddx = np.diff(dx)
    
    activity = np.var(sig)
    mobility = np.sqrt(np.var(dx) / (activity + 1e-10))
    complexity = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-10)) / (mobility + 1e-10)
    
    return {
        'hjorth_activity': activity,
        'hjorth_mobility': mobility,
        'hjorth_complexity': complexity
    }

def extract_statistical_features(sig):
    """Statistical features"""
    return {
        'mean': np.mean(sig),
        'std': np.std(sig),
        'var': np.var(sig),
        'median': np.median(sig),
        'min': np.min(sig),
        'max': np.max(sig),
        'range': np.ptp(sig),
        'rms': np.sqrt(np.mean(sig**2)),
        'skewness': stats.skew(sig),
        'kurtosis': stats.kurtosis(sig),
        'q25': np.percentile(sig, 25),
        'q75': np.percentile(sig, 75),
        'iqr': np.percentile(sig, 75) - np.percentile(sig, 25),
        'zero_crossings': np.sum(np.diff(np.sign(sig)) != 0),
        'energy': np.sum(sig**2),
        'curve_length': np.sum(np.abs(np.diff(sig)))
    }

def extract_all_features_fast(sig, fs):
    """Extract ALL features"""
    all_features = {}
    
    # Each category ~0.01 seconds
    all_features.update(extract_statistical_features(sig))
    all_features.update(extract_band_powers(sig, fs))
    all_features.update(extract_stft_features(sig, fs))
    all_features.update(extract_wavelet_features(sig))
    all_features.update(extract_entropy_features_fast(sig))
    all_features.update(extract_hjorth_parameters(sig))
    
    return all_features

# Process Datasets
def process_dataset(dataset_name, dataset_info):
    print(f"Processing: {dataset_info['name']}")
    
    fs = dataset_info['fs']
    
    # Load signals
    if dataset_name == 'epileptic_seizure':
        signals_path = Path(BASE_DIR) / 'outputs'
        signal_file = signals_path / 'full_processed_signals.npy'
        label_file = signals_path / 'full_labels.npy'
        
        if not (signal_file.exists() and label_file.exists()):
            print(f"  ✗ Signal files not found")
            return None
        
        signals = np.load(signal_file, allow_pickle=True)
        labels = np.load(label_file, allow_pickle=True)
    else:
        signals_path = Path(PROCESSED_DIR) / dataset_name
        
        if not (signals_path / 'preprocessed_signals.npy').exists():
            print(f"   No raw signals found")
            return None
        
        signals = np.load(signals_path / 'preprocessed_signals.npy', allow_pickle=True)
        labels = np.load(signals_path / 'labels.npy', allow_pickle=True)
    
    print(f"   Loaded {len(signals):,} signals")
    
    # Extract features with progress bar
    print(f"  Extracting features")
    features_list = []
    start_time = time.time()
    
    for i, sig in enumerate(signals):
        # Progress update every 50 signals
        if i % 50 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(signals) - i) / rate if rate > 0 else 0
            print(f"    Progress: {i:,}/{len(signals):,} ({100*i/len(signals):.1f}%) - "
                  f"{rate:.1f} signals/sec - ETA: {remaining/60:.1f} min")
        
        try:
            features = extract_all_features_fast(sig, fs)
            features['label'] = labels[i]
            features_list.append(features)
        except:
            continue
    
    elapsed_total = time.time() - start_time
    print(f"   Extracted {len(features_list):,} feature sets in {elapsed_total:.1f}s "
          f"({len(features_list)/elapsed_total:.1f} signals/sec)")
    
    # Create DataFrame
    features_df = pd.DataFrame(features_list)
    X = features_df.drop('label', axis=1).values
    y = features_df['label'].values
    feature_names = features_df.drop('label', axis=1).columns.tolist()
    
    print(f"  Total features: {len(feature_names)}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save
    output_path = Path(OUTPUT_DIR) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / 'X_train.npy', X_train_scaled)
    np.save(output_path / 'X_val.npy', X_val_scaled)
    np.save(output_path / 'X_test.npy', X_test_scaled)
    np.save(output_path / 'y_train.npy', y_train)
    np.save(output_path / 'y_val.npy', y_val)
    np.save(output_path / 'y_test.npy', y_test)
    joblib.dump(scaler, output_path / 'scaler.pkl')
    
    with open(output_path / 'feature_names.txt', 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    features_df.to_csv(output_path / 'all_features.csv', index=False)
    
    metadata = {
        'dataset': dataset_info['name'],
        'sampling_rate': fs,
        'total_samples': len(features_list),
        'total_features': len(feature_names),
        'classes': int(len(np.unique(y))),
        'splits': {
            'train': int(len(X_train)),
            'val': int(len(X_val)),
            'test': int(len(X_test))
        },
        'extraction_time_seconds': float(elapsed_total),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   Saved to: {output_path}")
    
    return metadata

# Main
if __name__ == '__main__':
    overall_start = time.time()
    results = {}
    
    for dataset_name, dataset_info in DATASETS.items():
        try:
            metadata = process_dataset(dataset_name, dataset_info)
            if metadata:
                results[dataset_name] = metadata
        except Exception as e:
            print(f"\n Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    overall_time = time.time() - overall_start
    print("FEATURE EXTRACTION COMPLETE!")
    print(f"Total time: {overall_time/60:.1f} minutes\n")
    
    for name, meta in results.items():
        print(f" {meta['dataset']}")
        print(f"    Samples: {meta['total_samples']:,}")
        print(f"    Features: {meta['total_features']}")
        print(f"    Time: {meta['extraction_time_seconds']:.1f}s\n")
    
    print(f"Output: {OUTPUT_DIR}")
