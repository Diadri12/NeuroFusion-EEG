"""
DUAL-BRANCH FUSION CNN - STEP 3: Dataset Class
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import pywt
from pathlib import Path
import json

class DualBranchEEGDataset(Dataset):
    def __init__(self, raw_signals, labels, n_classes,
                 transform_type='stft', transform_params=None,
                 augment=False, precomputed_transforms=None):

        # Normalize labels to start from 0
        labels = np.array(labels)
        min_label = labels.min()
        if min_label != 0:
            labels = labels - min_label

        # Remove invalid labels
        mask = (labels >= 0) & (labels < n_classes)
        if not mask.all():
            print(f" Removing {(~mask).sum()} invalid samples")
            raw_signals = raw_signals[mask]
            labels = labels[mask]

        self.raw_signals = raw_signals
        self.labels = torch.LongTensor(labels)
        self.n_classes = n_classes
        self.transform_type = transform_type
        self.transform_params = transform_params or {}
        self.augment = augment

        # Precompute or prepare transforms
        if precomputed_transforms is not None:
            print("Using precomputed transforms")
            self.transformed_features = precomputed_transforms[mask]
            self.precomputed = True
        else:
            print(f"Transforms will be computed on-the-fly using {transform_type}")
            self.precomputed = False
            # Compute one sample to determine output shape
            sample_transform = self._compute_transform(raw_signals[0])
            self.transform_shape = sample_transform.shape
            print(f"Transform output shape: {self.transform_shape}")

        print(f" Dataset ready: {len(self.labels)} samples")

    # Compute transform for a single signal
    def _compute_transform(self, signal_data):
        if self.transform_type == 'stft':
            return self._stft_transform(signal_data)
        elif self.transform_type == 'wavelet':
            return self._wavelet_transform(signal_data)
        elif self.transform_type == 'cwt':
            return self._cwt_transform(signal_data)
        elif self.transform_type == 'psd':
            return self._psd_transform(signal_data)
        elif self.transform_type == 'spectrogram':
            return self._spectrogram_transform(signal_data)
        else:
            raise ValueError(f"Unknown transform: {self.transform_type}")

    # Short-Time Fourier Transform
    def _stft_transform(self, signal_data):
        nperseg = self.transform_params.get('nperseg', 64)
        noverlap = self.transform_params.get('noverlap', 32)

        f, t, Zxx = signal.stft(signal_data, nperseg=nperseg, noverlap=noverlap)
        magnitude = np.abs(Zxx)

        # Return as (1, freq, time) for 2D conv or flatten for 1D
        if self.transform_params.get('flatten', False):
            return magnitude.flatten()
        else:
            return magnitude

    # Discrete Wavelet Transform
    def _wavelet_transform(self, signal_data):
        wavelet = self.transform_params.get('wavelet', 'db4')
        level = self.transform_params.get('level', 4)

        coeffs = pywt.wavedec(signal_data, wavelet, level=level)

        # Concatenate all coefficients
        features = np.concatenate([c.flatten() for c in coeffs])
        return features

    # Continuous Wavelet Transform - returns 2D scalogram
    def _cwt_transform(self, signal_data):
        scales = self.transform_params.get('scales', np.arange(1, 64))
        wavelet = self.transform_params.get('wavelet', 'morl')

        coefficients, _ = pywt.cwt(signal_data, scales, wavelet)
        return np.abs(coefficients)

    # Power Spectral Density
    def _psd_transform(self, signal_data):
        nperseg = self.transform_params.get('nperseg', 256)

        f, Pxx = signal.welch(signal_data, nperseg=nperseg)
        return Pxx
   # Spectrogram
    def _spectrogram_transform(self, signal_data):
        nperseg = self.transform_params.get('nperseg', 64)

        f, t, Sxx = signal.spectrogram(signal_data, nperseg=nperseg)

        # Return as 2D or flattened
        if self.transform_params.get('flatten', False):
            return Sxx.flatten()
        else:
            return Sxx

    # Apply data augmentation to raw signal
    def _augment_signal(self, signal_data):
        # Random noise injection
        if torch.rand(1) > 0.5:
            noise = np.random.randn(*signal_data.shape) * 0.01
            signal_data = signal_data + noise

        # Random amplitude scaling
        if torch.rand(1) > 0.5:
            scale = 0.9 + torch.rand(1).item() * 0.2
            signal_data = signal_data * scale

        # Random shift (circular)
        if torch.rand(1) > 0.7:
            shift = np.random.randint(-50, 50)
            signal_data = np.roll(signal_data, shift)

        return signal_data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get raw signal
        raw_signal = self.raw_signals[idx].copy()

        # Apply augmentation if enabled
        if self.augment:
            raw_signal = self._augment_signal(raw_signal)

        # Convert to tensor
        raw_tensor = torch.FloatTensor(raw_signal).unsqueeze(0)  # (1, signal_length)

        # Get or compute transformed features
        if self.precomputed:
            transformed = self.transformed_features[idx]
        else:
            transformed = self._compute_transform(raw_signal)

        # Convert to tensor with proper shape
        if transformed.ndim == 1:
            # 1D transform
            transformed_tensor = torch.FloatTensor(transformed).unsqueeze(0)
        elif transformed.ndim == 2:
            # 2D transform
            transformed_tensor = torch.FloatTensor(transformed).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected transform dimension: {transformed.ndim}")

        label = self.labels[idx]

        return raw_tensor, transformed_tensor, label

class PrecomputeTransforms:

    @staticmethod
    def compute_all_transforms(raw_signals, transform_type, transform_params=None):
        print(f"Precomputing {transform_type} transforms for {len(raw_signals)} samples")

        # Create temporary dataset to use transform methods
        temp_dataset = DualBranchEEGDataset(
            raw_signals=raw_signals[:1],  # Just one sample to initialize
            labels=np.array([0]),
            n_classes=2,
            transform_type=transform_type,
            transform_params=transform_params,
            precomputed_transforms=None
        )

        # Compute all transforms
        all_transforms = []
        for i, sig in enumerate(raw_signals):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(raw_signals)}")

            transformed = temp_dataset._compute_transform(sig)
            all_transforms.append(transformed)

        # Stack into array
        all_transforms = np.array(all_transforms)
        print(f" Precomputed transforms shape: {all_transforms.shape}")

        return all_transforms

    # Save precomputed transforms to disk
    @staticmethod
    def save_transforms(transforms, save_path):
        np.save(save_path, transforms)
        print(f" Saved transforms to {save_path}")

   # Load precomputed transforms from disk
    @staticmethod
    def load_transforms(load_path):
        transforms = np.load(load_path)
        print(f" Loaded transforms from {load_path}, shape: {transforms.shape}")
        return transforms

# Test the dataset class
if __name__ == "__main__":
    print("TESTING DUAL-BRANCH EEG DATASET")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    signal_length = 4096
    n_classes = 5

    raw_signals = np.random.randn(n_samples, signal_length)
    labels = np.random.randint(0, n_classes, n_samples)

    print(f"\nSynthetic data: {n_samples} samples, {signal_length} length, {n_classes} classes")

    # Test 1: On-the-fly transforms
    print("Test 1: On-the-fly STFT Transform")

    dataset_stft = DualBranchEEGDataset(
        raw_signals=raw_signals,
        labels=labels,
        n_classes=n_classes,
        transform_type='stft',
        transform_params={'nperseg': 64, 'noverlap': 32, 'flatten': False},
        augment=True
    )

    # Get a sample
    raw, transformed, label = dataset_stft[0]
    print(f" Sample 0:")
    print(f"  Raw shape: {raw.shape}")
    print(f"  Transformed shape: {transformed.shape}")
    print(f"  Label: {label}")

    # Test DataLoader
    dataloader = DataLoader(dataset_stft, batch_size=8, shuffle=True)
    raw_batch, transformed_batch, label_batch = next(iter(dataloader))
    print(f"\n DataLoader batch:")
    print(f"  Raw batch shape: {raw_batch.shape}")
    print(f"  Transformed batch shape: {transformed_batch.shape}")
    print(f"  Label batch shape: {label_batch.shape}")

    # Test 2: Different transforms
    print("Test 2: Different Transform Types")

    transforms_to_test = [
        ('wavelet', {'wavelet': 'db4', 'level': 4}),
        ('cwt', {'scales': np.arange(1, 32), 'wavelet': 'morl'}),
        ('psd', {'nperseg': 256}),
        ('spectrogram', {'nperseg': 64, 'flatten': False})
    ]

    for transform_type, params in transforms_to_test:
        dataset = DualBranchEEGDataset(
            raw_signals=raw_signals[:10],  # Use fewer samples for testing
            labels=labels[:10],
            n_classes=n_classes,
            transform_type=transform_type,
            transform_params=params,
            augment=False
        )

        raw, transformed, label = dataset[0]
        print(f" {transform_type.upper():15s} - Transformed shape: {transformed.shape}")

    # Test 3: Precomputed transforms
    print("Test 3: Precomputed Transforms")

    # Precompute STFT for all samples
    precomputed = PrecomputeTransforms.compute_all_transforms(
        raw_signals=raw_signals,
        transform_type='stft',
        transform_params={'nperseg': 64, 'noverlap': 32}
    )

    # Create dataset with precomputed transforms
    dataset_precomputed = DualBranchEEGDataset(
        raw_signals=raw_signals,
        labels=labels,
        n_classes=n_classes,
        transform_type='stft',  # This is ignored when precomputed is provided
        precomputed_transforms=precomputed,
        augment=False
    )

    raw, transformed, label = dataset_precomputed[0]
    print(f" Precomputed dataset sample:")
    print(f"  Transformed shape: {transformed.shape}")

    # Test 4: Save and load transforms
    print("Test 4: Save/Load Precomputed Transforms")

    save_path = "/tmp/test_transforms.npy"
    PrecomputeTransforms.save_transforms(precomputed, save_path)
    loaded_transforms = PrecomputeTransforms.load_transforms(save_path)

    print(f" Shapes match: {np.array_equal(precomputed, loaded_transforms)}")

    print(" ALL DATASET TESTS PASSED")
