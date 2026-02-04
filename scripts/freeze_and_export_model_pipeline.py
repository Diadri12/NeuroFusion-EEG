"""
FREEZE & EXPORT MODEL PIPELINE
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import joblib
from datetime import datetime

print("FREEZING MODEL FOR DEPLOYMENT")

BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
PHASE3_DIR = f"{BASE_DIR}/outputs/dual_branch_training"
PHASE4_DIR = f"{BASE_DIR}/outputs/error_diagnosis"

# Create deployment directory
DEPLOY_DIR = Path(BASE_DIR) / "deployment"
DEPLOY_DIR.mkdir(exist_ok=True, parents=True)

MODEL_DIR = DEPLOY_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

print(f" Deployment directory: {DEPLOY_DIR}")

# MODEL ARCHITECTURE DEFINITION
print("DEFINING MODEL ARCHITECTURE")

# CNN branch for raw signal processing
class BranchA_LayerNorm(nn.Module):
    def __init__(self, signal_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 7, 2, 3)
        self.norm1 = nn.GroupNorm(1, 32)
        self.conv2 = nn.Conv1d(32, 64, 5, 1, 2)
        self.norm2 = nn.GroupNorm(1, 64)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, 1)
        self.norm3 = nn.GroupNorm(1, 128)
        self.conv4 = nn.Conv1d(128, 256, 3, 1, 1)
        self.norm4 = nn.GroupNorm(1, 256)
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Sequential(nn.Linear(256, 64), nn.ReLU())
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.pool(self.relu(self.norm1(self.conv1(x)))))
        x = self.dropout(self.pool(self.relu(self.norm2(self.conv2(x)))))
        x = self.dropout(self.pool(self.relu(self.norm3(self.conv3(x)))))
        x = self.dropout(self.pool(self.relu(self.norm4(self.conv4(x)))))
        return self.embedding(self.gap(x).squeeze(-1))

# MLP branch for handcrafted features
class BranchB_Features(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

    def forward(self, x):
        return self.model(x)

# Complete dual-branch architecture
class DualBranchModel(nn.Module):
    def __init__(self, signal_length, n_features, n_classes):
        super().__init__()
        self.branch_a = BranchA_LayerNorm(signal_length)
        self.branch_b = BranchB_Features(n_features)
        self.classifier = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_classes)
        )

    def forward(self, signals, features):
        emb_a = self.branch_a(signals)
        emb_b = self.branch_b(features)
        fused = torch.cat([emb_a, emb_b], dim=1)
        return self.classifier(fused)

print(" Model architecture defined")

# LOAD TRAINED MODEL
print("LOADING TRAINED MODEL")

# Model configuration
signal_length = 256
n_features = 30
n_classes = 3

# Try to load best model
model_paths = [
    (Path(PHASE4_DIR) / "contrastive_pretraining_results" / "final_model.pt", "Contrastive"),
    (Path(PHASE4_DIR) / "balanced_sampling_results" / "balanced_model.pt", "Balanced"),
]

model = DualBranchModel(signal_length, n_features, n_classes)
model_loaded = False
model_source = "None"

for model_path, name in model_paths:
    if model_path.exists():
        try:
            print(f"\n Trying to load: {name}")
            checkpoint = torch.load(model_path, map_location='cpu')

            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=False)
            model.eval()

            model_loaded = True
            model_source = name
            print(f"   Successfully loaded {name} model")
            break
        except Exception as e:
            print(f"   Failed: {e}")

if not model_loaded:
    print("\n Could not load any model!")
    print("   Please check model paths")
    exit(1)

# LOAD PREPROCESSING COMPONENTS
print("LOADING PREPROCESSING COMPONENTS")

# Load feature scaler
scaler_path = Path(PHASE3_DIR) / "feature_scaler.pkl"
if scaler_path.exists():
    feature_scaler = joblib.load(scaler_path)
    print(f" Loaded feature scaler")
else:
    print(f" Feature scaler not found!")
    feature_scaler = None

# Load training statistics (for reference)
frozen_dir = Path(PHASE3_DIR) / "frozen"
if frozen_dir.exists():
    try:
        X_windowed = np.load(frozen_dir / "X_windowed.npy")
        y_windowed = np.load(frozen_dir / "y_windowed.npy")

        training_stats = {
            'n_samples': len(X_windowed),
            'signal_shape': X_windowed.shape,
            'class_distribution': np.bincount(y_windowed).tolist(),
            'signal_mean': float(X_windowed.mean()),
            'signal_std': float(X_windowed.std())
        }
        print(f" Loaded training statistics")
    except:
        training_stats = None
        print(f" Could not load training statistics")
else:
    training_stats = None

# EXPORT MODEL WEIGHTS
print("EXPORTING MODEL WEIGHTS")

# Save complete model state
model_export = {
    'model_state_dict': model.state_dict(),
    'model_architecture': {
        'signal_length': signal_length,
        'n_features': n_features,
        'n_classes': n_classes,
        'architecture_type': 'DualBranch_LayerNorm'
    },
    'model_source': model_source,
    'export_date': datetime.now().isoformat(),
    'pytorch_version': torch.__version__
}

torch.save(model_export, MODEL_DIR / "model_weights.pt")
print(f" Saved: model_weights.pt")

# Save model architecture as Python module
model_code = '''"""
Model Architecture for EEG Classification
Auto-generated - DO NOT MODIFY
"""

import torch
import torch.nn as nn

class BranchA_LayerNorm(nn.Module):
    """CNN branch for raw signal processing"""
    def __init__(self, signal_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 7, 2, 3)
        self.norm1 = nn.GroupNorm(1, 32)
        self.conv2 = nn.Conv1d(32, 64, 5, 1, 2)
        self.norm2 = nn.GroupNorm(1, 64)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, 1)
        self.norm3 = nn.GroupNorm(1, 128)
        self.conv4 = nn.Conv1d(128, 256, 3, 1, 1)
        self.norm4 = nn.GroupNorm(1, 256)
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Sequential(nn.Linear(256, 64), nn.ReLU())
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.pool(self.relu(self.norm1(self.conv1(x)))))
        x = self.dropout(self.pool(self.relu(self.norm2(self.conv2(x)))))
        x = self.dropout(self.pool(self.relu(self.norm3(self.conv3(x)))))
        x = self.dropout(self.pool(self.relu(self.norm4(self.conv4(x)))))
        return self.embedding(self.gap(x).squeeze(-1))

class BranchB_Features(nn.Module):
    """MLP branch for handcrafted features"""
    def __init__(self, n_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

    def forward(self, x):
        return self.model(x)

class DualBranchModel(nn.Module):
    """Complete dual-branch architecture"""
    def __init__(self, signal_length, n_features, n_classes):
        super().__init__()
        self.branch_a = BranchA_LayerNorm(signal_length)
        self.branch_b = BranchB_Features(n_features)
        self.classifier = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_classes)
        )

    def forward(self, signals, features):
        emb_a = self.branch_a(signals)
        emb_b = self.branch_b(features)
        fused = torch.cat([emb_a, emb_b], dim=1)
        return self.classifier(fused)
'''

with open(MODEL_DIR / "model_architecture.py", 'w') as f:
    f.write(model_code)

print(f" Saved: model_architecture.py")

# EXPORT PREPROCESSING PIPELINE
print("EXPORTING PREPROCESSING PIPELINE")

# Save feature scaler
if feature_scaler is not None:
    joblib.dump(feature_scaler, MODEL_DIR / "feature_scaler.pkl")
    print(f" Saved: feature_scaler.pkl")

# Save feature extraction code
feature_extraction_code = '''"""
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
'''

with open(MODEL_DIR / "feature_extraction.py", 'w') as f:
    f.write(feature_extraction_code)

print(f" Saved: feature_extraction.py")

# CREATE MODEL METADATA
print("CREATING MODEL METADATA")

metadata = {
    "model_info": {
        "name": "NeuroFusion-EEG Dual-Branch Model",
        "version": "1.0.0",
        "architecture": "DualBranch_LayerNorm",
        "source": model_source,
        "export_date": datetime.now().isoformat(),
        "frozen": True
    },

    "input_specifications": {
        "signal": {
            "shape": [1, signal_length],
            "dtype": "float32",
            "description": "Raw EEG signal",
            "length": signal_length,
            "sampling_rate": 173.61,
            "units": "microvolts (µV)",
            "preprocessing": "Resampled to 256 samples if different length"
        },
        "features": {
            "shape": [n_features],
            "dtype": "float32",
            "description": "Handcrafted features",
            "n_features": n_features,
            "normalization": "StandardScaler (fitted on training data)",
            "extraction": "See feature_extraction.py"
        }
    },

    "output_specifications": {
        "classes": n_classes,
        "class_names": ["Class 0", "Class 1", "Class 2"],
        "class_descriptions": {
            "0": "Healthy / Normal",
            "1": "Interictal / Abnormal non-seizure",
            "2": "Ictal / Seizure"
        },
        "output_format": {
            "logits": "Raw model outputs (3 values)",
            "probabilities": "Softmax probabilities (sum to 1)",
            "prediction": "Argmax of probabilities"
        }
    },

    "performance": {
        "within_dataset": {
            "macro_f1": 0.2763,
            "dataset": "Original training dataset"
        },
        "cross_dataset": {
            "bonn_f1": 0.1566,
            "retention": 0.567,
            "note": "Performance may vary on different datasets"
        }
    },

    "training_info": training_stats if training_stats else {},

    "requirements": {
        "python": ">=3.8",
        "pytorch": ">=1.10.0",
        "numpy": ">=1.21.0",
        "scipy": ">=1.7.0",
        "scikit-learn": ">=1.0.0"
    },

    "notes": [
        "Model is FROZEN - do not retrain",
        "Use provided preprocessing pipeline",
        "Input signals must be 256 samples long",
        "Features must be normalized with provided scaler",
        "Cross-dataset performance may be lower due to domain shift"
    ]
}

with open(MODEL_DIR / "model_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f" Saved: model_metadata.json")

# CREATE README
print("CREATING README")

readme = f'''# NeuroFusion-EEG Model Package

**Version:** 1.0.0
**Export Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Status:**  FROZEN (Do not retrain)

## Overview

This package contains the complete, frozen inference pipeline for the NeuroFusion-EEG dual-branch classification model.

## Contents

```
model/
├── model_weights.pt           # Trained model weights
├── model_architecture.py      # Model architecture definition
├── feature_scaler.pkl         # Feature normalization scaler
├── feature_extraction.py      # Feature extraction code
├── model_metadata.json        # Complete model specifications
└── README.md                  # This file
```

## Model Specifications

### Input Requirements

**Signal Input:**
- Shape: (1, 256)
- Type: float32
- Length: 256 samples
- Sampling Rate: 173.61 Hz
- Units: Microvolts (µV)

**Feature Input:**
- Shape: (30,)
- Type: float32
- Features: 30 handcrafted features
- Normalization: StandardScaler (provided)

### Output Format

**Classes:** 3
- Class 0: Healthy / Normal
- Class 1: Interictal / Abnormal non-seizure
- Class 2: Ictal / Seizure

**Output Types:**
- Logits: Raw model outputs (3 values)
- Probabilities: Softmax probabilities (sum to 1.0)
- Prediction: Argmax of probabilities

### Performance

- **Within-dataset F1:** 0.2763
- **Cross-dataset F1 (Bonn):** 0.1566 (56.7% retention)

**Note:** Performance on new datasets may vary due to domain shift.

## Quick Start

### Loading the Model

```python
import torch
from model_architecture import DualBranchModel

# Load model
model = DualBranchModel(signal_length=256, n_features=30, n_classes=3)
checkpoint = torch.load('model_weights.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Running Inference

```python
import numpy as np
import joblib
from feature_extraction import extract_features

# Load scaler
scaler = joblib.load('feature_scaler.pkl')

# Prepare input signal (256 samples)
signal = np.random.randn(256)  # Replace with actual EEG signal

# Extract features
features = extract_features(signal, fs=173.61)
features_scaled = scaler.transform(features.reshape(1, -1))

# Prepare tensors
signal_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0)  # (1, 1, 256)
features_tensor = torch.FloatTensor(features_scaled)  # (1, 30)

# Run inference
with torch.no_grad():
    outputs = model(signal_tensor, features_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    prediction = outputs.argmax(1).item()
    confidence = probabilities[0, prediction].item()

print(f"Prediction: Class {{prediction}}")
print(f"Confidence: {{confidence:.2%}}")
print(f"Probabilities: {{probabilities[0].numpy()}}")
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- scikit-learn >= 1.0.0

## Important Notes

1. ** Model is FROZEN** - Do not retrain or modify
2. **Preprocessing Required** - Always use provided feature extraction and scaler
3. **Signal Length** - Input must be exactly 256 samples
4. **Domain Shift** - Performance may degrade on datasets significantly different from training data
5. **Feature Normalization** - Must use the provided StandardScaler

## Troubleshooting

**Q: Model gives poor predictions on my data**
A: This may be due to domain shift. Consider:
- Checking signal preprocessing
- Verifying sampling rate matches (173.61 Hz)
- Using per-sample normalization instead of StandardScaler
- Fine-tuning on a small labeled set from your domain

**Q: Input shape mismatch**
A: Ensure:
- Signal is exactly 256 samples long
- Signal tensor shape is (1, 1, 256)
- Features tensor shape is (1, 30)

**Q: How to handle different signal lengths?**
A: Resample to 256 samples:
```python
if len(signal) != 256:
    indices = np.linspace(0, len(signal)-1, 256).astype(int)
    signal = signal[indices]
---

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Source Model:** {model_source}
**Export Script:** freeze_model.py
'''

with open(MODEL_DIR / "README.md", 'w') as f:
    f.write(readme)

print(f" Saved: README.md")

# CREATE VERSION FILE
version_info = {
    "version": "1.0.0",
    "export_date": datetime.now().isoformat(),
    "model_source": model_source,
    "frozen": True,
    "checksums": {
        "model_weights": "Not computed",
        "feature_scaler": "Not computed"
    }
}

with open(MODEL_DIR / "VERSION", 'w') as f:
    json.dump(version_info, f, indent=2)

print(f" Saved: VERSION")

# SUMMARY
print(" MODEL FREEZING COMPLETE!")

print(f"\n Model Status: FROZEN")
print(f"   Source: {model_source}")
print(f"   Version: 1.0.0")
print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\n This model is now production-ready!")
print(f"   Treat it as a black box service for inference.\n")
