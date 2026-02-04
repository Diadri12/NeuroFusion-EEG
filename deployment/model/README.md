# NeuroFusion-EEG Model Package

**Version:** 1.0.0  
**Export Date:** 2026-02-01 14:40:12  
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

print(f"Prediction: Class {prediction}")
print(f"Confidence: {confidence:.2%}")
print(f"Probabilities: {probabilities[0].numpy()}")
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

**Generated:** 2026-02-01 14:40:12  
**Source Model:** Contrastive  
**Export Script:** freeze_model.py
