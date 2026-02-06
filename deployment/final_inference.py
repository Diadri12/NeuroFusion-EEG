"""
EEG Seizure Classification Inference Script
"""

import os
from pathlib import Path
import numpy as np
import torch
import joblib
import importlib.util

#  Paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
MODEL_WEIGHTS = MODEL_DIR / "model_weights.pt"
FEATURE_EXTRACTION_PATH = MODEL_DIR / "feature_extraction.py"
MODEL_ARCH_PATH = MODEL_DIR / "model_architecture.py"

# Load feature extraction dynamically
spec = importlib.util.spec_from_file_location("feature_extraction", FEATURE_EXTRACTION_PATH)
feature_extraction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_extraction)
extract_features = feature_extraction.extract_features
print("feature_extraction.py loaded successfully")

# Load scaler
scaler = joblib.load(SCALER_PATH)
print("Scaler loaded successfully")

# Load model architecture dynamically
spec_arch = importlib.util.spec_from_file_location("model_architecture", MODEL_ARCH_PATH)
model_arch = importlib.util.module_from_spec(spec_arch)
spec_arch.loader.exec_module(model_arch)
DualBranchModel = model_arch.DualBranchModel
print("model_architecture.py loaded successfully")

# Initialize model
signal_length = 256   # Each EEG signal has 256 samples
n_features = 30       # 30 handcrafted features
n_classes = 3        # Example: seizure vs non-seizure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualBranchModel(signal_length, n_features, n_classes).to(device)
checkpoint = torch.load(MODEL_WEIGHTS, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Model loaded and set to eval mode")

# Prediction function
def predict(signal):
    """
    Predict seizure from a single EEG signal.

    Parameters:
    -----------
    signal : array-like, shape (256,)
        Raw EEG signal.

    Returns:
    --------
    dict : {
        "pred_class": int,
        "pred_prob": float
    }
    """
    signal = np.array(signal).flatten()
    if signal.shape[0] != signal_length:
        raise ValueError(f"Expected signal of length {signal_length}, got {signal.shape[0]}")

    # Extract features
    features = extract_features(signal)
    features_scaled = scaler.transform([features])  # shape (1, 30)

    # Convert to torch tensors
    sig_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,256)
    feat_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)                  # (1,30)

    with torch.no_grad():
        output = model(sig_tensor, feat_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
        pred_class = int(np.argmax(probs))
        pred_prob = float(probs[pred_class])

    return {"pred_class": pred_class, "pred_prob": pred_prob}

# Test run
if __name__ == "__main__":
    print("\nRunning a test prediction with a random signal")
    dummy_signal = np.random.randn(signal_length)
    result = predict(dummy_signal)
    print(f"Prediction result: {result}\n")
    print("Inference pipeline works successfully!")
