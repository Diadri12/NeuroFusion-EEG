"""
EpiGuard API - FastAPI Implementation
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025-2026
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
import json
import time
from scipy.stats import entropy

print("EpiGuard API - Starting")

# MODEL ARCHITECTURE
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

# FEATURE EXTRACTION
def extract_features(signal, fs=173.61):
    """Extract 30 handcrafted features from EEG signal"""
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

# LOAD MODEL AND SCALER
# Path to deployment folder
BASE_DIR = Path("/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG")
MODEL_DIR = BASE_DIR / "deployment" / "model"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n Loading model from: {MODEL_DIR}")

# Load model
try:
    model = DualBranchModel(signal_length=256, n_features=30, n_classes=3).to(device)
    checkpoint = torch.load(MODEL_DIR / "model_weights.pt",map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(" Model loaded successfully!")
except Exception as e:
    print(f" Failed to load model: {e}")
    exit(1)

# Load scaler
try:
    scaler = joblib.load(MODEL_DIR / "feature_scaler.pkl")
    print(" Scaler loaded successfully!")
except Exception as e:
    print(f" Failed to load scaler: {e}")
    exit(1)

# Load metadata
try:
    with open(MODEL_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    print(" Metadata loaded successfully!")
except Exception as e:
    print(f" Warning: Could not load metadata: {e}")
    # Create default metadata
    metadata = {
        "model_info": {"name": "NeuroFusion-EEG", "version": "1.0.0"},
        "output_specifications": {
            "class_names": ["Class 0", "Class 1", "Class 2"],
            "class_descriptions": {
                "0": "Healthy / Normal",
                "1": "Interictal / Abnormal non-seizure",
                "2": "Ictal / Seizure"
            }
        }
    }

print(f" All components loaded!")
print(f"   Device: {device}")
print(f"   Model: {metadata['model_info']['name']}")

# FASTAPI APP
app = FastAPI(
    title="EpiGuard API",
    description="EEG-based seizure detection API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  REQUEST/RESPONSE MODELS
class PredictionRequest(BaseModel):
    signal: List[float]
    sampling_rate: Optional[float] = 173.61

class PredictionResponse(BaseModel):
    prediction: dict
    probabilities: dict
    metadata: dict

# HELPER FUNCTIONS
def preprocess_signal(signal: np.ndarray, target_length: int = 256) -> np.ndarray:
    """Resample signal to target length"""
    signal = np.array(signal, dtype=np.float64).flatten()

    if len(signal) > target_length:
        indices = np.linspace(0, len(signal)-1, target_length).astype(int)
        signal = signal[indices]
    elif len(signal) < target_length:
        signal = np.pad(signal, (0, target_length - len(signal)))

    return signal

def validate_signal(signal: np.ndarray) -> tuple:
    """Validate signal quality"""
    if len(signal) < 50:
        return False, "Signal too short (minimum 50 samples required)"
    if np.isnan(signal).any():
        return False, "Signal contains NaN values"
    if np.isinf(signal).any():
        return False, "Signal contains infinite values"
    return True, "OK"

# API ENDPOINTS
@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "running",
        "model": metadata['model_info']['name'],
        "version": metadata['model_info']['version'],
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "info": "/info"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "device": str(device),
        "model_info": metadata['model_info']
    }

@app.get("/info")
async def model_info():
    """Get model information"""
    return metadata

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict seizure from EEG signal

    **Input:**
    - signal: List of float values (EEG signal)
    - sampling_rate: Optional, default 173.61 Hz

    **Output:**
    - prediction: Class and confidence
    - probabilities: All class probabilities
    - metadata: Processing info
    """
    try:
        start_time = time.time()

        # Convert to numpy array
        signal = np.array(request.signal, dtype=np.float64)

        # Validate signal
        is_valid, error_msg = validate_signal(signal)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        # Preprocess signal
        signal_preprocessed = preprocess_signal(signal, target_length=256)

        # Extract features
        features = extract_features(signal_preprocessed, fs=request.sampling_rate)

        # Normalize features
        features_normalized = scaler.transform(features.reshape(1, -1))[0]

        # Prepare tensors
        signal_tensor = torch.FloatTensor(signal_preprocessed).unsqueeze(0).unsqueeze(0).to(device)
        features_tensor = torch.FloatTensor(features_normalized).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output = model(signal_tensor, features_tensor)
            probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()

        # Get prediction
        class_id = int(probabilities.argmax())
        class_names = metadata['output_specifications']['class_names']
        class_descriptions = metadata['output_specifications']['class_descriptions']

        confidence = float(probabilities[class_id])

        processing_time = (time.time() - start_time) * 1000  # ms

        # Build response
        response = {
            "prediction": {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "class_description": class_descriptions[str(class_id)],
                "confidence": confidence
            },
            "probabilities": {
                class_names[i]: float(probabilities[i])
                for i in range(len(class_names))
            },
            "metadata": {
                "processing_time_ms": processing_time,
                "model_version": metadata['model_info']['version'],
                "high_confidence": confidence >= 0.6,
                "signal_length_input": len(request.signal),
                "signal_length_processed": len(signal_preprocessed),
                "sampling_rate": request.sampling_rate,
                "device": str(device)
            }
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
