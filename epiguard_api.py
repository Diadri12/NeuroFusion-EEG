
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

print("EPIGUARD API STARTING")

# Model Architecture
class BranchA_LayerNorm(nn.Module):
    def __init__(self, signal_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 7, 2, 3)
        self.norm1 = nn.GroupNorm(1,32)
        self.conv2 = nn.Conv1d(32,64,5,1,2)
        self.norm2 = nn.GroupNorm(1,64)
        self.conv3 = nn.Conv1d(64,128,3,1,1)
        self.norm3 = nn.GroupNorm(1,128)
        self.conv4 = nn.Conv1d(128,256,3,1,1)
        self.norm4 = nn.GroupNorm(1,256)
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Sequential(nn.Linear(256,64), nn.ReLU())
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
            nn.Linear(n_features,64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.15)
        )
    def forward(self, x):
        return self.model(x)

class DualBranchModel(nn.Module):
    def __init__(self, signal_length, n_features, n_classes):
        super().__init__()
        self.branch_a = BranchA_LayerNorm(signal_length)
        self.branch_b = BranchB_Features(n_features)
        self.classifier = nn.Sequential(
            nn.Linear(96,32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32,n_classes)
        )
    def forward(self, signals, features):
        return self.classifier(torch.cat([self.branch_a(signals), self.branch_b(features)],1))

# Feature Extraction
def extract_features(signal, fs=173.61):
    signal = np.array(signal).flatten().astype(np.float64)
    features = []
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal),1/fs)
    psd = np.abs(fft)**2
    
    for _, low, high in [('d',0.5,4), ('t',4,8), ('a',8,13), ('b',13,30), ('g',30,50)]:
        mask = (freqs >= low) & (freqs <= high)
        if np.any(mask):
            features.extend([float(np.sum(psd[mask])), float(np.mean(psd[mask])), 
                             float(np.std(psd[mask])), float(freqs[mask][np.argmax(psd[mask])])])
        else:
            features.extend([0.0]*4)
    
    features.extend([float(features[4]/(features[8]+1e-10)), 
                     float(features[8]/(features[12]+1e-10)), 
                     float(features[0]/(features[4]+1e-10))])
    features.extend([float(entropy(np.abs(fft[:len(fft)//2])+1e-10)),
                     float(np.std(signal)), float(np.std(np.diff(signal)))])
    
    activity = np.var(signal)
    diff1 = np.diff(signal)
    mobility = np.sqrt(np.var(diff1)/(activity+1e-10))
    diff2 = np.diff(diff1)
    complexity = np.sqrt(np.var(diff2)/(np.var(diff1)+1e-10))/(mobility+1e-10)
    features.extend([float(activity), float(mobility), float(complexity)])
    features.append(float(np.sum(np.diff(np.sign(signal))!=0)))
    
    return np.array(features, dtype=np.float64)

# Load model and scaler
BASE_DIR = Path("/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG")
MODEL_DIR = BASE_DIR / "deployment" / "model"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading from: {MODEL_DIR}")
model = DualBranchModel(256,30,3).to(device)

# FIX FOR PyTorch 2.6+ checkpoints
checkpoint = torch.load(MODEL_DIR/"model_weights.pt", map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()
print("Model loaded!")

scaler = joblib.load(MODEL_DIR/"feature_scaler.pkl")
print("Scaler loaded!")

with open(MODEL_DIR/"model_metadata.json") as f:
    metadata = json.load(f)
print("Metadata loaded!")

# FastAPI
app = FastAPI(title="EpiGuard", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                  allow_methods=["*"], allow_headers=["*"])

class PredictionRequest(BaseModel):
    signal: List[float]
    sampling_rate: Optional[float] = 173.61

def preprocess_signal(signal, target_length=256):
    signal = np.array(signal, dtype=np.float64).flatten()
    if len(signal) > target_length:
        signal = signal[np.linspace(0, len(signal)-1, target_length).astype(int)]
    elif len(signal) < target_length:
        signal = np.pad(signal, (0, target_length-len(signal)))
    return signal

@app.get("/health")
async def health():
    return {"status":"healthy","model_loaded":True,"device":str(device)}

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        signal = np.array(request.signal, dtype=np.float64)
        if len(signal)<50: raise HTTPException(400,"Signal too short")
        signal_preprocessed = preprocess_signal(signal)
        features = extract_features(signal_preprocessed, request.sampling_rate)
        features_normalized = scaler.transform(features.reshape(1,-1))[0]
        signal_tensor = torch.FloatTensor(signal_preprocessed).unsqueeze(0).unsqueeze(0).to(device)
        features_tensor = torch.FloatTensor(features_normalized).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(signal_tensor, features_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        class_id = int(probs.argmax())
        return {
            "prediction": {
                "class_id": class_id,
                "class_name": metadata['output_specifications']['class_names'][class_id],
                "confidence": float(probs[class_id])
            }
        }
    except Exception as e:
        raise HTTPException(500,str(e))

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
