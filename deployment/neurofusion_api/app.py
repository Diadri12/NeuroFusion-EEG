import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib, os, io, time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Config
MODEL_PATH  = os.getenv("MODEL_PATH",  "final_model.pt")
SCALER_PATH = os.getenv("SCALER_PATH", "feature_scaler.pkl")
DEVICE      = torch.device("cpu")
WINDOW_SIZE = 256

CLASS_NAMES = {0: "Interictal", 1: "Preictal", 2: "Ictal"}
CLASS_DESCRIPTIONS = {
    0: "Normal interictal state — no seizure activity detected.",
    1: "Preictal state — early warning, seizure onset may be imminent.",
    2: "Ictal state — active seizure activity detected.",
}
CLASS_COLOURS  = {0: "green",  1: "orange", 2: "red"}
CLASS_URGENCY  = {0: "low",    1: "high",   2: "critical"}

# ARCHITECTURE

def get_norm_layer(norm_type, num_channels, num_groups=8):
    if norm_type == "batch":
        return nn.BatchNorm1d(num_channels)
    elif norm_type == "group":
        return nn.GroupNorm(min(num_groups, num_channels), num_channels)
    return nn.Identity()


class SupConBranchA(nn.Module):
    def __init__(self, embedding_dim=64,
                 norm_type="group", num_groups=8):
        super().__init__()
        self.conv1     = nn.Conv1d(1,   32,  7, stride=2, padding=3)
        self.norm1     = get_norm_layer(norm_type, 32,  num_groups)
        self.conv2     = nn.Conv1d(32,  64,  5, stride=1, padding=2)
        self.norm2     = get_norm_layer(norm_type, 64,  num_groups)
        self.conv3     = nn.Conv1d(64,  128, 3, stride=1, padding=1)
        self.norm3     = get_norm_layer(norm_type, 128, num_groups)
        self.conv4     = nn.Conv1d(128, 256, 3, stride=1, padding=1)
        self.norm4     = get_norm_layer(norm_type, 256, num_groups)
        self.pool      = nn.MaxPool1d(2)
        self.gap       = nn.AdaptiveAvgPool1d(1)
        self.relu      = nn.ReLU()
        self.dropout   = nn.Dropout(0.3)
        self.embedding = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        x = self.dropout(self.pool(self.relu(self.norm1(self.conv1(x)))))
        x = self.dropout(self.pool(self.relu(self.norm2(self.conv2(x)))))
        x = self.dropout(self.pool(self.relu(self.norm3(self.conv3(x)))))
        x = self.dropout(self.pool(self.relu(self.norm4(self.conv4(x)))))
        return self.embedding(self.gap(x).squeeze(-1))


class SupConBranchB(nn.Module):
    def __init__(self, n_features=30, embedding_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

    def forward(self, x):
        return self.model(x)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=96, projection_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x):
        return F.normalize(self.projection(x), dim=1)


class DualBranchContrastive(nn.Module):
    def __init__(self, signal_length=256, n_features=30,
                 n_classes=3, norm_type="group",
                 num_groups=8, projection_dim=128):
        super().__init__()
        self.branch_a        = SupConBranchA(64, norm_type, num_groups)
        self.branch_b        = SupConBranchB(n_features, 32)
        self.projection_head = ProjectionHead(96, projection_dim)
        self.classifier      = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_classes),
        )

    def forward(self, signals, features, mode="classify"):
        fused = torch.cat(
            [self.branch_a(signals),
             self.branch_b(features)], dim=1)
        if mode == "contrastive":
            return self.projection_head(fused)
        return self.classifier(fused)

# FEATURE EXTRACTION

def extract_features(window: np.ndarray) -> np.ndarray:
    x   = window.astype(np.float64)
    eps = 1e-10

    mean_amp     = float(np.mean(x))
    std_dev      = float(np.std(x))
    skewness     = float(np.mean(((x - mean_amp) / (std_dev + eps)) ** 3))
    kurtosis     = float(np.mean(((x - mean_amp) / (std_dev + eps)) ** 4))
    zcr          = float(np.sum(np.diff(np.sign(x)) != 0)) / len(x)
    rms          = float(np.sqrt(np.mean(x ** 2)))
    peak_to_peak = float(np.max(x) - np.min(x))
    energy       = float(np.sum(x ** 2))
    variance     = float(np.var(x))
    iqr          = float(np.percentile(x, 75) - np.percentile(x, 25))

    dx         = np.diff(x)
    ddx        = np.diff(dx)
    activity   = np.var(x)
    mobility   = float(np.sqrt(np.var(dx)  / (activity   + eps)))
    complexity = float(np.sqrt(np.var(ddx) / (np.var(dx) + eps))
                       / (mobility + eps))
    line_length = float(np.sum(np.abs(np.diff(x))))

    N     = len(x)
    fs    = 256
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    psd   = (np.abs(np.fft.rfft(x)) ** 2) / N

    def band_power(lo, hi):
        return float(np.sum(psd[np.where((freqs >= lo) & (freqs < hi))]))

    delta = band_power(0.5,   4.0)
    theta = band_power(4.0,   8.0)
    alpha = band_power(8.0,  13.0)
    beta  = band_power(13.0, 30.0)
    gamma = band_power(30.0, 100.0)

    total_power      = float(np.sum(psd) + eps)
    psd_norm         = psd / total_power
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + eps)))
    cumulative       = np.cumsum(psd)
    spectral_edge    = float(
        freqs[np.searchsorted(cumulative, 0.95 * cumulative[-1])])
    low_high_ratio   = float((delta + theta) / (beta + gamma + eps))

    try:
        import pywt
        coeffs          = pywt.wavedec(x, "db4", level=4)
        wavelet_energy  = float(sum(np.sum(c ** 2) for c in coeffs))
        all_c           = np.concatenate(coeffs)
        p               = all_c ** 2 / (np.sum(all_c ** 2) + eps)
        wavelet_entropy = float(-np.sum(p * np.log2(p + eps)))
        wavelet_shannon = wavelet_entropy
    except ImportError:
        wavelet_energy  = energy
        wavelet_entropy = spectral_entropy
        wavelet_shannon = spectral_entropy

    hist, _     = np.histogram(x, bins=32, density=True)
    hist        = hist / (hist.sum() + eps)
    shannon_ent = float(-np.sum(hist * np.log2(hist + eps)))

    def permutation_entropy(sig, order=3, delay=1):
        perms = {}
        for i in range(len(sig) - (order - 1) * delay):
            pat = tuple(np.argsort(sig[i: i + order * delay: delay]))
            perms[pat] = perms.get(pat, 0) + 1
        total = sum(perms.values())
        probs = np.array(list(perms.values())) / total
        return float(-np.sum(probs * np.log2(probs + eps)))

    perm_ent = permutation_entropy(x)

    def sample_entropy(sig, m=2, r_factor=0.2):
        r, N = r_factor * np.std(sig) + eps, len(sig)
        def count_matches(tlen):
            count = 0
            for i in range(N - tlen):
                tmpl = sig[i: i + tlen]
                for j in range(i + 1, N - tlen):
                    if np.max(np.abs(sig[j: j + tlen] - tmpl)) < r:
                        count += 1
            return count
        A, B = count_matches(m + 1), count_matches(m)
        return float(-np.log((A + eps) / (B + eps)))

    samp_ent = sample_entropy(x)

    def hurst(sig):
        lags = range(2, min(20, len(sig) // 2))
        tau  = [np.std(np.subtract(sig[lag:], sig[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(np.array(tau) + eps), 1)
        return float(poly[0])

    def higuchi_fd(sig, kmax=10):
        L, N = [], len(sig)
        for k in range(1, kmax + 1):
            Lk = []
            for m in range(1, k + 1):
                s = sum(
                    abs(sig[m - 1 + i * k] - sig[m - 1 + (i - 1) * k])
                    for i in range(1, int((N - m) / k))
                )
                Lk.append(s * (N - 1) / (k * int((N - m) / k) * k))
            L.append(np.mean(Lk))
        if len(L) < 2:
            return 0.0
        poly = np.polyfit(
            np.log(range(1, kmax + 1)),
            np.log(np.array(L) + eps), 1)
        return float(-poly[0])

    hurst_exp = hurst(x)
    higuchi   = higuchi_fd(x)
    dists     = np.abs(np.diff(x))
    L_total   = np.sum(dists)
    d_max     = np.max(np.abs(x - x[0]))
    n_steps   = len(x) - 1
    katz_fd   = float(
        np.log10(n_steps) /
        (np.log10(d_max / (L_total + eps)) + np.log10(n_steps) + eps))

    features = np.array([
        mean_amp, std_dev, skewness, kurtosis,
        zcr, rms, peak_to_peak, energy,
        variance, iqr,
        mobility, complexity, line_length,
        delta, theta, alpha, beta, gamma,
        low_high_ratio,
        spectral_entropy, spectral_edge,
        wavelet_energy, wavelet_entropy, wavelet_shannon,
        shannon_ent, perm_ent, samp_ent,
        hurst_exp, higuchi, katz_fd,
    ], dtype=np.float64)

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

# LOAD MODEL AT STARTUP

print("Loading NeuroFusion-EEG model")
scaler = StandardScaler()
scaler.mean_          = np.load(os.path.join(os.path.dirname(__file__), "scaler_mean.npy"))
scaler.scale_         = np.load(os.path.join(os.path.dirname(__file__), "scaler_scale.npy"))
scaler.var_           = scaler.scale_ ** 2
scaler.n_features_in_ = len(scaler.mean_)
model      = DualBranchContrastive(
    signal_length=256, n_features=30, n_classes=3,
    norm_type="group", num_groups=8, projection_dim=128)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.to(DEVICE).eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"Model ready — {n_params:,} parameters")

# PREDICTION HELPERS

def run_prediction(window: np.ndarray) -> dict:
    features = extract_features(window)
    features = scaler.transform(features.reshape(1, -1))
    sig_t    = torch.tensor(window, dtype=torch.float32)                     .unsqueeze(0).unsqueeze(0).to(DEVICE)
    feat_t   = torch.tensor(features, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(sig_t, feat_t, mode="classify")
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_class = int(np.argmax(probs))
    return {
        "predicted_class" : pred_class,
        "predicted_label" : CLASS_NAMES[pred_class],
        "description"     : CLASS_DESCRIPTIONS[pred_class],
        "confidence"      : round(float(probs[pred_class]), 4),
        "urgency"         : CLASS_URGENCY[pred_class],
        "colour"          : CLASS_COLOURS[pred_class],
        "probabilities"   : {
            CLASS_NAMES[i]: round(float(probs[i]), 4)
            for i in range(3)
        },
    }


def extract_windows_from_csv(df: pd.DataFrame) -> np.ndarray:
    signal_col = None
    for candidate in ["Signal", "signal", "EEG", "eeg", "value"]:
        if candidate in df.columns:
            signal_col = candidate
            break
    if signal_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric signal column found in CSV.")
        signal_col = numeric_cols[0]

    signal  = df[signal_col].dropna().values.astype(np.float64)
    stride  = 20
    windows = np.array([
        signal[i * stride: i * stride + WINDOW_SIZE]
        for i in range((len(signal) - WINDOW_SIZE) // stride + 1)
        if i * stride + WINDOW_SIZE <= len(signal)
    ])
    return windows


def compute_overall_status(class_counts: dict, n_windows: int) -> dict:
    if class_counts[2] > 0:
        return {
            "status"  : "Ictal activity detected",
            "colour"  : "red",
            "urgency" : "critical",
            "advice"  : "Immediate medical attention recommended. "
                        "Please contact your healthcare provider.",
        }
    elif class_counts[1] > n_windows * 0.2:
        return {
            "status"  : "Preictal activity detected — early warning",
            "colour"  : "orange",
            "urgency" : "high",
            "advice"  : "Elevated seizure risk detected. "
                        "Monitor closely and consult your healthcare provider.",
        }
    else:
        return {
            "status"  : "No seizure activity detected",
            "colour"  : "green",
            "urgency" : "low",
            "advice"  : "No concerning EEG patterns detected at this time. "
                        "Continue regular monitoring.",
        }

# FASTAPI APP

app = FastAPI(
    title       = "NeuroFusion-EEG API",
    description = (
        "3-class EEG seizure classification — "
        "Interictal / Preictal / Ictal. "
        "Dual-Branch CNN with Supervised Contrastive Learning. "
        "Built for the NeuroFusion-EEG Final Year Project 2025-26."
    ),
    version = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# Request models

class SignalRequest(BaseModel):
    signal: List[float]


# Endpoints

@app.get("/")
def root():
    return {
        "name"    : "NeuroFusion-EEG API",
        "version" : "1.0.0",
        "status"  : "running",
        "endpoints": [
            "GET  /health",
            "GET  /info",
            "POST /predict",
            "POST /predict/csv",
            "POST /predict/batch",
        ],
    }


@app.get("/health")
def health():
    return {
        "status"  : "ok",
        "model"   : "DualBranchContrastive (SupCon + Balanced)",
        "classes" : list(CLASS_NAMES.values()),
        "version" : "1.0.0",
    }


@app.get("/info")
def info():
    return {
        "model_name"     : "NeuroFusion-EEG — SupCon + Balanced",
        "architecture"   : "Dual-Branch CNN (Signal Branch + Feature Branch)",
        "parameters"     : 187555,
        "window_size"    : WINDOW_SIZE,
        "n_features"     : 30,
        "classes"        : CLASS_NAMES,
        "descriptions"   : CLASS_DESCRIPTIONS,
        "performance"    : {
            "macro_f1"   : 0.2763,
            "accuracy"   : 0.2824,
            "auc"        : 0.5016,
            "test_n"     : 14450,
        },
        "dataset"        : (
            "EEG Epilepsy Diagnosis (signal source) + "
            "Epilepsy Federated EEG Dataset (labels + features)"
        ),
        "disclaimer"     : (
            "This tool is for research purposes only. "
            "It is not a medical device and must not be used "
            "for clinical diagnosis."
        ),
    }


@app.post("/predict")
def predict_single_window(request: SignalRequest):
    """
    Predict seizure state for a single EEG window.
    Requires exactly 256 float values.
    """
    if len(request.signal) != WINDOW_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Signal must be exactly {WINDOW_SIZE} values. "
                   f"Got {len(request.signal)}.")
    try:
        window = np.array(request.signal, dtype=np.float64)
        result = run_prediction(window)
        result["window_size"] = WINDOW_SIZE
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file containing EEG signal data.
    Expects a column named Signal (or first numeric column).
    Returns per-window predictions and an overall summary.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=422,
            detail="Only .csv files are accepted.")
    try:
        start_time = time.time()
        contents   = await file.read()
        df         = pd.read_csv(io.BytesIO(contents))
        windows    = extract_windows_from_csv(df)

        if len(windows) == 0:
            raise HTTPException(
                status_code=422,
                detail=f"Could not extract any windows. "
                       f"Signal must be at least {WINDOW_SIZE} samples.")

        # Batch inference
        results      = []
        class_counts = {0: 0, 1: 0, 2: 0}

        for i, window in enumerate(windows):
            r = run_prediction(window)
            r["window_index"] = i
            results.append(r)
            class_counts[r["predicted_class"]] += 1

        n_windows    = len(windows)
        elapsed      = round(time.time() - start_time, 2)
        overall      = compute_overall_status(class_counts, n_windows)
        dominant     = max(class_counts, key=class_counts.get)

        return {
            "file_name"      : file.filename,
            "total_windows"  : n_windows,
            "time_taken_sec" : elapsed,
            "overall_status" : overall["status"],
            "overall_colour" : overall["colour"],
            "overall_urgency": overall["urgency"],
            "advice"         : overall["advice"],
            "class_distribution" : {
                CLASS_NAMES[c]: {
                    "count" : class_counts[c],
                    "pct"   : round(class_counts[c] / n_windows * 100, 1),
                }
                for c in range(3)
            },
            "dominant_prediction" : {
                "class" : dominant,
                "label" : CLASS_NAMES[dominant],
            },
            "disclaimer"     : (
                "This tool is for research purposes only and is not "
                "a medical device. Always consult a healthcare professional."
            ),
            "windows"        : results,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch_windows(signals: List[SignalRequest]):
    """
    Predict seizure states for multiple EEG windows at once.
    Each item must contain exactly 256 float values.
    Maximum batch size: 500 windows.
    """
    if len(signals) == 0:
        raise HTTPException(status_code=422, detail="Empty batch.")
    if len(signals) > 500:
        raise HTTPException(
            status_code=422,
            detail="Batch size limited to 500 windows.")
    try:
        results      = []
        class_counts = {0: 0, 1: 0, 2: 0}
        for i, req in enumerate(signals):
            if len(req.signal) != WINDOW_SIZE:
                raise HTTPException(
                    status_code=422,
                    detail=f"Window {i}: expected {WINDOW_SIZE} values, "
                           f"got {len(req.signal)}.")
            window = np.array(req.signal, dtype=np.float64)
            r      = run_prediction(window)
            r["window_index"] = i
            results.append(r)
            class_counts[r["predicted_class"]] += 1

        n_windows = len(results)
        overall   = compute_overall_status(class_counts, n_windows)

        return {
            "count"          : n_windows,
            "overall_status" : overall["status"],
            "overall_colour" : overall["colour"],
            "overall_urgency": overall["urgency"],
            "advice"         : overall["advice"],
            "class_distribution" : {
                CLASS_NAMES[c]: {
                    "count" : class_counts[c],
                    "pct"   : round(class_counts[c] / n_windows * 100, 1),
                }
                for c in range(3)
            },
            "predictions"    : results,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
