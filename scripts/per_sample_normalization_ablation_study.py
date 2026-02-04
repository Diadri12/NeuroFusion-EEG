"""
PER-SAMPLE NORMALIZATION ABLATION STUDY
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from tqdm import tqdm
import joblib
from scipy.signal import resample

BASE_DIR = Path("/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG")
PHASE3_DIR = BASE_DIR / "outputs/dual_branch_training"
PHASE6_DIR = BASE_DIR / "outputs/advanced_analysis"

RESULTS_DIR = PHASE6_DIR / "normalization_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_LENGTH = 256
N_FEATURES = 30
N_CLASSES = 3
BATCH_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# DATA LOADER
def load_bonn_dataset(dataset_path):
    print("\nLoading Bonn EEG dataset (raw)")
    dataset_path = Path(dataset_path)

    label_map = {
        "Z": 0, "O": 0,
        "N": 1, "F": 1,
        "S": 2
    }

    signals, labels = [], []

    for file in dataset_path.rglob("*.txt"):
        label_char = file.stem[0].upper()
        if label_char not in label_map:
            continue

        sig = np.loadtxt(file)
        signals.append(sig)
        labels.append(label_map[label_char])

    if len(signals) == 0:
        raise RuntimeError(
            f"No Bonn EEG files loaded from {dataset_path}. "
            "Check directory structure."
        )

    signals = np.array(signals, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print(f" Loaded {signals.shape[0]} signals")
    return signals, labels

def preprocess_signals(signals, target_len):
    processed = []
    for sig in signals:
        if len(sig) != target_len:
            sig = resample(sig, target_len)
        processed.append(sig)
    return np.array(processed, dtype=np.float32)


def extract_features_simple(signals):
    features = []
    for sig in signals:
        feats = [
            np.mean(sig),
            np.std(sig),
            np.var(sig),
            np.max(sig),
            np.min(sig),
            np.median(sig),
            np.percentile(sig, 25),
            np.percentile(sig, 75),
            np.mean(np.abs(sig)),
            np.sqrt(np.mean(sig**2))
        ]
        # Pad to 30 features
        while len(feats) < 30:
            feats.append(0.0)
        features.append(feats)
    return np.array(features, dtype=np.float32)

# LOAD DATA
print("\nLOADING PREPROCESSED DATA")

cross_dataset_dir = PHASE6_DIR / "cross_dataset_generalization"
bonn_dir = cross_dataset_dir / "bonn_eeg"

if (bonn_dir / "signals.npy").exists():
    print(" Found saved Bonn files")
    bonn_signals = np.load(bonn_dir / "signals.npy")
    bonn_features = np.load(bonn_dir / "features.npy")
    bonn_labels = np.load(bonn_dir / "labels.npy")
else:
    print(" Saved Bonn files not found — loading from raw dataset")
    bonn_signals, bonn_labels = load_bonn_dataset(
        BASE_DIR / "datasets" / "Bonn_EEG_Time_Series"
    )
    bonn_signals = preprocess_signals(bonn_signals, SIGNAL_LENGTH)
    bonn_features = extract_features_simple(bonn_signals)

print(f" Bonn signals:  {bonn_signals.shape}")
print(f" Bonn features: {bonn_features.shape}")
print(f" Bonn labels:   {bonn_labels.shape}")

# NORMALIZATION METHODS
scaler = joblib.load(PHASE3_DIR / "feature_scaler.pkl")

def norm_standard(features):
    return scaler.transform(features)

def norm_per_sample(features):
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True) + 1e-10
    return (features - mean) / std

def norm_per_feature(features):
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-10
    return (features - mean) / std

NORMALIZERS = {
    "StandardScaler": norm_standard,
    "Per-Sample": norm_per_sample,
    "Per-Feature": norm_per_feature
}

# MODEL DEFINITION
class BranchA(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, 2, 3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, 1, 2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class BranchB(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class DualBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = BranchA()
        self.b = BranchB()
        self.cls = nn.Linear(128 + 32, N_CLASSES)

    def forward(self, s, f):
        return self.cls(torch.cat([self.a(s), self.b(f)], dim=1))


# LOAD TRAINED MODEL
model = DualBranch().to(device)
ckpt = torch.load(
    BASE_DIR / "outputs/error_diagnosis/contrastive_pretraining_results/final_model.pt",
    map_location=device
)
model.load_state_dict(ckpt, strict=False)
model.eval()

WITHIN_F1 = 0.2763

# ABLATION EXPERIMENT
results = {}

signals_tensor = torch.tensor(bonn_signals).unsqueeze(1).to(device)
labels_np = bonn_labels

for name, norm_fn in NORMALIZERS.items():
    print(f"\n{'='*60}\nNormalization: {name}")

    feats = norm_fn(bonn_features)
    feats_tensor = torch.tensor(feats).to(device)

    preds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(signals_tensor), BATCH_SIZE)):
            out = model(
                signals_tensor[i:i+BATCH_SIZE],
                feats_tensor[i:i+BATCH_SIZE]
            )
            preds.extend(out.argmax(1).cpu().numpy())

    preds = np.array(preds)

    f1 = f1_score(labels_np, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels_np, preds)
    retention = (f1 / WITHIN_F1) * 100
    cm = confusion_matrix(labels_np, preds)

    results[name] = {
        "accuracy": acc,
        "f1_macro": f1,
        "retention": retention,
        "confusion_matrix": cm.tolist()
    }

    print(f" Accuracy:  {acc:.4f}")
    print(f" Macro F1:  {f1:.4f}")
    print(f" Retention: {retention:.1f}%")
    print(" Confusion Matrix:")
    print(cm)

# SAVE RESULTS
df = pd.DataFrame([
    {
        "Method": k,
        "Macro F1": v["f1_macro"],
        "Retention (%)": v["retention"]
    }
    for k, v in results.items()
]).sort_values("Macro F1", ascending=False)

df.to_csv(RESULTS_DIR / "normalization_ablation.csv", index=False)

with open(RESULTS_DIR / "normalization_ablation.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n EXPERIMENT COMPLETE")
print(df)
