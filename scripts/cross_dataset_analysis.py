"""
CROSS-DATASET ANALYSIS
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, precision_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from tqdm import tqdm
import joblib

print("ALL-IN-ONE CROSS-DATASET ANALYSIS")

BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
DATASETS_DIR = f"{BASE_DIR}/datasets"
PHASE3_DIR = f"{BASE_DIR}/outputs/dual_branch_training"
PHASE6_DIR = f"{BASE_DIR}/outputs/advanced_analysis"

# Create output directories
cross_dataset_dir = Path(PHASE6_DIR) / "cross_dataset_generalization"
cross_dataset_dir.mkdir(exist_ok=True, parents=True)

norm_dir = cross_dataset_dir / "normalization_ablation"
norm_dir.mkdir(exist_ok=True, parents=True)

viz_dir = cross_dataset_dir / "feature_visualization"
viz_dir.mkdir(exist_ok=True, parents=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# LOAD BONN DATASET
print("LOADING BONN DATASET")

def load_bonn_dataset(dataset_path):
    print("\n Loading Bonn EEG dataset")

    dataset_path = Path(dataset_path)
    bonn_data = dataset_path / "Bonn_EEG_Time_Series"

    if not bonn_data.exists():
        print(f" Bonn dataset not found at {bonn_data}")
        return None, None

    extracted_dir = bonn_data / "extracted"

    # Extract if needed
    if not extracted_dir.exists():
        print("   Extracting zip files")
        import zipfile
        extracted_dir.mkdir(exist_ok=True)

        for zip_file in bonn_data.glob("*.zip"):
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    set_dir = extracted_dir / zip_file.stem
                    set_dir.mkdir(exist_ok=True)
                    zip_ref.extractall(set_dir)
                print(f" Extracted {zip_file.name}")
            except:
                pass

    # Load data
    sets = {
        'Z': (0, 'Healthy_EyesOpen'),
        'O': (0, 'Healthy_EyesClosed'),
        'N': (1, 'Interictal_Hippocampus'),
        'F': (1, 'Interictal_Epileptogenic'),
        'S': (2, 'Ictal_Seizure')
    }

    signals, labels = [], []

    for set_name, (label, desc) in sets.items():
        set_dir = extracted_dir / set_name
        if not set_dir.exists():
            set_dir = extracted_dir / set_name.lower()
        if not set_dir.exists():
            continue

        txt_files = list(set_dir.glob("*.txt")) or list(set_dir.rglob("*.txt"))

        for file in txt_files[:100]:  # 100 per class
            try:
                data = np.loadtxt(file)
                if len(data) > 0:
                    signals.append(data)
                    labels.append(label)
            except:
                continue

    if len(signals) == 0:
        return None, None

    # Standardize length
    min_length = min(len(s) for s in signals)
    signals = np.array([s[:min_length] for s in signals], dtype=np.float64)
    labels = np.array(labels, dtype=np.int64)

    print(f"   Loaded {len(signals)} signals, shape: {signals.shape}")
    print(f"   Class distribution: {np.bincount(labels)}")

    return signals, labels

bonn_signals, bonn_labels = load_bonn_dataset(DATASETS_DIR)

if bonn_signals is None:
    print("\n Cannot proceed without Bonn dataset")
    print("   Please check dataset path and structure")
    exit(1)

# PREPROCESS BONN
print("PREPROCESSING BONN SIGNALS")

signal_length = 256  # Your model's expected input length

# Resample signals to target length
def resample_signals(signals, target_length):
    resampled = []
    for sig in tqdm(signals, desc="Resampling", leave=False):
        if len(sig) > target_length:
            indices = np.linspace(0, len(sig)-1, target_length).astype(int)
            sig = sig[indices]
        elif len(sig) < target_length:
            sig = np.pad(sig, (0, target_length - len(sig)))
        resampled.append(sig)
    return np.array(resampled, dtype=np.float64)

bonn_signals = resample_signals(bonn_signals, signal_length)
print(f" Resampled to {bonn_signals.shape}")

# EXTRACT FEATURES
print("EXTRACTING FEATURES")

# Extract 30 handcrafted features
def extract_features(signals, fs=173.61):
    features_list = []

    for sig in tqdm(signals, desc="Feature extraction", leave=False):
        sig = np.array(sig).flatten().astype(np.float64)
        features = []

        # FFT
        fft = np.fft.fft(sig)
        freqs = np.fft.fftfreq(len(sig), 1/fs)
        psd = np.abs(fft)**2

        # 5 bands × 4 features = 20
        for low, high in [(0.5,4), (4,8), (8,13), (13,30), (30,50)]:
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                features.extend([
                    float(np.sum(psd[mask])),
                    float(np.mean(psd[mask])),
                    float(np.std(psd[mask])),
                    float(freqs[mask][np.argmax(psd[mask])])
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

        # Ratios (3)
        delta, theta, alpha, beta = features[0], features[4], features[8], features[12]
        features.extend([
            float(theta / (alpha + 1e-10)),
            float(alpha / (beta + 1e-10)),
            float(delta / (theta + 1e-10))
        ])

        # Entropy (3)
        features.append(float(entropy(np.abs(fft[:len(fft)//2]) + 1e-10)))
        features.append(float(np.std(sig)))
        features.append(float(np.std(np.diff(sig))))

        # Hjorth (3)
        activity = np.var(sig)
        diff1 = np.diff(sig)
        mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
        diff2 = np.diff(diff1)
        complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
        features.extend([float(activity), float(mobility), float(complexity)])

        # Zero-crossing (1) to reach 30
        features.append(float(np.sum(np.diff(np.sign(sig)) != 0)))

        features_list.append(features)

    return np.array(features_list, dtype=np.float64)

bonn_features_raw = extract_features(bonn_signals)
print(f" Extracted features: {bonn_features_raw.shape}")

# Save preprocessed data
bonn_dir = cross_dataset_dir / "bonn_eeg"
bonn_dir.mkdir(exist_ok=True, parents=True)

np.save(bonn_dir / "signals.npy", bonn_signals)
np.save(bonn_dir / "features.npy", bonn_features_raw)
np.save(bonn_dir / "labels.npy", bonn_labels)

print(f"Saved preprocessed Bonn data to {bonn_dir}")

# LOAD TRAINING DATA
print("LOADING TRAINING DATA")

frozen_dir = Path(PHASE3_DIR) / "frozen"

your_features = np.load(frozen_dir / "X_features.npy")
your_labels = np.load(frozen_dir / "y_features.npy")

print("Your dataset:")
print(" Features shape:", your_features.shape)
print(" Labels shape:", your_labels.shape)

assert your_features.ndim == 2 and your_features.shape[1] == 30, \
    f"Expected (N,30) features, got {your_features.shape}"

# Load scaler
scaler_path = Path(PHASE3_DIR) / "feature_scaler.pkl"
your_scaler = joblib.load(scaler_path)
print(f" Loaded StandardScaler")

# LOAD MODEL
print("LOADING TRAINED MODEL")

# Model architecture (LayerNorm version)
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
            nn.Linear(n_features, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.15)
        )

    def forward(self, x):
        return self.model(x)

class DualBranchModel(nn.Module):
    def __init__(self, signal_length, n_features, n_classes):
        super().__init__()
        self.branch_a = BranchA_LayerNorm(signal_length)
        self.branch_b = BranchB_Features(n_features)
        self.classifier = nn.Sequential(
            nn.Linear(96, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, n_classes)
        )

    def forward(self, signals, features):
        return self.classifier(torch.cat([self.branch_a(signals), self.branch_b(features)], 1))

# Load model
model_path = Path(f"{BASE_DIR}/outputs/error_diagnosis/contrastive_pretraining_results/final_model.pt")

model = DualBranchModel(signal_length, 30, 3).to(device)
checkpoint = torch.load(model_path, map_location=device)

if isinstance(checkpoint, dict):
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=False)
model.eval()

print(f" Model loaded successfully")

your_within_f1 = 0.2763  # Your within-dataset performance

#  EVALUATION WITH DIFFERENT NORMALIZATIONS
print("NORMALIZATION ABLATION")

results = {}

# Method 1: StandardScaler
print("\n Method 1: StandardScaler")
bonn_features_std = your_scaler.transform(bonn_features_raw)

signals_tensor = torch.FloatTensor(bonn_signals).unsqueeze(1).to(device)
features_tensor = torch.FloatTensor(bonn_features_std).to(device)

all_preds = []
with torch.no_grad():
    for i in tqdm(range(0, len(bonn_signals), 256), desc="Inference", leave=False):
        outputs = model(signals_tensor[i:i+256], features_tensor[i:i+256])
        all_preds.extend(outputs.argmax(1).cpu().numpy())

all_preds = np.array(all_preds)

f1 = f1_score(bonn_labels, all_preds, average='macro', zero_division=0)
acc = accuracy_score(bonn_labels, all_preds)
recalls = recall_score(bonn_labels, all_preds, average=None, zero_division=0)
cm = confusion_matrix(bonn_labels, all_preds)

print(f"   F1: {f1:.4f}, Retention: {(f1/your_within_f1)*100:.1f}%")

results['StandardScaler'] = {
    'f1': float(f1),
    'accuracy': float(acc),
    'retention': float((f1/your_within_f1)*100),
    'recalls': recalls.tolist(),
    'confusion_matrix': cm.tolist()
}

# Method 2: Per-Sample Normalization
print("\n Method 2: Per-Sample Normalization")
bonn_features_persample = (bonn_features_raw - bonn_features_raw.mean(axis=1, keepdims=True)) / \
                          (bonn_features_raw.std(axis=1, keepdims=True) + 1e-10)

features_tensor = torch.FloatTensor(bonn_features_persample).to(device)

all_preds = []
with torch.no_grad():
    for i in tqdm(range(0, len(bonn_signals), 256), desc="Inference", leave=False):
        outputs = model(signals_tensor[i:i+256], features_tensor[i:i+256])
        all_preds.extend(outputs.argmax(1).cpu().numpy())

all_preds = np.array(all_preds)

f1 = f1_score(bonn_labels, all_preds, average='macro', zero_division=0)
acc = accuracy_score(bonn_labels, all_preds)
recalls = recall_score(bonn_labels, all_preds, average=None, zero_division=0)
cm = confusion_matrix(bonn_labels, all_preds)

print(f"   F1: {f1:.4f}, Retention: {(f1/your_within_f1)*100:.1f}%")

results['Per-Sample'] = {
    'f1': float(f1),
    'accuracy': float(acc),
    'retention': float((f1/your_within_f1)*100),
    'recalls': recalls.tolist(),
    'confusion_matrix': cm.tolist()
}

# Create comparison table
df = pd.DataFrame({
    'Method': list(results.keys()),
    'Bonn F1': [results[m]['f1'] for m in results.keys()],
    'Retention (%)': [results[m]['retention'] for m in results.keys()],
    'Class 0 Recall': [results[m]['recalls'][0] for m in results.keys()],
    'Class 1 Recall': [results[m]['recalls'][1] for m in results.keys()],
    'Class 2 Recall': [results[m]['recalls'][2] for m in results.keys()]
})

print("ABLATION TABLE")
print(df.to_string(index=False))

df.to_csv(norm_dir / "normalization_ablation.csv", index=False)
print(f"\n Saved: {norm_dir / 'normalization_ablation.csv'}")

# FEATURE VISUALIZATION
print("FEATURE SPACE VISUALIZATION")

# Sample data
def sample_balanced(features, labels, n_per_class=150):
    sampled_feat, sampled_lab = [], []
    for c in np.unique(labels):
        mask = labels == c
        feat = features[mask]
        n = min(n_per_class, len(feat))
        idx = np.random.choice(len(feat), n, replace=False)
        sampled_feat.append(feat[idx])
        sampled_lab.extend([c] * n)
    return np.vstack(sampled_feat), np.array(sampled_lab)

your_feat_sample, your_lab_sample = sample_balanced(your_features, your_labels, 150)
bonn_feat_sample, bonn_lab_sample = sample_balanced(bonn_features_raw, bonn_labels, 150)

# Combine
all_features = np.vstack([your_feat_sample, bonn_feat_sample])
all_labels = np.hstack([your_lab_sample, bonn_lab_sample])
all_datasets = np.array(['Your Data']*len(your_feat_sample) + ['Bonn']*len(bonn_feat_sample))

print(f"\n Combined {len(all_features)} samples for visualization")

# PCA
print(" Running PCA")
pca = PCA(n_components=2, random_state=42)
features_pca = pca.fit_transform(all_features)
var_exp = pca.explained_variance_ratio_

print(f"   Variance explained: {var_exp[0]:.1%} + {var_exp[1]:.1%} = {var_exp.sum():.1%}")

# t-SNE
print(" Running t-SNE")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
features_tsne = tsne.fit_transform(all_features)

# Quantify domain shift
your_mask = all_datasets == 'Your Data'
bonn_mask = all_datasets == 'Bonn'

your_centroid = features_pca[your_mask].mean(axis=0)
bonn_centroid = features_pca[bonn_mask].mean(axis=0)

distance = np.linalg.norm(your_centroid - bonn_centroid)

print(f"\n Domain Shift Analysis:")
print(f"   Centroid distance (PCA): {distance:.4f}")

if distance > 1.0:
    print(f"   HIGH domain shift - explains poor generalization!")
else:
    print(f"   Moderate domain shift")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PCA by dataset
ax = axes[0]
colors = {'Your Data': '#2ecc71', 'Bonn': '#e74c3c'}
markers = {'Your Data': 'o', 'Bonn': 's'}

for dataset in ['Your Data', 'Bonn']:
    mask = all_datasets == dataset
    ax.scatter(features_pca[mask, 0], features_pca[mask, 1],
              c=colors[dataset], marker=markers[dataset],
              label=dataset, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)

ax.set_xlabel(f'PC1 ({var_exp[0]:.1%})', fontsize=11, fontweight='bold')
ax.set_ylabel(f'PC2 ({var_exp[1]:.1%})', fontsize=11, fontweight='bold')
ax.set_title('(a) PCA: Domain Shift Visualization', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# t-SNE by dataset
ax = axes[1]
for dataset in ['Your Data', 'Bonn']:
    mask = all_datasets == dataset
    ax.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
              c=colors[dataset], marker=markers[dataset],
              label=dataset, alpha=0.6, s=40, edgecolors='black', linewidth=0.5)

ax.set_xlabel('t-SNE 1', fontsize=11, fontweight='bold')
ax.set_ylabel('t-SNE 2', fontsize=11, fontweight='bold')
ax.set_title('(b) t-SNE: Domain Shift Visualization', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.suptitle(f'Feature Distribution Analysis\nDomain shift distance: {distance:.2f}',
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(viz_dir / "feature_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n Saved: {viz_dir / 'feature_distribution.png'}")

# SUMMARY
print(" CROSS-DATASET ANALYSIS COMPLETE!")
print(f"   Within-dataset F1: {your_within_f1:.4f}")
print(f"\n   StandardScaler:    F1={results['StandardScaler']['f1']:.4f} ({results['StandardScaler']['retention']:.1f}% retention)")
print(f"   Per-Sample Norm:   F1={results['Per-Sample']['f1']:.4f} ({results['Per-Sample']['retention']:.1f}% retention)")

improvement = results['Per-Sample']['f1'] - results['StandardScaler']['f1']
print(f"\n   Improvement: {improvement:+.4f}")

print(f"\n Domain Shift:")
print(f"   PCA distance: {distance:.4f}")
print(f"   Interpretation: {'HIGH shift - explains generalization challenge' if distance > 1.0 else 'Moderate shift'}")
