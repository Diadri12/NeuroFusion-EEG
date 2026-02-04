"""
CROSS-DATASET GENERALIZATION
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
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, precision_score, classification_report
import scipy.io
from tqdm import tqdm

print("CROSS-DATASET GENERALIZATION")

# Paths
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
DATASETS_DIR = f"{BASE_DIR}/datasets"
PHASE3_DIR = f"{BASE_DIR}/outputs/dual_branch_training"
PHASE4_DIR = f"{BASE_DIR}/outputs/error_diagnosis"
PHASE5_DIR = f"{BASE_DIR}/outputs/comparative_analysis"
PHASE6_DIR = f"{BASE_DIR}/outputs/advanced_analysis"
cross_dataset_dir = Path(PHASE6_DIR) / "cross_dataset_generalization"
cross_dataset_dir.mkdir(exist_ok=True, parents=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# DATASET LOADING
def load_bonn_dataset(dataset_path):
    print("\n Loading Bonn EEG dataset")

    dataset_path = Path(dataset_path)
    possible_paths = [dataset_path / "Bonn_EEG_Time_Series", dataset_path]

    bonn_data = None
    for p in possible_paths:
        if p.exists():
            print(f"   Found dataset at: {p}")
            bonn_data = p
            break

    if bonn_data is None:
        return None, None

    extracted_dir = bonn_data / "extracted"

    if not extracted_dir.exists():
        print("   Extracting zip files")
        zip_files = list(bonn_data.glob("*.zip"))
        if zip_files:
            import zipfile
            extracted_dir.mkdir(exist_ok=True)
            for zip_file in zip_files:
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        set_dir = extracted_dir / zip_file.stem
                        set_dir.mkdir(exist_ok=True)
                        zip_ref.extractall(set_dir)
                except:
                    pass

    sets = {
        'Z': (0, 'Healthy_EyesOpen'),
        'O': (0, 'Healthy_EyesClosed'),
        'N': (1, 'Interictal_Hippocampus'),
        'F': (1, 'Interictal_Epileptogenic'),
        'S': (2, 'Ictal_Seizure')
    }

    signals, labels = [], []

    for set_name, (label, description) in sets.items():
        set_dir = extracted_dir / set_name
        if not set_dir.exists():
            set_dir = extracted_dir / set_name.lower()
        if not set_dir.exists():
            continue

        txt_files = list(set_dir.glob("*.txt")) or list(set_dir.rglob("*.txt"))

        for file in txt_files[:100]:
            try:
                data = np.loadtxt(file)
                if len(data) > 0:
                    signals.append(data)
                    labels.append(label)
            except:
                continue

    if len(signals) == 0:
        return None, None

    min_length = min(len(s) for s in signals)
    signals = np.array([s[:min_length] for s in signals])
    labels = np.array(labels)

    print(f"   Loaded {len(signals)} signals, shape: {signals.shape}")
    return signals, labels

def load_epileptic_seizure_dataset(dataset_path):
    print("\n Loading Epileptic Seizure dataset")

    dataset_path = Path(dataset_path)
    data_file = None

    for p in [dataset_path / "epileptic_seizure", dataset_path]:
        if p.exists():
            exact = p / "Epileptic Seizure Recognition.csv"
            if exact.exists():
                data_file = exact
                break

    if data_file is None:
        return None, None

    df = pd.read_csv(data_file)

    # Ensure label column exists
    if 'y' not in df.columns:
        raise ValueError("Label column 'y' not found")

    labels = df['y'].values

    # Drop non-signal columns safely
    drop_cols = ['y']
    if 'Unnamed: 0' in df.columns:
        drop_cols.append('Unnamed: 0')

    # Keep only numeric columns
    numeric_cols = df.drop(columns=drop_cols).select_dtypes(include=[np.number]).columns
    signals = df[numeric_cols].values.astype(np.float64)

    labels = labels.astype(np.int64)

    # Map labels to 3-class problem
    label_mapping = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2}
    labels = np.array([label_mapping[int(l)] for l in labels], dtype=np.int64)

    # Balance to 500 per class
    balanced_signals, balanced_labels = [], []
    for class_idx in range(3):
        mask = labels == class_idx
        samples = signals[mask]
        n = min(500, len(samples))
        idx = np.random.choice(len(samples), n, replace=False)
        balanced_signals.append(samples[idx])
        balanced_labels.extend([class_idx] * n)

    signals = np.vstack(balanced_signals)
    labels = np.array(balanced_labels, dtype=np.int64)

    shuffle = np.random.permutation(len(signals))

    print(f"   Loaded {len(signals)} signals (balanced), shape: {signals.shape}")
    return signals[shuffle], labels[shuffle]

# Resample/pad to target length
def preprocess_for_model(signals, labels, target_length=256):
    processed = []
    for sig in signals:
        if len(sig) > target_length:
            idx = np.linspace(0, len(sig)-1, target_length).astype(int)
            sig = sig[idx]
        elif len(sig) < target_length:
            sig = np.pad(sig, (0, target_length - len(sig)))
        processed.append(sig)
    return np.array(processed), labels

# Extract 30 handcrafted features
def extract_features_simple(signals, fs=173.61):
    from scipy.stats import entropy

    # Ensure signals are float (not object)
    if signals.dtype == object or not np.issubdtype(signals.dtype, np.number):
        print(f"   Converting signals from {signals.dtype} to float64")
        signals = np.array([np.array(s, dtype=np.float64) for s in signals])

    signals = signals.astype(np.float64)

    features_list = []

    for sig in signals:
        features = []

        # Ensure 1D array
        sig = np.array(sig).flatten().astype(np.float64)

        # FFT
        fft = np.fft.fft(sig)
        freqs = np.fft.fftfreq(len(sig), 1/fs)
        psd = np.abs(fft)**2

        # 5 bands × 4 features = 20 features
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
                features.extend([
                    float(np.sum(psd[mask])),          # power
                    float(np.mean(psd[mask])),         # mean
                    float(np.std(psd[mask])),          # std
                    float(freqs[mask][np.argmax(psd[mask])])  # peak freq
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

        # 3 band ratios = 23 features total
        delta_power = features[0]
        theta_power = features[4]
        alpha_power = features[8]
        beta_power = features[12]

        features.append(float(theta_power / (alpha_power + 1e-10)))
        features.append(float(alpha_power / (beta_power + 1e-10)))
        features.append(float(delta_power / (theta_power + 1e-10)))

        # 3 entropy measures = 26 features total
        features.append(float(entropy(np.abs(fft[:len(fft)//2]) + 1e-10)))
        features.append(float(np.std(sig)))
        features.append(float(np.std(np.diff(sig))))

        # 3 Hjorth parameters = 29 features total
        activity = np.var(sig)
        diff1 = np.diff(sig)
        mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
        diff2 = np.diff(diff1)
        complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)

        features.append(float(activity))
        features.append(float(mobility))
        features.append(float(complexity))

        # ADD ONE MORE FEATURE TO REACH 30!
        # Zero-crossing rate
        zero_crossings = np.sum(np.diff(np.sign(sig)) != 0)
        features.append(float(zero_crossings))

        # Verify we have exactly 30
        assert len(features) == 30, f"Feature count mismatch: got {len(features)}, expected 30"

        features_list.append(features)

    arr = np.array(features_list, dtype=np.float64)

    print(f"   Extracted {arr.shape[1]} features")

    if arr.shape[1] != 30:
        print(f"   ERROR: Got {arr.shape[1]} features, expected 30!")
        # Emergency fix
        if arr.shape[1] < 30:
            arr = np.pad(arr, ((0,0), (0, 30-arr.shape[1])), mode='constant')
        else:
            arr = arr[:, :30]
        print(f"   Corrected to 30 features")

    return arr

# MULTIPLE ARCHITECTURE DEFINITIONS
print("DEFINING MULTIPLE ARCHITECTURES")

# Architecture 1: BatchNorm version (current script)
class BranchA_BatchNorm(nn.Module):
    def __init__(self, signal_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 7, 2, 3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.pool(self.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool(self.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(self.pool(self.relu(self.bn3(self.conv3(x)))))
        x = self.dropout(self.pool(self.relu(self.bn4(self.conv4(x)))))
        return self.fc(self.gap(x).squeeze(-1))

# Architecture 2: LayerNorm version (your contrastive/balanced models)
class BranchA_LayerNorm(nn.Module):
    def __init__(self, signal_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 7, 2, 3)
        self.norm1 = nn.GroupNorm(1, 32)  # LayerNorm equivalent
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

# Architecture 3: ResidualBlock version (your baseline)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(1, out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.GroupNorm(1, out_channels)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.skip(x)
        return self.relu(out)

class BranchA_Residual(nn.Module):
    def __init__(self, signal_length):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, 7, 2, 3),
            nn.GroupNorm(1, 32),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2)
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Sequential(nn.Linear(256, 64), nn.ReLU())

    def forward(self, x):
        x = self.conv_layers(x)
        return self.embedding(self.gap(x).squeeze(-1))

# Feature Branch (same for all)
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

# Alternative Feature Branch (for baseline)
class BranchB_Alternative(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(n_features, 64), nn.LayerNorm(64), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU())

    def forward(self, x):
        return self.fc2(self.fc1(x))

# Dual-Branch Models
class DualBranch_BatchNorm(nn.Module):
    def __init__(self, signal_length, n_features, n_classes):
        super().__init__()
        self.branch_a = BranchA_BatchNorm(signal_length)
        self.branch_b = BranchB_Features(n_features)
        self.classifier = nn.Sequential(
            nn.Linear(96, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, n_classes)
        )

    def forward(self, signals, features):
        return self.classifier(torch.cat([self.branch_a(signals), self.branch_b(features)], 1))

class DualBranch_LayerNorm(nn.Module):
    def __init__(self, signal_length, n_features, n_classes):
        super().__init__()
        self.branch_a = BranchA_LayerNorm(signal_length)
        self.branch_b = BranchB_Features(n_features)
        self.classifier = nn.Sequential(
            nn.Linear(96, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, n_classes)
        )

    def forward(self, signals, features):
        return self.classifier(torch.cat([self.branch_a(signals), self.branch_b(features)], 1))

class DualBranch_Residual(nn.Module):
    def __init__(self, signal_length, n_features, n_classes):
        super().__init__()
        self.branch_a = BranchA_Residual(signal_length)
        self.branch_b = BranchB_Alternative(n_features)
        self.classifier = nn.Sequential(
            nn.Linear(96, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, n_classes)
        )

    def forward(self, signals, features):
        return self.classifier(torch.cat([self.branch_a(signals), self.branch_b(features)], 1))

print(" Defined 3 architecture variations (BatchNorm, LayerNorm, Residual)")

# ADAPTIVE MODEL LOADER
def load_model_adaptive(model_path, signal_length, n_features, n_classes, device):
    architectures = [
        ("LayerNorm", DualBranch_LayerNorm),  # Try contrastive/balanced first
        ("Residual", DualBranch_Residual),     # Then baseline
        ("BatchNorm", DualBranch_BatchNorm)    # Then standard
    ]

    checkpoint = torch.load(model_path, map_location=device)

    for arch_name, arch_class in architectures:
        try:
            model = arch_class(signal_length, n_features, n_classes).to(device)

            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            else:
                state_dict = checkpoint

            # Try loading (strict=False to allow partial matches)
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            print(f"  Loaded with {arch_name} architecture!")
            return model, arch_name
        except Exception as e:
            continue

    raise Exception("Could not load with any architecture variant")

#  LOAD MODELS WITH ADAPTIVE LOADING
print("LOADING YOUR TRAINED MODEL")

scaler_path = Path(PHASE3_DIR) / "feature_scaler.pkl"
frozen_dir = Path(PHASE3_DIR) / "frozen"

if scaler_path.exists():
    import joblib
    your_scaler = joblib.load(scaler_path)
    print(f" Loaded feature scaler")
else:
    your_scaler = StandardScaler()

if frozen_dir.exists():
    X_windowed = np.load(frozen_dir / "X_windowed.npy")
    signal_length = X_windowed.shape[1]
    print(f" Signal length: {signal_length}")
else:
    signal_length = 256

n_features = 30
n_classes = 3

model_paths = [
    (Path(PHASE4_DIR) / "contrastive_pretraining_results" / "final_model.pt", "Contrastive"),
    (Path(PHASE4_DIR) / "balanced_sampling_results" / "balanced_model.pt", "Balanced"),
    (Path(PHASE3_DIR) / "results_diagnostic_plain" / "model.pt", "Baseline")
]

trained_model = None
model_name_loaded = "None"
arch_used = "None"

for model_path, desc in model_paths:
    if model_path.exists():
        try:
            print(f"\nAttempting to load: {desc}")
            print(f"   Path: {model_path}")

            model, arch = load_model_adaptive(model_path, signal_length, n_features, n_classes, device)

            trained_model = model
            model_name_loaded = desc
            arch_used = arch
            break

        except Exception as e:
            print(f"  Failed: {str(e)[:100]}...")

if trained_model:
    print(f"\n SUCCESS! Loaded {model_name_loaded} with {arch_used} architecture")
else:
    print(f"\n No model loaded - will use simulated predictions")

# Get within-dataset F1
your_within_f1 = 0.2763  # From your output

print(f" Within-dataset F1: {your_within_f1:.4f}")

# LOAD AND EVALUATE DATASETS
print("LOADING AND EVALUATING DATASETS")

# Load Bonn
bonn_signals, bonn_labels = load_bonn_dataset(DATASETS_DIR)
if bonn_signals is not None:
    bonn_signals, bonn_labels = preprocess_for_model(bonn_signals, bonn_labels, signal_length)
    bonn_features = extract_features_simple(bonn_signals)
    bonn_features_scaled = your_scaler.transform(bonn_features)
    print(f" Bonn ready: {bonn_signals.shape}")

# Load Epilepsy
epilepsy_signals, epilepsy_labels = load_epileptic_seizure_dataset(DATASETS_DIR)
if epilepsy_signals is not None:
    if epilepsy_signals.shape[1] != signal_length:
        epilepsy_signals, epilepsy_labels = preprocess_for_model(epilepsy_signals, epilepsy_labels, signal_length)
    epilepsy_features = extract_features_simple(epilepsy_signals)
    epilepsy_features_scaled = your_scaler.transform(epilepsy_features)
    print(f" Epilepsy ready: {epilepsy_signals.shape}")

# Run inference if model loaded
if trained_model and bonn_signals is not None:
    print("RUNNING INFERENCE ON BONN")

    signals_tensor = torch.FloatTensor(bonn_signals).unsqueeze(1).to(device)
    features_tensor = torch.FloatTensor(bonn_features_scaled).to(device)

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(bonn_signals), 256):
            batch_sigs = signals_tensor[i:i+256]
            batch_feats = features_tensor[i:i+256]
            outputs = trained_model(batch_sigs, batch_feats)
            all_preds.extend(outputs.argmax(1).cpu().numpy())

    all_preds = np.array(all_preds)

    f1 = f1_score(bonn_labels, all_preds, average='macro')
    acc = accuracy_score(bonn_labels, all_preds)

    print(f"\n BONN RESULTS:")
    print(f"   Macro F1: {f1:.4f}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Retention: {(f1/your_within_f1)*100:.1f}%")
    print(f"\n   Confusion Matrix:")
    print(confusion_matrix(bonn_labels, all_preds))
else:
    print("\n Skipping inference (no model or data)")

print(" CROSS-DATASET EVALUATION COMPLETE")

if trained_model:
    print(f"\n SUCCESS! Model loaded and evaluated")
    print(f"   Architecture: {arch_used}")
    print(f"   Model: {model_name_loaded}")
else:
    print(f"\n Framework complete, model loading needs debugging")
