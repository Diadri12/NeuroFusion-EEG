"""
PREPARE TRAINING PIPELINE
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from pathlib import Path
import json

print("TRAINING PIPELINE PREPARATION")

# Configuration
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
PHASE3_DIR = f"{BASE_DIR}/outputs/dual_branch_training"
frozen_dir = Path(PHASE3_DIR) / "frozen"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load Frozen Data
print("LOADING FROZEN DATA")

# Load data
X_windowed = np.load(frozen_dir / "X_windowed.npy")
y_windowed = np.load(frozen_dir / "y_windowed.npy")
X_features = np.load(frozen_dir / "X_features.npy")

print(f"Loaded windowed signals: {X_windowed.shape}")
print(f"Loaded features: {X_features.shape}")
print(f"Loaded labels: {y_windowed.shape}")

n_classes = len(np.unique(y_windowed))

# DEFINING DUAL-INPUT DATASET
print("DEFINING DUAL-INPUT DATASET")

# Dataset that returns both raw signals and engineered features
class DualInputDataset(Dataset):
    def __init__(self, signals, features, labels, feature_scaler=None, fit_scaler=False):
        self.signals = torch.FloatTensor(signals).unsqueeze(1)  # (N, 1, signal_length)
        self.labels = torch.LongTensor(labels)

        # Normalize features
        if feature_scaler is None and fit_scaler:
            self.feature_scaler = StandardScaler()
            features_normalized = self.feature_scaler.fit_transform(features)
        elif feature_scaler is not None:
            self.feature_scaler = feature_scaler
            features_normalized = self.feature_scaler.transform(features)
        else:
            self.feature_scaler = None
            features_normalized = features

        self.features = torch.FloatTensor(features_normalized)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.features[idx], self.labels[idx]

print("\n Dataset class defined:")
print("   Returns: (signal_window, engineered_features, label)")
print("   Signal shape: (1, signal_length)")
print(f"   Feature shape: ({X_features.shape[1]},)")
print("   Label: class index")

# Split Data
print("SPLITTING DATA")

# Split: 70% train, 15% val, 15% test
X_sig_train, X_sig_tmp, X_feat_train, X_feat_tmp, y_train, y_tmp = train_test_split(
    X_windowed, X_features, y_windowed,
    test_size=0.3, stratify=y_windowed, random_state=42
)

X_sig_val, X_sig_test, X_feat_val, X_feat_test, y_val, y_test = train_test_split(
    X_sig_tmp, X_feat_tmp, y_tmp,
    test_size=0.5, stratify=y_tmp, random_state=42
)

print(f"\n Data split:")
print(f"   Train: {len(X_sig_train):7,} samples ({100*len(X_sig_train)/len(X_windowed):.1f}%)")
print(f"   Val:   {len(X_sig_val):7,} samples ({100*len(X_sig_val)/len(X_windowed):.1f}%)")
print(f"   Test:  {len(X_sig_test):7,} samples ({100*len(X_sig_test)/len(X_windowed):.1f}%)")

# Check stratification
print(f"\n Class distribution per split:")
for split_name, split_labels in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    unique, counts = np.unique(split_labels, return_counts=True)
    print(f"   {split_name}:", end='')
    for cls, cnt in zip(unique, counts):
        pct = 100 * cnt / len(split_labels)
        print(f" Class {cls}={cnt} ({pct:.1f}%)", end='')
    print()

# Create Datasets
print("CREATING DATASETS")

# Create datasets (fit scaler on training data)
train_dataset = DualInputDataset(
    X_sig_train, X_feat_train, y_train,
    feature_scaler=None, fit_scaler=True
)

val_dataset = DualInputDataset(
    X_sig_val, X_feat_val, y_val,
    feature_scaler=train_dataset.feature_scaler, fit_scaler=False
)

test_dataset = DualInputDataset(
    X_sig_test, X_feat_test, y_test,
    feature_scaler=train_dataset.feature_scaler, fit_scaler=False
)

print(f"\n Created datasets with feature normalization")
print(f"   Scaler fit on training data")
print(f"   Applied to validation and test data")

# Verify dataset works
sample_signal, sample_features, sample_label = train_dataset[0]
print(f"\n Sample data:")
print(f"   Signal shape: {sample_signal.shape}")
print(f"   Features shape: {sample_features.shape}")
print(f"   Label: {sample_label.item()}")

# CREATING DATALOADERS
print("CREATING DATALOADERS")

BATCH_SIZE = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True if device.type == 'cuda' else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True if device.type == 'cuda' else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True if device.type == 'cuda' else False
)

print(f"\n Created dataloaders")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
print(f"   Test batches: {len(test_loader)}")

# Test dataloader
signals, features, labels = next(iter(train_loader))
print(f"\n Sample batch:")
print(f"   Signals: {signals.shape}")
print(f"   Features: {features.shape}")
print(f"   Labels: {labels.shape}")

# COMPUTING CLASS WEIGHTS
print("COMPUTING CLASS WEIGHTS")

# Compute balanced class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

print(f"\n Balanced class weights:")
for i, w in enumerate(class_weights):
    count = np.sum(y_train == i)
    print(f"   Class {i}: weight={w:.4f} ({count:,} samples)")

# Apply aggressive scaling
WEIGHT_EXPONENT = 1.5
class_weights_aggressive = class_weights ** WEIGHT_EXPONENT

print(f"\n Aggressive class weights (exponent={WEIGHT_EXPONENT}):")
for i, w in enumerate(class_weights_aggressive):
    ratio = w / class_weights_aggressive[0]
    print(f"   Class {i}: weight={w:.4f} (ratio to Class 0: {ratio:.2f}×)")

# Convert to tensor
class_weights_tensor = torch.FloatTensor(class_weights_aggressive).to(device)

print(f"\n Class weights computed and moved to {device}")

# Define Loss Function
print("DEFINING LOSS FUNCTION")

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

print(f"\n Loss function: Weighted CrossEntropyLoss")
print(f"   Weights: {class_weights_aggressive}")
print(f"   Purpose: Balance class importance during training")

# Define Metrics
print("DEFINING METRICS")

def compute_metrics(y_true, y_pred, split_name=""):
    metrics = {}

    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class metrics
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics['per_class_recall'] = per_class_recall.tolist()
    metrics['per_class_f1'] = per_class_f1.tolist()

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # Prediction distribution
    pred_counts = np.bincount(y_pred, minlength=n_classes)
    metrics['prediction_distribution'] = pred_counts.tolist()

    return metrics

#Print Metrics
def print_metrics(metrics, split_name=""):
    print(f"{split_name} METRICS")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"F1 (macro):    {metrics['f1_macro']:.4f}")

    print(f"\nPer-Class Metrics:")
    for i in range(n_classes):
        recall = metrics['per_class_recall'][i]
        f1 = metrics['per_class_f1'][i]
        status = "✓" if recall > 0 else "✗ WARNING!"
        print(f"  Class {i}: Recall={recall:.4f}, F1={f1:.4f} {status}")

    print(f"\nPrediction Distribution:")
    for i, count in enumerate(metrics['prediction_distribution']):
        print(f"  Class {i}: {count} predictions")

    # Check for zero-recall classes
    if any(r == 0 for r in metrics['per_class_recall']):
        print(f"\n WARNING: Some classes have ZERO recall!")
        print(f"   Training should be stopped and debugged")

print(f"\n  IMPORTANT:")
print(f"   Early stopping will use MACRO F1")
print(f"   If any class has recall=0, training stops")

# Save Pipeline Configuration
print("SAVING PIPELINE CONFIGURATION")

pipeline_config = {
    'data_split': {
        'train_samples': int(len(train_dataset)),
        'val_samples': int(len(val_dataset)),
        'test_samples': int(len(test_dataset)),
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'stratified': True
    },
    'batch_size': BATCH_SIZE,
    'feature_normalization': {
        'method': 'StandardScaler',
        'fit_on': 'training_data',
        'mean': train_dataset.feature_scaler.mean_.tolist(),
        'std': train_dataset.feature_scaler.scale_.tolist()
    },
    'class_weights': {
        'method': 'balanced_with_exponent',
        'exponent': WEIGHT_EXPONENT,
        'weights': class_weights_aggressive.tolist()
    },
    'loss_function': 'CrossEntropyLoss',
    'primary_metric': 'f1_macro',
    'critical_check': 'per_class_recall > 0 for all classes',
    'device': str(device)
}

config_path = Path(PHASE3_DIR) / "pipeline_config.json"
with open(config_path, 'w') as f:
    json.dump(pipeline_config, f, indent=2)

print(f" Saved: {config_path}")

# Save scaler for later use
import joblib
scaler_path = Path(PHASE3_DIR) / "feature_scaler.pkl"
joblib.dump(train_dataset.feature_scaler, scaler_path)
print(f" Saved: {scaler_path}")

# Pipeline Checklist
print("PIPELINE CHECKLIST")

checklist = {
    'Data loaded': True,
    'Dataset class defined': True,
    'Data split (stratified)': True,
    'Feature normalization': True,
    'Dataloaders created': True,
    'Class weights computed': True,
    'Loss function defined': True,
    'Metrics defined': True,
    'Configuration saved': True
}

print("\n Pipeline components:")
for item, status in checklist.items():
    print(f"   {'Yes' if status else 'No'} {item}")

print(f"\n Ready for training:")
print(f"   Training samples: {len(train_dataset):,}")
print(f"   Validation samples: {len(val_dataset):,}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Batches per epoch: {len(train_loader)}")
print(f"   Device: {device}")

print("\n PIPELINE READY ")
