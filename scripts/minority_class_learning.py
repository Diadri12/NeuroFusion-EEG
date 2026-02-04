"""
MINORITY CLASS LEARNING
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from scipy import signal as scipy_signal

print("MINORITY CLASS LEARNING")

# Configuration
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
PHASE2_DIR = f"{BASE_DIR}/outputs/weighted_training"
PHASE2_5_DIR = f"{BASE_DIR}/outputs/aggressive_minority_learning"
Path(PHASE2_5_DIR).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# Load Data
print("LOADING DATA")

signal_path = f"{BASE_DIR}/outputs/final_processed/epilepsy_122mb"
X = np.load(Path(signal_path) / "preprocessed_signals.npy")
y = np.load(Path(signal_path) / "labels.npy")

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

n_classes = len(np.unique(y))
print(f"Data loaded and split")
print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
print(f"  Classes: {n_classes}")

# Normalize labels
y_train_norm = y_train - y_train.min()
y_val_norm = y_val - y_val.min()
y_test_norm = y_test - y_test.min()

# Print class distribution
print(f"\nClass Distribution:")
for i in range(n_classes):
    count = np.sum(y_train_norm == i)
    pct = 100 * count / len(y_train_norm)
    print(f"  Class {i}: {count:6,} samples ({pct:5.2f}%)")

# AGGRESSIVE CLASS WEIGHTING
print("STEP 1: COMPUTING AGGRESSIVE CLASS WEIGHTS")

# Standard balanced weights
class_weights_standard = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_norm),
    y=y_train_norm
)

print("\nStandard Balanced Weights:")
for i, w in enumerate(class_weights_standard):
    print(f"  Class {i}: {w:.4f}")

# Aggressive weighting - exponentiate to emphasize minorities
EXPONENT = 1.5  # Can increase to 2.0 if needed
class_weights_aggressive = class_weights_standard ** EXPONENT

print(f"\nAggressive Weights (exponent={EXPONENT}):")
for i, w in enumerate(class_weights_aggressive):
    ratio = w / class_weights_aggressive[0]
    print(f"  Class {i}: {w:.4f} (ratio to Class 0: {ratio:.2f}x)")

class_weights_tensor = torch.tensor(class_weights_aggressive, dtype=torch.float).to(device)

# FREQUENCY-DOMAIN FEATURE EXTRACTION
print("EXTRACTING FREQUENCY-DOMAIN FEATURES")

def extract_stft_features(signal_batch, nperseg=64, noverlap=32):
    stft_features = []

    for i, sig in enumerate(signal_batch):
        if i % 10000 == 0 and i > 0:
            print(f"  Processing signal {i}/{len(signal_batch)}")

        # Compute STFT
        f, t, Zxx = scipy_signal.stft(sig, fs=178, nperseg=nperseg, noverlap=noverlap)

        # Take magnitude and apply log scaling
        mag = np.abs(Zxx)
        log_mag = np.log1p(mag)  # log(1 + x) for stability

        stft_features.append(log_mag)

    return np.array(stft_features)

print("\nExtracting STFT features for all splits")

X_train_freq = extract_stft_features(X_train)
X_val_freq = extract_stft_features(X_val)
X_test_freq = extract_stft_features(X_test)

print(f"\n STFT features extracted")
print(f"  Shape: {X_train_freq.shape} (samples, freq_bins, time_frames)")

# Save frequency features for Phase 3
freq_dir = Path(PHASE2_5_DIR) / "frequency_features"
freq_dir.mkdir(exist_ok=True)

np.save(freq_dir / "X_train_freq.npy", X_train_freq)
np.save(freq_dir / "X_val_freq.npy", X_val_freq)
np.save(freq_dir / "X_test_freq.npy", X_test_freq)

print(f" Frequency features saved to: {freq_dir}")

# IMPROVED MODEL WITH HIGHER DROPOUT
print("STEP 3: BUILDING IMPROVED MODEL")

class ImprovedTimeDomainModel(nn.Module):
    def __init__(self, n_classes=3, dropout=0.4):
        super().__init__()

        # Deeper convolution stack
        self.conv1 = nn.Conv1d(1, 32, 7, 2, 3)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, 5, 2, 2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, 3, 2, 1)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 256, 3, 2, 1)
        self.bn4 = nn.BatchNorm1d(256)

        # Classifier
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, n_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Conv blocks
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(self.relu(self.bn4(self.conv4(x))))

        # Global pooling and classification
        x = self.gap(x).squeeze(-1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

print(" Model architecture:")
print(f"  - 4 convolutional layers")
print(f"  - Dropout: 0.4 (increased from 0.2)")
print(f"  - Deeper feature extraction")

# STRATIFIED MINI-BATCH SAMPLER
print("CREATING STRATIFIED BATCH SAMPLER")

from torch.utils.data import WeightedRandomSampler

def create_stratified_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

train_sampler = create_stratified_sampler(y_train_norm)
print(" Stratified sampler created")
print("  Ensures balanced representation in each batch")

# TRAINING WITH ALL IMPROVEMENTS
print("STEP 5: TRAINING WITH AGGRESSIVE SETTINGS")

# Prepare dataloaders
train_ds = TensorDataset(
    torch.FloatTensor(X_train).unsqueeze(1),
    torch.LongTensor(y_train_norm)
)
val_ds = TensorDataset(
    torch.FloatTensor(X_val).unsqueeze(1),
    torch.LongTensor(y_val_norm)
)
test_ds = TensorDataset(
    torch.FloatTensor(X_test).unsqueeze(1),
    torch.LongTensor(y_test_norm)
)

# Use stratified sampler for training
train_dl = DataLoader(train_ds, batch_size=64, sampler=train_sampler)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

# Create model
model = ImprovedTimeDomainModel(n_classes=n_classes, dropout=0.4).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# Training configuration
MAX_EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-4  # Reduced from 1e-3

print(f"\nTraining Configuration:")
print(f"  Max epochs: {MAX_EPOCHS}")
print(f"  Patience: {PATIENCE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Batch size: 64")
print(f"  Dropout: 0.4")
print(f"  Class weights: aggressive (exponent={EXPONENT})")

# Training functions
def train_epoch(model, loader, opt, crit, device):
    model.train()
    loss_sum, correct, total = 0, 0, 0

    for raw, y in loader:
        raw, y = raw.to(device), y.to(device)
        out = model(raw)
        loss = crit(out, y)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_sum += loss.item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / len(loader), correct / total

def validate(model, loader, crit, device):
    model.eval()
    loss_sum = 0
    preds, labels = [], []

    with torch.no_grad():
        for raw, y in loader:
            raw, y = raw.to(device), y.to(device)
            out = model(raw)
            loss_sum += crit(out, y).item()
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    return loss_sum / len(loader), np.array(preds), np.array(labels)

# Initialize training
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_EPOCHS)

best_loss = float('inf')
best_state = None
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': []}

print("TRAINING STARTED")

start_time = time.time()

for epoch in range(MAX_EPOCHS):
    tr_loss, tr_acc = train_epoch(model, train_dl, optimizer, criterion, device)
    val_loss, val_preds, val_labels = validate(model, val_dl, criterion, device)

    history['train_loss'].append(tr_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(tr_acc)

    scheduler.step()

    # Check predictions distribution every 10 epochs
    if epoch % 10 == 0:
        pred_dist = np.bincount(val_preds, minlength=n_classes)
        pred_str = ", ".join([f"C{i}:{pred_dist[i]}" for i in range(n_classes)])
        print(f"Epoch {epoch:03d} | Train: {tr_loss:.4f} (acc={tr_acc:.4f}) | "
              f"Val: {val_loss:.4f} | Pred: [{pred_str}]")

    if val_loss < best_loss:
        best_loss = val_loss
        best_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print(f"\n Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_state)
elapsed = time.time() - start_time
print(f"\n Training completed in {elapsed/60:.1f} minutes")

# EVALUATION AND DIAGNOSTICS
print("EVALUATION AND DIAGNOSTICS")

# Test evaluation
test_loss, test_preds, test_labels = validate(model, test_dl, criterion, device)

# Compute metrics
acc = accuracy_score(test_labels, test_preds)
f1_weighted = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)
per_class_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)
cm = confusion_matrix(test_labels, test_preds)

print("TEST RESULTS")
print(f"Accuracy:      {acc:.4f}")
print(f"F1 (weighted): {f1_weighted:.4f}")
print(f"F1 (macro):    {f1_macro:.4f}")

print(f"\nPer-Class F1 Scores:")
for i, f1 in enumerate(per_class_f1):
    print(f"  Class {i}: {f1:.4f}")

# Critical diagnostic
print("CRITICAL DIAGNOSTIC CHECK")

predictions_per_class = np.bincount(test_preds, minlength=n_classes)
samples_per_class = np.bincount(test_labels, minlength=n_classes)

print("\nPrediction Distribution:")
for i in range(n_classes):
    predicted = predictions_per_class[i]
    actual = samples_per_class[i]
    pct = 100 * predicted / len(test_preds)

    if predicted > 0:
        status = " PREDICTING"
    else:
        status = " STILL IGNORED"

    print(f"  Class {i}: {predicted:5d} predictions ({pct:5.2f}%) | "
          f"actual: {actual:5d} | {status}")

all_classes_predicted = all(predictions_per_class > 0)

if all_classes_predicted:
    print("SUCCESS! All classes are now being predicted!")
else:
    print("WARNING: Some classes still not predicted")

# Save results and metrics
print("SAVING RESULTS")

# Save model
torch.save(model.state_dict(), Path(PHASE2_5_DIR) / "model_aggressive.pt")

# Save metrics
metrics = {
    'accuracy': float(acc),
    'f1_weighted': float(f1_weighted),
    'f1_macro': float(f1_macro),
    'per_class_f1': per_class_f1.tolist(),
    'predictions_per_class': predictions_per_class.tolist(),
    'samples_per_class': samples_per_class.tolist(),
    'all_classes_predicted': bool(all_classes_predicted),
    'config': {
        'exponent': EXPONENT,
        'dropout': 0.4,
        'learning_rate': LEARNING_RATE,
        'max_epochs': MAX_EPOCHS,
        'patience': PATIENCE
    }
}

with open(Path(PHASE2_5_DIR) / "metrics_aggressive.json", 'w') as f:
    json.dump(metrics, f, indent=2)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('True')
axes[0, 0].set_title('Confusion Matrix')

# Normalized confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[0, 1], vmin=0, vmax=1)
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('True')
axes[0, 1].set_title('Confusion Matrix (Normalized)')

# Training curves
axes[1, 0].plot(history['train_loss'], label='Train Loss')
axes[1, 0].plot(history['val_loss'], label='Val Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('Training Progress')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Prediction distribution
axes[1, 1].bar(range(n_classes), predictions_per_class, alpha=0.7, label='Predicted')
axes[1, 1].bar(range(n_classes), samples_per_class, alpha=0.5, label='Actual')
axes[1, 1].set_xlabel('Class')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Prediction vs Actual Distribution')
axes[1, 1].legend()
axes[1, 1].set_xticks(range(n_classes))

plt.suptitle('Minority Class Learning', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(Path(PHASE2_5_DIR) / "aggressive_results.png", dpi=300, bbox_inches='tight')
plt.close()

print(f" Saved model, metrics, and visualizations")
print(f"\nOutput directory: {PHASE2_5_DIR}")
