"""
CLASS-WEIGHTED TRAINING
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

print("CLASS-WEIGHTED TRAINING")

# Configuration
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
PHASE1_DIR = f"{BASE_DIR}/outputs/problem_analysis"
PHASE2_DIR = f"{BASE_DIR}/outputs/weighted_training"
Path(PHASE2_DIR).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load Problem Analysis
print("Loading Problem Analysis")

analysis_path = Path(PHASE1_DIR) / "analysis_data.json"
if not analysis_path.exists():
    print(" ERROR: Problem analysis not found!")
    exit(1)

with open(analysis_path, 'r') as f:
    analysis = json.load(f)

print(f" Problem Analysis loaded")
print(f"  Imbalance ratio: {analysis['imbalance_ratio']:.2f}:1")
print(f"  Severity: {analysis['severity']}")
print(f"  Needs weighting: {analysis['needs_class_weighting']}\n")

if not analysis['needs_class_weighting']:
    print(" Class weighting not critical for this dataset")

# Load Data
print("Loading Data")

signal_path = f"{BASE_DIR}/outputs/final_processed/epilepsy_122mb"
X = np.load(Path(signal_path) / "preprocessed_signals.npy")
y = np.load(Path(signal_path) / "labels.npy")

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

n_classes = len(np.unique(y))
print(f" Data loaded and split")
print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
print(f"  Classes: {n_classes}\n")

# Compute Class Weights
print("Computing Balanced Class Weights")

# Normalize labels
y_train_norm = y_train - y_train.min()
y_val_norm = y_val - y_val.min()
y_test_norm = y_test - y_test.min()

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_norm),
    y=y_train_norm
)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

print("CLASS WEIGHTS:")
for i, w in enumerate(class_weights):
    original_count = np.sum(y_train_norm == i)
    print(f"  Class {i}: weight = {w:.4f} (has {original_count:,} samples)")

# Model Definition
class BranchA_TimeDomain(nn.Module):
    def __init__(self, out_feat=128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 7, 2, 3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 5, 2, 2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, 2, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc = nn.Linear(128, out_feat)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.drop(self.pool(self.relu(self.bn1(self.conv1(x)))))
        x = self.drop(self.pool(self.relu(self.bn2(self.conv2(x)))))
        x = self.drop(self.relu(self.bn3(self.conv3(x))))
        x = self.gap(x).squeeze(-1)
        return self.relu(self.fc(x))

# Training Functions
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
    preds, labels, probs = [], [], []

    with torch.no_grad():
        for raw, y in loader:
            raw, y = raw.to(device), y.to(device)
            out = model(raw)
            loss_sum += crit(out, y).item()
            p = torch.softmax(out, 1)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(p.cpu().numpy())

    return loss_sum / len(loader), np.array(preds), np.array(labels), np.array(probs)

def train_model(model, train_dl, val_dl, criterion, epochs=50, patience=10, lr=1e-3):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': []}

    print(f"Training: {epochs} epochs (max), patience={patience}\n")
    start_time = time.time()

    for epoch in range(epochs):
        tr_loss, tr_acc = train_epoch(model, train_dl, opt, criterion, device)
        val_loss, _, _, _ = validate(model, val_dl, criterion, device)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc)

        sched.step()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
            improved = "Yes"
        else:
            patience_counter += 1
            improved = "No"

        if epoch % 2 == 0:
            print(f"Epoch {epoch:03d} | Train: {tr_loss:.4f} ({tr_acc:.4f}) | "
                  f"Val: {val_loss:.4f} | Patience: {patience_counter}/{patience} {improved}")

        if patience_counter >= patience:
            print(f"\n Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    elapsed = time.time() - start_time
    print(f" Training completed in {elapsed/60:.1f} minutes\n")

    return model, history

def compute_metrics(y_true, y_pred, y_probs, n_classes):
    m = {}
    m['accuracy'] = accuracy_score(y_true, y_pred)
    m['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    m['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    m['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    m['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    m['per_class_f1'] = per_class_f1.tolist()
    m['cm'] = confusion_matrix(y_true, y_pred)

    return m

def print_detailed_metrics(m, name):
    print(f"{'='*70}")
    print(f"{name} - Test Results")
    print(f"{'='*70}")
    print(f"Accuracy:      {m['accuracy']:.4f}")
    print(f"F1 (weighted): {m['f1_weighted']:.4f}")
    print(f"F1 (macro):    {m['f1_macro']:.4f}")
    print(f"Precision:     {m['precision']:.4f}")
    print(f"Recall:        {m['recall']:.4f}")

    print(f"\nPer-Class F1 Scores:")
    for i, f1 in enumerate(m['per_class_f1']):
        print(f"  Class {i}: {f1:.4f}")

# Train with class weighting
print("Training Baseline with Class Weighting")

# Prepare data
train_ds = TensorDataset(torch.FloatTensor(X_train).unsqueeze(1), torch.LongTensor(y_train_norm))
val_ds = TensorDataset(torch.FloatTensor(X_val).unsqueeze(1), torch.LongTensor(y_val_norm))
test_ds = TensorDataset(torch.FloatTensor(X_test).unsqueeze(1), torch.LongTensor(y_test_norm))

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64)
test_dl = DataLoader(test_ds, batch_size=64)

# Create model
baseline_weighted = nn.Sequential(
    BranchA_TimeDomain(128),
    nn.Linear(128, n_classes)
).to(device)

# Train with weighted loss
criterion_weighted = nn.CrossEntropyLoss(weight=class_weights_tensor)
baseline_weighted, hist_weighted = train_model(
    baseline_weighted, train_dl, val_dl, criterion_weighted,
    epochs=50, patience=10, lr=1e-3
)

# Evaluate
_, preds_w, labels_w, probs_w = validate(baseline_weighted, test_dl, criterion_weighted, device)
metrics_weighted = compute_metrics(labels_w, preds_w, probs_w, n_classes)
print_detailed_metrics(metrics_weighted, "Baseline (Weighted)")

# Save results
weighted_dir = Path(PHASE2_DIR) / "baseline_weighted"
weighted_dir.mkdir(exist_ok=True)

torch.save(baseline_weighted.state_dict(), weighted_dir / "model.pt")

with open(weighted_dir / "metrics.json", 'w') as f:
    json.dump({k: v for k, v in metrics_weighted.items() if k not in ['cm', 'per_class_f1']}, f, indent=2, default=float)

# Comparison and Visualization
print("Comparison with Previous Results")

# Try to load previous unweighted results
old_results_path = Path(f"{BASE_DIR}/outputs/dual_branch_experiments/epilepsy_122mb/baseline/metrics.json")

if old_results_path.exists():
    with open(old_results_path, 'r') as f:
        metrics_unweighted = json.load(f)

    print("\nBEFORE vs AFTER Class Weighting:")
    print(f"{'Metric':<20} {'Before':<12} {'After':<12} {'Change':<12}")

    for metric in ['accuracy', 'f1', 'precision', 'recall']:
        before = metrics_unweighted.get(metric, 0)
        after = metrics_weighted.get(metric if metric != 'f1' else 'f1_weighted', 0)
        change = after - before
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"{metric.capitalize():<20} {before:<12.4f} {after:<12.4f} {arrow} {abs(change):<.4f}")

else:
    print("\n No previous baseline found for comparison")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix
cm_weighted = metrics_weighted['cm']
sns.heatmap(cm_weighted, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].set_title('Confusion Matrix (Weighted)')

# Normalized confusion matrix
cm_norm = cm_weighted.astype('float') / cm_weighted.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[1], vmin=0, vmax=1)
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].set_title('Confusion Matrix (Normalized)')

plt.suptitle('Class-Weighted Training Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(weighted_dir / "results.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n Saved: results.png")

# Diagnostic Check
print("Diagnostic Check")

print("\n Checking if minority classes are predicted:")

predictions_per_class = np.bincount(preds_w, minlength=n_classes)
samples_per_class = np.bincount(labels_w, minlength=n_classes)

for i in range(n_classes):
    predicted = predictions_per_class[i]
    actual = samples_per_class[i]
    status = " FIXED" if predicted > 0 else " STILL IGNORED"
    print(f"Class {i}: {predicted:5d} predictions (actual: {actual:5d}) {status}")

all_predicted = all(predictions_per_class > 0)
if all_predicted:
    print("\n SUCCESS! All classes are now being predicted")
else:
    print("\n Some classes still not predicted")

print("PHASE 2 COMPLETE")

print(f"\n Output Directory: {PHASE2_DIR}")
