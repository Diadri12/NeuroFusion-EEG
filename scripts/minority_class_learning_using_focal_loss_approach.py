"""
MINORITY CLASS LEARNING: FOCAL LOSS APPROACH
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time

print("MINORITY CLASS LEARNING: FOCAL LOSS APPROACH")

# Configuration
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
PHASE2_5_DIR = f"{BASE_DIR}/outputs/aggressive_minority_learning"
FOCAL_DIR = f"{PHASE2_5_DIR}/focal_loss_variant"
Path(FOCAL_DIR).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# FOCAL LOSS IMPLEMENTATION
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights tensor
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)  # probability of correct class

        focal_weight = (1 - p) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

print("\n Focal Loss implemented")

# Load Data
print("LOADING DATA")

signal_path = f"{BASE_DIR}/outputs/final_processed/epilepsy_122mb"
X = np.load(Path(signal_path) / "preprocessed_signals.npy")
y = np.load(Path(signal_path) / "labels.npy")

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

n_classes = len(np.unique(y))
print(f"Data loaded: {len(X_train):,} train, {len(X_val):,} val, {len(X_test):,} test")

# Normalize labels
y_train_norm = y_train - y_train.min()
y_val_norm = y_val - y_val.min()
y_test_norm = y_test - y_test.min()

# Compute class weights for focal loss alpha
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_norm),
    y=y_train_norm
)

# For focal loss, combine with aggressive scaling
ALPHA_SCALE = 2.0  # Make minority class weights even stronger
class_weights_focal = class_weights ** ALPHA_SCALE
class_weights_tensor = torch.tensor(class_weights_focal, dtype=torch.float).to(device)

print(f"\nFocal Loss Alpha Weights (scale={ALPHA_SCALE}):")
for i, w in enumerate(class_weights_focal):
    count = np.sum(y_train_norm == i)
    print(f"  Class {i}: alpha={w:.4f} ({count:,} samples)")

# Model Definition
class ImprovedTimeDomainModel(nn.Module):
    def __init__(self, n_classes=3, dropout=0.5):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, 7, 2, 3)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, 5, 2, 2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, 3, 2, 1)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 256, 3, 2, 1)
        self.bn4 = nn.BatchNorm1d(256)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, n_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(self.relu(self.bn4(self.conv4(x))))

        x = self.gap(x).squeeze(-1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# Prepare Data
print("PREPARING DATALOADERS")

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

# Stratified sampler
class_counts = np.bincount(y_train_norm)
sample_weights = 1.0 / class_counts[y_train_norm]
train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_dl = DataLoader(train_ds, batch_size=64, sampler=train_sampler)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

print(" Stratified sampling enabled")
print(" Batch size: 64")

# Training Setup
print("TRAINING WITH FOCAL LOSS")

model = ImprovedTimeDomainModel(n_classes=n_classes, dropout=0.5).to(device)
criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

MAX_EPOCHS = 100
PATIENCE = 15

print(f"\nConfiguration:")
print(f"  Loss: Focal Loss (gamma=2.0)")
print(f"  Optimizer: AdamW (lr=1e-4)")
print(f"  Max epochs: {MAX_EPOCHS}")
print(f"  Patience: {PATIENCE}")
print(f"  Dropout: 0.5")

# Training Loop
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

# Evaluation
print("EVALUATION")

test_loss, test_preds, test_labels = validate(model, test_dl, criterion, device)

acc = accuracy_score(test_labels, test_preds)
f1_weighted = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)
per_class_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)
cm = confusion_matrix(test_labels, test_preds)

print(f"\nTest Results:")
print(f"  Accuracy:      {acc:.4f}")
print(f"  F1 (weighted): {f1_weighted:.4f}")
print(f"  F1 (macro):    {f1_macro:.4f}")

print(f"\nPer-Class F1:")
for i, f1 in enumerate(per_class_f1):
    print(f"  Class {i}: {f1:.4f}")

# Critical Diagnostic
print("PREDICTION DISTRIBUTION")

predictions_per_class = np.bincount(test_preds, minlength=n_classes)
samples_per_class = np.bincount(test_labels, minlength=n_classes)

for i in range(n_classes):
    predicted = predictions_per_class[i]
    actual = samples_per_class[i]
    pct = 100 * predicted / len(test_preds)
    status = "Yes" if predicted > 0 else "No"
    print(f"  {status} Class {i}: {predicted:5d} predictions ({pct:5.2f}%) | actual: {actual:5d}")

all_classes_predicted = all(predictions_per_class > 0)
if all_classes_predicted:
    print("SUCCESS! All classes predicted with Focal Loss")
else:
    print("WARNING: Some classes still not predicted")

# Save Results
torch.save(model.state_dict(), Path(FOCAL_DIR) / "model_focal.pt")

metrics = {
    'accuracy': float(acc),
    'f1_weighted': float(f1_weighted),
    'f1_macro': float(f1_macro),
    'per_class_f1': per_class_f1.tolist(),
    'predictions_per_class': predictions_per_class.tolist(),
    'all_classes_predicted': bool(all_classes_predicted)
}

with open(Path(FOCAL_DIR) / "metrics_focal.json", 'w') as f:
    json.dump(metrics, f, indent=2)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix (Focal Loss)')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[1], vmin=0, vmax=1)
axes[1].set_title('Normalized Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

axes[2].bar(range(n_classes), predictions_per_class, alpha=0.7, label='Predicted')
axes[2].bar(range(n_classes), samples_per_class, alpha=0.5, label='Actual')
axes[2].set_title('Prediction Distribution')
axes[2].set_xlabel('Class')
axes[2].set_ylabel('Count')
axes[2].legend()

plt.suptitle('Focal Loss Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(Path(FOCAL_DIR) / "focal_results.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n Results saved to: {FOCAL_DIR}")
