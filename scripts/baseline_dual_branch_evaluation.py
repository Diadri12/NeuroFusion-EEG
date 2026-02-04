"""
BASELINE DUAL-BRANCH EVALUATION
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from tqdm import tqdm
import joblib
from datetime import datetime

print("BASELINE DUAL-BRANCH")

# Configuration - BASELINE SETTINGS (DO NOT MODIFY)
BASELINE_CONFIG = {
    # Architecture - Simple dual-branch
    'branch_a_depth': 'medium',      # Standard CNN depth
    'use_residual': True,            # Basic residual connections
    'use_bilstm': False,             # NO BiLSTM (baseline)

    # Normalization
    'normalization': 'group',        # CPU-stable
    'num_groups': 8,

    # Regularization
    'dropout_strategy': 'adaptive',
    'base_dropout': 0.3,
    'min_dropout': 0.1,

    # Loss - Weighted CrossEntropy (conservative)
    'loss_function': 'weighted_ce',  # NOT focal (baseline)
    'weight_scale': 1.5,             # Moderate class weights

    # Training
    'max_epochs': 100,
    'patience': 15,
    'learning_rate': 1e-4,
    'batch_size': 64,
    'gradient_clip': 1.0,

    # Device
    'force_cpu': False
}

# Paths
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
PHASE3_DIR = f"{BASE_DIR}/outputs/dual_branch_training"
frozen_dir = Path(PHASE3_DIR) / "frozen"

# Output directory - LOCKED after first run
diagnostic_dir = Path(PHASE3_DIR) / "BASELINE_DUAL_BRANCH_EVALUATION"
diagnostic_dir.mkdir(exist_ok=True, parents=True)

# Check if already run
lock_file = diagnostic_dir / "DIAGNOSTIC_LOCKED.txt"
if lock_file.exists():
    print("  DUAL BRANCH EVALUATION FILE ALREADY RUN AND LOCKED")
    with open(lock_file) as f:
        print(f.read())

    response = input("\nRe-run anyway? This will overwrite locked results. (yes/no): ")
    if response.lower() != 'yes':
        print("\nExiting. Use phase3_representation_learning.py for next experiment.")
        exit(0)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() and not BASELINE_CONFIG['force_cpu'] else 'cpu')
print(f"\nDevice: {device}")

if device.type == 'cpu' and BASELINE_CONFIG['normalization'] == 'batch':
    BASELINE_CONFIG['normalization'] = 'group'

# LOAD DATA
print("LOADING DATA")

X_windowed = np.load(frozen_dir / "X_windowed.npy")
y_windowed = np.load(frozen_dir / "y_windowed.npy")
X_features = np.load(frozen_dir / "X_features.npy")
scaler = joblib.load(Path(PHASE3_DIR) / "feature_scaler.pkl")

n_classes = len(np.unique(y_windowed))
signal_length = X_windowed.shape[1]
n_features = X_features.shape[1]

print(f"\n Data loaded:")
print(f"   Signals: {X_windowed.shape}")
print(f"   Features: {X_features.shape}")
print(f"   Classes: {n_classes}")

# Split
X_sig_train, X_sig_tmp, X_feat_train, X_feat_tmp, y_train, y_tmp = train_test_split(
    X_windowed, X_features, y_windowed, test_size=0.3, stratify=y_windowed, random_state=42
)
X_sig_val, X_sig_test, X_feat_val, X_feat_test, y_val, y_test = train_test_split(
    X_sig_tmp, X_feat_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
)

print(f"\n Split: Train={len(X_sig_train):,}, Val={len(X_sig_val):,}, Test={len(X_sig_test):,}")

# DATASET
class DualInputDataset(Dataset):
    def __init__(self, signals, features, labels, feature_scaler):
        self.signals = torch.FloatTensor(signals).unsqueeze(1)
        self.features = torch.FloatTensor(feature_scaler.transform(features))
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.features[idx], self.labels[idx]

train_dataset = DualInputDataset(X_sig_train, X_feat_train, y_train, scaler)
val_dataset = DualInputDataset(X_sig_val, X_feat_val, y_val, scaler)
test_dataset = DualInputDataset(X_sig_test, X_feat_test, y_test, scaler)

train_loader = DataLoader(train_dataset, batch_size=BASELINE_CONFIG['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BASELINE_CONFIG['batch_size'], shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BASELINE_CONFIG['batch_size'], shuffle=False, num_workers=0)

# BASELINE MODEL (NO BiLSTM)
print("BASELINE DUAL-BRANCH MODEL")

def get_norm_layer(norm_type, num_channels, num_groups=8):
    if norm_type == 'batch':
        return nn.BatchNorm1d(num_channels)
    elif norm_type == 'group':
        return nn.GroupNorm(min(num_groups, num_channels), num_channels)
    else:
        return nn.Identity()

# Branch A: Standard CNN (NO BiLSTM)
class BaselineBranchA(nn.Module):
    def __init__(self, signal_length, embedding_dim=64, norm_type='group', num_groups=8):
        super().__init__()

        # 4 conv layers (medium depth)
        self.conv1 = nn.Conv1d(1, 32, 7, 2, 3)
        self.norm1 = get_norm_layer(norm_type, 32, num_groups)

        self.conv2 = nn.Conv1d(32, 64, 5, 1, 2)
        self.norm2 = get_norm_layer(norm_type, 64, num_groups)

        self.conv3 = nn.Conv1d(64, 128, 3, 1, 1)
        self.norm3 = get_norm_layer(norm_type, 128, num_groups)

        self.conv4 = nn.Conv1d(128, 256, 3, 1, 1)
        self.norm4 = get_norm_layer(norm_type, 256, num_groups)

        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.pool(self.relu(self.norm1(self.conv1(x)))))
        x = self.dropout(self.pool(self.relu(self.norm2(self.conv2(x)))))
        x = self.dropout(self.pool(self.relu(self.norm3(self.conv3(x)))))
        x = self.dropout(self.pool(self.relu(self.norm4(self.conv4(x)))))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

# Branch B: Feature encoder
class BaselineBranchB(nn.Module):
    def __init__(self, n_features, embedding_dim=32):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

    def forward(self, x):
        return self.model(x)

# Baseline dual-branch model
class BaselineDualBranch(nn.Module):
    def __init__(self, signal_length, n_features, n_classes, config):
        super().__init__()

        self.branch_a = BaselineBranchA(
            signal_length,
            embedding_dim=64,
            norm_type=config['normalization'],
            num_groups=config['num_groups']
        )

        self.branch_b = BaselineBranchB(n_features, embedding_dim=32)

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

model = BaselineDualBranch(signal_length, n_features, n_classes, BASELINE_CONFIG).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n Baseline model created:")
print(f"   Parameters: {total_params:,}")
print(f"   Branch A: CNN only (NO BiLSTM)")
print(f"   Branch B: Feature encoder")
print(f"   Architecture: Medium depth, residual connections")

# LOSS FUNCTION - WEIGHTED CE
print("LOSS FUNCTION")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Moderate scaling
class_weights_scaled = class_weights ** BASELINE_CONFIG['weight_scale']

print(f"\n Class weights (scale={BASELINE_CONFIG['weight_scale']}):")
for i, w in enumerate(class_weights_scaled):
    print(f"   Class {i}: {w:.4f}")

class_weights_tensor = torch.FloatTensor(class_weights_scaled).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

print(f"\n Loss: Weighted CrossEntropyLoss (baseline)")

# TRAINING
optimizer = optim.AdamW(model.parameters(), lr=BASELINE_CONFIG['learning_rate'], weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, BASELINE_CONFIG['max_epochs'])

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for signals, features, labels in tqdm(loader, desc="Training", leave=False):
        signals, features, labels = signals.to(device), features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(signals, features)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), BASELINE_CONFIG['gradient_clip'])
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for signals, features, labels in tqdm(loader, desc="Validating", leave=False):
            signals, features, labels = signals.to(device), features.to(device), labels.to(device)
            outputs = model(signals, features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)

print("TRAINING BASELINE")

history = {
    'train_loss': [], 'train_f1': [],
    'val_loss': [], 'val_f1': [], 'val_recalls': [],
    'learning_rate': []
}

best_f1 = 0.0
patience_counter = 0
best_state = None

print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val F1':<10} {'Recalls':<30} {'Status'}")

start_time = time.time()

for epoch in range(BASELINE_CONFIG['max_epochs']):
    train_loss, train_preds, train_labels = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device)

    scheduler.step()

    train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
    val_recalls = recall_score(val_labels, val_preds, average=None, zero_division=0)

    history['train_loss'].append(train_loss)
    history['train_f1'].append(train_f1)
    history['val_loss'].append(val_loss)
    history['val_f1'].append(val_f1)
    history['val_recalls'].append(val_recalls.tolist())
    history['learning_rate'].append(optimizer.param_groups[0]['lr'])

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = model.state_dict().copy()
        patience_counter = 0
        status = "YES"
    else:
        patience_counter += 1
        status = f"({patience_counter}/{BASELINE_CONFIG['patience']})"

    recall_str = f"[{val_recalls[0]:.3f}, {val_recalls[1]:.3f}, {val_recalls[2]:.3f}]"

    if epoch % 5 == 0 or epoch == BASELINE_CONFIG['max_epochs'] - 1:
        print(f"{epoch:<8} {train_loss:<12.4f} {val_loss:<12.4f} {val_f1:<10.4f} {recall_str:<30} {status}")

    if patience_counter >= BASELINE_CONFIG['patience']:
        print(f"\n Early stopping at epoch {epoch}")
        break

if best_state:
    model.load_state_dict(best_state)

elapsed = time.time() - start_time
print(f"\n Training completed in {elapsed/60:.1f} minutes")

# EVALUATION
print("BASELINE EVALUATION")

_, test_preds, test_labels = validate(model, test_loader, criterion, device)

# Compute all metrics
test_acc = accuracy_score(test_labels, test_preds)
test_f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)
test_f1_weighted = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
test_precision = precision_score(test_labels, test_preds, average='macro', zero_division=0)
test_recall = recall_score(test_labels, test_preds, average='macro', zero_division=0)
test_per_class_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)
test_per_class_recall = recall_score(test_labels, test_preds, average=None, zero_division=0)
test_per_class_precision = precision_score(test_labels, test_preds, average=None, zero_division=0)
test_cm = confusion_matrix(test_labels, test_preds)

print(f"\n BASELINE PERFORMANCE:")
print(f"   Accuracy:          {test_acc:.4f}")
print(f"   Precision (macro): {test_precision:.4f}")
print(f"   Recall (macro):    {test_recall:.4f}")
print(f"   F1 (macro):        {test_f1_macro:.4f}")
print(f"   F1 (weighted):     {test_f1_weighted:.4f}")

print(f"\n Per-Class Performance:")
for i in range(n_classes):
    print(f"   Class {i}: P={test_per_class_precision[i]:.4f}, "
          f"R={test_per_class_recall[i]:.4f}, F1={test_per_class_f1[i]:.4f}")

print(f"\n Confusion Matrix:")
print(test_cm)

# SAVE RESULTS
print("SAVING BASELINE RESULTS")

# Save model
torch.save(model.state_dict(), diagnostic_dir / "baseline_model.pt")

# Save comprehensive results
baseline_results = {
    'experiment': 'Phase 3 Diagnostic - Baseline Dual-Branch',
    'timestamp': datetime.now().isoformat(),
    'configuration': BASELINE_CONFIG,

    # Model architecture
    'architecture': {
        'type': 'Dual-Branch CNN',
        'branch_a': 'CNN (4 conv layers, NO BiLSTM)',
        'branch_b': 'MLP (2 dense layers)',
        'fusion': 'Concatenation',
        'total_parameters': total_params
    },

    # Performance metrics
    'test_performance': {
        'accuracy': float(test_acc),
        'precision_macro': float(test_precision),
        'recall_macro': float(test_recall),
        'f1_macro': float(test_f1_macro),
        'f1_weighted': float(test_f1_weighted)
    },

    # Per-class metrics
    'per_class_metrics': {
        f'class_{i}': {
            'precision': float(test_per_class_precision[i]),
            'recall': float(test_per_class_recall[i]),
            'f1': float(test_per_class_f1[i])
        }
        for i in range(n_classes)
    },

    # Confusion matrix
    'confusion_matrix': test_cm.tolist(),

    # Training details
    'training': {
        'total_epochs': len(history['train_loss']),
        'best_val_f1': float(best_f1),
        'training_time_minutes': float(elapsed / 60)
    }
}

with open(diagnostic_dir / "baseline_results.json", 'w') as f:
    json.dump(baseline_results, f, indent=2)

# Save history
with open(diagnostic_dir / "baseline_history.json", 'w') as f:
    json.dump(history, f, indent=2)

# Classification report
report = classification_report(test_labels, test_preds,
                               target_names=[f'Class {i}' for i in range(n_classes)])
with open(diagnostic_dir / "classification_report.txt", 'w') as f:
    f.write("BASELINE DUAL-BRANCH CLASSIFICATION REPORT\n")
    f.write("="*50 + "\n\n")
    f.write(report)

print(f" Results saved to: {diagnostic_dir}")

# VISUALIZATION
print("VISUALIZATION")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Training curves
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history['train_loss'], label='Train', alpha=0.7, linewidth=2)
ax1.plot(history['val_loss'], label='Validation', alpha=0.7, linewidth=2)
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('(a) Training Loss', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history['train_f1'], label='Train', alpha=0.7, linewidth=2)
ax2.plot(history['val_f1'], label='Validation', alpha=0.7, linewidth=2)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Macro F1 Score', fontsize=11)
ax2.set_title('(b) F1 Score Progression', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Per-class recalls
ax3 = fig.add_subplot(gs[0, 2])
val_recalls_array = np.array(history['val_recalls'])
for i in range(n_classes):
    ax3.plot(val_recalls_array[:, i], label=f'Class {i}', linewidth=2)
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('Recall', fontsize=11)
ax3.set_title('(c) Per-Class Recall', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Confusion matrix
ax4 = fig.add_subplot(gs[1, :2])
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar_kws={'label': 'Count'})
ax4.set_xlabel('Predicted Class', fontsize=11)
ax4.set_ylabel('True Class', fontsize=11)
ax4.set_title('(d) Confusion Matrix', fontsize=12, fontweight='bold')

# Normalized confusion matrix
ax5 = fig.add_subplot(gs[1, 2])
cm_norm = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax5, vmin=0, vmax=1,
            cbar_kws={'label': 'Proportion'})
ax5.set_xlabel('Predicted Class', fontsize=11)
ax5.set_ylabel('True Class', fontsize=11)
ax5.set_title('(e) Normalized Confusion Matrix', fontsize=12, fontweight='bold')

# Performance bar chart
ax6 = fig.add_subplot(gs[2, :])
metrics_names = ['Precision', 'Recall', 'F1-Score']
class_0_metrics = [test_per_class_precision[0], test_per_class_recall[0], test_per_class_f1[0]]
class_1_metrics = [test_per_class_precision[1], test_per_class_recall[1], test_per_class_f1[1]]
class_2_metrics = [test_per_class_precision[2], test_per_class_recall[2], test_per_class_f1[2]]

x = np.arange(len(metrics_names))
width = 0.25

ax6.bar(x - width, class_0_metrics, width, label='Class 0', alpha=0.8)
ax6.bar(x, class_1_metrics, width, label='Class 1', alpha=0.8)
ax6.bar(x + width, class_2_metrics, width, label='Class 2', alpha=0.8)

ax6.set_ylabel('Score', fontsize=11)
ax6.set_title('(f) Per-Class Performance Metrics', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics_names)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_ylim(0, 1)

plt.suptitle(f'Baseline Dual-Branch CNN - Macro F1: {test_f1_macro:.4f}',
             fontsize=14, fontweight='bold')
plt.savefig(diagnostic_dir / "baseline_results.png", dpi=300, bbox_inches='tight')
plt.close()

print(f" Figure saved: baseline_results.png")

# CREATE LOCK FILE
print("LOCKING DIAGNOSTIC RUN")

thesis_summary = f"""
BASELINE DUAL-BRANCH (LOCKED)
=================================================
Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

THESIS SUMMARY:
"Baseline dual-branch CNN with weighted CrossEntropy shows partial class
separability with macro F1 of {test_f1_macro:.4f}. The architecture combines
convolutional feature learning (Branch A) with handcrafted EEG features
(Branch B), achieving {test_per_class_recall[0]:.2%}, {test_per_class_recall[1]:.2%},
and {test_per_class_recall[2]:.2%} recall across the three classes respectively."

KEY RESULTS:
- Macro F1: {test_f1_macro:.4f}
- Accuracy: {test_acc:.4f}
- All classes predicted: {all(r > 0 for r in test_per_class_recall)}
- Architecture: CNN (NO BiLSTM) + Feature Encoder

LIMITATIONS IDENTIFIED:
- Limited temporal dependency modeling (CNN only)
- Moderate performance on minority classes
- No long-range pattern capture

NEXT STEP:
→ Phase 3 Representation Learning (BiLSTM enhancement)
   Expected: Improved temporal modeling → Higher macro F1

FILES FOR THESIS:
- baseline_results_THESIS.png (Figure)
- baseline_results.json (Metrics table)
- classification_report.txt (Detailed results)
"""

with open(lock_file, 'w') as f:
    f.write(thesis_summary)

print(thesis_summary)

print("BASELINE DIAGNOSTIC COMPLETE AND LOCKED")
