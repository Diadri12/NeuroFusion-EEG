"""
CLASS-BALANCED SAMPLING (DATA-LEVEL INTERVENTION)
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from tqdm import tqdm
import joblib
from datetime import datetime

print("CLASS-BALANCED SAMPLING")
print("\n Data-level intervention: Balanced batch composition")

# Configuration
BALANCED_CONFIG = {
    # Architecture - same as baseline for fair comparison
    'branch_a_depth': 'medium',
    'use_residual': True,
    'use_bilstm': False,  # Start without BiLSTM

    # Normalization
    'normalization': 'group',
    'num_groups': 8,

    # Regularization
    'dropout_strategy': 'adaptive',
    'base_dropout': 0.3,
    'min_dropout': 0.1,

    # Loss - PLAIN CrossEntropy (no weights!)
    'loss_function': 'plain_ce',  # Sampling handles balance
    'use_label_smoothing': 0.1,   # Slight smoothing

    # Training with balanced sampling
    'max_epochs': 120,
    'patience': 20,
    'learning_rate': 1e-4,
    'batch_size': 64,
    'gradient_clip': 1.0,

    # KEY: Balanced sampling
    'use_balanced_sampler': True,

    # Device
    'force_cpu': False
}

print(" Configuration:")
print(f"   use_balanced_sampler: {BALANCED_CONFIG['use_balanced_sampler']} ← KEY CHANGE")
print(f"   loss_function: {BALANCED_CONFIG['loss_function']} (no class weights!)")
print(f"   label_smoothing: {BALANCED_CONFIG['use_label_smoothing']}")

# Paths
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
PHASE3_DIR = f"{BASE_DIR}/outputs/dual_branch_training"
PHASE4_DIR = f"{BASE_DIR}/outputs/error_diagnosis"
balanced_dir = Path(PHASE4_DIR) / "balanced_sampling_results"
balanced_dir.mkdir(exist_ok=True, parents=True)

device = torch.device('cuda' if torch.cuda.is_available() and not BALANCED_CONFIG['force_cpu'] else 'cpu')
print(f"\nDevice: {device}\n")

# LOAD DATA
print("LOADING DATA")

frozen_dir = Path(PHASE3_DIR) / "frozen"

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

# Same split as Phase 3 for fair comparison
X_sig_train, X_sig_tmp, X_feat_train, X_feat_tmp, y_train, y_tmp = train_test_split(
    X_windowed, X_features, y_windowed, test_size=0.3, stratify=y_windowed, random_state=42
)
X_sig_val, X_sig_test, X_feat_val, X_feat_test, y_val, y_test = train_test_split(
    X_sig_tmp, X_feat_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
)

print(f"\n Split (same as Phase 3):")
print(f"   Train: {len(X_sig_train):,}, Val: {len(X_sig_val):,}, Test: {len(X_sig_test):,}")

# BALANCED SAMPLER CREATION
print("CREATING BALANCED SAMPLER")

# Compute class weights for sampler (inverse frequency)
class_counts = np.bincount(y_train)
class_weights_sampler = 1.0 / class_counts
sample_weights = class_weights_sampler[y_train]

print(f"\n Class distribution in training set:")
for i, count in enumerate(class_counts):
    print(f"   Class {i}: {count:,} samples ({100*count/len(y_train):.1f}%)")

print(f"\n Sampler weights (inverse frequency):")
for i, weight in enumerate(class_weights_sampler):
    print(f"   Class {i}: {weight:.4f}")

print(f"\n Balanced sampler will:")
print(f"   - Oversample minority classes")
print(f"   - Undersample majority class")
print(f"   - Ensure balanced batches")

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

# Create balanced sampler for training
balanced_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BALANCED_CONFIG['batch_size'],
    sampler=balanced_sampler,  # ← KEY: Using balanced sampler
    num_workers=0
)

val_loader = DataLoader(val_dataset, batch_size=BALANCED_CONFIG['batch_size'], shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BALANCED_CONFIG['batch_size'], shuffle=False, num_workers=0)

# Verify balanced sampling
print(f"\n Verifying balanced sampling:")
sample_batch_labels = []
for signals, features, labels in train_loader:
    sample_batch_labels.extend(labels.numpy())
    if len(sample_batch_labels) >= 1000:
        break

sample_batch_dist = np.bincount(sample_batch_labels, minlength=n_classes)
print(f"   First 1000 samples distribution:")
for i, count in enumerate(sample_batch_dist):
    print(f"      Class {i}: {count} ({100*count/len(sample_batch_labels):.1f}%)")
print(f"   → Much more balanced than original!")

# MODEL
print("BUILDING MODEL")

def get_norm_layer(norm_type, num_channels, num_groups=8):
    if norm_type == 'batch':
        return nn.BatchNorm1d(num_channels)
    elif norm_type == 'group':
        return nn.GroupNorm(min(num_groups, num_channels), num_channels)
    else:
        return nn.Identity()

class BaselineBranchA(nn.Module):
    def __init__(self, signal_length, embedding_dim=64, norm_type='group', num_groups=8):
        super().__init__()

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

class DualBranchBalanced(nn.Module):
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

model = DualBranchBalanced(signal_length, n_features, n_classes, BALANCED_CONFIG).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n Model created (same architecture as baseline):")
print(f"   Parameters: {total_params:,}")
print(f"   Only change: Balanced sampling, NOT architecture")

# LOSS FUNCTION - PLAIN CE WITH LABEL SMOOTHING
print("LOSS FUNCTION")

# Plain CrossEntropy with label smoothing (NO class weights - sampling handles it)
criterion = nn.CrossEntropyLoss(label_smoothing=BALANCED_CONFIG['use_label_smoothing'])

print(f"\n Loss: Plain CrossEntropyLoss")
print(f"   Label smoothing: {BALANCED_CONFIG['use_label_smoothing']}")
print(f"   NO class weights (balanced sampling handles imbalance)")

# TRAINING
optimizer = optim.AdamW(model.parameters(), lr=BALANCED_CONFIG['learning_rate'], weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, BALANCED_CONFIG['max_epochs'])

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), BALANCED_CONFIG['gradient_clip'])
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

print("TRAINING WITH BALANCED SAMPLING")

print(f"\n  Expected behavior:")
print(f"   - Balanced gradient updates from all classes")
print(f"   - Per-class recalls should be more even")
print(f"   - May sacrifice overall accuracy for balance")

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

for epoch in range(BALANCED_CONFIG['max_epochs']):
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
        status = f"({patience_counter}/{BALANCED_CONFIG['patience']})"

    recall_str = f"[{val_recalls[0]:.3f}, {val_recalls[1]:.3f}, {val_recalls[2]:.3f}]"

    if epoch % 5 == 0 or epoch == BALANCED_CONFIG['max_epochs'] - 1:
        print(f"{epoch:<8} {train_loss:<12.4f} {val_loss:<12.4f} {val_f1:<10.4f} {recall_str:<30} {status}")

    if patience_counter >= BALANCED_CONFIG['patience']:
        print(f"\n Early stopping at epoch {epoch}")
        break

if best_state:
    model.load_state_dict(best_state)

elapsed = time.time() - start_time
print(f"\n Training completed in {elapsed/60:.1f} minutes")

# EVALUATION
print("EVALUATION WITH BALANCED SAMPLING")

_, test_preds, test_labels = validate(model, test_loader, criterion, device)

test_acc = accuracy_score(test_labels, test_preds)
test_f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)
test_f1_weighted = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
test_precision = precision_score(test_labels, test_preds, average='macro', zero_division=0)
test_recall = recall_score(test_labels, test_preds, average='macro', zero_division=0)
test_per_class_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)
test_per_class_recall = recall_score(test_labels, test_preds, average=None, zero_division=0)
test_per_class_precision = precision_score(test_labels, test_preds, average=None, zero_division=0)
test_cm = confusion_matrix(test_labels, test_preds)
test_pred_dist = np.bincount(test_preds, minlength=n_classes)

print(f"\n Test Performance (Balanced Sampling):")
print(f"   Accuracy:          {test_acc:.4f}")
print(f"   Precision (macro): {test_precision:.4f}")
print(f"   Recall (macro):    {test_recall:.4f}")
print(f"   F1 (macro):        {test_f1_macro:.4f}")

print(f"\n Per-Class Performance:")
for i in range(n_classes):
    actual = np.sum(test_labels == i)
    print(f"   Class {i}: P={test_per_class_precision[i]:.4f}, "
          f"R={test_per_class_recall[i]:.4f}, F1={test_per_class_f1[i]:.4f}, "
          f"Pred={test_pred_dist[i]}/{actual}")

print(f"\n Prediction Distribution:")
for i, count in enumerate(test_pred_dist):
    print(f"   Class {i}: {count} predictions")

all_predicted = all(r > 0 for r in test_per_class_recall)

# COMPARE WITH BASELINE
print("COMPARISON WITH BASELINE")

baseline_path = Path(PHASE3_DIR) / "results_diagnostic_plain" / "results.json"
if baseline_path.exists():
    with open(baseline_path) as f:
        baseline_results = json.load(f)

    baseline_f1 = baseline_results['test_f1_macro']
    baseline_recalls = baseline_results['test_per_class_recall']

    improvement_f1 = test_f1_macro - baseline_f1
    improvement_pct = (improvement_f1 / baseline_f1) * 100 if baseline_f1 > 0 else 0

    print(f"\n Macro F1 Comparison:")
    print(f"   Baseline (weighted loss):   {baseline_f1:.4f}")
    print(f"   Balanced sampling:          {test_f1_macro:.4f}")
    print(f"   Absolute improvement:       {improvement_f1:+.4f}")
    print(f"   Relative improvement:       {improvement_pct:+.1f}%")

    print(f"\n Per-Class Recall Comparison:")
    for i in range(n_classes):
        base_r = baseline_recalls[i]
        balanced_r = test_per_class_recall[i]
        impr = balanced_r - base_r
        print(f"   Class {i}: {base_r:.4f} → {balanced_r:.4f} ({impr:+.4f})")

    # Check if balanced sampling helped
    if all_predicted and not all(r > 0 for r in baseline_recalls):
        print(f"\n SUCCESS: Balanced sampling enabled all classes to be predicted!")
    elif improvement_f1 > 0.05:
        print(f"\n SIGNIFICANT IMPROVEMENT with balanced sampling")
    elif improvement_f1 > 0:
        print(f"\n Modest improvement with balanced sampling")
    else:
        print(f"\n No improvement over baseline")
else:
    print("\n  Baseline results not found")
    improvement_f1 = None

# Save results
balanced_results = {
    'experiment': 'Class-Balanced Sampling',
    'timestamp': datetime.now().isoformat(),
    'configuration': BALANCED_CONFIG,

    'test_performance': {
        'accuracy': float(test_acc),
        'precision_macro': float(test_precision),
        'recall_macro': float(test_recall),
        'f1_macro': float(test_f1_macro),
        'f1_weighted': float(test_f1_weighted)
    },

    'per_class_metrics': {
        f'class_{i}': {
            'precision': float(test_per_class_precision[i]),
            'recall': float(test_per_class_recall[i]),
            'f1': float(test_per_class_f1[i])
        }
        for i in range(n_classes)
    },

    'confusion_matrix': test_cm.tolist(),
    'prediction_distribution': test_pred_dist.tolist(),
    'all_classes_predicted': bool(all_predicted),

    'training': {
        'total_epochs': len(history['train_loss']),
        'best_val_f1': float(best_f1),
        'training_time_minutes': float(elapsed / 60)
    },

    'comparison_to_baseline': {
        'baseline_f1': float(baseline_f1) if baseline_path.exists() else None,
        'balanced_f1': float(test_f1_macro),
        'absolute_improvement': float(improvement_f1) if improvement_f1 is not None else None,
        'relative_improvement_percent': float(improvement_pct) if improvement_f1 is not None else None
    }
}

with open(balanced_dir / "balanced_sampling_results.json", 'w') as f:
    json.dump(balanced_results, f, indent=2)

with open(balanced_dir / "balanced_sampling_history.json", 'w') as f:
    json.dump(history, f, indent=2)

torch.save(model.state_dict(), balanced_dir / "balanced_model.pt")

print(f"\n Results saved to: {balanced_dir}")

print(f"\n Final Results:")
print(f"   Macro F1: {test_f1_macro:.4f}")
print(f"   All classes predicted: {all_predicted}")
if improvement_f1 is not None:
    print(f"   Improvement: {improvement_f1:+.4f} ({improvement_pct:+.1f}%)")

print("\n Balanced sampling intervention complete!")
