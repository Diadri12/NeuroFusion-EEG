"""
REPRESENTATION LEARNING WITH BiLSTM
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
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from tqdm import tqdm
import joblib

print("REPRESENTATION LEARNING WITH BiLSTM")
print("\n Adding temporal dependency modeling")

# CONFIGURATION
CONFIG = {
    # Architecture - KEY CHANGE
    'branch_a_depth': 'medium',
    'use_residual': True,
    'use_bilstm': True,  # ← ENABLED for representation learning

    # Normalization
    'normalization': 'group',
    'num_groups': 8,

    # Regularization
    'dropout_strategy': 'adaptive',
    'base_dropout': 0.3,
    'min_dropout': 0.1,

    # Loss function - Focal for better minority class learning
    'loss_function': 'focal',
    'focal_gamma': 2.0,      # Standard focusing
    'focal_alpha_scale': 1.5,  # Moderate scaling

    # Training - Adjusted for BiLSTM
    'max_epochs': 150,        # More epochs (BiLSTM slower to converge)
    'patience': 25,           # More patience
    'learning_rate': 5e-5,    # Lower LR for stability
    'batch_size': 32,         # Smaller batches (BiLSTM memory intensive)
    'gradient_clip': 0.5,     # Tighter clipping

    # Device
    'force_cpu': False
}

print(" Configuration:")
for key, value in CONFIG.items():
    if key in ['use_bilstm', 'loss_function', 'learning_rate', 'batch_size']:
        print(f"   {key}: {value} ← Key parameter")
    else:
        print(f"   {key}: {value}")

# Paths
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
PHASE3_DIR = f"{BASE_DIR}/outputs/dual_branch_training"
frozen_dirs = list(Path(PHASE3_DIR).glob("frozen_*"))
if not frozen_dirs:
    frozen_dirs = [Path(PHASE3_DIR) / "frozen"]
frozen_dir = frozen_dirs[0]

output_dir = Path(PHASE3_DIR) / "results_representation_learning_bilstm"
output_dir.mkdir(exist_ok=True, parents=True)

device = torch.device('cuda' if torch.cuda.is_available() and not CONFIG['force_cpu'] else 'cpu')
print(f"\n Device: {device}")
print(f" Output: {output_dir}")

if device.type == 'cpu':
    print("\n  WARNING: BiLSTM training on CPU will be SLOW")

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

# Split
X_sig_train, X_sig_tmp, X_feat_train, X_feat_tmp, y_train, y_tmp = train_test_split(
    X_windowed, X_features, y_windowed, test_size=0.3, stratify=y_windowed, random_state=42
)
X_sig_val, X_sig_test, X_feat_val, X_feat_test, y_val, y_test = train_test_split(
    X_sig_tmp, X_feat_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
)

print(f" Split: Train={len(X_sig_train):,}, Val={len(X_sig_val):,}, Test={len(X_sig_test):,}")

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

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

# MODEL WITH BiLSTM
print("BUILDING MODEL WITH BiLSTM")

def get_norm_layer(norm_type, num_channels, num_groups=8):
    if norm_type == 'batch':
        return nn.BatchNorm1d(num_channels)
    elif norm_type == 'group':
        return nn.GroupNorm(min(num_groups, num_channels), num_channels)
    else:
        return nn.Identity()

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_type='group', num_groups=8, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.norm1 = get_norm_layer(norm_type, out_channels, num_groups)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.norm2 = get_norm_layer(norm_type, out_channels, num_groups)

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                get_norm_layer(norm_type, out_channels, num_groups)
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += identity
        out = self.relu(out)
        return out

# Branch A with BiLSTM for temporal dependencies
class BranchA_WithBiLSTM(nn.Module):
    def __init__(self, signal_length, embedding_dim=64, norm_type='group', num_groups=8):
        super().__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(1, 32, 7, 2, 3)
        self.norm1 = get_norm_layer(norm_type, 32, num_groups)

        self.conv2 = nn.Conv1d(32, 64, 5, 1, 2)
        self.norm2 = get_norm_layer(norm_type, 64, num_groups)

        self.conv3 = nn.Conv1d(64, 128, 3, 1, 1)
        self.norm3 = get_norm_layer(norm_type, 128, num_groups)

        self.conv4 = nn.Conv1d(128, 256, 3, 1, 1)
        self.norm4 = get_norm_layer(norm_type, 256, num_groups)

        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

        # BiLSTM for temporal dependencies
        self.bilstm = nn.LSTM(256, 128, num_layers=2, bidirectional=True,
                             batch_first=True, dropout=0.2)

        # Global pooling and embedding
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, embedding_dim),  # 128*2 from bidirectional
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # CNN feature extraction
        x = self.dropout(self.pool(self.relu(self.norm1(self.conv1(x)))))
        x = self.dropout(self.pool(self.relu(self.norm2(self.conv2(x)))))
        x = self.dropout(self.pool(self.relu(self.norm3(self.conv3(x)))))
        x = self.dropout(self.pool(self.relu(self.norm4(self.conv4(x)))))

        # BiLSTM for temporal modeling
        x = x.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
        x, _ = self.bilstm(x)   # (B, L, 256)
        x = x.permute(0, 2, 1)  # (B, 256, L)

        # Global pooling and embedding
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)

        return x

class BranchB_FeatureEncoder(nn.Module):
    def __init__(self, n_features, embedding_dim=32):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class DualBranchWithBiLSTM(nn.Module):
    def __init__(self, signal_length, n_features, n_classes, norm_type='group', num_groups=8):
        super().__init__()

        self.branch_a = BranchA_WithBiLSTM(signal_length, 64, norm_type, num_groups)
        self.branch_b = BranchB_FeatureEncoder(n_features, 32)

        self.classifier = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, signals, features):
        emb_a = self.branch_a(signals)
        emb_b = self.branch_b(features)
        fused = torch.cat([emb_a, emb_b], dim=1)
        return self.classifier(fused)

model = DualBranchWithBiLSTM(signal_length, n_features, n_classes,
                             CONFIG['normalization'], CONFIG['num_groups']).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n Model created with BiLSTM:")
print(f"   Total parameters: {total_params:,}")
print(f"   BiLSTM layers: 2")
print(f"   BiLSTM hidden: 128 (256 bidirectional)")

# LOSS FUNCTION
print("LOSS FUNCTION")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights_scaled = class_weights ** CONFIG['focal_alpha_scale']

print(f"\n Class weights (focal alpha scale={CONFIG['focal_alpha_scale']}):")
for i, w in enumerate(class_weights_scaled):
    print(f"   Class {i}: {w:.4f}")

class_weights_tensor = torch.FloatTensor(class_weights_scaled).to(device)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()

criterion = FocalLoss(alpha=class_weights_tensor, gamma=CONFIG['focal_gamma'])

print(f"\n Loss: Focal Loss (gamma={CONFIG['focal_gamma']})")

# OPTIMIZER
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['max_epochs'])

print(f" Optimizer: AdamW (lr={CONFIG['learning_rate']})")
print(f" Scheduler: CosineAnnealingLR")

# TRAINING
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
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

print("TRAINING WITH BiLSTM")

print(f"\n  Expected behavior:")
print(f"   - First 10-20 epochs: Lower metrics (BiLSTM learning)")
print(f"   - After epoch 20: Gradual F1 improvement")
print(f"   - Slower per-epoch time due to BiLSTM")

history = {
    'train_loss': [], 'train_f1': [],
    'val_loss': [], 'val_f1': [], 'val_recalls': [],
    'learning_rate': []
}

best_f1 = 0.0
patience_counter = 0
best_state = None

print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val F1':<10} {'Recalls':<35} {'Status'}")

start_time = time.time()

for epoch in range(CONFIG['max_epochs']):
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
        status = "BEST"
    else:
        patience_counter += 1
        status = f"({patience_counter}/{CONFIG['patience']})"

    recall_str = f"[{val_recalls[0]:.3f}, {val_recalls[1]:.3f}, {val_recalls[2]:.3f}]"

    if epoch % 5 == 0 or epoch == CONFIG['max_epochs'] - 1:
        print(f"{epoch:<8} {train_loss:<12.4f} {val_loss:<12.4f} {val_f1:<10.4f} {recall_str:<35} {status}")

    if patience_counter >= CONFIG['patience']:
        print(f"\n Early stopping at epoch {epoch}")
        break

if best_state:
    model.load_state_dict(best_state)

elapsed = time.time() - start_time
print(f"\n Training completed in {elapsed/60:.1f} minutes")
print(f"   Best validation F1: {best_f1:.4f}")

# EVALUATION
print("EVALUATION")

test_loss, test_preds, test_labels = validate(model, test_loader, criterion, device)

test_acc = accuracy_score(test_labels, test_preds)
test_f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)
test_f1_weighted = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
test_recalls = recall_score(test_labels, test_preds, average=None, zero_division=0)
test_per_class_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)
test_cm = confusion_matrix(test_labels, test_preds)
test_pred_dist = np.bincount(test_preds, minlength=n_classes)

print(f"\n Test Performance:")
print(f"   Accuracy:      {test_acc:.4f}")
print(f"   F1 (weighted): {test_f1_weighted:.4f}")
print(f"   F1 (macro):    {test_f1_macro:.4f}")

print(f"\n Per-Class Metrics:")
for i in range(n_classes):
    actual = np.sum(test_labels == i)
    status = "✓" if test_recalls[i] > 0 else "✗"
    print(f"   {status} Class {i}: Recall={test_recalls[i]:.4f}, F1={test_per_class_f1[i]:.4f}, "
          f"Pred={test_pred_dist[i]}/{actual}")

print(f"\n Confusion Matrix:")
print(test_cm)

all_predicted = all(r > 0 for r in test_recalls)

if all_predicted and test_f1_macro > 0.4:
    print(" SUCCESS: BiLSTM improved temporal modeling")
elif all_predicted:
    print(" PARTIAL: All classes predicted but improvement limited")
else:
    print(" BiLSTM did not resolve class collapse")

# COMPARE WITH BASELINE
baseline_path = Path(PHASE3_DIR) / "LOCKED_BASELINE_RESULTS" / "results.json"
if baseline_path.exists():
    with open(baseline_path) as f:
        baseline_results = json.load(f)

    print("COMPARISON WITH BASELINE")

    baseline_f1 = baseline_results['test_f1_macro']
    improvement = test_f1_macro - baseline_f1
    improvement_pct = (improvement / baseline_f1) * 100

    print(f"\n Macro F1 Comparison:")
    print(f"   Baseline (CNN only): {baseline_f1:.4f}")
    print(f"   BiLSTM:              {test_f1_macro:.4f}")
    print(f"   Improvement:         {improvement:+.4f} ({improvement_pct:+.1f}%)")

    if improvement > 0.05:
        print(f"\n BiLSTM shows clear improvement!")
    elif improvement > 0:
        print(f"\n BiLSTM shows modest improvement")
    else:
        print(f"\n BiLSTM did not improve over baseline")

# Save results
results = {
    'approach': 'Representation Learning with BiLSTM',
    'config': CONFIG,
    'test_accuracy': float(test_acc),
    'test_f1_macro': float(test_f1_macro),
    'test_f1_weighted': float(test_f1_weighted),
    'test_per_class_recall': test_recalls.tolist(),
    'test_per_class_f1': test_per_class_f1.tolist(),
    'test_confusion_matrix': test_cm.tolist(),
    'all_classes_predicted': bool(all_predicted),
    'total_epochs': len(history['train_loss']),
    'training_time_minutes': float(elapsed / 60),
    'total_parameters': total_params
}

with open(output_dir / "results.json", 'w') as f:
    json.dump(results, f, indent=2)

with open(output_dir / "history.json", 'w') as f:
    json.dump(history, f, indent=2)

torch.save(model.state_dict(), output_dir / "model_bilstm.pt")

print(f"\n Results saved to: {output_dir}")

print("REPRESENTATION LEARNING COMPLETE")
