"""
COMPREHENSIVE EEG CLASSIFICATION TRAINING SCRIPT
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

# Configurations

# Select mode
MODE = 'diagnostic_plain'  # Options: 'diagnostic_plain', 'signal_only', 'feature_only', 'dual_branch', 'dual_branch_focal'

# Training configuration
CONFIG = {
    # Architecture
    'branch_a_depth': 'medium',      # 'shallow', 'medium', 'deep'
    'use_residual': True,            # Residual connections
    'use_bilstm': False,             # Add BiLSTM (slow)

    # Normalization
    'normalization': 'group',        # 'batch', 'group', 'layer'
    'num_groups': 8,

    # Regularization
    'dropout_strategy': 'adaptive',  # 'fixed', 'adaptive'
    'base_dropout': 0.3,
    'min_dropout': 0.1,

    # Loss function
    'loss_function': 'weighted_ce',  # 'weighted_ce', 'focal'
    'focal_gamma': 1.5,              # For focal loss
    'focal_alpha_scale': 1.5,        # For focal loss

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

# Output directory based on mode
output_dir = Path(PHASE3_DIR) / f"results_{MODE}"
output_dir.mkdir(exist_ok=True, parents=True)

print(f"COMPREHENSIVE EEG TRAINING - MODE: {MODE.upper()}")

device = torch.device('cuda' if torch.cuda.is_available() and not CONFIG['force_cpu'] else 'cpu')
print(f"\nDevice: {device}")
print(f"Mode: {MODE}")
print(f"Output: {output_dir}\n")

# Adjust normalization for CPU
if device.type == 'cpu' and CONFIG['normalization'] == 'batch':
    print(" Switching from BatchNorm to GroupNorm for CPU stability")
    CONFIG['normalization'] = 'group'

# Load Data
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

# Split data
X_sig_train, X_sig_tmp, X_feat_train, X_feat_tmp, y_train, y_tmp = train_test_split(
    X_windowed, X_features, y_windowed, test_size=0.3, stratify=y_windowed, random_state=42
)
X_sig_val, X_sig_test, X_feat_val, X_feat_test, y_val, y_test = train_test_split(
    X_sig_tmp, X_feat_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
)

print(f"\n Data split:")
print(f"   Train: {len(X_sig_train):,} | Val: {len(X_sig_val):,} | Test: {len(X_sig_test):,}")

# Dataset Classes
class DualInputDataset(Dataset):
    """Dataset for dual-branch models"""
    def __init__(self, signals, features, labels, feature_scaler):
        self.signals = torch.FloatTensor(signals).unsqueeze(1)
        self.features = torch.FloatTensor(feature_scaler.transform(features))
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.features[idx], self.labels[idx]

class SignalOnlyDataset(Dataset):
    """Dataset for signal-only models"""
    def __init__(self, signals, labels):
        self.signals = torch.FloatTensor(signals).unsqueeze(1)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

class FeatureOnlyDataset(Dataset):
    """Dataset for feature-only models"""
    def __init__(self, features, labels, feature_scaler):
        self.features = torch.FloatTensor(feature_scaler.transform(features))
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create datasets based on mode
if MODE == 'signal_only':
    train_dataset = SignalOnlyDataset(X_sig_train, y_train)
    val_dataset = SignalOnlyDataset(X_sig_val, y_val)
    test_dataset = SignalOnlyDataset(X_sig_test, y_test)
elif MODE == 'feature_only':
    train_dataset = FeatureOnlyDataset(X_feat_train, y_train, scaler)
    val_dataset = FeatureOnlyDataset(X_feat_val, y_val, scaler)
    test_dataset = FeatureOnlyDataset(X_feat_test, y_test, scaler)
else:  # dual_branch modes
    train_dataset = DualInputDataset(X_sig_train, X_feat_train, y_train, scaler)
    val_dataset = DualInputDataset(X_sig_val, X_feat_val, y_val, scaler)
    test_dataset = DualInputDataset(X_sig_test, X_feat_test, y_test, scaler)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

# Model Architectures
def get_norm_layer(norm_type, num_channels, num_groups=8):
    """Get normalization layer"""
    if norm_type == 'batch':
        return nn.BatchNorm1d(num_channels)
    elif norm_type == 'group':
        return nn.GroupNorm(min(num_groups, num_channels), num_channels)
    elif norm_type == 'layer':
        return nn.GroupNorm(1, num_channels)
    else:
        return nn.Identity()

class ResidualBlock1D(nn.Module):
    """Residual block for 1D convolutions"""
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

class BranchA_TemporalLearner(nn.Module):
    """Branch A: Temporal signal learner"""
    def __init__(self, signal_length, embedding_dim=64, depth='medium', use_residual=True,
                 use_bilstm=False, norm_type='group', num_groups=8, dropout_strategy='adaptive'):
        super().__init__()

        self.use_bilstm = use_bilstm

        # Dropout rates
        if dropout_strategy == 'adaptive':
            dropout_rates = [0.30, 0.25, 0.20, 0.15, 0.10]
        else:
            dropout_rates = [0.3] * 5

        # Architecture based on depth
        if depth == 'shallow':
            channels = [1, 32, 64, 128]
            kernels = [7, 5, 3]
        elif depth == 'medium':
            channels = [1, 32, 64, 128, 256]
            kernels = [7, 5, 3, 3]
        else:  # deep
            channels = [1, 32, 64, 128, 256, 512]
            kernels = [7, 5, 5, 3, 3]

        # Build conv layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        for i in range(len(kernels)):
            if use_residual and i > 0:
                layer = ResidualBlock1D(
                    channels[i], channels[i+1], kernels[i], stride=1,
                    norm_type=norm_type, num_groups=num_groups, dropout=dropout_rates[i]
                )
            else:
                layer = nn.Sequential(
                    nn.Conv1d(channels[i], channels[i+1], kernels[i],
                             stride=2 if i == 0 else 1, padding=kernels[i]//2),
                    get_norm_layer(norm_type, channels[i+1], num_groups),
                    nn.ReLU(),
                    nn.Dropout(dropout_rates[i])
                )

            self.conv_layers.append(layer)
            self.pool_layers.append(nn.MaxPool1d(2))

        # Optional BiLSTM
        if use_bilstm:
            self.bilstm = nn.LSTM(channels[-1], 128, bidirectional=True, batch_first=True)
            lstm_output_dim = 256
        else:
            self.bilstm = None
            lstm_output_dim = channels[-1]

        # Global pooling and embedding
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rates[-1])
        )

    def forward(self, x):
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x)
            x = pool(x)

        if self.bilstm is not None:
            x = x.permute(0, 2, 1)
            x, _ = self.bilstm(x)
            x = x.permute(0, 2, 1)

        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x

# Branch B: Feature encoder
class BranchB_FeatureEncoder(nn.Module):
    def __init__(self, n_features, embedding_dim=32, norm_type='group', dropout_strategy='adaptive'):
        super().__init__()

        if dropout_strategy == 'adaptive':
            dropout_rates = [0.25, 0.15]
        else:
            dropout_rates = [0.3, 0.2]

        if norm_type == 'batch':
            norm_layer1 = nn.BatchNorm1d(64)
        else:
            norm_layer1 = nn.LayerNorm(64)

        self.fc1 = nn.Sequential(
            nn.Linear(n_features, 64),
            norm_layer1,
            nn.ReLU(),
            nn.Dropout(dropout_rates[0])
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1])
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Complete dual-branch model
class DualBranchModel(nn.Module):
    def __init__(self, signal_length, n_features, n_classes, config):
        super().__init__()

        embedding_dim_a = 64
        embedding_dim_b = 32

        self.branch_a = BranchA_TemporalLearner(
            signal_length=signal_length,
            embedding_dim=embedding_dim_a,
            depth=config['branch_a_depth'],
            use_residual=config['use_residual'],
            use_bilstm=config['use_bilstm'],
            norm_type=config['normalization'],
            num_groups=config['num_groups'],
            dropout_strategy=config['dropout_strategy']
        )

        self.branch_b = BranchB_FeatureEncoder(
            n_features=n_features,
            embedding_dim=embedding_dim_b,
            norm_type=config['normalization'],
            dropout_strategy=config['dropout_strategy']
        )

        fusion_dim = embedding_dim_a + embedding_dim_b

        if config['dropout_strategy'] == 'adaptive':
            final_dropout = config['min_dropout']
        else:
            final_dropout = 0.4

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Dropout(final_dropout),
            nn.Linear(32, n_classes)
        )

    def forward(self, signals, features):
        emb_a = self.branch_a(signals)
        emb_b = self.branch_b(features)
        fused = torch.cat([emb_a, emb_b], dim=1)
        return self.classifier(fused)

# Signal-only CNN model
class SignalOnlyModel(nn.Module):
    def __init__(self, signal_length, n_classes, norm_type='group', num_groups=8):
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

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.pool(self.relu(self.norm1(self.conv1(x)))))
        x = self.dropout(self.pool(self.relu(self.norm2(self.conv2(x)))))
        x = self.dropout(self.pool(self.relu(self.norm3(self.conv3(x)))))
        x = self.dropout(self.pool(self.relu(self.norm4(self.conv4(x)))))
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)
# Feature-only MLP model
class FeatureOnlyModel(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        return self.model(x)

# CREATE MODEL BASED ON MODE
print("CREATING MODEL")

if MODE == 'signal_only':
    model = SignalOnlyModel(signal_length, n_classes, CONFIG['normalization'], CONFIG['num_groups']).to(device)
    print("\n Signal-only CNN created")

elif MODE == 'feature_only':
    model = FeatureOnlyModel(n_features, n_classes).to(device)
    print("\n Feature-only MLP created")

else:  # dual_branch modes
    model = DualBranchModel(signal_length, n_features, n_classes, CONFIG).to(device)
    print(f"\n Dual-branch model created")
    print(f"   Depth: {CONFIG['branch_a_depth']}")
    print(f"   Residual: {CONFIG['use_residual']}")
    print(f"   BiLSTM: {CONFIG['use_bilstm']}")

total_params = sum(p.numel() for p in model.parameters())
print(f"   Total parameters: {total_params:,}")

# LOSS FUNCTION
print("LOSS FUNCTION")

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

print(f"\n Base balanced weights:")
for i, w in enumerate(class_weights):
    print(f"   Class {i}: {w:.4f}")

# Apply scaling based on mode and loss
if MODE == 'diagnostic_plain':
    # No scaling for diagnostic mode
    class_weights_final = class_weights
    loss_name = "Plain Weighted CrossEntropy"

elif MODE == 'dual_branch_focal':
    # Aggressive for focal
    class_weights_final = class_weights ** CONFIG['focal_alpha_scale']
    loss_name = f"Focal Loss (gamma={CONFIG['focal_gamma']})"

else:
    # Moderate scaling for others
    if CONFIG['loss_function'] == 'focal':
        class_weights_final = class_weights ** CONFIG['focal_alpha_scale']
        loss_name = f"Focal Loss (gamma={CONFIG['focal_gamma']})"
    else:
        class_weights_final = class_weights ** 1.5
        loss_name = "Weighted CrossEntropy"

print(f"\n Final class weights:")
for i, w in enumerate(class_weights_final):
    print(f"   Class {i}: {w:.4f}")

class_weights_tensor = torch.FloatTensor(class_weights_final).to(device)

# Create loss function
class FocalLoss(nn.Module):
    """Focal Loss"""
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

if CONFIG['loss_function'] == 'focal' and MODE != 'diagnostic_plain':
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=CONFIG['focal_gamma'])
else:
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

print(f"\n Loss: {loss_name}")

# OPTIMIZER AND SCHEDULER
# Adjust learning rate for feature-only
lr = CONFIG['learning_rate'] * 10 if MODE == 'feature_only' else CONFIG['learning_rate']

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, CONFIG['max_epochs'])

print(f"\n Optimizer: AdamW (lr={lr})")
print(f" Scheduler: CosineAnnealingLR")

# TRAINING FUNCTIONS
def train_epoch(model, loader, optimizer, criterion, device, mode):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Training", leave=False):
        if mode == 'signal_only':
            signals, labels = batch
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
        elif mode == 'feature_only':
            features, labels = batch
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
        else:  # dual_branch
            signals, features, labels = batch
            signals, features, labels = signals.to(device), features.to(device), labels.to(device)
            outputs = model(signals, features)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)

def validate(model, loader, criterion, device, mode):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            if mode == 'signal_only':
                signals, labels = batch
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
            elif mode == 'feature_only':
                features, labels = batch
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
            else:  # dual_branch
                signals, features, labels = batch
                signals, features, labels = signals.to(device), features.to(device), labels.to(device)
                outputs = model(signals, features)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)

# TRAINING LOOP
print("TRAINING")

history = {
    'train_loss': [], 'train_f1': [],
    'val_loss': [], 'val_f1': [], 'val_recalls': [], 'val_pred_dist': [],
    'learning_rate': []
}

best_f1 = 0.0
patience_counter = 0
best_state = None

print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val F1':<10} {'Recalls':<35} {'Status'}")

start_time = time.time()

for epoch in range(CONFIG['max_epochs']):
    # Train
    train_loss, train_preds, train_labels = train_epoch(
        model, train_loader, optimizer, criterion, device, MODE
    )

    # Validate
    val_loss, val_preds, val_labels = validate(
        model, val_loader, criterion, device, MODE
    )

    scheduler.step()

    # Metrics
    train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
    val_recalls = recall_score(val_labels, val_preds, average=None, zero_division=0)
    val_pred_dist = np.bincount(val_preds, minlength=n_classes)

    # Save history
    history['train_loss'].append(train_loss)
    history['train_f1'].append(train_f1)
    history['val_loss'].append(val_loss)
    history['val_f1'].append(val_f1)
    history['val_recalls'].append(val_recalls.tolist())
    history['val_pred_dist'].append(val_pred_dist.tolist())
    history['learning_rate'].append(optimizer.param_groups[0]['lr'])

    # Track best
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = model.state_dict().copy()
        patience_counter = 0
        status = "BEST"
    else:
        patience_counter += 1
        status = f"({patience_counter}/{CONFIG['patience']})"

    # Format output
    recall_str = f"[{val_recalls[0]:.3f}, {val_recalls[1]:.3f}, {val_recalls[2]:.3f}]"
    zero_classes = [i for i, r in enumerate(val_recalls) if r == 0]
    if zero_classes:
        recall_str += f"   Zero: {zero_classes}"

    # Print
    if epoch % 5 == 0 or epoch == CONFIG['max_epochs'] - 1:
        print(f"{epoch:<8} {train_loss:<12.4f} {val_loss:<12.4f} {val_f1:<10.4f} {recall_str:<35} {status}")

    # Early stopping
    if patience_counter >= CONFIG['patience']:
        print(f"\n Early stopping at epoch {epoch}")
        break

    # Emergency stop
    if epoch > 30 and all(r == 0 for r in val_recalls[1:]):
        print(f"\n Emergency stop: Minority classes not learning")
        break

# Load best model
if best_state:
    model.load_state_dict(best_state)

elapsed = time.time() - start_time
print(f"\n Training completed in {elapsed/60:.1f} minutes")
print(f"   Best validation F1: {best_f1:.4f}")

# EVALUATION
print("TEST EVALUATION")

test_loss, test_preds, test_labels = validate(model, test_loader, criterion, device, MODE)

# Compute metrics
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
    status = "Yes" if test_recalls[i] > 0 else "No"
    print(f"   {status} Class {i}: Recall={test_recalls[i]:.4f}, F1={test_per_class_f1[i]:.4f}, "
          f"Pred={test_pred_dist[i]}/{actual}")

print(f"\n Confusion Matrix:")
print(test_cm)

# Diagnosis
all_predicted = all(r > 0 for r in test_recalls)
diverse_predictions = len([d for d in test_pred_dist if d > 100]) >= 2

print("DIAGNOSIS")

print(f"\n All classes predicted: {all_predicted}")
print(f" Diverse predictions: {diverse_predictions}")
print(f" Macro F1: {test_f1_macro:.4f}")

if all_predicted and test_f1_macro > 0.3:
    print(f"\n SUCCESS: Model learned multiple classes effectively")
elif all_predicted:
    print(f"\n PARTIAL: All classes predicted but performance could improve")
else:
    print(f"\n FAILURE: Some classes not learned")

# Mode-specific recommendations
if MODE == 'diagnostic_plain' and all_predicted:
    print(f"\n Recommendation:")
    print(f"   → Use weighted CrossEntropy for final training")
    print(f"   → Train full dual-branch model")

elif MODE == 'signal_only' and all_predicted:
    print(f"\n Recommendation:")
    print(f"   → Signals ARE informative (F1={test_f1_macro:.4f})")
    print(f"   → Dual-branch will add features for improvement")

elif MODE == 'feature_only' and all_predicted:
    print(f"\n Recommendation:")
    print(f"   → Features ARE informative (F1={test_f1_macro:.4f})")
    print(f"   → Dual-branch will add temporal patterns")

# SAVE RESULTS
print("SAVING RESULTS")

# Save model
torch.save(model.state_dict(), output_dir / "model.pt")
torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'mode': MODE,
    'history': history
}, output_dir / "complete_checkpoint.pt")

# Save metrics
results = {
    'mode': MODE,
    'config': CONFIG,
    'test_accuracy': float(test_acc),
    'test_f1_macro': float(test_f1_macro),
    'test_f1_weighted': float(test_f1_weighted),
    'test_per_class_recall': test_recalls.tolist(),
    'test_per_class_f1': test_per_class_f1.tolist(),
    'test_confusion_matrix': test_cm.tolist(),
    'test_prediction_distribution': test_pred_dist.tolist(),
    'all_classes_predicted': bool(all_predicted),
    'best_val_f1': float(best_f1),
    'total_epochs': len(history['train_loss']),
    'training_time_minutes': float(elapsed / 60),
    'total_parameters': total_params
}

with open(output_dir / "results.json", 'w') as f:
    json.dump(results, f, indent=2)

with open(output_dir / "history.json", 'w') as f:
    json.dump(history, f, indent=2)

# Classification report
report = classification_report(test_labels, test_preds, target_names=[f'Class {i}' for i in range(n_classes)])
with open(output_dir / "classification_report.txt", 'w') as f:
    f.write(report)

print(f"\n Model saved: {output_dir / 'model.pt'}")
print(f" Results saved: {output_dir / 'results.json'}")
print(f" History saved: {output_dir / 'history.json'}")

# VISUALIZATION
print("CREATING VISUALIZATIONS")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Training curves
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history['train_loss'], label='Train', alpha=0.7)
ax1.plot(history['val_loss'], label='Val', alpha=0.7)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history['train_f1'], label='Train', alpha=0.7)
ax2.plot(history['val_f1'], label='Val', alpha=0.7)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('F1 Score')
ax2.set_title('Macro F1 Score')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(history['learning_rate'])
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Learning Rate')
ax3.set_title('Learning Rate Schedule')
ax3.grid(True, alpha=0.3)

# Per-class recalls
ax4 = fig.add_subplot(gs[1, :])
val_recalls_array = np.array(history['val_recalls'])
for i in range(n_classes):
    ax4.plot(val_recalls_array[:, i], label=f'Class {i}', linewidth=2)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Recall')
ax4.set_title('Per-Class Recall During Training')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)

# Confusion matrices
ax5 = fig.add_subplot(gs[2, 0])
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=ax5)
ax5.set_xlabel('Predicted')
ax5.set_ylabel('True')
ax5.set_title('Confusion Matrix')

ax6 = fig.add_subplot(gs[2, 1])
cm_norm = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', ax=ax6, vmin=0, vmax=1)
ax6.set_xlabel('Predicted')
ax6.set_ylabel('True')
ax6.set_title('Normalized Confusion Matrix')

# Prediction distribution
ax7 = fig.add_subplot(gs[2, 2])
actual_dist = [np.sum(test_labels == i) for i in range(n_classes)]
x = np.arange(n_classes)
width = 0.35
ax7.bar(x - width/2, test_pred_dist, width, label='Predicted', alpha=0.8)
ax7.bar(x + width/2, actual_dist, width, label='Actual', alpha=0.8)
ax7.set_xlabel('Class')
ax7.set_ylabel('Count')
ax7.set_title('Prediction Distribution')
ax7.set_xticks(x)
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'{MODE.upper()} - Test F1: {test_f1_macro:.4f}', fontsize=16, fontweight='bold')
plt.savefig(output_dir / "results.png", dpi=300, bbox_inches='tight')
plt.close()

print(f" Visualization saved: {output_dir / 'results.png'}")

print(f"\n Results saved to: {output_dir}")
