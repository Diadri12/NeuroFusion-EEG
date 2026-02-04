"""
SUPERVISED CONTRASTIVE PRETRAINING
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from tqdm import tqdm
import joblib
from datetime import datetime

print("SUPERVISED CONTRASTIVE PRETRAINING")
print("\n Representation-level refinement")
print(" Two-stage training: SupCon pretraining → Supervised fine-tuning\n")

# CONFIGURATION
CONTRASTIVE_CONFIG = {
    # Architecture (same as baseline for fair comparison)
    'branch_a_depth': 'medium',
    'use_residual': True,
    'use_bilstm': False,
    'normalization': 'group',
    'num_groups': 8,

    # Contrastive pretraining parameters
    'pretrain_epochs': 60,
    'pretrain_batch_size': 128,  # Large for more negative pairs
    'pretrain_lr': 1e-4,
    'temperature': 0.07,  # Standard for SupCon
    'projection_dim': 128,

    # Data augmentation
    'use_augmentation': True,
    'noise_std': 0.02,
    'time_warp_strength': 0.1,
    'scale_range': (0.9, 1.1),
    'feature_mask_ratio': 0.1,

    # Fine-tuning parameters
    'finetune_epochs': 80,
    'finetune_batch_size': 64,
    'finetune_lr': 5e-5,
    'use_balanced_sampler': True,
    'label_smoothing': 0.1,

    # Training
    'patience': 20,
    'gradient_clip': 1.0,
    'weight_decay': 1e-4,

    'force_cpu': False
}

print(" Configuration:")
print(f"   Pretrain epochs: {CONTRASTIVE_CONFIG['pretrain_epochs']}")
print(f"   Pretrain batch: {CONTRASTIVE_CONFIG['pretrain_batch_size']} (large for negatives)")
print(f"   Temperature: {CONTRASTIVE_CONFIG['temperature']}")
print(f"   Projection dim: {CONTRASTIVE_CONFIG['projection_dim']}")
print(f"   Fine-tune with balanced sampling: {CONTRASTIVE_CONFIG['use_balanced_sampler']}")

# Paths
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
PHASE3_DIR = f"{BASE_DIR}/outputs/dual_branch_training"
PHASE4_DIR = f"{BASE_DIR}/outputs/error_diagnosis"
contrastive_dir = Path(PHASE4_DIR) / "contrastive_pretraining_results"
contrastive_dir.mkdir(exist_ok=True, parents=True)

device = torch.device('cuda' if torch.cuda.is_available() and not CONTRASTIVE_CONFIG['force_cpu'] else 'cpu')
print(f"\nDevice: {device}")

if device.type == 'cpu':
    print(" WARNING: Training on CPU will be slower")
    print("   Estimated time: 3-4 hours total")

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
print(f"   Classes: {n_classes}")

# Same split as all previous experiments
X_sig_train, X_sig_tmp, X_feat_train, X_feat_tmp, y_train, y_tmp = train_test_split(
    X_windowed, X_features, y_windowed, test_size=0.3, stratify=y_windowed, random_state=42
)
X_sig_val, X_sig_test, X_feat_val, X_feat_test, y_val, y_test = train_test_split(
    X_sig_tmp, X_feat_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
)

print(f"\n Split (same as Phase 3 for fair comparison):")
print(f"   Train: {len(X_sig_train):,}")
print(f"   Val: {len(X_sig_val):,}")
print(f"   Test: {len(X_sig_test):,}")

# DATA AUGMENTATION FUNCTIONS
print("DATA AUGMENTATION")

# Data augmentation for EEG signals and features
class EEGAugmenter:
    def __init__(self, config):
        self.noise_std = config['noise_std']
        self.time_warp_strength = config['time_warp_strength']
        self.scale_range = config['scale_range']
        self.feature_mask_ratio = config['feature_mask_ratio']

    # Augment EEG signal with multiple transformations
    def augment_signal(self, signal):
        signal = signal.copy().astype(np.float32)

        # Add Gaussian noise (50% probability)
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, self.noise_std, signal.shape)
            signal = signal + noise

        # Amplitude scaling (50% probability)
        if np.random.rand() < 0.5:
            scale = np.random.uniform(*self.scale_range)
            signal = signal * scale

        # Time warping (30% probability)
        if np.random.rand() < 0.3:
            # Simple shift-based warping
            shift = int(len(signal) * self.time_warp_strength * (np.random.rand() - 0.5))
            signal = np.roll(signal, shift)

        return signal

    # Augment handcrafted features
    def augment_features(self, features):
        features = features.copy().astype(np.float32)

        # Small Gaussian noise (50% probability)
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.01, features.shape)
            features = features + noise

        # Random feature masking (30% probability)
        if np.random.rand() < 0.3:
            mask_count = int(len(features) * self.feature_mask_ratio)
            if mask_count > 0:
                mask_idx = np.random.choice(len(features), mask_count, replace=False)
                features[mask_idx] = 0

        return features

augmenter = EEGAugmenter(CONTRASTIVE_CONFIG)

print(f"\n Augmentation configured:")
print(f"   Signal augmentation:")
print(f"     - Gaussian noise (std={CONTRASTIVE_CONFIG['noise_std']})")
print(f"     - Amplitude scaling ({CONTRASTIVE_CONFIG['scale_range']})")
print(f"     - Time warping (strength={CONTRASTIVE_CONFIG['time_warp_strength']})")
print(f"   Feature augmentation:")
print(f"     - Small perturbations")
print(f"     - Random masking ({CONTRASTIVE_CONFIG['feature_mask_ratio']*100:.0f}%)")

# SUPERVISED CONTRASTIVE LOSS
print("SUPERVISED CONTRASTIVE LOSS (SupCon)")

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss

    Paper: Khosla et al., NeurIPS 2020
    "Supervised Contrastive Learning"

    Formula:
    L_i = -1/|P(i)| * sum_{p in P(i)} log[ exp(z_i · z_p / τ) / sum_{a in A(i)} exp(z_i · z_a / τ) ]

    Where:
    - z_i, z_p, z_a: L2-normalized embeddings
    - P(i): positive pairs (same class as i)
    - A(i): all samples except i
    - τ: temperature hyperparameter

    Effect:
    - Pulls embeddings of same class together
    - Pushes embeddings of different classes apart
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        # Normalize features (important for contrastive learning)
        features = F.normalize(features, dim=1)

        # Create positive pair mask (same class)
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(device)

        # Mask out diagonal (self-contrast)
        logits_mask = torch.scatter(
            torch.ones_like(mask_positive),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask_positive = mask_positive * logits_mask

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask_positive * log_prob).sum(1) / (mask_positive.sum(1) + 1e-10)

        # Loss (negative of mean log probability)
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss

supcon_criterion = SupConLoss(temperature=CONTRASTIVE_CONFIG['temperature'])

print(f"\n SupCon Loss defined:")
print(f"   Temperature: {CONTRASTIVE_CONFIG['temperature']}")
print(f"   Positive pairs: Same-class samples")
print(f"   Negative pairs: Different-class samples")
print(f"   Effect: Maximizes inter-class distance, minimizes intra-class distance")

# MODEL ARCHITECTURE
print("MODEL ARCHITECTURE")

# Get normalization layer
def get_norm_layer(norm_type, num_channels, num_groups=8):
    if norm_type == 'batch':
        return nn.BatchNorm1d(num_channels)
    elif norm_type == 'group':
        return nn.GroupNorm(min(num_groups, num_channels), num_channels)
    else:
        return nn.Identity()

# Branch A: CNN for signal embeddings
class BranchA_CNN(nn.Module):
    def __init__(self, signal_length, embedding_dim=64, norm_type='group', num_groups=8):
        super().__init__()

        # 4 Convolutional blocks
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.norm1 = get_norm_layer(norm_type, 32, num_groups)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.norm2 = get_norm_layer(norm_type, 64, num_groups)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.norm3 = get_norm_layer(norm_type, 128, num_groups)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.norm4 = get_norm_layer(norm_type, 256, num_groups)

        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.embedding = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Conv blocks
        x = self.dropout(self.pool(self.relu(self.norm1(self.conv1(x)))))
        x = self.dropout(self.pool(self.relu(self.norm2(self.conv2(x)))))
        x = self.dropout(self.pool(self.relu(self.norm3(self.conv3(x)))))
        x = self.dropout(self.pool(self.relu(self.norm4(self.conv4(x)))))

        # Global pooling
        x = self.gap(x).squeeze(-1)

        # Embedding
        x = self.embedding(x)

        return x

# Branch B: MLP for feature embeddings
class BranchB_Features(nn.Module):
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

#  Projection head for contrastive learning
class ProjectionHead(nn.Module):
    """
    Maps fused embeddings to projection space where contrastive loss is computed
    This is discarded after pretraining
    """
    def __init__(self, input_dim, projection_dim=128):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        # Project and L2 normalize
        projected = self.projection(x)
        normalized = F.normalize(projected, dim=1)
        return normalized

# Dual-branch model with projection head
class DualBranchContrastive(nn.Module):
    """
    Used for:
    1. Contrastive pretraining (with projection head)
    2. Classification fine-tuning (without projection head)
    """
    def __init__(self, signal_length, n_features, n_classes, config):
        super().__init__()

        # Encoder branches
        self.branch_a = BranchA_CNN(
            signal_length,
            embedding_dim=64,
            norm_type=config['normalization'],
            num_groups=config['num_groups']
        )

        self.branch_b = BranchB_Features(
            n_features,
            embedding_dim=32
        )

        # Projection head (for contrastive pretraining only)
        self.projection_head = ProjectionHead(
            input_dim=96,  # 64 + 32
            projection_dim=config['projection_dim']
        )

        # Classifier (for fine-tuning)
        self.classifier = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_classes)
        )

    def forward(self, signals, features, mode='classify'):
        # Get embeddings from both branches
        emb_a = self.branch_a(signals)
        emb_b = self.branch_b(features)

        # Fuse embeddings
        fused = torch.cat([emb_a, emb_b], dim=1)

        if mode == 'contrastive':
            # For contrastive learning
            return self.projection_head(fused)
        else:
            # For classification
            return self.classifier(fused)

# Create model
model = DualBranchContrastive(
    signal_length,
    n_features,
    n_classes,
    CONTRASTIVE_CONFIG
).to(device)

total_params = sum(p.numel() for p in model.parameters())
projection_params = sum(p.numel() for p in model.projection_head.parameters())
encoder_params = total_params - projection_params

print(f"\n Model created:")
print(f"   Total parameters: {total_params:,}")
print(f"   Encoder (branches A+B): {encoder_params:,}")
print(f"   Projection head: {projection_params:,}")
print(f"   Projection dimension: {CONTRASTIVE_CONFIG['projection_dim']}")

# DATASET FOR CONTRASTIVE PRETRAINING
print("CREATING DATASETS")

# Dataset with augmentation for contrastive learning
class ContrastiveDataset(Dataset):
    def __init__(self, signals, features, labels, feature_scaler, augmenter):
        self.signals = signals
        self.features = features
        self.labels = labels
        self.scaler = feature_scaler
        self.augmenter = augmenter

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        features = self.features[idx]
        label = self.labels[idx]

        # Apply augmentation
        signal_aug = self.augmenter.augment_signal(signal)
        features_aug = self.augmenter.augment_features(features)

        # Convert to tensors
        signal_tensor = torch.FloatTensor(signal_aug).unsqueeze(0)
        features_tensor = torch.FloatTensor(
            self.scaler.transform(features_aug.reshape(1, -1))[0]
        )
        label_tensor = torch.LongTensor([label])[0]

        return signal_tensor, features_tensor, label_tensor

# Standard dataset without augmentation for fine-tuning
class StandardDataset(Dataset):
    def __init__(self, signals, features, labels, feature_scaler):
        self.signals = torch.FloatTensor(signals).unsqueeze(1)
        self.features = torch.FloatTensor(feature_scaler.transform(features))
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.features[idx], self.labels[idx]

# Create datasets
pretrain_dataset = ContrastiveDataset(
    X_sig_train, X_feat_train, y_train, scaler, augmenter
)

pretrain_loader = DataLoader(
    pretrain_dataset,
    batch_size=CONTRASTIVE_CONFIG['pretrain_batch_size'],
    shuffle=True,
    num_workers=0,
    drop_last=True  # Ensure consistent batch size
)

print(f"\n Contrastive dataset created:")
print(f"   Samples: {len(pretrain_dataset):,}")
print(f"   Batch size: {CONTRASTIVE_CONFIG['pretrain_batch_size']}")
print(f"   Batches per epoch: {len(pretrain_loader)}")

# CONTRASTIVE PRETRAINING
print("CONTRASTIVE PRETRAINING")
print(f"\n  This stage learns better embeddings")
print(f" No classification loss - pure representation learning")

# Optimizer and scheduler
pretrain_optimizer = optim.AdamW(
    model.parameters(),
    lr=CONTRASTIVE_CONFIG['pretrain_lr'],
    weight_decay=CONTRASTIVE_CONFIG['weight_decay']
)

pretrain_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    pretrain_optimizer,
    T_max=CONTRASTIVE_CONFIG['pretrain_epochs']
)

# Training history
pretrain_history = {
    'contrastive_loss': [],
    'learning_rate': []
}

print(f"\n{'Epoch':<8} {'Loss':<12} {'LR':<12}")

pretrain_start = time.time()

for epoch in range(CONTRASTIVE_CONFIG['pretrain_epochs']):
    model.train()
    epoch_loss = 0

    for signals, features, labels in tqdm(pretrain_loader, desc=f"Pretrain {epoch}", leave=False):
        signals = signals.to(device)
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass through projection head
        projections = model(signals, features, mode='contrastive')

        # Compute contrastive loss
        loss = supcon_criterion(projections, labels)

        # Backward pass
        pretrain_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONTRASTIVE_CONFIG['gradient_clip'])
        pretrain_optimizer.step()

        epoch_loss += loss.item()

    pretrain_scheduler.step()

    # Record metrics
    avg_loss = epoch_loss / len(pretrain_loader)
    current_lr = pretrain_optimizer.param_groups[0]['lr']

    pretrain_history['contrastive_loss'].append(avg_loss)
    pretrain_history['learning_rate'].append(current_lr)

    # Print progress
    if epoch % 10 == 0 or epoch == CONTRASTIVE_CONFIG['pretrain_epochs'] - 1:
        print(f"{epoch:<8} {avg_loss:<12.4f} {current_lr:<12.6f}")

pretrain_elapsed = time.time() - pretrain_start

print(f"\n Contrastive pretraining completed")
print(f"   Time: {pretrain_elapsed/60:.1f} minutes")
print(f"   Final contrastive loss: {pretrain_history['contrastive_loss'][-1]:.4f}")

# Save pretrained model
torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONTRASTIVE_CONFIG,
    'history': pretrain_history
}, contrastive_dir / "pretrained_checkpoint.pt")

print(f" Pretrained model saved")

# SUPERVISED FINE-TUNING WITH BALANCED SAMPLING
print("SUPERVISED FINE-TUNING")

print(f"\n Using pretrained encoder")
print(f" Balanced sampling + Classification loss")

# Create standard datasets (no augmentation for fine-tuning)
finetune_train = StandardDataset(X_sig_train, X_feat_train, y_train, scaler)
finetune_val = StandardDataset(X_sig_val, X_feat_val, y_val, scaler)
finetune_test = StandardDataset(X_sig_test, X_feat_test, y_test, scaler)

# Create balanced sampler
class_counts = np.bincount(y_train)
class_weights_sampler = 1.0 / class_counts
sample_weights = class_weights_sampler[y_train]

print(f"\n Class distribution in training:")
for i, count in enumerate(class_counts):
    print(f"   Class {i}: {count:,} samples ({100*count/len(y_train):.1f}%)")

balanced_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Create dataloaders
finetune_train_loader = DataLoader(
    finetune_train,
    batch_size=CONTRASTIVE_CONFIG['finetune_batch_size'],
    sampler=balanced_sampler,
    num_workers=0
)

finetune_val_loader = DataLoader(
    finetune_val,
    batch_size=CONTRASTIVE_CONFIG['finetune_batch_size'],
    shuffle=False,
    num_workers=0
)

finetune_test_loader = DataLoader(
    finetune_test,
    batch_size=CONTRASTIVE_CONFIG['finetune_batch_size'],
    shuffle=False,
    num_workers=0
)

print(f"\n Balanced sampler created")
print(f"   Will oversample minority classes")
print(f"   Will undersample majority class")

# Classification loss with label smoothing
finetune_criterion = nn.CrossEntropyLoss(
    label_smoothing=CONTRASTIVE_CONFIG['label_smoothing']
)

# Optimizer with lower learning rate
finetune_optimizer = optim.AdamW(
    model.parameters(),
    lr=CONTRASTIVE_CONFIG['finetune_lr'],
    weight_decay=CONTRASTIVE_CONFIG['weight_decay']
)

finetune_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    finetune_optimizer,
    T_max=CONTRASTIVE_CONFIG['finetune_epochs']
)

print(f"\n Fine-tuning setup:")
print(f"   Loss: CrossEntropy + label smoothing ({CONTRASTIVE_CONFIG['label_smoothing']})")
print(f"   Learning rate: {CONTRASTIVE_CONFIG['finetune_lr']} (lower than pretraining)")
print(f"   Balanced sampling: {CONTRASTIVE_CONFIG['use_balanced_sampler']}")

# Training functions
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for signals, features, labels in tqdm(loader, desc="Training", leave=False):
        signals = signals.to(device)
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(signals, features, mode='classify')
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONTRASTIVE_CONFIG['gradient_clip'])
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for signals, features, labels in tqdm(loader, desc="Validating", leave=False):
            signals = signals.to(device)
            features = features.to(device)
            labels = labels.to(device)

            logits = model(signals, features, mode='classify')
            loss = criterion(logits, labels)

            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), np.array(all_preds), np.array(all_labels)

# Fine-tuning loop
finetune_history = {
    'train_loss': [], 'train_f1': [],
    'val_loss': [], 'val_f1': [], 'val_recalls': [],
    'learning_rate': []
}

best_f1 = 0.0
patience_counter = 0
best_state = None

print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val F1':<10} {'Recalls':<30} {'Status'}")

finetune_start = time.time()

for epoch in range(CONTRASTIVE_CONFIG['finetune_epochs']):
    # Train
    train_loss, train_preds, train_labels = train_epoch(
        model, finetune_train_loader, finetune_optimizer, finetune_criterion, device
    )

    # Validate
    val_loss, val_preds, val_labels = validate(
        model, finetune_val_loader, finetune_criterion, device
    )

    finetune_scheduler.step()

    # Compute metrics
    train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
    val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
    val_recalls = recall_score(val_labels, val_preds, average=None, zero_division=0)

    # Save history
    finetune_history['train_loss'].append(train_loss)
    finetune_history['train_f1'].append(train_f1)
    finetune_history['val_loss'].append(val_loss)
    finetune_history['val_f1'].append(val_f1)
    finetune_history['val_recalls'].append(val_recalls.tolist())
    finetune_history['learning_rate'].append(finetune_optimizer.param_groups[0]['lr'])

    # Track best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = model.state_dict().copy()
        patience_counter = 0
        status = "BEST"
    else:
        patience_counter += 1
        status = f"({patience_counter}/{CONTRASTIVE_CONFIG['patience']})"

    # Format output
    recall_str = f"[{val_recalls[0]:.3f}, {val_recalls[1]:.3f}, {val_recalls[2]:.3f}]"

    # Print progress
    if epoch % 5 == 0 or epoch == CONTRASTIVE_CONFIG['finetune_epochs'] - 1:
        print(f"{epoch:<8} {train_loss:<12.4f} {val_loss:<12.4f} {val_f1:<10.4f} {recall_str:<30} {status}")

    # Early stopping
    if patience_counter >= CONTRASTIVE_CONFIG['patience']:
        print(f"\n Early stopping at epoch {epoch}")
        break

# Load best model
if best_state:
    model.load_state_dict(best_state)

finetune_elapsed = time.time() - finetune_start
total_elapsed = pretrain_elapsed + finetune_elapsed

print(f"\n Fine-tuning completed")
print(f"   Time: {finetune_elapsed/60:.1f} minutes")
print(f"   Best validation F1: {best_f1:.4f}")
print(f"\n Total training time: {total_elapsed/60:.1f} minutes")

# FINAL EVALUATION
print("FINAL EVALUATION")

# Evaluate on test set
_, test_preds, test_labels = validate(model, finetune_test_loader, finetune_criterion, device)

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
test_pred_dist = np.bincount(test_preds, minlength=n_classes)

print(f"\n Test Performance:")
print(f"   Accuracy:          {test_acc:.4f}")
print(f"   Precision (macro): {test_precision:.4f}")
print(f"   Recall (macro):    {test_recall:.4f}")
print(f"   F1 (macro):        {test_f1_macro:.4f}")
print(f"   F1 (weighted):     {test_f1_weighted:.4f}")

print(f"\n Per-Class Performance:")
for i in range(n_classes):
    actual = np.sum(test_labels == i)
    status = "YES" if test_per_class_recall[i] > 0 else "NO"
    print(f"   {status} Class {i}: P={test_per_class_precision[i]:.4f}, "
          f"R={test_per_class_recall[i]:.4f}, F1={test_per_class_f1[i]:.4f}, "
          f"Pred={test_pred_dist[i]}/{actual}")

print(f"\n Confusion Matrix:")
print(test_cm)

all_predicted = all(r > 0 for r in test_per_class_recall)

# COMPARISON WITH PREVIOUS EXPERIMENTS
print("COMPARISON WITH PREVIOUS EXPERIMENTS")

comparisons = {}

# Load baseline
baseline_path = Path(PHASE3_DIR) / "phase3_diagnostic_BASELINE" / "baseline_results.json"
if baseline_path.exists():
    with open(baseline_path) as f:
        baseline = json.load(f)
    comparisons['Baseline (CNN)'] = baseline['test_performance']['f1_macro']

# Load balanced sampling
balanced_path = Path(PHASE4_DIR) / "balanced_sampling_results" / "balanced_sampling_results.json"
if balanced_path.exists():
    with open(balanced_path) as f:
        balanced = json.load(f)
    comparisons['Balanced Sampling'] = balanced['test_performance']['f1_macro']

# Current
comparisons['Contrastive (This)'] = test_f1_macro

print(f"\n Macro F1 Comparison:")
print(f"   {'Experiment':<25} {'Macro F1':<10} {'Improvement'}")

baseline_f1 = comparisons.get('Baseline (CNN)', None)
for name, f1 in comparisons.items():
    if baseline_f1 and name != 'Baseline (CNN)':
        improvement = f1 - baseline_f1
        improvement_pct = (improvement / baseline_f1) * 100 if baseline_f1 > 0 else 0
        print(f"   {name:<25} {f1:<10.4f} {improvement:+.4f} ({improvement_pct:+.1f}%)")
    else:
        print(f"   {name:<25} {f1:<10.4f} {'Baseline' if name == 'Baseline (CNN)' else ''}")

if baseline_f1:
    final_improvement = test_f1_macro - baseline_f1
    final_improvement_pct = (final_improvement / baseline_f1) * 100 if baseline_f1 > 0 else 0

    print(f"\n Final improvement over baseline:")
    print(f"   Absolute: {final_improvement:+.4f}")
    print(f"   Relative: {final_improvement_pct:+.1f}%")

    if test_f1_macro > 0.30:
        print(f"\n EXCELLENT: Macro F1 > 0.30 achieved!")
        print(f"   This represents strong performance for imbalanced EEG data")
    elif test_f1_macro > 0.20:
        print(f"\n STRONG: Significant improvement demonstrated")
        print(f"   Contrastive learning effectively improved separability")
    elif test_f1_macro > 0.15:
        print(f"\n GOOD: Measurable improvement over baseline")
        print(f"   Validates the contrastive pretraining approach")
    elif final_improvement > 0:
        print(f"\n Modest but positive improvement")
        print(f"   Demonstrates value of representation learning")

# SAVE RESULTS
print("SAVING RESULTS")

# Comprehensive results
results = {
    'experiment': 'Supervised Contrastive Pretraining',
    'timestamp': datetime.now().isoformat(),
    'configuration': CONTRASTIVE_CONFIG,

    'stage1_pretraining': {
        'epochs': len(pretrain_history['contrastive_loss']),
        'loss_type': 'SupCon',
        'temperature': CONTRASTIVE_CONFIG['temperature'],
        'final_loss': float(pretrain_history['contrastive_loss'][-1]),
        'time_minutes': float(pretrain_elapsed / 60)
    },

    'stage2_finetuning': {
        'epochs': len(finetune_history['train_loss']),
        'best_val_f1': float(best_f1),
        'balanced_sampling': CONTRASTIVE_CONFIG['use_balanced_sampler'],
        'time_minutes': float(finetune_elapsed / 60)
    },

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

    'comparisons': comparisons,

    'total_time_minutes': float(total_elapsed / 60),
    'improvement_over_baseline': {
        'absolute': float(final_improvement) if baseline_f1 else None,
        'relative_percent': float(final_improvement_pct) if baseline_f1 else None
    }
}

# Save everything
with open(contrastive_dir / "contrastive_results.json", 'w') as f:
    json.dump(results, f, indent=2)

with open(contrastive_dir / "pretrain_history.json", 'w') as f:
    json.dump(pretrain_history, f, indent=2)

with open(contrastive_dir / "finetune_history.json", 'w') as f:
    json.dump(finetune_history, f, indent=2)

# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONTRASTIVE_CONFIG,
    'test_performance': results['test_performance']
}, contrastive_dir / "final_model.pt")

print(f"\n All results saved to: {contrastive_dir}")
print(f"   - contrastive_results.json (complete metrics)")
print(f"   - pretrain_history.json (pretraining curves)")
print(f"   - finetune_history.json (fine-tuning curves)")
print(f"   - final_model.pt (trained model)")

# SUMMARY
print("CONTRASTIVE PRETRAINING")
print(f"\n Final Results:")
print(f"   Macro F1: {test_f1_macro:.4f}")
print(f"   All classes predicted: {all_predicted}")
print(f"   Training time: {total_elapsed/60:.1f} minutes")
if baseline_f1:
    print(f"   Result: {final_improvement_pct:+.1f}% improvement over baseline")

print(f"\n Contrastive pretraining complete!")
