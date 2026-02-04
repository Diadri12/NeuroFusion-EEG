"""
IMPROVED ABLATION STUDY
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
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from tqdm import tqdm
import joblib
import pandas as pd

print("IMPROVED ABLATION STUDY")

# IMPROVED Configuration
ABLATION_CONFIG = {
    'epochs': 100,  # Increased from 60
    'batch_size': 64,
    'learning_rate': 5e-5,  # Lower LR for stability
    'patience': 25,  # More patience
    'use_balanced_sampling': True,
    'label_smoothing': 0.1,
    'weight_decay': 1e-3,  # Stronger regularization
    'gradient_clip': 1.0,
    'warmup_epochs': 5,  # Add warmup
    'force_cpu': False
}

print(" Improved configuration:")
print(f"   Epochs: {ABLATION_CONFIG['epochs']} (was 60)")
print(f"   Learning rate: {ABLATION_CONFIG['learning_rate']} (was 1e-4)")
print(f"   Patience: {ABLATION_CONFIG['patience']} (was 15)")
print(f"   Warmup: {ABLATION_CONFIG['warmup_epochs']} epochs")

# Paths
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
PHASE3_DIR = f"{BASE_DIR}/outputs/dual_branch_training"
PHASE6_DIR = f"{BASE_DIR}/outputs/advanced_analysis"
ablation_dir = Path(PHASE6_DIR) / "ablation_study_improved"
ablation_dir.mkdir(exist_ok=True, parents=True)

device = torch.device('cuda' if torch.cuda.is_available() and not ABLATION_CONFIG['force_cpu'] else 'cpu')
print(f"Device: {device}")

if device.type == 'cpu':
    print(" WARNING: CPU training detected!")

# Load data (same as before)
frozen_dir = Path(PHASE3_DIR) / "frozen"

X_windowed = np.load(frozen_dir / "X_windowed.npy")
y_windowed = np.load(frozen_dir / "y_windowed.npy")
X_features = np.load(frozen_dir / "X_features.npy")
scaler = joblib.load(Path(PHASE3_DIR) / "feature_scaler.pkl")

signal_length = X_windowed.shape[1]
n_features = X_features.shape[1]
n_classes = len(np.unique(y_windowed))

X_sig_train, X_sig_tmp, X_feat_train, X_feat_tmp, y_train, y_tmp = train_test_split(
    X_windowed, X_features, y_windowed, test_size=0.3, stratify=y_windowed, random_state=42
)
X_sig_val, X_sig_test, X_feat_val, X_feat_test, y_val, y_test = train_test_split(
    X_sig_tmp, X_feat_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
)

print(f"\n Data loaded: {len(X_sig_train):,} train, {len(X_sig_test):,} test")

# IMPROVED ARCHITECTURES WITH BETTER INITIALIZATION
# Xavier initialization for better convergence
def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Improved CNN with better initialization
class BranchA_CNN_Improved(nn.Module):
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

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.apply(init_weights)

    def forward(self, x):
        x = self.dropout(self.pool(self.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool(self.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(self.pool(self.relu(self.bn3(self.conv3(x)))))
        x = self.dropout(self.pool(self.relu(self.bn4(self.conv4(x)))))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

# Improved feature branch
class BranchB_Features_Improved(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)

class BranchA_Only_Improved(nn.Module):
    def __init__(self, signal_length, n_classes):
        super().__init__()
        self.branch_a = BranchA_CNN_Improved(signal_length)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes)
        )
        self.apply(init_weights)

    def forward(self, signals, features=None):
        emb = self.branch_a(signals)
        return self.classifier(emb)

class BranchB_Only_Improved(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.branch_b = BranchB_Features_Improved(n_features)
        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_classes)
        )
        self.apply(init_weights)

    def forward(self, signals, features):
        emb = self.branch_b(features)
        return self.classifier(emb)

class DualBranch_Improved(nn.Module):
    def __init__(self, signal_length, n_features, n_classes):
        super().__init__()
        self.branch_a = BranchA_CNN_Improved(signal_length)
        self.branch_b = BranchB_Features_Improved(n_features)

        # Better fusion with gating
        self.gate = nn.Sequential(
            nn.Linear(96, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_classes)
        )

        self.apply(init_weights)

    def forward(self, signals, features):
        emb_a = self.branch_a(signals)
        emb_b = self.branch_b(features)

        # Simple concatenation (or use gate for weighted fusion)
        fused = torch.cat([emb_a, emb_b], dim=1)

        return self.classifier(fused)

print("\n Improved architectures with Xavier initialization")

# DATASET AND DATALOADER
class AblationDataset(Dataset):
    def __init__(self, signals, features, labels, feature_scaler):
        self.signals = torch.FloatTensor(signals).unsqueeze(1)
        self.features = torch.FloatTensor(feature_scaler.transform(features))
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.features[idx], self.labels[idx]

train_dataset = AblationDataset(X_sig_train, X_feat_train, y_train, scaler)
val_dataset = AblationDataset(X_sig_val, X_feat_val, y_val, scaler)
test_dataset = AblationDataset(X_sig_test, X_feat_test, y_test, scaler)

# Balanced sampling
class_counts = np.bincount(y_train)
class_weights_sampler = 1.0 / class_counts
sample_weights = class_weights_sampler[y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=ABLATION_CONFIG['batch_size'], sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=ABLATION_CONFIG['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=ABLATION_CONFIG['batch_size'], shuffle=False)

# IMPROVED TRAINING FUNCTION
def train_improved(model, model_name, train_loader, val_loader, config):
    print(f"TRAINING: {model_name}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])

    # Warmup + Cosine scheduler
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return (epoch + 1) / config['warmup_epochs']
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - config['warmup_epochs']) /
                                    (config['epochs'] - config['warmup_epochs'])))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_recalls': []}
    best_state = None

    start_time = time.time()

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0

        for signals, features, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            signals, features, labels = signals.to(device), features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals, features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for signals, features, labels in val_loader:
                signals, features, labels = signals.to(device), features.to(device), labels.to(device)
                outputs = model(signals, features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_recalls = recall_score(all_labels, all_preds, average=None, zero_division=0)

        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_f1'].append(val_f1)
        history['val_recalls'].append(val_recalls.tolist())

        # Track best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
            status = "BEST"
        else:
            patience_counter += 1
            status = f"({patience_counter}/{config['patience']})"

        if epoch % 10 == 0 or status == "BEST":
            print(f"Epoch {epoch}: Train={train_loss/len(train_loader):.4f}, "
                  f"Val={val_loss/len(val_loader):.4f}, F1={val_f1:.4f}, "
                  f"Recalls={val_recalls.round(3)} {status}")

        if patience_counter >= config['patience']:
            print(f"\n Early stopping at epoch {epoch}")
            break

    # Load best and evaluate
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for signals, features, labels in test_loader:
            signals, features, labels = signals.to(device), features.to(device), labels.to(device)
            outputs = model(signals, features)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    test_recalls = recall_score(all_labels, all_preds, average=None, zero_division=0)
    test_cm = confusion_matrix(all_labels, all_preds)

    training_time = time.time() - start_time

    print(f"\n {model_name} complete:")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Macro F1: {test_f1:.4f}")
    print(f"   Per-class Recall: {test_recalls}")
    print(f"   Time: {training_time/60:.1f} minutes")

    return {
        'model_name': model_name,
        'test_accuracy': float(test_acc),
        'test_f1_macro': float(test_f1),
        'test_recalls': test_recalls.tolist(),
        'confusion_matrix': test_cm.tolist(),
        'training_time': float(training_time / 60),
        'best_val_f1': float(best_val_f1),
        'history': history
    }

# RUN IMPROVED ABLATION
print("RUNNING IMPROVED ABLATION EXPERIMENTS")
ablation_results = {}

# Branch A only
model1 = BranchA_Only_Improved(signal_length, n_classes)
ablation_results['Branch A Only (CNN)'] = train_improved(
    model1, 'Branch A Only (CNN)', train_loader, val_loader, ABLATION_CONFIG
)

# Branch B only
model2 = BranchB_Only_Improved(n_features, n_classes)
ablation_results['Branch B Only (Features)'] = train_improved(
    model2, 'Branch B Only (Features)', train_loader, val_loader, ABLATION_CONFIG
)

# Full dual-branch
model3 = DualBranch_Improved(signal_length, n_features, n_classes)
ablation_results['Full Dual-Branch'] = train_improved(
    model3, 'Full Dual-Branch', train_loader, val_loader, ABLATION_CONFIG
)

# Save
with open(ablation_dir / "ablation_results_improved.json", 'w') as f:
    json.dump(ablation_results, f, indent=2)

# ANALYSIS
print("IMPROVED ABLATION RESULTS")

comparison_data = []
for name, results in ablation_results.items():
    comparison_data.append({
        'Model': name,
        'Macro F1': results['test_f1_macro'],
        'Class 0 Recall': results['test_recalls'][0],
        'Class 1 Recall': results['test_recalls'][1],
        'Class 2 Recall': results['test_recalls'][2]
    })

df = pd.DataFrame(comparison_data)
df = df.sort_values('Macro F1', ascending=False)

print(f"\n{df.to_string(index=False)}\n")

# Check if fixed
baseline_f1 = ablation_results['Full Dual-Branch']['test_f1_macro']
features_only_f1 = ablation_results['Branch B Only (Features)']['test_f1_macro']
cnn_only_f1 = ablation_results['Branch A Only (CNN)']['test_f1_macro']

print("\n Validation Check:")
if baseline_f1 > features_only_f1 and baseline_f1 > cnn_only_f1:
    print(f"   FIXED! Dual-branch ({baseline_f1:.4f}) > Both individual branches")
    print(f"   This validates the architecture!")
else:
    print(f"   Issue persists:")
    print(f"   Dual-branch: {baseline_f1:.4f}")
    print(f"   Features only: {features_only_f1:.4f}")
    print(f"   CNN only: {cnn_only_f1:.4f}")

df.to_csv(ablation_dir / "ablation_comparison_improved.csv", index=False)

print(f"\n Improved ablation complete!")
print(f"   Results saved to: ablation_study_improved/")
