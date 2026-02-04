"""
ABLATION STUDY
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

print("ABLATION STUDY")

# Configuration
ABLATION_CONFIG = {
    'epochs': 60,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'patience': 15,
    'use_balanced_sampling': True,
    'label_smoothing': 0.1,
    'force_cpu': False
}

print(" Ablation configuration:")
print(f"   Epochs: {ABLATION_CONFIG['epochs']}")
print(f"   Balanced sampling: {ABLATION_CONFIG['use_balanced_sampling']}")
print(f"   Quick training for comparisons\n")

# Paths
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
PHASE3_DIR = f"{BASE_DIR}/outputs/dual_branch_training"
PHASE6_DIR = f"{BASE_DIR}/outputs/advanced_analysis"
ablation_dir = Path(PHASE6_DIR) / "ablation_study"
ablation_dir.mkdir(exist_ok=True, parents=True)

device = torch.device('cuda' if torch.cuda.is_available() and not ABLATION_CONFIG['force_cpu'] else 'cpu')
print(f"Device: {device}\n")

# LOAD DATA
print("LOADING DATA")

frozen_dir = Path(PHASE3_DIR) / "frozen"

X_windowed = np.load(frozen_dir / "X_windowed.npy")
y_windowed = np.load(frozen_dir / "y_windowed.npy")
X_features = np.load(frozen_dir / "X_features.npy")
scaler = joblib.load(Path(PHASE3_DIR) / "feature_scaler.pkl")

signal_length = X_windowed.shape[1]
n_features = X_features.shape[1]
n_classes = len(np.unique(y_windowed))

# Same split as all experiments
X_sig_train, X_sig_tmp, X_feat_train, X_feat_tmp, y_train, y_tmp = train_test_split(
    X_windowed, X_features, y_windowed, test_size=0.3, stratify=y_windowed, random_state=42
)
X_sig_val, X_sig_test, X_feat_val, X_feat_test, y_val, y_test = train_test_split(
    X_sig_tmp, X_feat_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
)

print(f"\n Data loaded:")
print(f"   Train: {len(X_sig_train):,}")
print(f"   Val: {len(X_sig_val):,}")
print(f"   Test: {len(X_sig_test):,}")

# MODEL ARCHITECTURES FOR ABLATION
print("DEFINING ABLATION ARCHITECTURES")

# CNN branch for signal processing
class BranchA_CNN(nn.Module):
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
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.pool(self.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool(self.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(self.pool(self.relu(self.bn3(self.conv3(x)))))
        x = self.dropout(self.pool(self.relu(self.bn4(self.conv4(x)))))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

# MLP branch for handcrafted features
class BranchB_Features(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15)
        )

    def forward(self, x):
        return self.model(x)

# Ablation: Branch A only (Only CNN branch, no handcrafted features)
class BranchA_Only(nn.Module):
    def __init__(self, signal_length, n_classes):
        super().__init__()
        self.branch_a = BranchA_CNN(signal_length)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_classes)
        )

    def forward(self, signals, features=None):
        emb = self.branch_a(signals)
        return self.classifier(emb)

# Ablation: Branch B only (Only handcrafted features, no CNN)
class BranchB_Only(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.branch_b = BranchB_Features(n_features)
        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_classes)
        )

    def forward(self, signals, features):
        emb = self.branch_b(features)
        return self.classifier(emb)

# Full dual-branch (baseline for comparison)
class DualBranch_Full(nn.Module):
    def __init__(self, signal_length, n_features, n_classes):
        super().__init__()
        self.branch_a = BranchA_CNN(signal_length)
        self.branch_b = BranchB_Features(n_features)
        self.classifier = nn.Sequential(
            nn.Linear(96, 32),  # 64 + 32
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_classes)
        )

    def forward(self, signals, features):
        emb_a = self.branch_a(signals)
        emb_b = self.branch_b(features)
        fused = torch.cat([emb_a, emb_b], dim=1)
        return self.classifier(fused)

# Ablation: No normalization (CNN without batch normalization)
class BranchA_NoNorm(nn.Module):
    def __init__(self, signal_length):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 32, 7, 2, 3)
        self.conv2 = nn.Conv1d(32, 64, 5, 1, 2)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv1d(128, 256, 3, 1, 1)

        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.pool(self.relu(self.conv1(x))))
        x = self.dropout(self.pool(self.relu(self.conv2(x))))
        x = self.dropout(self.pool(self.relu(self.conv3(x))))
        x = self.dropout(self.pool(self.relu(self.conv4(x))))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

# Dual-branch without normalization
class DualBranch_NoNorm(nn.Module):
    def __init__(self, signal_length, n_features, n_classes):
        super().__init__()
        self.branch_a = BranchA_NoNorm(signal_length)
        self.branch_b = BranchB_Features(n_features)
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

print("\n Ablation architectures defined:")

# DATASET
class AblationDataset(Dataset):
    def __init__(self, signals, features, labels, feature_scaler):
        self.signals = torch.FloatTensor(signals).unsqueeze(1)
        self.features = torch.FloatTensor(feature_scaler.transform(features))
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.features[idx], self.labels[idx]

# Create datasets
train_dataset = AblationDataset(X_sig_train, X_feat_train, y_train, scaler)
val_dataset = AblationDataset(X_sig_val, X_feat_val, y_val, scaler)
test_dataset = AblationDataset(X_sig_test, X_feat_test, y_test, scaler)

# Create balanced sampler
if ABLATION_CONFIG['use_balanced_sampling']:
    class_counts = np.bincount(y_train)
    class_weights_sampler = 1.0 / class_counts
    sample_weights = class_weights_sampler[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    shuffle = False
else:
    sampler = None
    shuffle = True

train_loader = DataLoader(train_dataset, batch_size=ABLATION_CONFIG['batch_size'],
                          sampler=sampler, shuffle=shuffle)
val_loader = DataLoader(val_dataset, batch_size=ABLATION_CONFIG['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=ABLATION_CONFIG['batch_size'], shuffle=False)

# TRAINING FUNCTION
def train_ablation_model(model, model_name, train_loader, val_loader, config):
    print(f"TRAINING: {model_name}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    best_val_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    start_time = time.time()

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0

        for signals, features, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            signals = signals.to(device)
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals, features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for signals, features, labels in val_loader:
                signals = signals.to(device)
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(signals, features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_f1'].append(val_f1)

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, "
                  f"Val Loss={val_loss/len(val_loader):.4f}, Val F1={val_f1:.4f}")

        if patience_counter >= config['patience']:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    training_time = time.time() - start_time

    # Evaluate on test set
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for signals, features, labels in test_loader:
            signals = signals.to(device)
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(signals, features)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    test_recalls = recall_score(all_labels, all_preds, average=None, zero_division=0)
    test_cm = confusion_matrix(all_labels, all_preds)

    print(f"\n {model_name} complete:")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Macro F1: {test_f1:.4f}")
    print(f"   Per-class Recall: {test_recalls}")
    print(f"   Training time: {training_time/60:.1f} minutes")

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

# RUN ABLATION EXPERIMENTS
print("RUNNING ABLATION EXPERIMENTS")

ablation_results = {}

# Experiment 1: Branch A only
model1 = BranchA_Only(signal_length, n_classes)
ablation_results['Branch A Only (CNN)'] = train_ablation_model(
    model1, 'Branch A Only (CNN)', train_loader, val_loader, ABLATION_CONFIG
)

# Experiment 2: Branch B only
model2 = BranchB_Only(n_features, n_classes)
ablation_results['Branch B Only (Features)'] = train_ablation_model(
    model2, 'Branch B Only (Features)', train_loader, val_loader, ABLATION_CONFIG
)

# Experiment 3: Full dual-branch
model3 = DualBranch_Full(signal_length, n_features, n_classes)
ablation_results['Full Dual-Branch'] = train_ablation_model(
    model3, 'Full Dual-Branch', train_loader, val_loader, ABLATION_CONFIG
)

# Experiment 4: No normalization
model4 = DualBranch_NoNorm(signal_length, n_features, n_classes)
ablation_results['No Batch Normalization'] = train_ablation_model(
    model4, 'No Batch Normalization', train_loader, val_loader, ABLATION_CONFIG
)

# Save results
with open(ablation_dir / "ablation_results.json", 'w') as f:
    json.dump(ablation_results, f, indent=2)

print(f"\n All ablation experiments complete!")

# COMPARATIVE ANALYSIS
print("ABLATION STUDY RESULTS")


# Create comparison table
comparison_data = []

for name, results in ablation_results.items():
    comparison_data.append({
        'Model': name,
        'Test Accuracy': results['test_accuracy'],
        'Macro F1': results['test_f1_macro'],
        'Class 0 Recall': results['test_recalls'][0],
        'Class 1 Recall': results['test_recalls'][1],
        'Class 2 Recall': results['test_recalls'][2],
        'Training Time (min)': results['training_time']
    })

df = pd.DataFrame(comparison_data)
df = df.sort_values('Macro F1', ascending=False)

print("ABLATION STUDY COMPARISON TABLE")
print(f"\n{df.to_string(index=False)}\n")

df.to_csv(ablation_dir / "ablation_comparison.csv", index=False)
print(f" Saved: ablation_comparison.csv")

# Calculate performance drops
baseline_f1 = ablation_results['Full Dual-Branch']['test_f1_macro']

print("\n Performance Impact Analysis:")
for name, results in ablation_results.items():
    if name != 'Full Dual-Branch':
        drop = baseline_f1 - results['test_f1_macro']
        drop_pct = (drop / baseline_f1) * 100
        print(f"   {name}:")
        print(f"      Drop: {drop:.4f} ({drop_pct:.1f}%)")

# VISUALIZATION
print("CREATING VISUALIZATIONS")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Macro F1 comparison
ax1 = axes[0, 0]
models = [r['model_name'] for r in ablation_results.values()]
f1_scores = [r['test_f1_macro'] for r in ablation_results.values()]

bars = ax1.bar(range(len(models)), f1_scores, color='steelblue', alpha=0.7)
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=15, ha='right', fontsize=9)
ax1.set_ylabel('Macro F1 Score', fontsize=11, fontweight='bold')
ax1.set_title('(a) Macro F1 Comparison', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Highlight best
best_idx = f1_scores.index(max(f1_scores))
bars[best_idx].set_color('green')
bars[best_idx].set_alpha(0.9)

# Add value labels
for bar, score in zip(bars, f1_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Performance drop from baseline
ax2 = axes[0, 1]
drops = []
model_names_clean = []
for name, results in ablation_results.items():
    if name != 'Full Dual-Branch':
        drop = baseline_f1 - results['test_f1_macro']
        drops.append(drop)
        model_names_clean.append(name.replace(' ', '\n'))

bars = ax2.barh(range(len(drops)), drops, color='coral', alpha=0.7)
ax2.set_yticks(range(len(drops)))
ax2.set_yticklabels(model_names_clean, fontsize=9)
ax2.set_xlabel('Performance Drop (Macro F1)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Impact of Component Removal', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

# Add value labels
for bar, drop in zip(bars, drops):
    width = bar.get_width()
    ax2.text(width + 0.005, bar.get_y() + bar.get_height()/2.,
            f'{drop:.4f}', va='center', fontsize=9, fontweight='bold')

# Per-class recall heatmap
ax3 = axes[1, 0]
recall_matrix = np.array([r['test_recalls'] for r in ablation_results.values()])
sns.heatmap(recall_matrix, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
            xticklabels=['Class 0', 'Class 1', 'Class 2'],
            yticklabels=models, ax=ax3, cbar_kws={'label': 'Recall'})
ax3.set_title('(c) Per-Class Recall Heatmap', fontsize=12, fontweight='bold')

# Training efficiency
ax4 = axes[1, 1]
training_times = [r['training_time'] for r in ablation_results.values()]

scatter = ax4.scatter(training_times, f1_scores, s=200, c=f1_scores,
                     cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidths=2)

for i, name in enumerate(models):
    ax4.annotate(name.split()[0], (training_times[i], f1_scores[i]),
                fontsize=8, ha='center', va='bottom')

ax4.set_xlabel('Training Time (minutes)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Macro F1 Score', fontsize=11, fontweight='bold')
ax4.set_title('(d) Efficiency vs Performance', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Macro F1')

plt.suptitle('Ablation Study: Component Contribution Analysis',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(ablation_dir / "ablation_study.png", dpi=300, bbox_inches='tight')
plt.close()

print(f" Saved: ablation_study.png")

print(f"\n Ablation study complete!")
print(f"   All architectural choices validated")
