"""
DUAL-BRANCH CNN - COMPLETE TRAINING PIPELINE
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

print("DUAL-BRANCH CNN - COMPLETE TRAINING PIPELINE")

# Configuration

BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
OUTPUT_DIR = f"{BASE_DIR}/outputs/dual_branch_experiments"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

HYPERPARAMS = {
    'batch_size': 64,
    'epochs': 50,
    'patience': 10,
    'lr': 1e-3,
    'branch_feat': 128
}

# Model Architectures
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

class BranchB_FrequencyDomain(nn.Module):
    def __init__(self, in_shape, out_feat=128):
        super().__init__()
        # Detect if input is 2D (freq x time) or 1D
        self.is_2d = len(in_shape) > 1 and in_shape[1] > 1

        if self.is_2d:
            # 2D CNN for STFT/Spectrogram (freq x time)
            print(f"  Using 2D convolutions for shape {in_shape}")
            self.conv1 = nn.Conv2d(in_shape[0], 32, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool = nn.MaxPool2d(2)
            self.gap = nn.AdaptiveAvgPool2d(1)
        else:
            # 1D CNN for flattened features
            print(f"  Using 1D convolutions for shape {in_shape}")
            self.conv1 = nn.Conv1d(in_shape[0], 32, 7, 2, 3)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32, 64, 5, 2, 2)
            self.bn2 = nn.BatchNorm1d(64)
            self.conv3 = nn.Conv1d(64, 128, 3, 2, 1)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool = nn.MaxPool1d(2)
            self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(128, out_feat)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.drop(self.pool(self.relu(self.bn1(self.conv1(x)))))
        x = self.drop(self.pool(self.relu(self.bn2(self.conv2(x)))))
        x = self.drop(self.relu(self.bn3(self.conv3(x))))

        if self.is_2d:
            x = self.gap(x).squeeze(-1).squeeze(-1)
        else:
            x = self.gap(x).squeeze(-1)

        return self.relu(self.fc(x))

class DualBranchCNN(nn.Module):
    def __init__(self, n_classes, transform_shape, fusion='concat'):
        super().__init__()
        self.branch_a = BranchA_TimeDomain(128)
        self.branch_b = BranchB_FrequencyDomain(transform_shape, 128)
        self.fusion = fusion

        if fusion == 'concat':
            self.fc1 = nn.Linear(256, 128)
        elif fusion == 'attention':
            self.attn_a = nn.Linear(128, 1)
            self.attn_b = nn.Linear(128, 1)
            self.fc1 = nn.Linear(256, 128)

        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)

    def forward(self, raw, trans):
        fa = self.branch_a(raw)
        fb = self.branch_b(trans)

        if self.fusion == 'concat':
            f = torch.cat([fa, fb], 1)
        elif self.fusion == 'attention':
            aa = torch.sigmoid(self.attn_a(fa))
            ab = torch.sigmoid(self.attn_b(fb))
            s = aa + ab
            f = torch.cat([fa * aa / s, fb * ab / s], 1)

        x = self.drop(self.relu(self.fc1(f)))
        x = self.drop(self.relu(self.fc2(x)))
        return self.fc3(x)

print(" Models defined\n")

# Dataset Class
class DualBranchDataset(Dataset):
    def __init__(self, X, y, n_classes, augment=False):
        # Normalize labels
        y = y - y.min()
        mask = (y >= 0) & (y < n_classes)
        self.X = X[mask]
        self.y = torch.LongTensor(y[mask])
        self.augment = augment

        # Compute STFT for one sample to get shape
        f, t, Zxx = signal.stft(self.X[0], nperseg=64, noverlap=32)
        self.stft_shape = np.abs(Zxx).shape
        print(f"  Dataset: {len(self.y)} samples, STFT shape: {self.stft_shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()

        # Augmentation
        if self.augment and np.random.rand() > 0.5:
            x = x + np.random.randn(*x.shape) * 0.01

        # Raw signal
        raw = torch.FloatTensor(x).unsqueeze(0)  # (1, signal_length)

        # STFT transform
        f, t, Zxx = signal.stft(x, nperseg=64, noverlap=32)
        stft = np.abs(Zxx)
        trans = torch.FloatTensor(stft).unsqueeze(0)  # (1, freq, time)

        return raw, trans, self.y[idx]

print(" Dataset class defined\n")

# Training Function
def train_epoch(model, loader, opt, crit, device, is_dual=False):
    model.train()
    loss_sum = 0
    correct = 0
    total = 0

    for batch in loader:
        if is_dual:
            raw, trans, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            out = model(raw, trans)
        else:
            raw, y = batch[0].to(device), batch[-1].to(device)
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

# Validate the model
def validate(model, loader, crit, device, is_dual=False):
    model.eval()
    loss_sum = 0
    preds, labels, probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            if is_dual:
                raw, trans, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                out = model(raw, trans)
            else:
                raw, y = batch[0].to(device), batch[-1].to(device)
                out = model(raw)

            loss_sum += crit(out, y).item()
            p = torch.softmax(out, 1)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())
            probs.extend(p.cpu().numpy())

    return loss_sum / len(loader), np.array(preds), np.array(labels), np.array(probs)

# Complete training loop with early stopping
def train_model(model, train_dl, val_dl, hyperparams, device, is_dual=False):
    opt = optim.AdamW(model.parameters(), lr=hyperparams['lr'], weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, hyperparams['epochs'])

    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\nTraining for {hyperparams['epochs']} epochs (patience={hyperparams['patience']})")
    start_time = time.time()

    for epoch in range(hyperparams['epochs']):
        tr_loss, tr_acc = train_epoch(model, train_dl, opt, crit, device, is_dual)
        val_loss, _, _, _ = validate(model, val_dl, crit, device, is_dual)

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
                  f"Val: {val_loss:.4f} | Patience: {patience_counter}/{hyperparams['patience']} {improved}")

        if patience_counter >= hyperparams['patience']:
            print(f" Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    total_time = time.time() - start_time
    print(f" Training completed in {total_time/60:.1f} minutes")

    return model, history

# Compute all evaluation metrics
def compute_metrics(y_true, y_pred, y_probs, n_classes):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    metrics['cm'] = cm

    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs[:, 1])
        except:
            metrics['roc_auc'] = None

    return metrics

# Print metrics
def print_metrics(metrics, name):
    print(f"{name} - Test Set Results")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    if 'sensitivity' in metrics:
        print(f"Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
    if metrics.get('roc_auc'):
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

# Save model, metrics, and plots
def save_results(model, metrics, history, save_dir, name):
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), save_dir / "model.pt")

    # Save metrics
    with open(save_dir / "metrics.json", 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                   for k, v in metrics.items() if k != 'cm'}, f, indent=2)

    # Save history
    with open(save_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Confusion matrix
    cm = metrics['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix')

    plt.suptitle(name, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "results.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f" Results saved to {save_dir}")

print(" Training functions defined\n")

# Main

if __name__ == "__main__":
    print("STARTING EXPERIMENTS")

    # Load data
    print("Loading data")
    signal_path = f"{BASE_DIR}/outputs/final_processed/epilepsy_122mb"
    X = np.load(Path(signal_path) / "preprocessed_signals.npy")
    y = np.load(Path(signal_path) / "labels.npy")

    print(f" Data loaded: {X.shape}")
    print(f" Unique labels: {np.unique(y)}\n")

    # Split data
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    n_classes = len(np.unique(y))
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Classes: {n_classes}\n")

    exp_dir = Path(OUTPUT_DIR) / "epilepsy_122mb"
    results = []

    # EXPERIMENT 1: BASELINE CNN (Single Branch - Raw Only
    print("EXPERIMENT 1: BASELINE CNN (Raw EEG Only)")

    # Normalize labels
    y_train_norm = y_train - y_train.min()
    y_val_norm = y_val - y_val.min()
    y_test_norm = y_test - y_test.min()

    # Create datasets
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

    train_dl = DataLoader(train_ds, batch_size=HYPERPARAMS['batch_size'], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=HYPERPARAMS['batch_size'])
    test_dl = DataLoader(test_ds, batch_size=HYPERPARAMS['batch_size'])

    # Create model
    baseline = nn.Sequential(
        BranchA_TimeDomain(HYPERPARAMS['branch_feat']),
        nn.Linear(HYPERPARAMS['branch_feat'], n_classes)
    ).to(device)

    # Train
    baseline, hist = train_model(baseline, train_dl, val_dl, HYPERPARAMS, device, is_dual=False)

    # Evaluate
    _, preds, labels, probs = validate(baseline, test_dl, nn.CrossEntropyLoss(), device, is_dual=False)
    m1 = compute_metrics(labels, preds, probs, n_classes)
    print_metrics(m1, "Baseline CNN")
    save_results(baseline, m1, hist, exp_dir / "baseline_cnn", "Baseline CNN")

    results.append({'Experiment': 'Baseline CNN', **{k: v for k, v in m1.items() if k != 'cm'}})

    #EXPERIMENTS 2 & 3: DUAL-BRANCH CNN
    print("PREPARING DUAL-BRANCH DATASETS")

    # Create dual-branch datasets
    train_ds_dual = DualBranchDataset(X_train, y_train, n_classes, augment=True)
    val_ds_dual = DualBranchDataset(X_val, y_val, n_classes, augment=False)
    test_ds_dual = DualBranchDataset(X_test, y_test, n_classes, augment=False)

    train_dl_dual = DataLoader(train_ds_dual, batch_size=HYPERPARAMS['batch_size'], shuffle=True)
    val_dl_dual = DataLoader(val_ds_dual, batch_size=HYPERPARAMS['batch_size'])
    test_dl_dual = DataLoader(test_ds_dual, batch_size=HYPERPARAMS['batch_size'])

    # Get transform shape
    transform_shape = train_ds_dual.stft_shape
    print(f"\n Transform shape for Branch B: (1, {transform_shape[0]}, {transform_shape[1]})\n")

    # Run both fusion experiments
    for fusion_type in ['concat', 'attention']:
        print(f"EXPERIMENT: DUAL-BRANCH CNN ({fusion_type.upper()} Fusion)")

        # Create model
        dual_model = DualBranchCNN(
            n_classes=n_classes,
            transform_shape=(1,) + transform_shape,
            fusion=fusion_type
        ).to(device)

        # Train
        dual_model, hist = train_model(dual_model, train_dl_dual, val_dl_dual,
                                       HYPERPARAMS, device, is_dual=True)

        # Evaluate
        _, preds, labels, probs = validate(dual_model, test_dl_dual,
                                          nn.CrossEntropyLoss(), device, is_dual=True)
        metrics = compute_metrics(labels, preds, probs, n_classes)
        print_metrics(metrics, f"Dual-Branch {fusion_type.capitalize()}")
        save_results(dual_model, metrics, hist, exp_dir / f"dual_{fusion_type}",
                    f"Dual-Branch {fusion_type.capitalize()}")

        results.append({
            'Experiment': f'Dual-Branch {fusion_type.capitalize()}',
            **{k: v for k, v in metrics.items() if k != 'cm'}
        })

    # Create Comparison Report
    print("FINAL COMPARISON")

    df = pd.DataFrame(results)
    df.to_csv(exp_dir / "comparison.csv", index=False)

    print(df.to_string(index=False))

    print(" ALL EXPERIMENTS COMPLETE")
    print(f"\nResults saved to: {exp_dir}")
