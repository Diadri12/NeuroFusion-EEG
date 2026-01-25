"""
CNN Baseline Implementation
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
from collections import Counter
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CNN BASELINE IMPLEMENTATION FOR EEG SEIZURE DETECTION")
print(f"Device: {device}")

BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
OUTPUT_DIR = f"{BASE_DIR}/outputs/dl_optimized"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Dataset Configuration
DATASETS_CONFIG = {
    "bonn_eeg": {
        "name": "Bonn EEG",
        "signal_path": f"{BASE_DIR}/outputs/final_processed/bonn_eeg",
        "signal_file": "preprocessed_signals.npy",
        "label_file": "labels.npy",
        "n_classes": 5,
        "label_map": {0: "Non-seizure", 1: "Seizure"},
        "pre_split": False
    },
    "epileptic_seizure": {
        "name": "Epileptic Seizure",
        "signal_path": f"{BASE_DIR}/outputs",
        "signal_file": "full_processed_signals.npy",
        "label_file": "full_labels.npy",
        "n_classes": 5,
        "label_map": {
            0: "Seizure", 1: "Tumor", 2: "Healthy",
            3: "Eyes Closed", 4: "Eyes Open"
        },
        "pre_split": False
    },
    "epilepsy_122mb": {
        "name": "Epilepsy 122MB",
        "signal_path": f"{BASE_DIR}/outputs/final_processed/epilepsy_122mb",
        "signal_file": "preprocessed_signals.npy",
        "label_file": "labels.npy",
        "n_classes": 2,
        "label_map": {0: "Non-seizure", 1: "Seizure"},
        "pre_split": False
    },
    "merged_epileptic_bonn": {
        "name": "Merged Epileptic + Bonn",
        "signal_path": f"{BASE_DIR}/outputs/merged_datasets/merged_epileptic_bonn",
        "signal_file": "X_train.npy",
        "val_file": "X_val.npy",
        "test_file": "X_test.npy",
        "label_train": "y_train.npy",
        "label_val": "y_val.npy",
        "label_test": "y_test.npy",
        "n_classes": 2,
        "label_map": {0: "Non-seizure", 1: "Seizure"},
        "pre_split": True
    }
}

# Lightweight CNN Model (CPU-Optimized)
class EEG_CNN_Optimized(nn.Module):
    def __init__(self, n_classes, dropout=0.5):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, 7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Block 2
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Block 3
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv1d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Dataset with Light Augmentation
class EEGDataset(Dataset):
    def __init__(self, signals, labels, n_classes, augment=False, dataset_name="EEG"):
        labels = np.array(labels)
        min_label = labels.min()
        
        if min_label != 0:
            print(f"Adjusting labels for {dataset_name}: subtracting offset {min_label}")
        labels = labels - min_label
        
        mask = (labels >= 0) & (labels < n_classes)
        if not mask.all():
            print(f"Removing {(~mask).sum()} invalid samples from {dataset_name}")
            signals = signals[mask]
            labels = labels[mask]
        
        self.signals = torch.FloatTensor(signals).unsqueeze(1)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        
        print(f" {dataset_name}: {len(self.labels)} samples | Labels: {self.labels.min()}-{self.labels.max()}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x, y = self.signals[idx], self.labels[idx]
        
        if self.augment and torch.rand(1) > 0.5:
            # Light noise injection
            x = x + torch.randn_like(x) * 0.01
        
        return x, y

# Data Loading
def load_dataset(cfg):
    path = Path(cfg["signal_path"])

    if cfg["pre_split"]:
        X_train = np.load(path / cfg["signal_file"])
        X_val = np.load(path / cfg["val_file"])
        X_test = np.load(path / cfg["test_file"])
        y_train = np.load(path / cfg["label_train"])
        y_val = np.load(path / cfg["label_val"])
        y_test = np.load(path / cfg["label_test"])
    else:
        X = np.load(path / cfg["signal_file"])
        y = np.load(path / cfg["label_file"])

        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
        )

    return X_train, X_val, X_test, y_train, y_val, y_test

# Weighted Sampler
def get_weighted_sampler(labels, n_classes):
    labels_np = labels.numpy()
    class_counts = Counter(labels_np)
    
    weights = []
    for i in range(n_classes):
        count = class_counts.get(i, 1)
        weights.append(1.0 / count)
    
    weights = torch.tensor(weights, dtype=torch.float)
    sample_weights = weights[labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

# Simple Cosine Annealing
class CosineScheduler:
    def __init__(self, optimizer, total_epochs, lr_min=1e-6):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.lr_min = lr_min
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch):
        lr = self.lr_min + (self.base_lr - self.lr_min) * 0.5 * (1 + np.cos(np.pi * epoch / self.total_epochs))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# Fast Training Function
def train_model(model, train_loader, val_loader, n_classes, epochs=50, patience=10):
    start_time = time.time()
    
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Compute class weights
    train_labels = []
    for _, y in train_loader:
        train_labels.extend(y.numpy())
    
    class_counts = Counter(train_labels)
    weights = []
    for i in range(n_classes):
        count = class_counts.get(i, 1)
        weights.append(1.0 / count)
    
    weights = torch.tensor(weights, dtype=torch.float).to(device)
    weights = weights / weights.sum() * n_classes
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    
    scheduler = CosineScheduler(opt, total_epochs=epochs)
    
    best_state, best_loss = None, float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    
    print(f"Starting training for {epochs} epochs (max) with patience={patience}")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        tr_loss, tr_correct, tr_total = 0, 0, 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            opt.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            
            tr_loss += loss.item()
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_total += y.size(0)
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                val_loss += loss_fn(out, y).item()
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)
        
        # Metrics
        tr_loss /= len(train_loader)
        val_loss /= len(val_loader)
        tr_acc = tr_correct / tr_total
        val_acc = val_correct / val_total
        lr = scheduler.step(epoch)
        
        epoch_time = time.time() - epoch_start
        
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 2 == 0 or patience_counter == patience - 1:
            print(f"Epoch {epoch:03d} | Train: {tr_loss:.4f} ({tr_acc:.4f}) | "
                  f"Val: {val_loss:.4f} ({val_acc:.4f}) | "
                  f"LR: {lr:.6f} | Patience: {patience_counter}/{patience} | "
                  f"Time: {epoch_time:.1f}s")
        
        if patience_counter >= patience:
            print(f"\n Early stopping at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    print(f"\n Training completed in {total_time/60:.1f} minutes ({total_time:.1f}s)")
    
    model.load_state_dict(best_state)
    return model, history

def evaluate(model, loader, n_classes):
    model.eval()
    preds, labels, probs = [], [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            out = model(X)
            p = torch.softmax(out, 1)
            preds.extend(p.argmax(1).cpu().numpy())
            labels.extend(y.numpy())
            probs.extend(p.cpu().numpy())

    preds = np.array(preds)
    labels = np.array(labels)
    probs = np.array(probs)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(labels, probs[:, 1]) if n_classes == 2 else None
    }

    return metrics, confusion_matrix(labels, preds)

# Visualization
def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history["val_acc"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].grid(True)
    
    axes[2].plot(history["lr"])
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(True)
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# Main
if __name__ == "__main__":
    total_start = time.time()
    results_all = []

    for key, cfg in DATASETS_CONFIG.items():
        print(f"TRAINING ON: {cfg['name']}")

        dataset_start = time.time()
        
        X_tr, X_v, X_te, y_tr, y_v, y_te = load_dataset(cfg)

        train_ds = EEGDataset(X_tr, y_tr, cfg["n_classes"], augment=True, dataset_name=f"{cfg['name']} Train")
        val_ds = EEGDataset(X_v, y_v, cfg["n_classes"], augment=False, dataset_name=f"{cfg['name']} Val")
        test_ds = EEGDataset(X_te, y_te, cfg["n_classes"], augment=False, dataset_name=f"{cfg['name']} Test")

        sampler = get_weighted_sampler(train_ds.labels, cfg["n_classes"])
        train_dl = DataLoader(train_ds, batch_size=128, sampler=sampler, num_workers=0)
        val_dl = DataLoader(val_ds, batch_size=128, num_workers=0)
        test_dl = DataLoader(test_ds, batch_size=128, num_workers=0)

        model = EEG_CNN_Optimized(cfg["n_classes"]).to(device)
        print(f"\n Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        model, hist = train_model(model, train_dl, val_dl, cfg["n_classes"], epochs=50, patience=10)

        print("\nEvaluate on test set")
        metrics, cm = evaluate(model, test_dl, cfg["n_classes"])

        out_dir = Path(OUTPUT_DIR) / key
        out_dir.mkdir(exist_ok=True)

        torch.save(model.state_dict(), out_dir / "model.pt")
        
        with open(out_dir / "results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        with open(out_dir / "history.json", "w") as f:
            json.dump(hist, f, indent=2)
        
        plot_training_history(hist, out_dir / "training_history.png")
        plot_confusion_matrix(cm, list(cfg["label_map"].values()), out_dir / "confusion_matrix.png")

        results_all.append({
            "Dataset": cfg["name"],
            **{k: f"{v:.4f}" if v is not None else "N/A" for k, v in metrics.items()}
        })
        
        dataset_time = time.time() - dataset_start
        
        print(f"\n{'='*70}")
        print(f"Test Results for {cfg['name']}:")
        for k, v in metrics.items():
            if v is not None:
                print(f"  {k.capitalize()}: {v:.4f}")
        print(f"\n Dataset completed in {dataset_time/60:.1f} minutes")
        print(f"{'='*70}")

    df = pd.DataFrame(results_all)
    df.to_csv(f"{OUTPUT_DIR}/cnn_comparison.csv", index=False)

    total_time = time.time() - total_start

    print("CNN TRAINING COMPLETE")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f}s)")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"\nComparison Summary:")
    print(df.to_string(index=False))
