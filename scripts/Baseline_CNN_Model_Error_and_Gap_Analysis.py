"""
Baseline CNN Model Error and Gap Analysis
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("BASELINE CNN - ERROR & GAP ANALYSIS")

BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"

# Auto-detect baseline directory
possible_dirs = [
    f"{BASE_DIR}/outputs/dl_optimized",
    f"{BASE_DIR}/outputs/dl_baseline"
]

BASELINE_DIR = None
for dir_path in possible_dirs:
    if Path(dir_path).exists():
        has_models = any((Path(dir_path) / key / "model.pt").exists() 
                        for key in ["bonn_eeg", "epileptic_seizure", "epilepsy_122mb", "merged_epileptic_bonn"])
        if has_models:
            BASELINE_DIR = dir_path
            print(f" Found baseline models in: {dir_path}")
            break

if BASELINE_DIR is None:
    print(" Could not find baseline models!")
    exit(1)

ANALYSIS_DIR = f"{BASE_DIR}/outputs/error_analysis"
Path(ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Configuration
DATASETS_CONFIG = {
    "bonn_eeg": {
        "name": "Bonn EEG",
        "signal_path": f"{BASE_DIR}/outputs/final_processed/bonn_eeg",
        "signal_file": "preprocessed_signals.npy",
        "label_file": "labels.npy",
        "n_classes": 5,
        "label_map": {0: "Set A", 1: "Set B", 2: "Set C", 3: "Set D", 4: "Set E (Seizure)"},
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

# CNN Model Implementation
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

class EEGDataset(Dataset):
    def __init__(self, signals, labels, n_classes):
        labels = np.array(labels)
        min_label = labels.min()
        labels = labels - min_label
        
        mask = (labels >= 0) & (labels < n_classes)
        signals = signals[mask]
        labels = labels[mask]
        
        self.signals = torch.FloatTensor(signals).unsqueeze(1)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

def load_dataset(cfg):
    path = Path(cfg["signal_path"])
    
    if cfg["pre_split"]:
        X_test = np.load(path / cfg["test_file"])
        y_test = np.load(path / cfg["label_test"])
    else:
        from sklearn.model_selection import train_test_split
        X = np.load(path / cfg["signal_file"])
        y = np.load(path / cfg["label_file"])
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )
    
    return X_test, y_test

def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            out = model(X)
            probs = torch.softmax(out, 1)
            
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(cm, labels, save_path, title):
    """Enhanced confusion matrix with percentages"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax1,
                cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('True', fontsize=12)
    ax1.set_title(f'{title} - Absolute Counts', fontsize=14, fontweight='bold')
    
    # Normalized (percentages)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', 
                xticklabels=labels, yticklabels=labels, ax=ax2,
                cbar_kws={'label': 'Percentage'}, vmin=0, vmax=1)
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('True', fontsize=12)
    ax2.set_title(f'{title} - Normalized', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_performance(class_metrics, save_path, title):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['precision', 'recall', 'f1-score']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        classes = list(class_metrics.keys())
        values = [class_metrics[c][metric] for c in classes]
        
        bars = axes[idx].bar(range(len(classes)), values, color=color, alpha=0.7, edgecolor='black')
        axes[idx].set_xticks(range(len(classes)))
        axes[idx].set_xticklabels(classes, rotation=45, ha='right')
        axes[idx].set_ylabel(metric.capitalize(), fontsize=12)
        axes[idx].set_title(f'{metric.capitalize()} by Class', fontsize=12, fontweight='bold')
        axes[idx].set_ylim([0, 1.05])
        axes[idx].axhline(y=np.mean(values), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(values):.3f}')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_misclassification_patterns(cm, labels, save_path, title):
    # Get off-diagonal elements (misclassifications)
    n_classes = len(labels)
    misclass = cm.copy()
    np.fill_diagonal(misclass, 0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(misclass, annot=True, fmt='d', cmap='Reds',
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'label': 'Misclassifications'})
    
    ax.set_xlabel('Predicted As', fontsize=12)
    ax.set_ylabel('Actually Was', fontsize=12)
    ax.set_title(f'{title} - Misclassification Patterns', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_confidence(probs, preds, labels, save_path, title):
    correct_mask = (preds == labels)
    
    # Get confidence (max probability)
    confidence = np.max(probs, axis=1)
    
    correct_conf = confidence[correct_mask]
    incorrect_conf = confidence[~correct_mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(correct_conf, bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
    axes[0].hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    axes[0].set_xlabel('Prediction Confidence', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    data = [correct_conf, incorrect_conf]
    bp = axes[1].boxplot(data, labels=['Correct', 'Incorrect'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[1].set_ylabel('Prediction Confidence', fontsize=12)
    axes[1].set_title('Confidence Comparison', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add statistics
    stats_text = f"Correct: μ={np.mean(correct_conf):.3f}, σ={np.std(correct_conf):.3f}\n"
    stats_text += f"Incorrect: μ={np.mean(incorrect_conf):.3f}, σ={np.std(incorrect_conf):.3f}"
    axes[1].text(0.5, 0.02, stats_text, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'{title} - Prediction Confidence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_observations(dataset_name, cm, class_metrics, overall_acc, n_classes, label_map):
    observations = []
    observations.append(f"## {dataset_name} - Analysis\n")
    
    # Overall performance
    observations.append(f"**Overall Accuracy**: {overall_acc:.4f}\n")
    
    # Class imbalance detection
    class_samples = cm.sum(axis=1)
    max_samples = class_samples.max()
    min_samples = class_samples.min()
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
    
    if imbalance_ratio > 3:
        observations.append(f" **Class Imbalance Detected**: Ratio {imbalance_ratio:.2f}:1")
        observations.append(f"  - Majority class: {class_samples.argmax()} ({max_samples} samples)")
        observations.append(f"  - Minority class: {class_samples.argmin()} ({min_samples} samples)\n")
    
    # Per-class performance
    observations.append("**Class-wise Performance**:")
    worst_class = None
    worst_f1 = 1.0
    
    for class_name, metrics in class_metrics.items():
        f1 = metrics['f1-score']
        observations.append(f"  - {class_name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={f1:.3f}")
        
        if f1 < worst_f1:
            worst_f1 = f1
            worst_class = class_name
    
    observations.append(f"\n**Weakest Class**: {worst_class} (F1={worst_f1:.3f})")
    
    # Misclassification patterns
    observations.append("\n**Common Misclassifications**:")
    misclass = cm.copy()
    np.fill_diagonal(misclass, 0)
    
    # Find top 3 confusion pairs
    confusions = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and misclass[i, j] > 0:
                confusions.append((i, j, misclass[i, j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)
    for i, j, count in confusions[:3]:
        true_label = label_map.get(i, f"Class {i}")
        pred_label = label_map.get(j, f"Class {j}")
        observations.append(f"  - {true_label} → {pred_label}: {count} errors")
    
    # Hypothesis for failures
    observations.append("\n**Potential Failure Modes**:")
    
    if imbalance_ratio > 3:
        observations.append("  1. **Class imbalance** - Model biased toward majority class")
    
    if worst_f1 < 0.7:
        observations.append(f"  2. **Poor {worst_class} detection** - May need class-specific features")
    
    if n_classes > 2:
        observations.append("  3. **Multi-class complexity** - Binary decomposition might help")
    
    observations.append("  4. **Limited temporal modeling** - CNN lacks explicit sequence modeling")
    observations.append("  5. **Raw signal variability** - May benefit from time-frequency features")
    
    return "\n".join(observations)

# Main Analysis
if __name__ == "__main__":
    all_observations = []
    summary_stats = []
    
    for dataset_key, cfg in DATASETS_CONFIG.items():
        print(f"ANALYZING: {cfg['name']}")
        
        # Check if model exists
        model_path = Path(BASELINE_DIR) / dataset_key / "model.pt"
        if not model_path.exists():
            print(f" Skipping - model not found at {model_path}")
            continue
        
        # Load data
        print("Loading test data")
        X_test, y_test = load_dataset(cfg)
        test_ds = EEGDataset(X_test, y_test, cfg["n_classes"])
        test_dl = DataLoader(test_ds, batch_size=64)
        
        # Load model
        print("Loading trained model")
        model = EEG_CNN_Optimized(cfg["n_classes"]).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Get predictions
        print("Generating predictions")
        preds, labels, probs = get_predictions(model, test_dl, device)
        
        # Determine which classes are actually present
        unique_classes = np.unique(np.concatenate([labels, preds]))
        present_classes = sorted(unique_classes.tolist())
        class_names = [cfg["label_map"].get(i, f"Class {i}") for i in present_classes]
        
        print(f"Classes present in test set: {present_classes}")
        print(f"Class names: {class_names}")
        
        # Compute confusion matrix
        cm = confusion_matrix(labels, preds, labels=present_classes)
        
        # Get classification report
        report = classification_report(labels, preds, labels=present_classes,
                                      target_names=class_names, 
                                      output_dict=True, zero_division=0)
        
        # Extract class-wise metrics
        class_metrics = {name: report[name] for name in class_names}
        
        # Create output directory
        output_dir = Path(ANALYSIS_DIR) / dataset_key
        output_dir.mkdir(exist_ok=True)
        
        # Generate plots
        print("Generating visualizations")
        
        plot_confusion_matrix(
            cm, class_names,
            output_dir / "confusion_matrix_detailed.png",
            cfg['name']
        )
        
        plot_class_performance(
            class_metrics,
            output_dir / "class_performance.png",
            cfg['name']
        )
        
        plot_misclassification_patterns(
            cm, class_names,
            output_dir / "misclassification_patterns.png",
            cfg['name']
        )
        
        analyze_confidence(
            probs, preds, labels,
            output_dir / "confidence_analysis.png",
            cfg['name']
        )
        
        # Generate observations
        print("Generating observations")
        observations = generate_observations(
            cfg['name'], cm, class_metrics, 
            report['accuracy'], len(present_classes), 
            {i: cfg['label_map'].get(i, f"Class {i}") for i in present_classes}
        )
        
        # Save observations
        with open(output_dir / "observations.md", 'w') as f:
            f.write(observations)
        
        all_observations.append(observations)
        
        # Save detailed metrics
        with open(output_dir / "detailed_metrics.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Collect summary stats
        summary_stats.append({
            'Dataset': cfg['name'],
            'Accuracy': report['accuracy'],
            'Macro Avg F1': report['macro avg']['f1-score'],
            'Weighted Avg F1': report['weighted avg']['f1-score'],
            'Worst Class F1': min([class_metrics[c]['f1-score'] for c in class_names]),
            'Best Class F1': max([class_metrics[c]['f1-score'] for c in class_names])
        })
        
        print(f"\n Analysis complete for {cfg['name']}")
        print(f"  Plots saved to: {output_dir}")
    
    # Create report
    print("CREATING REPORT")
    
    with open(Path(ANALYSIS_DIR) / "master_observations.md", 'w') as f:
        f.write("# Baseline CNN - Error & Gap Analysis\n\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        f.write("---\n\n")
        f.write("\n\n---\n\n".join(all_observations))
        
        f.write("\n\n---\n\n")
        f.write("## Overall Conclusions\n\n")
        f.write("**Key Findings Across All Datasets:**\n\n")
        f.write("1. **Temporal Modeling Gap**: CNN lacks explicit temporal/sequential modeling\n")
        f.write("   - Recommendation: Add LSTM/GRU or Transformer layers\n\n")
        f.write("2. **Class Imbalance Issues**: Several datasets show significant imbalance\n")
        f.write("   - Recommendation: Implement focal loss or advanced sampling strategies\n\n")
        f.write("3. **Raw Signal Limitations**: Direct convolution may miss important patterns\n")
        f.write("   - Recommendation: Add attention mechanisms or multi-scale feature extraction\n\n")
        f.write("4. **Cross-Dataset Variability**: Performance varies significantly\n")
        f.write("   - Recommendation: Domain adaptation or transfer learning approaches\n\n")
    
    # Save summary table
    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv(Path(ANALYSIS_DIR) / "summary_statistics.csv", index=False)
    
    print("\n Report created")
    print("ANALYSIS COMPLETE")
    print(f"\nResults saved to: {ANALYSIS_DIR}")
    print(df_summary.to_string(index=False))
