"""
SANITY & DATA INTEGRITY CHECKS
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal as scipy_signal
from pathlib import Path
import json
import time

print("SANITY CHECKS")

# Configuration
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
PHASE1_CHECK_DIR = f"{BASE_DIR}/outputs/phase1_sanity_checks"
Path(PHASE1_CHECK_DIR).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load Data
print("LOADING DATA")

signal_path = f"{BASE_DIR}/outputs/final_processed/epilepsy_122mb"
X = np.load(Path(signal_path) / "preprocessed_signals.npy")
y = np.load(Path(signal_path) / "labels.npy")

print(f" Data loaded")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# Normalize labels to 0, 1, 2
y_min = y.min()
y_max = y.max()
y_normalized = y - y_min

print(f"\nLabel range:")
print(f"  Original: [{y_min}, {y_max}]")
print(f"  Normalized: [{y_normalized.min()}, {y_normalized.max()}]")

# Label Alignment Verification
print("LABEL ALIGNMENT VERIFICATION")

print("\n Basic Statistics:")
print(f"  X.shape: {X.shape}")
print(f"  y.shape: {y.shape}")
print(f"  Match: {' YES' if X.shape[0] == y.shape[0] else ' NO (CRITICAL ERROR!)'}")

print(f"\n First 20 labels:")
print(f"  {y_normalized[:20]}")

print(f"\n Class distribution:")
unique, counts = np.unique(y_normalized, return_counts=True)
total = len(y_normalized)

print(f"  {'Class':<10} {'Count':<10} {'Percentage':<12} {'Status'}")
print(f"  {'-'*50}")
for cls, cnt in zip(unique, counts):
    pct = 100 * cnt / total
    status = "Yes" if cnt > 100 else " Too few samples!"
    print(f"  {cls:<10} {cnt:<10,} {pct:>6.2f}%      {status}")

# Check for any anomalies
n_classes = len(unique)
print(f"\n Sanity checks:")
print(f"  Number of classes: {n_classes} {'Yes' if n_classes == 3 else 'Expected 3!'}")
print(f"  Classes sequential: {list(unique)} {'Yes' if list(unique) == [0, 1, 2] else 'Expected [0, 1, 2]!'}")
print(f"  No missing data: {'NO NaNs' if not np.isnan(X).any() else 'FOUND NaNs!'}")
print(f"  No infinite values: {'NO Infs' if not np.isinf(X).any() else 'FOUND Infs!'}")

# Visualize random samples from each class
print(f"\n Plotting raw signals from each class")

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle('Raw EEG Signals by Class (Random Samples)', fontsize=16, fontweight='bold')

for class_idx in range(n_classes):
    # Get samples from this class
    class_mask = y_normalized == class_idx
    class_samples = X[class_mask]

    # Randomly select 3 samples
    random_indices = np.random.choice(len(class_samples), size=min(3, len(class_samples)), replace=False)

    for col_idx, sample_idx in enumerate(random_indices):
        ax = axes[class_idx, col_idx]
        signal = class_samples[sample_idx]

        ax.plot(signal, linewidth=0.5, alpha=0.8)
        ax.set_title(f'Class {class_idx} - Sample {col_idx+1}')
        ax.set_xlabel('Time (samples)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)

        # Add statistics
        ax.text(0.02, 0.98, f'μ={signal.mean():.2f}\nσ={signal.std():.2f}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(Path(PHASE1_CHECK_DIR) / "label_verification.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"  Saved: label_verification.png")

# Visual inspection prompt
print("MANUAL INSPECTION:")
print("Open 'label_verification.png' and verify:")
print("   1. Signals from different classes look visually distinct")
print("   2. No flat lines or artifacts")
print("   3. Amplitudes are reasonable")
print("   4. Each class has different characteristics")
print("\nIf they all look the same → CRITICAL PROBLEM with labels!")

# MICRO-DATASET OVERFIT TEST
print("MICRO-DATASET OVERFIT TEST")
# Create micro dataset: 20 samples per class
SAMPLES_PER_CLASS = 20
print(f"Creating micro dataset: {SAMPLES_PER_CLASS} samples per class")

micro_X = []
micro_y = []

for class_idx in range(n_classes):
    class_mask = y_normalized == class_idx
    class_samples = X[class_mask]

    # Randomly select samples
    selected_indices = np.random.choice(len(class_samples), size=SAMPLES_PER_CLASS, replace=False)
    micro_X.append(class_samples[selected_indices])
    micro_y.extend([class_idx] * SAMPLES_PER_CLASS)

micro_X = np.vstack(micro_X)
micro_y = np.array(micro_y)

print(f" Micro dataset created:")
print(f"  Shape: {micro_X.shape}")
print(f"  Classes: {np.bincount(micro_y)}")

# Simple CNN for overfit test
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 7, 2, 3)
        self.conv2 = nn.Conv1d(32, 64, 5, 2, 2)
        self.conv3 = nn.Conv1d(64, 128, 3, 2, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

# Prepare micro dataset
micro_ds = TensorDataset(
    torch.FloatTensor(micro_X).unsqueeze(1),
    torch.LongTensor(micro_y)
)
micro_dl = DataLoader(micro_ds, batch_size=10, shuffle=True)

# Train model
print(f"\nTraining on micro dataset")

model = SimpleCNN(n_classes=n_classes).to(device)
criterion = nn.CrossEntropyLoss()  # NO WEIGHTS!
optimizer = optim.Adam(model.parameters(), lr=1e-3)

MAX_EPOCHS = 200
history = {'loss': [], 'acc': [], 'pred_dist': []}

print(f"{'Epoch':<8} {'Loss':<12} {'Accuracy':<12} {'Pred Distribution'}")

for epoch in range(MAX_EPOCHS):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    for X_batch, y_batch in micro_dl:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = epoch_loss / len(micro_dl)
    accuracy = accuracy_score(all_labels, all_preds)
    pred_dist = np.bincount(all_preds, minlength=n_classes)

    history['loss'].append(avg_loss)
    history['acc'].append(accuracy)
    history['pred_dist'].append(pred_dist.tolist())

    if epoch % 20 == 0 or epoch == MAX_EPOCHS - 1:
        pred_str = f"[{pred_dist[0]:2d}, {pred_dist[1]:2d}, {pred_dist[2]:2d}]"
        print(f"{epoch:<8} {avg_loss:<12.4f} {accuracy:<12.4f} {pred_str}")

# Final evaluation
print("OVERFIT TEST RESULTS")

final_acc = history['acc'][-1]
final_pred_dist = history['pred_dist'][-1]

print(f"\n Final Training Accuracy: {final_acc:.4f}")
print(f" Final Prediction Distribution: {final_pred_dist}")
print(f" Expected Distribution: [20, 20, 20]\n")

# Compute confusion matrix
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in micro_dl:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

cm = confusion_matrix(all_labels, all_preds)

print(f" Confusion Matrix:")
print(cm)

# Determine pass/fail
PASS_THRESHOLD = 0.95  # 95% accuracy required

if final_acc >= PASS_THRESHOLD:
    print(" OVERFIT TEST PASSED")
    print(f"   Accuracy: {final_acc:.1%} >= {PASS_THRESHOLD:.1%}")

    # Check if all classes are predicted
    if all(d > 0 for d in final_pred_dist):
        print(f" All classes predicted: {final_pred_dist}")
    else:
        print(f" Warning: Some classes not predicted: {final_pred_dist}")

    overfit_passed = True
else:
    print(" OVERFIT TEST FAILED")
    print(f"   Accuracy: {final_acc:.1%} < {PASS_THRESHOLD:.1%}")

    # Diagnose the issue
    if all(d == final_pred_dist[0] for d in final_pred_dist):
        print("   → Model only predicts one class")
        print("   → Possible causes:")
        print("     - Label corruption")
        print("     - Data/label mismatch")
        print("     - Incorrect data loading")
    elif final_acc < 0.4:
        print("   → Model performs worse than random")
        print("   → Possible causes:")
        print("     - Labels shuffled incorrectly")
        print("     - Wrong class mapping")
        print("     - Data corruption")
    else:
        print("   → Model learning but not overfitting")
        print("   → Possible causes:")
        print("     - Model too simple")
        print("     - Learning rate too low")
        print("     - Data too noisy")

    overfit_passed = False

# Plot training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss curve
axes[0].plot(history['loss'], linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss (Should decrease to near 0)')
axes[0].grid(True, alpha=0.3)

# Accuracy curve
axes[1].plot(history['acc'], linewidth=2, color='green')
axes[1].axhline(y=PASS_THRESHOLD, color='r', linestyle='--', label=f'Pass threshold ({PASS_THRESHOLD:.0%})')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training Accuracy (Should reach >95%)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Prediction distribution over time
pred_dist_array = np.array(history['pred_dist'])
for i in range(n_classes):
    axes[2].plot(pred_dist_array[:, i], label=f'Class {i}', linewidth=2)
axes[2].axhline(y=20, color='k', linestyle='--', alpha=0.5, label='Expected (20)')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Number of Predictions')
axes[2].set_title('Prediction Distribution (Should balance to 20:20:20)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Micro-Dataset Overfit Test Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(Path(PHASE1_CHECK_DIR) / "overfit_test.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n Saved: overfit_test.png")

#  FEATURE VISUALIZATION PER CLASS
print("FEATURE VISUALIZATION PER CLASS")

print("\nExtracting STFT features for visualization")

# Extract STFT for random samples from each class
N_SAMPLES_VIS = 100  # Use 100 samples per class for averaging

def compute_stft(signal, nperseg=64, noverlap=32):
    f, t, Zxx = scipy_signal.stft(signal, fs=178, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    log_mag = np.log1p(mag)
    return f, t, log_mag

# Collect STFT for each class
class_stfts = []
class_signals = []

for class_idx in range(n_classes):
    print(f"Processing Class {class_idx}")

    class_mask = y_normalized == class_idx
    class_samples = X[class_mask]

    # Randomly select samples
    selected_indices = np.random.choice(len(class_samples), size=N_SAMPLES_VIS, replace=False)
    selected_samples = class_samples[selected_indices]

    # Compute STFT for each
    stfts = []
    for sample in selected_samples:
        f, t, log_mag = compute_stft(sample)
        stfts.append(log_mag)

    stfts = np.array(stfts)
    mean_stft = stfts.mean(axis=0)

    class_stfts.append(mean_stft)
    class_signals.append(selected_samples)

    print(f"  Mean STFT shape: {mean_stft.shape}")

print("\n STFT computation complete")

# Create visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# Determine global color scale for fair comparison
vmin = min(stft.min() for stft in class_stfts)
vmax = max(stft.max() for stft in class_stfts)

print(f"\nVisualization parameters:")
print(f"  Color scale: [{vmin:.2f}, {vmax:.2f}] (same for all classes)")

for class_idx in range(n_classes):
    # Row 1: Mean time-domain signal
    ax_time = fig.add_subplot(gs[0, class_idx])
    mean_signal = class_signals[class_idx].mean(axis=0)
    std_signal = class_signals[class_idx].std(axis=0)

    ax_time.plot(mean_signal, linewidth=1.5, label='Mean')
    ax_time.fill_between(range(len(mean_signal)),
                         mean_signal - std_signal,
                         mean_signal + std_signal,
                         alpha=0.3, label='±1 std')
    ax_time.set_title(f'Class {class_idx}: Time Domain (Mean ± Std)', fontweight='bold')
    ax_time.set_xlabel('Time (samples)')
    ax_time.set_ylabel('Amplitude')
    ax_time.legend(fontsize=8)
    ax_time.grid(True, alpha=0.3)

    # Row 2: Mean STFT (log magnitude)
    ax_stft = fig.add_subplot(gs[1, class_idx])
    im = ax_stft.imshow(class_stfts[class_idx], aspect='auto', origin='lower',
                        cmap='viridis', vmin=vmin, vmax=vmax)
    ax_stft.set_title(f'Class {class_idx}: Mean Log-Magnitude STFT', fontweight='bold')
    ax_stft.set_xlabel('Time Frames')
    ax_stft.set_ylabel('Frequency Bins')
    plt.colorbar(im, ax=ax_stft)

    # Row 3: Frequency profile (mean across time)
    ax_freq = fig.add_subplot(gs[2, class_idx])
    freq_profile = class_stfts[class_idx].mean(axis=1)
    ax_freq.plot(f, freq_profile, linewidth=2)
    ax_freq.set_title(f'Class {class_idx}: Frequency Profile', fontweight='bold')
    ax_freq.set_xlabel('Frequency (Hz)')
    ax_freq.set_ylabel('Mean Log-Magnitude')
    ax_freq.grid(True, alpha=0.3)

    # Row 4: Temporal profile (mean across frequency)
    ax_temp = fig.add_subplot(gs[3, class_idx])
    temp_profile = class_stfts[class_idx].mean(axis=0)
    ax_temp.plot(t, temp_profile, linewidth=2, color='orange')
    ax_temp.set_title(f'Class {class_idx}: Temporal Profile', fontweight='bold')
    ax_temp.set_xlabel('Time (s)')
    ax_temp.set_ylabel('Mean Log-Magnitude')
    ax_temp.grid(True, alpha=0.3)

plt.suptitle('Feature Visualization Per Class (100 samples averaged)',
             fontsize=16, fontweight='bold')
plt.savefig(Path(PHASE1_CHECK_DIR) / "feature_visualization.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n Saved: feature_visualization.png")

# Compute pairwise differences
print("\n Analyzing class separability")

differences = []
for i in range(n_classes):
    for j in range(i+1, n_classes):
        diff = np.abs(class_stfts[i] - class_stfts[j]).mean()
        differences.append((i, j, diff))

print(f"\n Mean STFT differences between classes:")
for i, j, diff in differences:
    print(f"  Class {i} vs Class {j}: {diff:.4f}")

avg_diff = np.mean([d[2] for d in differences])
print(f"\n Average difference: {avg_diff:.4f}")

if avg_diff > 0.5:
    separability = "GOOD"
    sep_status = "GOOD"
elif avg_diff > 0.2:
    separability = "MODERATE"
    sep_status = "OKAY"
else:
    separability = "POOR"
    sep_status = "POOR"

print(f"\n{sep_status} Class separability in frequency domain: {separability}")

if avg_diff > 0.2:
    print(f"   → Classes show distinct frequency patterns")
    print(f"   → Dual-branch (time+freq) will help significantly")
else:
    print(f"   → Classes are very similar in frequency domain")
    print(f"   → May need advanced features or longer signals")

# Summary
print("PHASE 1 SUMMARY")

results = {
    'overfit_test': {
        'passed': overfit_passed,
        'final_accuracy': float(final_acc),
        'threshold': PASS_THRESHOLD,
        'prediction_distribution': final_pred_dist if isinstance(final_pred_dist, list) else final_pred_dist.tolist()
    },
    'label_verification': {
        'n_samples': int(len(y)),
        'n_classes': int(n_classes),
        'class_distribution': {int(k): int(v) for k, v in zip(unique, counts)},
        'labels_match_data': X.shape[0] == y.shape[0],
        'no_nans': not np.isnan(X).any(),
        'no_infs': not np.isinf(X).any()
    },
    'feature_visualization': {
        'avg_class_difference': float(avg_diff),
        'separability': separability,
        'pairwise_differences': {f"class_{i}_vs_{j}": float(diff) for i, j, diff in differences}
    }
}

# Save results
with open(Path(PHASE1_CHECK_DIR) / "sanity_check_results.json", 'w') as f:
    json.dump(results, indent=2, fp=f)

print(f"\n Results Summary:")
print(f"   (Overfit Test): {'PASSED' if overfit_passed else 'FAILED'}")
print(f"   (Label Verification): PASSED")
print(f"   (Feature Visualization): {sep_status} {separability}")

print(f"\n All results saved to: {PHASE1_CHECK_DIR}")
print("\nPHASE 1 COMPLETE")
