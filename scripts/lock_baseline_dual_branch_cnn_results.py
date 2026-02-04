"""
Lock baseline dual-branch CNN results
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import json
import shutil
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

print("Lock baseline dual-branch CNN results")

# Paths
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
PHASE3_DIR = f"{BASE_DIR}/outputs/dual_branch_training"

# Find diagnostic results
diagnostic_dir = Path(PHASE3_DIR) / "results_diagnostic_plain"

if not diagnostic_dir.exists():
    print("\n ERROR: Diagnostic results not found!")
    print(f"   Expected: {diagnostic_dir}")
    exit(1)

print(f"\n Found diagnostic results: {diagnostic_dir}")

# CREATE LOCKED BASELINE DIRECTORY
print("CREATING LOCKED BASELINE")

baseline_dir = Path(PHASE3_DIR) / "LOCKED_BASELINE_RESULTS"
baseline_dir.mkdir(exist_ok=True)

# Timestamp
timestamp = datetime.now().isoformat()

print(f"\n Baseline directory: {baseline_dir}")

# LOAD AND SUMMARIZE RESULTS
print("LOADING DIAGNOSTIC RESULTS")

# Load metrics
with open(diagnostic_dir / "results.json") as f:
    results = json.load(f)

print(f"\n Diagnostic Performance:")
print(f"   Accuracy:      {results['test_accuracy']:.4f}")
print(f"   F1 (macro):    {results['test_f1_macro']:.4f}")
print(f"   F1 (weighted): {results['test_f1_weighted']:.4f}")

print(f"\n Per-Class Recall:")
for i, recall in enumerate(results['test_per_class_recall']):
    status = "Yes" if recall > 0 else "No"
    print(f"   {status} Class {i}: {recall:.4f}")

print(f"\n Per-Class F1:")
for i, f1 in enumerate(results['test_per_class_f1']):
    print(f"   Class {i}: {f1:.4f}")

all_predicted = results.get('all_classes_predicted', False)
print(f"\n All classes predicted: {all_predicted}")

# SUMMARY
print("CREATING SUMMARY")

# Determine characterization
macro_f1 = results['test_f1_macro']
if macro_f1 > 0.5:
    characterization = "good class separability"
elif macro_f1 > 0.3:
    characterization = "partial class separability"
elif macro_f1 > 0.15:
    characterization = "limited class separability"
else:
    characterization = "minimal class separability"

thesis_summary = f"""
SUMMARY

Date: {timestamp}

APPROACH:
---------
Baseline dual-branch CNN with weighted CrossEntropyLoss

Architecture:
  - Branch A: Temporal CNN (medium depth, 4 conv layers)
    * Conv1D layers: 32 → 64 → 128 → 256 channels
    * Global average pooling
    * 64-dimensional embedding

  - Branch B: Feature encoder (2 dense layers)
    * Input: 30 engineered EEG features
      - Band powers (delta, theta, alpha, beta, gamma)
      - Relative band powers
      - Band ratios (theta/alpha, delta/beta, theta/beta)
      - Spectral entropy, Hjorth parameters
      - Statistical features
    * 32-dimensional embedding

  - Fusion: Concatenation (96-dim) → Classifier
  - Total parameters: {results.get('total_parameters', 'N/A'):,}

Loss Function:
  - Weighted CrossEntropyLoss
  - Balanced class weights
  - No aggressive scaling

Training:
  - Epochs: {results.get('total_epochs', 'N/A')}
  - Time: {results.get('training_time_minutes', 'N/A'):.1f} minutes
  - Early stopping on validation macro F1

RESULTS:
--------
Test Performance:
  - Accuracy:      {results['test_accuracy']:.4f}
  - F1 (macro):    {results['test_f1_macro']:.4f}
  - F1 (weighted): {results['test_f1_weighted']:.4f}

Per-Class Performance:
  - Class 0: Recall={results['test_per_class_recall'][0]:.4f}, F1={results['test_per_class_f1'][0]:.4f}
  - Class 1: Recall={results['test_per_class_recall'][1]:.4f}, F1={results['test_per_class_f1'][1]:.4f}
  - Class 2: Recall={results['test_per_class_recall'][2]:.4f}, F1={results['test_per_class_f1'][2]:.4f}

Prediction Distribution:
  - Class 0: {results['test_prediction_distribution'][0]} predictions
  - Class 1: {results['test_prediction_distribution'][1]} predictions
  - Class 2: {results['test_prediction_distribution'][2]} predictions

All classes predicted: {all_predicted}

INTERPRETATION:
--------------
Baseline dual-branch CNN with weighted CrossEntropy shows {characterization}.

The model demonstrates {"successful" if all_predicted else "limited"} multi-class
learning with macro F1 of {results['test_f1_macro']:.4f}.

{"All three classes receive predictions, indicating the model has learned discriminative patterns for each class." if all_predicted else "Some classes are not being predicted, suggesting the need for improved representation learning."}

Key observations:
  - Temporal CNN (Branch A) captures local signal patterns
  - Feature encoder (Branch B) preserves domain knowledge
  - Fusion layer combines both representations
  - {"Performance is balanced across classes" if max(results['test_per_class_f1']) - min(results['test_per_class_f1']) < 0.3 else "Performance varies significantly across classes"}

LIMITATIONS:
-----------
  - CNN-only temporal learning may miss long-range dependencies
  - No recurrent mechanism to capture temporal dynamics
  - EEG signals have inherent temporal structure not fully exploited

NEXT STEPS:
----------
  - Add BiLSTM to Branch A for temporal dependency modeling
  - This represents a shift to representation learning
  - Expected improvement in minority class performance


CONFUSION MATRIX:
----------------
{np.array(results['test_confusion_matrix'])}

Normalized (by true class):
{(np.array(results['test_confusion_matrix']).astype('float') / np.array(results['test_confusion_matrix']).sum(axis=1)[:, np.newaxis])}
"""

# Save thesis summary
with open(baseline_dir / "SUMMARY.txt", 'w') as f:
    f.write(thesis_summary)

print(f" Created: SUMMARY.txt")

# COPY KEY FILES
print("COPYING KEY FILES")

# Copy results
shutil.copy(diagnostic_dir / "results.json", baseline_dir / "results.json")
print(f" Copied: results.json")

# Copy confusion matrix visualization
if (diagnostic_dir / "results.png").exists():
    shutil.copy(diagnostic_dir / "results.png", baseline_dir / "results.png")
    print(f" Copied: results.png")

# Copy model if exists
if (diagnostic_dir / "model.pt").exists():
    shutil.copy(diagnostic_dir / "model.pt", baseline_dir / "baseline_model.pt")
    print(f" Copied: baseline_model.pt")

# Copy classification report if exists
if (diagnostic_dir / "classification_report.txt").exists():
    shutil.copy(diagnostic_dir / "classification_report.txt",
                baseline_dir / "classification_report.txt")
    print(f" Copied: classification_report.txt")

# VISUALIZATIONS
print("VISUALIZATIONS")
cm = np.array(results['test_confusion_matrix'])
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion matrix (raw counts)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            cbar_kws={'label': 'Count'})
axes[0].set_xlabel('Predicted Class', fontsize=12)
axes[0].set_ylabel('True Class', fontsize=12)
axes[0].set_title('Confusion Matrix\n(Absolute Counts)', fontsize=14, fontweight='bold')

# Confusion matrix (normalized)
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[1],
            vmin=0, vmax=1, cbar_kws={'label': 'Proportion'})
axes[1].set_xlabel('Predicted Class', fontsize=12)
axes[1].set_ylabel('True Class', fontsize=12)
axes[1].set_title('Confusion Matrix\n(Normalized by True Class)', fontsize=14, fontweight='bold')

# Performance bar chart
classes = ['Class 0', 'Class 1', 'Class 2']
recalls = results['test_per_class_recall']
f1_scores = results['test_per_class_f1']

x = np.arange(len(classes))
width = 0.35

bars1 = axes[2].bar(x - width/2, recalls, width, label='Recall', alpha=0.8)
bars2 = axes[2].bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8)

axes[2].set_xlabel('Class', fontsize=12)
axes[2].set_ylabel('Score', fontsize=12)
axes[2].set_title('Per-Class Performance', fontsize=14, fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(classes)
axes[2].legend(fontsize=11)
axes[2].set_ylim([0, 1])
axes[2].grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

plt.suptitle(f'Phase 3 Diagnostic Baseline Results\nMacro F1: {macro_f1:.4f}',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(baseline_dir / "THESIS_FIGURE_Baseline.png", dpi=300, bbox_inches='tight')
plt.close()

print(f" Created: THESIS_FIGURE_Baseline.png")

# CREATE LOCK FILE
print("CREATING LOCK FILE")

lock_info = {
    'locked_on': timestamp,
    'source_directory': str(diagnostic_dir),
    'baseline_characterization': characterization,
    'macro_f1': macro_f1,
    'all_classes_predicted': all_predicted,
    'total_parameters': results.get('total_parameters', None),
    'training_epochs': results.get('total_epochs', None),
    'training_time_minutes': results.get('training_time_minutes', None),
    'purpose': 'Phase 3 Diagnostic Baseline for thesis comparison',
    'next_experiment': 'Representation Learning with BiLSTM',
    'thesis_section': 'Baseline Approach - Dual-Branch CNN',
    'files': [
        'results.json',
        'THESIS_SUMMARY.txt',
        'THESIS_FIGURE_Baseline.png',
        'classification_report.txt',
        'baseline_model.pt'
    ]
}

with open(baseline_dir / "LOCK_INFO.json", 'w') as f:
    json.dump(lock_info, f, indent=2)

print(f" Created: LOCK_INFO.json")
print("STEP 1 COMPLETE - BASELINE LOCKED")
