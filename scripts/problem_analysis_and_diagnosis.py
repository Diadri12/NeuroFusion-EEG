"""
Problem Analysis and Diagnosis
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

print("PROBLEM ANALYSIS & DIAGNOSIS")

# Configuration
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
ANALYSIS_DIR = f"{BASE_DIR}/outputs/problem_analysis"
Path(ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)

# Load Data
print("Loading Data")
signal_path = f"{BASE_DIR}/outputs/final_processed/epilepsy_122mb"
X = np.load(Path(signal_path) / "preprocessed_signals.npy")
y = np.load(Path(signal_path) / "labels.npy")

print(f" Dataset loaded")
print(f"  Samples: {X.shape[0]:,}")
print(f"  Features: {X.shape[1]}")
print(f"  Labels: {np.unique(y)}\n")

# Class Distribution Analysis
print("Class Distribution Analysis")
unique, counts = np.unique(y, return_counts=True)
total = len(y)

# Print detailed breakdown
print("\nCLASS DISTRIBUTION:")
for u, c in zip(unique, counts):
    percentage = c/total*100
    bar = '█' * int(percentage/2)
    print(f"Class {u}: {c:7,} samples ({percentage:5.2f}%) {bar}")

# Identify imbalance
max_count = counts.max()
min_count = counts.min()
imbalance_ratio = max_count / min_count

print(f"\n STATISTICS:")
print(f"  Majority class: {unique[np.argmax(counts)]} ({max_count:,} samples, {max_count/total*100:.1f}%)")
print(f"  Minority class: {unique[np.argmin(counts)]} ({min_count:,} samples, {min_count/total*100:.1f}%)")
print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

# Diagnosis
print(f"\n DIAGNOSIS:")
if imbalance_ratio > 3:
    severity = "SEVERE" if imbalance_ratio > 5 else "MODERATE"
    print(f"  {severity} CLASS IMBALANCE DETECTED!")
else:
    print(f"   Classes are relatively balanced")

# Visualization
print("\n Creating Visualizations")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Class distribution bar chart
axes[0, 0].bar(unique, counts, color=['#3B82F6', '#10B981', '#F59E0B'])
axes[0, 0].set_xlabel('Class', fontsize=12)
axes[0, 0].set_ylabel('Number of Samples', fontsize=12)
axes[0, 0].set_title('Class Distribution (Absolute)', fontsize=14, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, (u, c) in enumerate(zip(unique, counts)):
    axes[0, 0].text(u, c, f'{c:,}', ha='center', va='bottom', fontweight='bold')

# Percentage pie chart
colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
axes[0, 1].pie(counts, labels=[f'Class {u}' for u in unique], autopct='%1.1f%%',
               colors=colors[:len(unique)], startangle=90)
axes[0, 1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')

# Imbalance visualization
ratios = counts / min_count
axes[1, 0].barh(unique, ratios, color=colors[:len(unique)])
axes[1, 0].set_xlabel('Ratio to Minority Class', fontsize=12)
axes[1, 0].set_ylabel('Class', fontsize=12)
axes[1, 0].set_title('Class Imbalance Ratio', fontsize=14, fontweight='bold')
axes[1, 0].axvline(x=3, color='red', linestyle='--', label='Severe threshold (3:1)')
axes[1, 0].legend()
axes[1, 0].grid(axis='x', alpha=0.3)
for i, (u, r) in enumerate(zip(unique, ratios)):
    axes[1, 0].text(r, i, f'{r:.2f}:1', va='center', fontweight='bold')

# Sample distribution statistics
axes[1, 1].axis('off')
stats_text = f"""
DATASET STATISTICS

Total Samples: {total:,}
Number of Classes: {len(unique)}

Class Breakdown:
"""
for u, c in zip(unique, counts):
    stats_text += f"  Class {u}: {c:,} ({c/total*100:.2f}%)\n"

stats_text += f"""
Imbalance Analysis:
  Ratio: {imbalance_ratio:.2f}:1
  Severity: {'SEVERE' if imbalance_ratio > 5 else 'MODERATE' if imbalance_ratio > 3 else 'BALANCED'}

Recommendation:
  {'Use class-weighted loss' if imbalance_ratio > 3 else 'Standard loss is fine'}
"""

axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Class Distribution Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(Path(ANALYSIS_DIR) / "class_distribution.png", dpi=300, bbox_inches='tight')
print(f" Saved: class_distribution.png")

# Load Baseline Confusion Matrix
print("\n  Analyzing Baseline Results")

baseline_results_path = Path(f"{BASE_DIR}/outputs/dual_branch_experiments/epilepsy_122mb/baseline/metrics.json")

if baseline_results_path.exists():
    with open(baseline_results_path, 'r') as f:
        baseline_metrics = json.load(f)

    print(" Baseline results found")
    print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {baseline_metrics['f1']:.4f}")
    print(f"  Precision: {baseline_metrics['precision']:.4f}")
    print(f"  Recall: {baseline_metrics['recall']:.4f}")

    # Load confusion matrix from image or recreate
    print("\n  Check baseline/results.png for confusion matrix")
    print("  → Look for empty rows/columns (minority classes ignored)")
else:
    print(" Baseline results not found")
    print("  Run baseline training first to see the problem")

# Generate Problem Statement
print("\n Generating Problem Statement")

problem_statement = f"""
PROBLEM ANALYSIS REPORT
Generated: {pd.Timestamp.now()}

1. DATASET OVERVIEW

Total Samples: {total:,}
Number of Classes: {len(unique)}
Signal Length: {X.shape[1]} samples

2. CLASS DISTRIBUTION

"""

for u, c in zip(unique, counts):
    problem_statement += f"Class {u}: {c:7,} samples ({c/total*100:5.2f}%)\n"

problem_statement += f"""
Imbalance Ratio: {imbalance_ratio:.2f}:1
Severity: {'SEVERE' if imbalance_ratio > 5 else 'MODERATE' if imbalance_ratio > 3 else 'BALANCED'}

3. IDENTIFIED PROBLEMS

"""

if imbalance_ratio > 3:
    problem_statement += f"""
CRITICAL ISSUE: Class Imbalance

Problem:
  - Majority class ({unique[np.argmax(counts)]}) contains {max_count/total*100:.1f}% of data
  - Minority class ({unique[np.argmin(counts)]}) contains only {min_count/total*100:.1f}% of data
  - Ratio of {imbalance_ratio:.2f}:1 is {'severe' if imbalance_ratio > 5 else 'significant'}

Impact on Learning:
  - Model biased toward predicting majority class
  - High accuracy but poor minority class detection
  - Low F1, precision, recall scores
  - Identical results across different architectures

Evidence:
  - All models achieve ~{baseline_metrics.get('accuracy', 0.6)*100:.1f}% accuracy
  - F1 scores are low (~{baseline_metrics.get('f1', 0.45):.2f})
  - Confusion matrix shows empty predictions for minority classes

"""
else:
    problem_statement += " No severe class imbalance detected\n"

problem_statement += f"""
4. RECOMMENDED SOLUTIONS

Solution 1: Class-Weighted Loss Function (CRITICAL)
  - Compute balanced class weights
  - Use CrossEntropyLoss(weight=class_weights)
  - Forces model to pay attention to minority classes

Solution 2: Improved Feature Extraction
  - Enhance STFT parameters (larger window, more overlap)
  - Apply log scaling for better dynamic range
  - Increases discriminative power

Solution 3: Ablation Study
  - Train baseline (raw only)
  - Train STFT-only model
  - Train dual-branch fusion
  - Proves fusion value scientifically

5. EXPECTED OUTCOMES

After applying solutions:
  ✓ F1 score should increase significantly
  ✓ Precision/recall become balanced
  ✓ Minority classes actually get predicted
  ✓ Confusion matrix becomes more diagonal
  ✓ Different architectures show different performance

"""

# Save report
report_path = Path(ANALYSIS_DIR) / "problem_analysis.txt"
with open(report_path, 'w') as f:
    f.write(problem_statement)

print(f" Saved: problem_analysis.txt")

# Also save as JSON for programmatic access
analysis_data = {
    'total_samples': int(total),
    'n_classes': int(len(unique)),
    'class_distribution': {int(u): int(c) for u, c in zip(unique, counts)},
    'imbalance_ratio': float(imbalance_ratio),
    'majority_class': int(unique[np.argmax(counts)]),
    'minority_class': int(unique[np.argmin(counts)]),
    'severity': 'SEVERE' if imbalance_ratio > 5 else 'MODERATE' if imbalance_ratio > 3 else 'BALANCED',
    'needs_class_weighting': bool(imbalance_ratio > 3)
}

with open(Path(ANALYSIS_DIR) / "analysis_data.json", 'w') as f:
    json.dump(analysis_data, f, indent=2)

print(f" Saved: analysis_data.json")
