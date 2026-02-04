"""
ERROR PATTERN ANALYSIS
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import pandas as pd

print("ERROR PATTERN ANALYSIS")

# Paths
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
PHASE3_DIR = f"{BASE_DIR}/outputs/dual_branch_training"
PHASE4_DIR = f"{BASE_DIR}/outputs/error_diagnosis"
PHASE5_DIR = f"{BASE_DIR}/outputs/comparative_analysis"
error_analysis_dir = Path(PHASE5_DIR) / "error_pattern_analysis"
error_analysis_dir.mkdir(exist_ok=True, parents=True)

# LOAD CONFUSION MATRICES
print("LOADING CONFUSION MATRICES")

# Load Baseline
baseline_path = Path(PHASE3_DIR) / "results_diagnostic_plain" / "results.json"
if not baseline_path.exists():
    print(" ERROR: Baseline results not found!")
    exit(1)

with open(baseline_path) as f:
    baseline_data = json.load(f)

baseline_cm = np.array(
    baseline_data.get('test_confusion_matrix')
    or baseline_data.get('confusion_matrix')
)

baseline_f1 = (
    baseline_data.get('test_f1_macro')
    or baseline_data.get('test_performance', {}).get('f1_macro')
)

print(f" Loaded Baseline")
print(f"   Macro F1: {baseline_f1:.4f}")
print(f"   Confusion Matrix:\n{baseline_cm}")

# Load SupCon
supcon_path = Path(PHASE4_DIR) / "contrastive_pretraining_results" / "contrastive_results.json"
if not supcon_path.exists():
    print(" ERROR: SupCon results not found!")
    exit(1)

with open(supcon_path) as f:
    supcon_data = json.load(f)

supcon_cm = np.array(supcon_data['confusion_matrix'])
supcon_f1 = supcon_data['test_performance']['f1_macro']

print(f"\n Loaded SupCon + Balanced")
print(f"   Macro F1: {supcon_f1:.4f}")
print(f"   Confusion Matrix:\n{supcon_cm}")

# NORMALIZE CONFUSION MATRICES
print("NORMALIZING CONFUSION MATRICES")

# Normalize by true class (rows)
baseline_cm_norm = baseline_cm.astype('float') / baseline_cm.sum(axis=1, keepdims=True)
supcon_cm_norm = supcon_cm.astype('float') / supcon_cm.sum(axis=1, keepdims=True)

print("\nBaseline (Normalized):")
print(baseline_cm_norm)

print("\nSupCon (Normalized):")
print(supcon_cm_norm)

# CALCULATE ERROR PATTERNS
print("ANALYZING ERROR PATTERNS")

class_names = ['Class 0', 'Class 1', 'Class 2']

# Calculate misclassification rates
baseline_errors = 1 - np.diag(baseline_cm_norm)
supcon_errors = 1 - np.diag(supcon_cm_norm)

print(f"\n Per-Class Error Rates:")
print(f"{'Class':<10} {'Baseline':<12} {'SupCon':<12} {'Change':<12} {'Status'}")

error_improvements = []
for i in range(3):
    change = supcon_errors[i] - baseline_errors[i]
    status = "Improved" if change < -0.01 else "Worse" if change > 0.01 else "→ Similar"
    print(f"{class_names[i]:<10} {baseline_errors[i]:<12.4f} {supcon_errors[i]:<12.4f} "
          f"{change:+.4f}      {status}")
    error_improvements.append({
        'class': class_names[i],
        'baseline_error': baseline_errors[i],
        'supcon_error': supcon_errors[i],
        'change': change,
        'improved': change < -0.01
    })

# CONFUSION PAIR ANALYSIS
print("CONFUSION PAIR ANALYSIS")
print("\nBaseline Confusion Patterns:")

confusion_pairs = []

for i in range(3):
    for j in range(3):
        if i != j:
            baseline_conf = baseline_cm_norm[i, j]
            supcon_conf = supcon_cm_norm[i, j]
            change = supcon_conf - baseline_conf

            if baseline_conf > 0.1:  # Only report significant confusions
                print(f"   {class_names[i]} → {class_names[j]}: "
                      f"{baseline_conf:.2%} (Baseline) → {supcon_conf:.2%} (SupCon), "
                      f"Change: {change:+.2%}")

                confusion_pairs.append({
                    'true_class': class_names[i],
                    'predicted_class': class_names[j],
                    'baseline_rate': baseline_conf,
                    'supcon_rate': supcon_conf,
                    'change': change,
                    'improved': change < -0.01
                })

# IDENTIFY KEY IMPROVEMENTS AND PERSISTENT ISSUES
print("KEY IMPROVEMENTS & PERSISTENT ISSUES")

# Sort confusion pairs by magnitude of improvement
confusion_pairs_sorted = sorted(confusion_pairs, key=lambda x: x['change'])

improvements = [cp for cp in confusion_pairs_sorted if cp['improved']]
worsened = [cp for cp in confusion_pairs_sorted if cp['change'] > 0.01]
persistent = [cp for cp in confusion_pairs if cp['supcon_rate'] > 0.15 and abs(cp['change']) < 0.01]

print(f"\n IMPROVEMENTS (confusion reduced):")
if improvements:
    for cp in improvements[:3]:  # Top 3
        print(f"   {cp['true_class']} → {cp['predicted_class']}: "
              f"{cp['baseline_rate']:.2%} → {cp['supcon_rate']:.2%} ({cp['change']:+.2%})")
else:
    print("   None significant")

print(f"\n WORSENED (confusion increased):")
if worsened:
    for cp in worsened[:3]:
        print(f"   {cp['true_class']} → {cp['predicted_class']}: "
              f"{cp['baseline_rate']:.2%} → {cp['supcon_rate']:.2%} ({cp['change']:+.2%})")
else:
    print("   None significant")

print(f"\n  PERSISTENT ISSUES (high confusion remains):")
if persistent:
    for cp in persistent:
        print(f"   {cp['true_class']} → {cp['predicted_class']}: "
              f"{cp['supcon_rate']:.2%} (no significant change)")
else:
    print("   None significant")

# VISUALIZATIONS
print("CREATING VISUALIZATIONS")

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Baseline Confusion Matrix
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(baseline_cm_norm, annot=True, fmt='.2%', cmap='Reds', vmin=0, vmax=1,
            xticklabels=class_names, yticklabels=class_names, ax=ax1,
            cbar_kws={'label': 'Proportion'})
ax1.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
ax1.set_ylabel('True Class', fontsize=11, fontweight='bold')
ax1.set_title(f'(a) Baseline\nMacro F1: {baseline_f1:.4f}', fontsize=12, fontweight='bold')

# SupCon Confusion Matrix
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(supcon_cm_norm, annot=True, fmt='.2%', cmap='Reds', vmin=0, vmax=1,
            xticklabels=class_names, yticklabels=class_names, ax=ax2,
            cbar_kws={'label': 'Proportion'})
ax2.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
ax2.set_ylabel('True Class', fontsize=11, fontweight='bold')
ax2.set_title(f'(b) SupCon + Balanced\nMacro F1: {supcon_f1:.4f}', fontsize=12, fontweight='bold')

# Difference Matrix
ax3 = fig.add_subplot(gs[0, 2])
diff_matrix = supcon_cm_norm - baseline_cm_norm
sns.heatmap(diff_matrix, annot=True, fmt='+.2%', cmap='RdYlGn', center=0, vmin=-0.3, vmax=0.3,
            xticklabels=class_names, yticklabels=class_names, ax=ax3,
            cbar_kws={'label': 'Change'})
ax3.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
ax3.set_ylabel('True Class', fontsize=11, fontweight='bold')
ax3.set_title('(c) Improvement\n(SupCon - Baseline)', fontsize=12, fontweight='bold')

# Per-Class Error Rates
ax4 = fig.add_subplot(gs[1, 0])
x = np.arange(3)
width = 0.35

bars1 = ax4.bar(x - width/2, baseline_errors, width, label='Baseline', alpha=0.7, color='#e74c3c')
bars2 = ax4.bar(x + width/2, supcon_errors, width, label='SupCon', alpha=0.7, color='#2ecc71')

ax4.set_xlabel('Class', fontsize=11, fontweight='bold')
ax4.set_ylabel('Error Rate', fontsize=11, fontweight='bold')
ax4.set_title('(d) Per-Class Error Rates', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(class_names)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}', ha='center', va='bottom', fontsize=9)

# Confusion Pair Changes
ax5 = fig.add_subplot(gs[1, 1:])
if confusion_pairs:
    # Create pair labels
    pair_labels = [f"{cp['true_class'][6]} → {cp['predicted_class'][6]}"
                   for cp in confusion_pairs]
    changes = [cp['change'] for cp in confusion_pairs]

    colors_pairs = ['green' if c < 0 else 'red' for c in changes]

    bars = ax5.barh(range(len(pair_labels)), changes, color=colors_pairs, alpha=0.7)
    ax5.set_yticks(range(len(pair_labels)))
    ax5.set_yticklabels(pair_labels, fontsize=9)
    ax5.set_xlabel('Change in Confusion Rate (SupCon - Baseline)', fontsize=11, fontweight='bold')
    ax5.set_title('(e) Confusion Pattern Changes', fontsize=12, fontweight='bold')
    ax5.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax5.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, change) in enumerate(zip(bars, changes)):
        label_x = change + 0.005 if change > 0 else change - 0.005
        ha = 'left' if change > 0 else 'right'
        ax5.text(label_x, bar.get_y() + bar.get_height()/2.,
                f'{change:+.2%}', ha=ha, va='center', fontsize=8, fontweight='bold')

plt.suptitle('Error Pattern Analysis: Baseline vs SupCon + Balanced',
             fontsize=15, fontweight='bold')
plt.savefig(error_analysis_dir / "error_pattern_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved: error_pattern_analysis.png")

print(f"\n Key insights:")
if improvements:
    print(f"   Best improvement: {improvements[0]['true_class']} → {improvements[0]['predicted_class']}")
    print(f"      {improvements[0]['baseline_rate']:.2%} → {improvements[0]['supcon_rate']:.2%}")

if persistent:
    print(f"   Persistent issue: {persistent[0]['true_class']} → {persistent[0]['predicted_class']}")
    print(f"      Remains at {persistent[0]['supcon_rate']:.2%}")

print(f"\n Error pattern analysis complete!")
