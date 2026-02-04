"""
FINAL MODEL SELECTION
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("FINAL MODEL SELECTION")
print("\n Selecting final model with scientific justification\n")

# Paths
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
PHASE3_DIR = f"{BASE_DIR}/outputs/dual_branch_training"
PHASE4_DIR = f"{BASE_DIR}/outputs/error_diagnosis"
PHASE5_DIR = f"{BASE_DIR}/outputs/comparative_analysis"
selection_dir = Path(PHASE5_DIR) / "final_model_selection"
selection_dir.mkdir(exist_ok=True, parents=True)

# LOAD ALL MODELS
print("LOADING ALL MODELS")

candidates = {}

# Baseline
baseline_path = Path(PHASE3_DIR) / "results_diagnostic_plain" / "results.json"
if baseline_path.exists():
    with open(baseline_path) as f:
        data = json.load(f)
    candidates['Baseline (Weighted CE)'] = {
        'macro_f1': data['test_f1_macro'],
        'accuracy': data['test_accuracy'],
        'per_class_recall': data['test_per_class_recall'],
        'per_class_f1': data['test_per_class_f1'],
        'all_predicted': data.get('all_classes_predicted', False),
        'approach': 'Dual-branch CNN + Weighted CE',
        'model_file': str(Path(PHASE3_DIR) / "results_diagnostic_plain" / "model.pt")
    }

    print(f" Baseline: F1={candidates['Baseline (Weighted CE)']['macro_f1']:.4f}")

# Balanced Sampling
balanced_path = Path(PHASE4_DIR) / "balanced_sampling_results" / "balanced_sampling_results.json"
if balanced_path.exists():
    with open(balanced_path) as f:
        data = json.load(f)
    candidates['Balanced Sampling'] = {
        'macro_f1': data['test_performance']['f1_macro'],
        'accuracy': data['test_performance']['accuracy'],
        'per_class_recall': [
            data['per_class_metrics'][f'class_{i}']['recall'] for i in range(3)
        ],
        'per_class_f1': [
            data['per_class_metrics'][f'class_{i}']['f1'] for i in range(3)
        ],
        'all_predicted': data.get('all_classes_predicted', False),
        'approach': 'Dual-branch CNN + Balanced Sampling',
        'model_file': str(Path(PHASE4_DIR) / "balanced_sampling_results" / "balanced_model.pt")
    }

    print(f" Balanced Sampling: F1={candidates['Balanced Sampling']['macro_f1']:.4f}")

# SupCon + Balanced
supcon_path = Path(PHASE4_DIR) / "contrastive_pretraining_results" / "contrastive_results.json"
if supcon_path.exists():
    with open(supcon_path) as f:
        data = json.load(f)
    candidates['SupCon + Balanced'] = {
        'macro_f1': data['test_performance']['f1_macro'],
        'accuracy': data['test_performance']['accuracy'],
        'per_class_recall': [
            data['per_class_metrics'][f'class_{i}']['recall'] for i in range(3)
        ],
        'per_class_f1': [
            data['per_class_metrics'][f'class_{i}']['f1'] for i in range(3)
        ],
        'all_predicted': data.get('all_classes_predicted', False),
        'approach': 'Contrastive Pretraining + Balanced Sampling',
        'model_file': str(Path(PHASE4_DIR) / "contrastive_pretraining_results" / "final_model.pt")
    }

    print(f" SupCon + Balanced: F1={candidates['SupCon + Balanced']['macro_f1']:.4f}")
print(f"\n Loaded {len(candidates)} candidate models")

# EVALUATION CRITERIA
print("EVALUATION CRITERIA")

criteria_weights = {
    'macro_f1': 0.25,          # Raw performance (25%)
    'class_balance': 0.30,      # Balance across classes (30%)
    'methodological_rigor': 0.25,  # Scientific justification (25%)
    'practical_applicability': 0.20  # Real-world usability (20%)
}

print("\n Evaluation criteria (weighted):")
for criterion, weight in criteria_weights.items():
    print(f"   {criterion.replace('_', ' ').title()}: {weight*100:.0f}%")

# SCORE MODELS
print("SCORING MODELS")

scores = {}

for name, data in candidates.items():
    score_breakdown = {}

    # Macro F1 Score (normalized to 0-1)
    max_f1 = max([c['macro_f1'] for c in candidates.values()])
    score_breakdown['macro_f1'] = data['macro_f1'] / max_f1 if max_f1 > 0 else 0

    # Class Balance Score (inverse of recall variance)
    recall_variance = np.var(data['per_class_recall'])
    # Lower variance = better balance, normalize inversely
    max_variance = max([np.var(c['per_class_recall']) for c in candidates.values()])
    score_breakdown['class_balance'] = (1 - recall_variance / max_variance) if max_variance > 0 else 1

    # Bonus for predicting all classes
    if data['all_predicted']:
        score_breakdown['class_balance'] = min(1.0, score_breakdown['class_balance'] * 1.2)

    # Methodological Rigor Score
    if 'SupCon' in name:
        score_breakdown['methodological_rigor'] = 1.0  # Most advanced
    elif 'Balanced' in name:
        score_breakdown['methodological_rigor'] = 0.7  # Data-level intervention
    else:
        score_breakdown['methodological_rigor'] = 0.5  # Baseline

    # Practical Applicability Score
    # Based on: all classes predicted, balanced performance
    if data['all_predicted']:
        min_recall = min(data['per_class_recall'])
        score_breakdown['practical_applicability'] = min_recall * 2  # Scale up
    else:
        score_breakdown['practical_applicability'] = 0.0  # Unusable if not predicting all classes

    # Calculate weighted total
    total_score = sum(score_breakdown[criterion] * criteria_weights[criterion]
                     for criterion in criteria_weights.keys())

    scores[name] = {
        'breakdown': score_breakdown,
        'total': total_score
    }

    print(f"\n{name}:")
    print(f"   Macro F1 Score:         {score_breakdown['macro_f1']:.3f}")
    print(f"   Class Balance Score:    {score_breakdown['class_balance']:.3f}")
    print(f"   Methodological Rigor:   {score_breakdown['methodological_rigor']:.3f}")
    print(f"   Practical Application:  {score_breakdown['practical_applicability']:.3f}")
    print(f"   ─────────────────────")
    print(f"   TOTAL SCORE:            {total_score:.3f}")

# SELECT FINAL MODEL
print("FINAL MODEL SELECTION")
# Find best by total score
recommended_model = max(scores.items(), key=lambda x: x[1]['total'])[0]
recommended_score = scores[recommended_model]['total']

# Also identify best raw F1
best_f1_model = max(candidates.items(), key=lambda x: x[1]['macro_f1'])[0]
best_f1_value = candidates[best_f1_model]['macro_f1']

print(f"\n RECOMMENDED FINAL MODEL:")
print(f"   {recommended_model}")
print(f"   Total Score: {recommended_score:.3f}")
print(f"   Macro F1: {candidates[recommended_model]['macro_f1']:.4f}")
print(f"   All Classes: {'Yes' if candidates[recommended_model]['all_predicted'] else 'No'}")

if recommended_model != best_f1_model:
    print(f"\n NOTE: This is NOT the highest F1 model")
    print(f"   Highest F1: {best_f1_model} ({best_f1_value:.4f})")
    print(f"   BUT: Scientific criteria favor {recommended_model}")
    print(f"   Reason: Better balance + methodological justification")

#  CREATE SELECTION SUMMARY
print("CREATING SELECTION SUMMARY")

# Create comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Evaluation Scores Radar Chart
ax1 = axes[0, 0]
categories = list(criteria_weights.keys())
categories_clean = [c.replace('_', ' ').title() for c in categories]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for name in candidates.keys():
    values = [scores[name]['breakdown'][c] for c in categories]
    values += values[:1]

    linestyle = '-' if name == recommended_model else '--'
    linewidth = 3 if name == recommended_model else 1.5
    alpha = 1.0 if name == recommended_model else 0.5

    ax1.plot(angles, values, 'o-', linewidth=linewidth, linestyle=linestyle,
            label=name, alpha=alpha)
    ax1.fill(angles, values, alpha=0.1 if name == recommended_model else 0.05)

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(categories_clean, fontsize=9)
ax1.set_ylim(0, 1)
ax1.set_title('(a) Multi-Criteria Evaluation', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
ax1.grid(True)

# Total Scores
ax2 = axes[0, 1]
names = list(candidates.keys())
total_scores = [scores[name]['total'] for name in names]
colors = ['#2ecc71' if name == recommended_model else '#95a5a6' for name in names]

bars = ax2.barh(range(len(names)), total_scores, color=colors, alpha=0.7)
ax2.set_yticks(range(len(names)))
ax2.set_yticklabels(names, fontsize=9)
ax2.set_xlabel('Total Evaluation Score', fontsize=10, fontweight='bold')
ax2.set_title('(b) Overall Rankings', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 1)
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for bar, score in zip(bars, total_scores):
    ax2.text(score + 0.02, bar.get_y() + bar.get_height()/2.,
            f'{score:.3f}', va='center', fontweight='bold', fontsize=10)

# Performance vs Balance
ax3 = axes[1, 0]
f1_scores = [candidates[name]['macro_f1'] for name in names]
balance_scores = [scores[name]['breakdown']['class_balance'] for name in names]

for i, name in enumerate(names):
    color = '#2ecc71' if name == recommended_model else '#3498db'
    marker = 'o' if candidates[name]['all_predicted'] else 'x'
    size = 300 if name == recommended_model else 150

    ax3.scatter(f1_scores[i], balance_scores[i], s=size, c=color,
               marker=marker, alpha=0.7, edgecolors='black', linewidths=2)
    ax3.annotate(name.split()[0], (f1_scores[i], balance_scores[i]),
                fontsize=8, ha='center', va='bottom')

ax3.set_xlabel('Macro F1', fontsize=10, fontweight='bold')
ax3.set_ylabel('Class Balance Score', fontsize=10, fontweight='bold')
ax3.set_title('(c) Performance vs Balance Tradeoff', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Add quadrant lines
ax3.axvline(x=np.median(f1_scores), color='gray', linestyle='--', alpha=0.5)
ax3.axhline(y=np.median(balance_scores), color='gray', linestyle='--', alpha=0.5)

# Per-Class Recall Comparison
ax4 = axes[1, 1]
class_names = ['Class 0', 'Class 1', 'Class 2']
x = np.arange(len(class_names))
width = 0.25

for i, name in enumerate(names):
    offset = (i - 1) * width
    alpha = 1.0 if name == recommended_model else 0.5
    ax4.bar(x + offset, candidates[name]['per_class_recall'], width,
           label=name, alpha=alpha)

ax4.set_xlabel('Class', fontsize=10, fontweight='bold')
ax4.set_ylabel('Recall', fontsize=10, fontweight='bold')
ax4.set_title('(d) Per-Class Recall Comparison', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(class_names)
ax4.legend(fontsize=8)
ax4.set_ylim(0, 1)
ax4.grid(axis='y', alpha=0.3)

plt.suptitle(f'Final Model Selection: {recommended_model}',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(selection_dir / "final_model_selection.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved: final_model_selection.png")

# Save selection data
selection_data = {
    'selected_model': recommended_model,
    'selection_date': datetime.now().isoformat(),
    'total_score': float(recommended_score),
    'performance_metrics': candidates[recommended_model],
    'evaluation_breakdown': {k: float(v) for k, v in scores[recommended_model]['breakdown'].items()},
    'alternatives_considered': {
        name: {
            'total_score': float(scores[name]['total']),
            'macro_f1': float(data['macro_f1']),
            'all_predicted': data['all_predicted']
        }
        for name, data in candidates.items() if name != recommended_model
    },
    'selection_criteria': criteria_weights
}

with open(selection_dir / "final_model_selection.json", 'w') as f:
    json.dump(selection_data, f, indent=2)

print(f"Saved: final_model_selection.json")
print(f"\n Final model selection complete!")
