"""
DEFINE DUAL-BRANCH ARCHITECTURE
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

print("DUAL-BRANCH ARCHITECTURE DESIGN")

# Configuration
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
PHASE3_DIR = f"{BASE_DIR}/outputs/dual_branch_training"
frozen_dir = Path(PHASE3_DIR) / "frozen"

# Load frozen data info
with open(frozen_dir / "FREEZE_MANIFEST.json", 'r') as f:
    manifest = json.load(f)

signal_length = manifest['data_summary']['signal_length']
n_features = manifest['data_summary']['n_features']
n_classes = manifest['data_summary']['n_classes']

print(f" Input specifications from frozen experiments:")
print(f"   Signal length: {signal_length}")
print(f"   Feature count: {n_features}")
print(f"   Output classes: {n_classes}")

# BRANCH A: TEMPORAL SIGNAL LEARNER
print("BRANCH A: TEMPORAL SIGNAL LEARNER")

print("\n Purpose: Learn temporal patterns from raw windowed signals")
print(" Detect local seizure patterns, waveform morphologies")

branch_a_design = {
    'name': 'Temporal Signal Learner',
    'input_shape': (signal_length, 1),
    'architecture': 'Conv1D CNN',
    'purpose': 'Learn local temporal patterns',
    'layers': []
}

# Design Conv blocks
print(f"\n Architecture design:")
print(f"\n   Input: ({signal_length}, 1) - Raw windowed signal")

# Conv Block 1
print(f"\n   Block 1: Initial feature extraction")
print(f"      Conv1D(in=1, out=32, kernel=7, stride=2, padding=3)")
print(f"      → BatchNorm1d(32)")
print(f"      → ReLU()")
print(f"      → MaxPool1d(2)")
print(f"      Output shape: ({signal_length//4}, 32)")
print(f"      Purpose: Capture basic waveform shapes")

branch_a_design['layers'].append({
    'block': 1,
    'type': 'conv',
    'in_channels': 1,
    'out_channels': 32,
    'kernel_size': 7,
    'stride': 2,
    'purpose': 'Basic waveform shapes',
    'output_length': signal_length // 4
})

# Conv Block 2
conv2_length = signal_length // 4 // 2
print(f"\n   Block 2: Mid-level features")
print(f"      Conv1D(in=32, out=64, kernel=5, stride=1, padding=2)")
print(f"      → BatchNorm1d(64)")
print(f"      → ReLU()")
print(f"      → MaxPool1d(2)")
print(f"      Output shape: ({conv2_length}, 64)")
print(f"      Purpose: Pattern combinations")

branch_a_design['layers'].append({
    'block': 2,
    'type': 'conv',
    'in_channels': 32,
    'out_channels': 64,
    'kernel_size': 5,
    'stride': 1,
    'purpose': 'Pattern combinations',
    'output_length': conv2_length
})

# Conv Block 3
conv3_length = conv2_length // 2
print(f"\n   Block 3: High-level features")
print(f"      Conv1D(in=64, out=128, kernel=3, stride=1, padding=1)")
print(f"      → BatchNorm1d(128)")
print(f"      → ReLU()")
print(f"      → MaxPool1d(2)")
print(f"      Output shape: ({conv3_length}, 128)")
print(f"      Purpose: Abstract temporal patterns")

branch_a_design['layers'].append({
    'block': 3,
    'type': 'conv',
    'in_channels': 64,
    'out_channels': 128,
    'kernel_size': 3,
    'stride': 1,
    'purpose': 'Abstract patterns',
    'output_length': conv3_length
})

# Global Average Pooling
print(f"\n   Global Average Pooling:")
print(f"      AdaptiveAvgPool1d(1)")
print(f"      Output shape: (128,)")
print(f"      Purpose: Aggregate temporal information")

branch_a_design['layers'].append({
    'block': 'gap',
    'type': 'pooling',
    'output_dim': 128,
    'purpose': 'Temporal aggregation'
})

# Embedding layer
embedding_dim_a = 64
print(f"\n   Embedding layer:")
print(f"      Linear(128 → {embedding_dim_a})")
print(f"      → ReLU()")
print(f"      → Dropout(0.3)")
print(f"      Output: {embedding_dim_a}-dimensional learned representation")

branch_a_design['embedding_dim'] = embedding_dim_a
branch_a_design['total_params_estimate'] = (
    1*32*7 + 32*64*5 + 64*128*3 + 128*embedding_dim_a
)

print(f"\n    Branch A Summary:")
print(f"      Input: {signal_length} samples")
print(f"      Output: {embedding_dim_a}-dim embedding")
print(f"      Params: ~{branch_a_design['total_params_estimate']:,}")
print(f"      Depth: 3 conv layers + GAP + embedding")

# BRANCH B: PHYSIOLOGICAL FEATURE ENCODER
print("BRANCH B: PHYSIOLOGICAL FEATURE ENCODER")

print("\n Purpose: Preserve domain knowledge")
print(" Stabilize training with expert-crafted features")

branch_b_design = {
    'name': 'Physiological Feature Encoder',
    'input_shape': (n_features,),
    'architecture': 'Shallow MLP',
    'purpose': 'Encode domain knowledge',
    'layers': []
}

print(f"\n Architecture design:")
print(f"\n   Input: ({n_features},) - Engineered EEG features")

# Hidden layer 1
hidden_dim = 64
print(f"\n   Layer 1: Feature compression")
print(f"      Linear({n_features} → {hidden_dim})")
print(f"      → BatchNorm1d({hidden_dim})")
print(f"      → ReLU()")
print(f"      → Dropout(0.3)")
print(f"      Purpose: Compress while preserving information")

branch_b_design['layers'].append({
    'layer': 1,
    'type': 'dense',
    'in_features': n_features,
    'out_features': hidden_dim,
    'purpose': 'Feature compression'
})

# Embedding layer
embedding_dim_b = 32
print(f"\n   Layer 2: Embedding")
print(f"      Linear({hidden_dim} → {embedding_dim_b})")
print(f"      → ReLU()")
print(f"      → Dropout(0.2)")
print(f"      Output: {embedding_dim_b}-dimensional feature embedding")

branch_b_design['layers'].append({
    'layer': 2,
    'type': 'dense',
    'in_features': hidden_dim,
    'out_features': embedding_dim_b,
    'purpose': 'Feature embedding'
})

branch_b_design['embedding_dim'] = embedding_dim_b
branch_b_design['total_params_estimate'] = (
    n_features*hidden_dim + hidden_dim*embedding_dim_b
)

print(f"\n    Branch B Summary:")
print(f"      Input: {n_features} features")
print(f"      Output: {embedding_dim_b}-dim embedding")
print(f"      Params: ~{branch_b_design['total_params_estimate']:,}")
print(f"      Depth: 2 dense layers (shallow by design)")

# FUSION LAYER: COMBINING LEARNED + HANDCRAFTED
print("FUSION LAYER: COMBINING LEARNED + HANDCRAFTED")

fusion_design = {
    'type': 'concatenation',
    'input_dim_a': embedding_dim_a,
    'input_dim_b': embedding_dim_b,
    'fusion_dim': embedding_dim_a + embedding_dim_b,
    'purpose': 'Combine complementary representations'
}

print(f"\n Fusion strategy:")
print(f"\n   Method: Concatenation (⊕)")
print(f"      Branch A output: {embedding_dim_a}-dim")
print(f"      Branch B output: {embedding_dim_b}-dim")
print(f"      Fused: {fusion_design['fusion_dim']}-dim")

# CLASSIFIER HEAD
print("CLASSIFIER HEAD")

classifier_design = {
    'input_dim': fusion_design['fusion_dim'],
    'hidden_dim': 32,
    'output_dim': n_classes,
    'layers': []
}

print(f"\n Classifier design:")
print(f"\n   Input: {fusion_design['fusion_dim']}-dim fused embedding")

print(f"\n   Layer 1: Final representation")
print(f"      Linear({fusion_design['fusion_dim']} → {classifier_design['hidden_dim']})")
print(f"      → ReLU()")
print(f"      → Dropout(0.4)")

classifier_design['layers'].append({
    'layer': 1,
    'in_features': fusion_design['fusion_dim'],
    'out_features': classifier_design['hidden_dim'],
    'purpose': 'Final representation'
})

print(f"\n   Output layer:")
print(f"      Linear({classifier_design['hidden_dim']} → {n_classes})")
print(f"      Output: Class logits")

classifier_design['layers'].append({
    'layer': 'output',
    'in_features': classifier_design['hidden_dim'],
    'out_features': n_classes,
    'purpose': 'Classification'
})

# COMPLETE DUAL-BRANCH ARCHITECTURE
print("COMPLETE DUAL-BRANCH ARCHITECTURE")

architecture_spec = {
    'name': 'EEG Dual-Branch Classifier',
    'branch_a': branch_a_design,
    'branch_b': branch_b_design,
    'fusion': fusion_design,
    'classifier': classifier_design,
    'total_parameters': (
        branch_a_design['total_params_estimate'] +
        branch_b_design['total_params_estimate'] +
        fusion_design['fusion_dim'] * classifier_design['hidden_dim'] +
        classifier_design['hidden_dim'] * n_classes
    )
}

print(f"\n Architecture Summary:")
print(f"\n   Branch A (Temporal Learner):")
print(f"      Input: ({signal_length}, 1)")
print(f"      Layers: 3 Conv1D + GAP + Embedding")
print(f"      Output: {embedding_dim_a}-dim")
print(f"      Params: ~{branch_a_design['total_params_estimate']:,}")

print(f"\n   Branch B (Feature Encoder):")
print(f"      Input: ({n_features},)")
print(f"      Layers: 2 Dense")
print(f"      Output: {embedding_dim_b}-dim")
print(f"      Params: ~{branch_b_design['total_params_estimate']:,}")

print(f"\n   Fusion:")
print(f"      Method: Concatenation")
print(f"      Output: {fusion_design['fusion_dim']}-dim")

print(f"\n   Classifier:")
print(f"      Layers: 2 Dense")
print(f"      Output: {n_classes} classes")

print(f"\n   Total estimated parameters: ~{architecture_spec['total_parameters']:,}")

# Design Rationale
print("DESIGN RATIONALE")

rationale = f"""
DUAL-BRANCH ARCHITECTURE RATIONALE

1. Why Two Branches?
   - Phase 2 showed {manifest['phase2_results']['separability'].lower()} statistical separability
   - Handcrafted features alone are insufficient
   - Raw signals contain patterns beyond statistical features
   - Dual approach combines domain knowledge (stable) + learned patterns (flexible)

2. Branch A Design (Temporal Learner):
   - 3 conv layers: Hierarchical feature learning
     * Layer 1 (kernel=7): Basic waveform shapes
     * Layer 2 (kernel=5): Pattern combinations
     * Layer 3 (kernel=3): Abstract temporal patterns
   - Global Average Pooling: Translation-invariant representation
   - Output: {embedding_dim_a}-dim learned embedding

3. Branch B Design (Feature Encoder):
   - Only 2 layers: Features already meaningful
   - Shallow to avoid overfitting on handcrafted features
   - Purpose: Compress and preserve domain knowledge
   - Output: {embedding_dim_b}-dim feature embedding

4. Fusion Strategy:
   - Concatenation allows classifier to learn optimal weighting
   - Preserves both representations without assuming equal importance
   - Simple and effective for first iteration

5. Model Capacity:
   - ~{architecture_spec['total_parameters']:,} parameters
   - With {manifest['data_summary']['n_samples']:,} samples
   - Ratio: ~{manifest['data_summary']['n_samples'] / architecture_spec['total_parameters']:.1f} samples per parameter
   - Sufficient to prevent overfitting with proper regularization

6. Regularization Strategy:
   - Batch normalization in all branches
   - Dropout (0.2-0.4) increasing toward output
   - Early stopping on validation macro F1
   - No layer freezing - end-to-end training
"""

print(rationale)

# Save rationale
with open(Path(PHASE3_DIR) / "ARCHITECTURE_RATIONALE.txt", 'w') as f:
    f.write(rationale)

# SAVING ARCHITECTURE SPECIFICATION
print("SAVING ARCHITECTURE SPECIFICATION")

# Save detailed spec
spec_path = Path(PHASE3_DIR) / "architecture_spec.json"
with open(spec_path, 'w') as f:
    json.dump(architecture_spec, f, indent=2)

print(f"Saved: {spec_path}")

# Create Architecture Diagram
print("CREATING ARCHITECTURE DIAGRAM")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.5, 'EEG Dual-Branch Architecture',
        ha='center', fontsize=18, fontweight='bold')

# Branch A
ax.text(2.5, 10.5, 'Branch A: Temporal Learner',
        ha='center', fontsize=12, fontweight='bold', color='blue')

# Branch A boxes
branch_a_y = 10
boxes_a = [
    (f'Input\n({signal_length}, 1)', 9.5),
    ('Conv1D\n32 filters', 8.5),
    ('Conv1D\n64 filters', 7.5),
    ('Conv1D\n128 filters', 6.5),
    ('GAP', 5.5),
    (f'Embedding\n{embedding_dim_a}-dim', 4.5)
]

for label, y in boxes_a:
    box = FancyBboxPatch((1.5, y-0.3), 2, 0.6,
                         boxstyle="round,pad=0.1",
                         edgecolor='blue', facecolor='lightblue', linewidth=2)
    ax.add_patch(box)
    ax.text(2.5, y, label, ha='center', va='center', fontsize=9)

    # Arrow to next
    if y > 4.5:
        ax.arrow(2.5, y-0.35, 0, -0.5, head_width=0.2, head_length=0.1,
                fc='blue', ec='blue')

# Branch B
ax.text(7.5, 10.5, 'Branch B: Feature Encoder',
        ha='center', fontsize=12, fontweight='bold', color='green')

# Branch B boxes
boxes_b = [
    (f'Features\n({n_features},)', 9.5),
    (f'Dense\n{hidden_dim} units', 8),
    (f'Embedding\n{embedding_dim_b}-dim', 6.5)
]

for label, y in boxes_b:
    box = FancyBboxPatch((6.5, y-0.3), 2, 0.6,
                         boxstyle="round,pad=0.1",
                         edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(box)
    ax.text(7.5, y, label, ha='center', va='center', fontsize=9)

    # Arrow to next
    if y > 6.5:
        next_y = 6.5 if y == 8 else 8
        ax.arrow(7.5, y-0.35, 0, -(y-next_y-0.4), head_width=0.2, head_length=0.1,
                fc='green', ec='green')

# Fusion
ax.text(5, 3.5, 'Fusion (Concatenation)',
        ha='center', fontsize=12, fontweight='bold', color='purple')

fusion_box = FancyBboxPatch((3.5, 2.7), 3, 0.6,
                           boxstyle="round,pad=0.1",
                           edgecolor='purple', facecolor='plum', linewidth=2)
ax.add_patch(fusion_box)
ax.text(5, 3, f'{fusion_design["fusion_dim"]}-dim Fused',
        ha='center', va='center', fontsize=10)

# Arrows to fusion
ax.arrow(2.5, 4.15, 1.5, -1, head_width=0.2, head_length=0.1,
        fc='blue', ec='blue', linestyle='--')
ax.arrow(7.5, 6.15, -1.5, -2.8, head_width=0.2, head_length=0.1,
        fc='green', ec='green', linestyle='--')

# Classifier
ax.text(5, 2, 'Classifier',
        ha='center', fontsize=12, fontweight='bold', color='red')

classifier_box = FancyBboxPatch((3.5, 1), 3, 0.6,
                               boxstyle="round,pad=0.1",
                               edgecolor='red', facecolor='lightcoral', linewidth=2)
ax.add_patch(classifier_box)
ax.text(5, 1.3, f'Dense → {n_classes} classes',
        ha='center', va='center', fontsize=10)

# Arrow to classifier
ax.arrow(5, 2.65, 0, -0.95, head_width=0.2, head_length=0.1,
        fc='purple', ec='purple')

# Add legend
legend_elements = [
    mpatches.Patch(color='lightblue', label='Temporal Learning'),
    mpatches.Patch(color='lightgreen', label='Feature Encoding'),
    mpatches.Patch(color='plum', label='Fusion'),
    mpatches.Patch(color='lightcoral', label='Classification')
]
ax.legend(handles=legend_elements, loc='lower center', ncol=4,
         frameon=True, fontsize=9)

plt.tight_layout()
plt.savefig(Path(PHASE3_DIR) / "architecture_diagram.png", dpi=300, bbox_inches='tight')
plt.close()

print(f" Saved: architecture_diagram.png")
print("\n COMPLETE - ARCHITECTURE DESIGNED ")
