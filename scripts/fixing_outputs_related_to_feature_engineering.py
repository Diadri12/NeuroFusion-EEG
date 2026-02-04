"""
FIXING OTPUTS PERTAINING TO WINDOWING AND FEATURE ENGINEERING
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
from pathlib import Path
import json
import shutil
from datetime import datetime

print("FIXING OTPUTS PERTAINING TO WINDOWING AND FEATURE ENGINEERING")

# Configuration
BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
PHASE2_DIR = f"{BASE_DIR}/outputs/feature_engineering"
PHASE3_DIR = f"{BASE_DIR}/outputs/dual_branch_training"
Path(PHASE3_DIR).mkdir(parents=True, exist_ok=True)

# Verifying completeness
print("VERIFYING COMPLETENESS")

required_files = {
    'windowed_signals': Path(PHASE2_DIR) / "windowed_data" / "X_windowed.npy",
    'windowed_labels': Path(PHASE2_DIR) / "windowed_data" / "y_windowed.npy",
    'engineered_features': Path(PHASE2_DIR) / "X_features.npy",
    'feature_labels': Path(PHASE2_DIR) / "y_features.npy",
    'feature_names': Path(PHASE2_DIR) / "feature_names.json",
    'phase2_summary': Path(PHASE2_DIR) / "phase2_summary.json"
}

print("\n Checking required files")
all_present = True
for name, path in required_files.items():
    if path.exists():
        size_mb = path.stat().st_size / (1024**2)
        print(f"  {name:<25} ({size_mb:.1f} MB)")
    else:
        print(f"  {name:<25} MISSING!")
        all_present = False

if not all_present:
    print("\n ERROR: Incomplete!")

print("\n All required files present")

# Load and Validate Data
print("LOADING AND VALIDATING DATA")

# Load data
X_windowed = np.load(required_files['windowed_signals'])
y_windowed = np.load(required_files['windowed_labels'])
X_features = np.load(required_files['engineered_features'])
y_features = np.load(required_files['feature_labels'])

with open(required_files['feature_names'], 'r') as f:
    feature_names = json.load(f)

with open(required_files['phase2_summary'], 'r') as f:
    phase2_summary = json.load(f)

print("\n Data shapes:")
print(f"  Windowed signals:    {X_windowed.shape}")
print(f"  Engineered features: {X_features.shape}")
print(f"  Labels:              {y_windowed.shape}")

# Validation checks
print("\n Running validation checks")

checks_passed = True

# Label alignment
if not np.array_equal(y_windowed, y_features):
    print("  Labels mismatch between windowed and features!")
    checks_passed = False
else:
    print("  Label alignment verified")

# Sample count
if len(X_windowed) != len(X_features):
    print(f"  Sample count mismatch: {len(X_windowed)} vs {len(X_features)}")
    checks_passed = False
else:
    print(f"  Sample count consistent: {len(X_windowed):,}")

# No NaN/Inf in signals
if np.isnan(X_windowed).any() or np.isinf(X_windowed).any():
    print("  NaN/Inf found in windowed signals!")
    checks_passed = False
else:
    print("  No NaN/Inf in windowed signals")

# No NaN/Inf in features
if np.isnan(X_features).any() or np.isinf(X_features).any():
    print("  NaN/Inf found in engineered features!")
    checks_passed = False
else:
    print("  No NaN/Inf in engineered features")

# Feature count matches names
if X_features.shape[1] != len(feature_names):
    print(f"  Feature count mismatch: {X_features.shape[1]} vs {len(feature_names)}")
    checks_passed = False
else:
    print(f"  Feature names match feature count: {len(feature_names)}")

# Class distribution
n_classes = len(np.unique(y_windowed))
if n_classes != 3:
    print(f"  Warning: Expected 3 classes, found {n_classes}")

unique, counts = np.unique(y_windowed, return_counts=True)
min_samples = counts.min()
if min_samples < 100:
    print(f"  Warning: Minimum class has only {min_samples} samples")
else:
    print(f"  All classes have sufficient samples (min: {min_samples:,})")

if not checks_passed:
    print("\n Validation failed!")
    exit(1)

print("\n All validation checks passed")

# Seperability Analysis
print("SEPARABILITY ANALYSIS")

sep = phase2_summary['separability']
avg_dist = sep['avg_normalized_distance']
separability = sep['separability']
ready = sep['ready_for_training']

print(f"\n Statistical Separability:")
print(f"  Average normalized distance: {avg_dist:.4f}")
print(f"  Classification: {separability}")
print(f"  Ready for training: {ready}")

pairwise_distances = sep.get("pairwise_distances", {})

print(f"\n Pairwise class distances:")
for key, value in pairwise_distances.items():
    print(f"  {key}: {value:.4f}")

# Determine motivation for deep learning
if avg_dist > 0.5:
    motivation = "Excellent statistical separation, but deep learning may further improve"
elif avg_dist > 0.2:
    motivation = "Good statistical separation, deep learning will add learned patterns"
elif avg_dist > 0.1:
    motivation = "Moderate statistical separation, deep learning is beneficial"
else:
    motivation = "Insufficient statistical separation, REQUIRING representation learning"

print(f"   {motivation}")

# Create Frozen Snapshot
print("CREATING FROZEN SNAPSHOT")

# Create frozen directory
frozen_dir = Path(PHASE3_DIR) / "frozen"
frozen_dir.mkdir(exist_ok=True)

# Create symlinks (or copies) to Phase 2 outputs
print("\n Creating snapshot")

snapshot_files = {
    'X_windowed.npy': X_windowed,
    'y_windowed.npy': y_windowed,
    'X_features.npy': X_features,
    'y_features.npy': y_features
}

for filename, data in snapshot_files.items():
    np.save(frozen_dir / filename, data)
    print(f"   Saved: {filename}")

# Copy metadata
shutil.copy(required_files['feature_names'], frozen_dir / "feature_names.json")
shutil.copy(required_files['phase2_summary'], frozen_dir / "phase2_summary.json")

print(f"   Saved: feature_names.json")
print(f"   Saved: phase2_summary.json")

# Create Freeze Manifest
print("CREATING FREEZE MANIFEST")

freeze_manifest = {
    'freeze_timestamp': datetime.now().isoformat(),
    'phase2_directory': str(PHASE2_DIR),
    'frozen_snapshot': str(frozen_dir),
    'data_summary': {
        'n_samples': int(len(X_windowed)),
        'signal_length': int(X_windowed.shape[1]),
        'n_features': int(X_features.shape[1]),
        'n_classes': int(n_classes),
        'class_distribution': {int(k): int(v) for k, v in zip(unique, counts)}
    },
    'phase2_results': {
        'avg_normalized_distance': float(avg_dist),
        'separability': separability,
        'ready_for_training': bool(ready)
    },
    'motivation_for_phase3': motivation,
    'locked_files': list(snapshot_files.keys()) + ['feature_names.json', 'phase2_summary.json'],
    'warnings': []
}

# Add warnings if needed
if avg_dist < 0.1:
    freeze_manifest['warnings'].append(
        "Low separability (<0.1) - deep learning will be challenging"
    )
if min_samples < 1000:
    freeze_manifest['warnings'].append(
        f"Low sample count for minority class ({min_samples}) - consider augmentation"
    )

# Save manifest
manifest_path = frozen_dir / "FREEZE_MANIFEST.json"
with open(manifest_path, 'w') as f:
    json.dump(freeze_manifest, f, indent=2)

print(f"\n Freeze manifest created: {manifest_path}")


# Create Lock File
print("CREATING LOCK FILE")

lock_file = Path(PHASE3_DIR) / "LOCKED.txt"
with open(lock_file, 'w') as f:
    f.write(f"Outputs locked on: {datetime.now().isoformat()}\n")
    f.write(f"\n")
    f.write(f"DO NOT MODIFY:\n")
    for file in freeze_manifest['locked_files']:
        f.write(f"  - {file}\n")
    f.write(f"\n")
    f.write(f"Frozen snapshot location:\n")
    f.write(f"  {frozen_dir}\n")

print(f" Lock file created: {lock_file}")

# Summary

print(f"\n Outputs have been frozen and validated")
print(f"\n Frozen snapshot location:")
print(f"   {frozen_dir}")

print(f"\n Snapshot contains:")
for file in freeze_manifest['locked_files']:
    print(f"   - {file}")

print(f"\n Dataset summary:")
print(f"   Samples: {len(X_windowed):,}")
print(f"   Signal length: {X_windowed.shape[1]}")
print(f"   Features: {X_features.shape[1]}")
print(f"   Classes: {n_classes}")

print(f"\n Separability: {separability}")
print(f"   Average distance: {avg_dist:.4f}")

# Print manifest for verification
print("\n Freeze Manifest Preview:")
print(f"   Timestamp: {freeze_manifest['freeze_timestamp']}")
print(f"   Samples: {freeze_manifest['data_summary']['n_samples']:,}")
print(f"   Separability: {freeze_manifest['phase2_results']['separability']}")
print(f"   Ready: {freeze_manifest['phase2_results']['ready_for_training']}")

if freeze_manifest['warnings']:
    print(f"\n Warnings:")
    for warning in freeze_manifest['warnings']:
        print(f"   - {warning}")
