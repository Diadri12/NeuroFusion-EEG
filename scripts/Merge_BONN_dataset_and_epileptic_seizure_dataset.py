"""
Merge 02 datasets(Bonn_egg Dataset and Epileptic seizure dataset)
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print(f"DATASET MERGING FOR CNN BASELINE")

BASE_DIR = '/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG'
OUTPUT_DIR = f'{BASE_DIR}/outputs/merged_datasets'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Dataset Configurations

DATASETS_TO_MERGE = {
    'epileptic_seizure': {
        'path': f'{BASE_DIR}/outputs',  # Direct outputs folder
        'signal_file': 'full_processed_signals.npy',  # Custom filename
        'label_file': 'full_labels.npy',  # Custom filename
        'name': 'Epileptic Seizure Dataset',
        'sampling_rate': 173.61,
        'signal_length': 178,
        'label_map': {
            1: 'Seizure activity',
            2: 'Tumor area',
            3: 'Healthy brain',
            4: 'Eyes closed',
            5: 'Eyes open'
        },
        'binary_mapping': {  # For binary seizure detection
            1: 1,  # Seizure -> 1
            2: 0,  # Non-seizure -> 0
            3: 0,  # Non-seizure -> 0
            4: 0,  # Non-seizure -> 0
            5: 0   # Non-seizure -> 0
        }
    },
    'bonn_eeg': {
        'path': f'{BASE_DIR}/outputs/final_processed/bonn_eeg',
        'signal_file': 'preprocessed_signals.npy',  # Standard filename
        'label_file': 'labels.npy',  # Standard filename
        'name': 'Bonn EEG Dataset',
        'sampling_rate': 173.61,
        'signal_length': 4097,
        'label_map': {
            1: 'Seizure',
            2: 'Non-seizure'
        },
        'binary_mapping': {
        1: 1,  # Seizure
        2: 0,  # Non-seizure
        4: 0,  # Non-seizure
        5: 0   # Non-seizure
    }
    },
    'epilepsy_122mb': {
        'path': f'{BASE_DIR}/outputs/final_processed/epilepsy_122mb',
        'signal_file': 'preprocessed_signals.npy',  # Standard filename
        'label_file': 'labels.npy',  # Standard filename
        'name': 'Epilepsy 122MB Dataset',
        'sampling_rate': 178,
        'signal_length': 4097,
        'label_map': {
            1: 'Seizure',
            2: 'Non-seizure'
        },
        'binary_mapping': {
            1: 1,  # Seizure -> 1
            2: 0   # Non-seizure -> 0
        }
    }
}

# Signal Resampling

def resample_signal(signal, target_length):
    from scipy import interpolate

    current_length = len(signal)
    if current_length == target_length:
        return signal

    # Create interpolation function
    x_old = np.linspace(0, 1, current_length)
    x_new = np.linspace(0, 1, target_length)
    f = interpolate.interp1d(x_old, signal, kind='linear')

    return f(x_new)

def pad_or_truncate(signal, target_length):
    current_length = len(signal)

    if current_length == target_length:
        return signal
    elif current_length > target_length:
        # Truncate from center
        start = (current_length - target_length) // 2
        return signal[start:start + target_length]
    else:
        # Pad with zeros
        pad_width = target_length - current_length
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return np.pad(signal, (pad_left, pad_right), mode='constant')

# Load and Merge Datasets
def load_dataset(dataset_name, dataset_info):
    print(f"Loading: {dataset_info['name']}")

    path = Path(dataset_info['path'])

    # Check if dataset exists
    if not path.exists():
        print(f" Dataset path not found at: {path}")
        return None

    # Get signal and label file names (handles different naming conventions)
    signal_filename = dataset_info.get('signal_file', 'preprocessed_signals.npy')
    label_filename = dataset_info.get('label_file', 'labels.npy')

    signals_file = path / signal_filename
    labels_file = path / label_filename

    print(f"  Looking for:")
    print(f"    Signals: {signals_file}")
    print(f"    Labels:  {labels_file}")

    if not signals_file.exists():
        print(f" Signal file not found: {signal_filename}")
        return None

    if not labels_file.exists():
        print(f" Label file not found: {label_filename}")
        return None

    signals = np.load(signals_file, allow_pickle=True)
    labels = np.load(labels_file, allow_pickle=True)

    print(f" Loaded {len(signals):,} samples")
    print(f"  Signal shape: {signals.shape}")
    print(f"  Signal length: {signals.shape[1]}")
    print(f"  Sampling rate: {dataset_info['sampling_rate']} Hz")

    # Show label distribution
    print(f"\n  Label distribution:")
    for label_val in sorted(np.unique(labels)):
        count = np.sum(labels == label_val)
        label_name = dataset_info['label_map'].get(label_val, f'Class {label_val}')
        print(f"    {label_val}: {label_name:25s} {count:>6,} ({count/len(labels)*100:>5.1f}%)")

    return {
        'signals': signals,
        'labels': labels,
        'dataset_name': dataset_name,
        'info': dataset_info
    }

#Merge Datasets
def merge_datasets(datasets_to_merge, target_length=None, merge_strategy='binary',
                   method='resample'):
    print(f"MERGING DATASETS")
    print(f"Strategy: {merge_strategy}")
    print(f"Length normalization: {method}")

    all_signals = []
    all_labels = []
    all_sources = []

    # Determine target length
    if target_length is None:
        lengths = []
        for dataset_name, dataset_info in datasets_to_merge.items():
            path = Path(dataset_info['path'])
            signal_filename = dataset_info.get('signal_file', 'preprocessed_signals.npy')
            signal_file = path / signal_filename

            if signal_file.exists():
                sig = np.load(signal_file, allow_pickle=True)
                lengths.append(sig.shape[1])

        target_length = min(lengths) if method == 'truncate' else max(lengths)
        print(f"\n Auto-detected target length: {target_length}")

    # Load and process each dataset
    for dataset_name, dataset_info in datasets_to_merge.items():
        dataset = load_dataset(dataset_name, dataset_info)

        if dataset is None:
            continue

        signals = dataset['signals']
        labels = dataset['labels']

        # Normalize signal length
        print(f"\n  Processing signals")
        processed_signals = []

        current_length = signals.shape[1]
        if current_length != target_length:
            print(f"    Normalizing from {current_length} to {target_length} timepoints")

            for i, sig in enumerate(signals):
                if method == 'resample':
                    processed_sig = resample_signal(sig, target_length)
                else:  # pad/truncate
                    processed_sig = pad_or_truncate(sig, target_length)
                processed_signals.append(processed_sig)

                if (i + 1) % 1000 == 0:
                    print(f"      [{(i+1)/len(signals)*100:5.1f}%] {i+1:,}/{len(signals):,}")

            signals = np.array(processed_signals)

        # Map labels based on strategy
        if merge_strategy == 'binary':
            # Use binary mapping from config
            binary_map = dataset_info['binary_mapping']
            mapped_labels = []
            for label in labels:
              if label not in binary_map:
                raise ValueError(
                    f"Label {label} not defined in binary_mapping for {dataset_info['name']}"
                )
              mapped_labels.append(binary_map[label])
            mapped_labels = np.array(mapped_labels)
            print(f"  Mapped to binary: Seizure={np.sum(mapped_labels==1):,}, Non-seizure={np.sum(mapped_labels==0):,}")
        else:
            # Keep original labels but add offset to avoid conflicts
            # This would need more sophisticated handling
            mapped_labels = labels.copy()

        all_signals.append(signals)
        all_labels.append(mapped_labels)
        all_sources.extend([dataset_name] * len(signals))

        print(f"   Added {len(signals):,} samples from {dataset_info['name']}")

    # Combine all datasets
    if not all_signals:
        print("\n No datasets loaded!")
        return None

    print(f"COMBINING DATASETS")

    merged_signals = np.vstack(all_signals)
    merged_labels = np.hstack(all_labels)
    merged_sources = np.array(all_sources)

    print(f" Merged dataset created")
    print(f"  Total samples: {len(merged_signals):,}")
    print(f"  Signal shape: {merged_signals.shape}")
    print(f"  Unique labels: {np.unique(merged_labels)}")

    # Show distribution
    print(f"\n  Combined label distribution:")
    for label_val in sorted(np.unique(merged_labels)):
        count = np.sum(merged_labels == label_val)
        label_name = 'Seizure' if label_val == 1 else 'Non-seizure'
        print(f"    {label_val}: {label_name:25s} {count:>7,} ({count/len(merged_labels)*100:>5.1f}%)")

    print(f"\n  Source distribution:")
    for source in np.unique(merged_sources):
        count = np.sum(merged_sources == source)
        source_info = datasets_to_merge[source]
        print(f"    {source_info['name']:30s} {count:>7,} ({count/len(merged_sources)*100:>5.1f}%)")

    return {
        'signals': merged_signals,
        'labels': merged_labels,
        'sources': merged_sources,
        'target_length': target_length,
        'merge_strategy': merge_strategy,
        'method': method
    }

# Create train/val/test splits
def create_splits(merged_data, train_size=0.70, val_size=0.15, test_size=0.15):
    print(f"CREATING TRAIN/VAL/TEST SPLITS")
    print(f"Split ratio: {train_size*100:.0f}% / {val_size*100:.0f}% / {test_size*100:.0f}%")

    X = merged_data['signals']
    y = merged_data['labels']

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(val_size + test_size),
        stratify=y,
        random_state=42
    )

    # Second split: val vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_size/(val_size + test_size),
        stratify=y_temp,
        random_state=42
    )

    print(f"\n Split sizes:")
    print(f"  Train:      {len(X_train):>7,} samples ({len(X_train)/len(X)*100:>5.1f}%)")
    print(f"  Validation: {len(X_val):>7,} samples ({len(X_val)/len(X)*100:>5.1f}%)")
    print(f"  Test:       {len(X_test):>7,} samples ({len(X_test)/len(X)*100:>5.1f}%)")

    # Verify stratification
    print(f"\n  Class distribution verification:")
    print(f"    {'Split':<12} {'Seizure':>12} {'Non-seizure':>15}")
    print(f"    {'-'*42}")

    for split_name, split_y in [('Train', y_train), ('Validation', y_val), ('Test', y_test)]:
        seizure_pct = (split_y == 1).sum() / len(split_y) * 100
        non_seizure_pct = (split_y == 0).sum() / len(split_y) * 100
        print(f"    {split_name:<12} {seizure_pct:>11.1f}% {non_seizure_pct:>14.1f}%")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }

# Save Merged Dataset and splits
def save_merged_dataset(merged_data, splits, output_name):
    print(f"SAVING MERGED DATASET")

    output_path = Path(OUTPUT_DIR) / output_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Save raw merged data
    np.save(output_path / 'merged_signals.npy', merged_data['signals'])
    np.save(output_path / 'merged_labels.npy', merged_data['labels'])
    np.save(output_path / 'merged_sources.npy', merged_data['sources'])

    print(f" Saved merged data:")
    print(f"  merged_signals.npy ({merged_data['signals'].nbytes/1024**2:.1f} MB)")
    print(f"  merged_labels.npy ({merged_data['labels'].nbytes/1024:.1f} KB)")
    print(f"  merged_sources.npy")

    # Save splits
    np.save(output_path / 'X_train.npy', splits['X_train'])
    np.save(output_path / 'y_train.npy', splits['y_train'])
    np.save(output_path / 'X_val.npy', splits['X_val'])
    np.save(output_path / 'y_val.npy', splits['y_val'])
    np.save(output_path / 'X_test.npy', splits['X_test'])
    np.save(output_path / 'y_test.npy', splits['y_test'])

    print(f"\n Saved train/val/test splits:")
    print(f"  X_train.npy ({splits['X_train'].nbytes/1024**2:.1f} MB)")
    print(f"  X_val.npy ({splits['X_val'].nbytes/1024**2:.1f} MB)")
    print(f"  X_test.npy ({splits['X_test'].nbytes/1024**2:.1f} MB)")

    # Save metadata
    metadata = {
        'dataset_name': output_name,
        'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': int(len(merged_data['signals'])),
        'signal_length': int(merged_data['target_length']),
        'merge_strategy': merged_data['merge_strategy'],
        'normalization_method': merged_data['method'],
        'num_classes': int(len(np.unique(merged_data['labels']))),
        'label_map': {
            0: 'Non-seizure',
            1: 'Seizure'
        },
        'source_datasets': list(DATASETS_TO_MERGE.keys()),
        'source_distribution': {
            source: int(np.sum(merged_data['sources'] == source))
            for source in np.unique(merged_data['sources'])
        },
        'class_distribution': {
            'seizure': int(np.sum(merged_data['labels'] == 1)),
            'non_seizure': int(np.sum(merged_data['labels'] == 0))
        },
        'splits': {
            'train_samples': int(len(splits['X_train'])),
            'val_samples': int(len(splits['X_val'])),
            'test_samples': int(len(splits['X_test'])),
            'train_ratio': 0.70,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            'random_seed': 42
        }
    }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n Saved metadata.json")

    # Create visualization
    create_merge_visualizations(merged_data, splits, output_path)

    print(f" MERGED DATASET SAVED")
    print(f"Location: {output_path}")
    print(f"Total size: {(merged_data['signals'].nbytes + merged_data['labels'].nbytes)/1024**2:.1f} MB")

    return output_path

# Visualizations
def create_merge_visualizations(merged_data, splits, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Source distribution
    ax = axes[0, 0]
    source_counts = pd.Series(merged_data['sources']).value_counts()
    source_names = [DATASETS_TO_MERGE[s]['name'] for s in source_counts.index]
    ax.bar(range(len(source_counts)), source_counts.values, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(source_counts)))
    ax.set_xticklabels(source_names, rotation=15, ha='right')
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title('Source Dataset Distribution', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    # Class distribution
    ax = axes[0, 1]
    class_counts = pd.Series(merged_data['labels']).value_counts().sort_index()
    class_names = ['Non-seizure', 'Seizure']
    colors = ['#2ecc71', '#e74c3c']
    ax.bar(class_names, class_counts.values, color=colors, edgecolor='black')
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title('Class Distribution', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

    # Split distribution
    ax = axes[1, 0]
    split_data = {
        'Train': [np.sum(splits['y_train'] == 1), np.sum(splits['y_train'] == 0)],
        'Val': [np.sum(splits['y_val'] == 1), np.sum(splits['y_val'] == 0)],
        'Test': [np.sum(splits['y_test'] == 1), np.sum(splits['y_test'] == 0)]
    }
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, [d[0] for d in split_data.values()], width, label='Seizure', color='#e74c3c', edgecolor='black')
    ax.bar(x + width/2, [d[1] for d in split_data.values()], width, label='Non-seizure', color='#2ecc71', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(split_data.keys())
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title('Train/Val/Test Split Distribution', fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Sample signals
    ax = axes[1, 1]
    # Plot one signal from each class
    for label_val in [0, 1]:
        idx = np.where(merged_data['labels'] == label_val)[0][0]
        signal = merged_data['signals'][idx]
        time = np.arange(len(signal)) / 173.61  # Approximate time
        label_name = 'Seizure' if label_val == 1 else 'Non-seizure'
        color = '#e74c3c' if label_val == 1 else '#2ecc71'
        ax.plot(time, signal, label=label_name, alpha=0.7, linewidth=1, color=color)
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Normalized Amplitude', fontweight='bold')
    ax.set_title('Sample Signals (One per Class)', fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f'Merged Dataset Analysis - {len(merged_data["signals"]):,} Total Samples',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'merge_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization: merge_analysis.png")
    plt.close()

# Main
if __name__ == '__main__':
    print(f"\nDatasets to merge:")
    for name, info in DATASETS_TO_MERGE.items():
        print(f"  • {info['name']}")
        print(f"    Path: {info['path']}")
        print(f"    Signal length: {info['signal_length']}")
        print(f"    Sampling rate: {info['sampling_rate']} Hz")

    print(f"CREATING MERGED DATASET: EPILEPTIC_SEIZURE + BONN_EEG")

    subset_datasets = {
        'epileptic_seizure': DATASETS_TO_MERGE['epileptic_seizure'],
        'bonn_eeg': DATASETS_TO_MERGE['bonn_eeg']
    }

    merged_subset = merge_datasets(
        subset_datasets,
        target_length=178,
        merge_strategy='binary',
        method='resample'
    )

    if merged_subset:
        splits_subset = create_splits(merged_subset)
        output_path_subset = save_merged_dataset(
            merged_subset,
            splits_subset,
            'merged_epileptic_bonn'
        )

        print(f"\n Successfully created: merged_epileptic_bonn")
        print(f"   Samples: {len(merged_subset['signals']):,}")
        print(f"   Classes: Binary (Seizure vs Non-seizure)")
        print(f"   Signal length: {merged_subset['target_length']}")

    print(f" DATASET MERGING COMPLETE")
    print(f"\nMerged datasets created in: {OUTPUT_DIR}")
