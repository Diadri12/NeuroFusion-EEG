"""
Apply Advanced Feature Engineering to Baseline Model
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

print("BASELINE ML MODELS ON ADVANCED FEATURES")

BASE_DIR = '/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG'

USE_ADVANCED_FEATURES = True  # Set to False to use basic features

if USE_ADVANCED_FEATURES:
    PROCESSED_DIR = f'{BASE_DIR}/outputs/advanced_features'  # Advanced features (84 features)
    MODELS_DIR = f'{BASE_DIR}/outputs/advanced_models'
    RESULTS_DIR = f'{BASE_DIR}/outputs/advanced_results'
    print("\n Using ADVANCED FEATURES")
else:
    PROCESSED_DIR = f'{BASE_DIR}/outputs/final_processed'  # Basic features (25-51 features)
    MODELS_DIR = f'{BASE_DIR}/outputs/baseline_models'
    RESULTS_DIR = f'{BASE_DIR}/outputs/baseline_results'
    print("\n Using BASIC FEATURES")

Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

DATASETS = ['bonn_eeg', 'epilepsy_122mb', 'epileptic_seizure']  # Excluded: epilepsy_diagnosis
RANDOM_STATE = 42

# Dataset exclusion note
EXCLUDED_DATASETS = {
    'epilepsy_diagnosis': {
        'reason': 'Not meaningful for seizure detection',
        'details': '500 classes, 1 feature, accuracy ≈ random'
    }
}

# SAMPLING CONFIGURATION
ENABLE_SAMPLING = True
MAX_TRAIN_SAMPLES = 50000
MAX_VAL_SAMPLES = 10000
MAX_TEST_SAMPLES = 10000

print(f"Sampling: {'ENABLED' if ENABLE_SAMPLING else 'DISABLED'}")
if ENABLE_SAMPLING:
    print(f"  Max samples: Train={MAX_TRAIN_SAMPLES:,}, Val={MAX_VAL_SAMPLES:,}, Test={MAX_TEST_SAMPLES:,}")

print(f"\n  Excluded Datasets:")
for ds, info in EXCLUDED_DATASETS.items():
    print(f"  • {ds}: {info['reason']}")
    print(f"    ({info['details']})")

# Define Baseline Models

def get_baseline_models():
    models = {
        'Logistic_Regression': LogisticRegression(
            solver='liblinear',
            random_state=RANDOM_STATE,
            max_iter=1000,
            class_weight="balanced"
        ),
        'SVM_Linear': CalibratedClassifierCV(
            LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=5000
            ),
            method="sigmoid",
            cv=3
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"
        )
    }
    return models

print("\nSelected Models:")
for name in get_baseline_models().keys():
    print(f"  • {name}")

# Load Data
def sample_data_stratified(X, y, max_samples, random_state=42):
    if len(X) <= max_samples:
        return X, y
    
    print(f"    Sampling from {len(X):,} to {max_samples:,} samples")
    X_sampled, _, y_sampled, _ = train_test_split(
        X, y,
        train_size=max_samples,
        stratify=y,
        random_state=random_state
    )
    return X_sampled, y_sampled

def load_dataset(dataset_name):
    print(f"\n  Load {dataset_name}")
    
    # Handle different directory structures for epileptic_seizure
    if dataset_name == 'epileptic_seizure':
        if USE_ADVANCED_FEATURES:
            # Advanced features location
            dataset_path = Path(PROCESSED_DIR) / dataset_name
        else:
            # Basic features location - directly in outputs/
            dataset_path = Path(BASE_DIR) / 'outputs/train_val_test_splits/Epileptic Seizure Dataset'
    else:
        # Standard location for bonn_eeg and epilepsy_122mb
        dataset_path = Path(PROCESSED_DIR) / dataset_name
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"    Dataset not found at: {dataset_path}")
        return None
    
    # Load data
    try:
        X_train = np.load(dataset_path / 'X_train.npy', allow_pickle=True)
        X_val = np.load(dataset_path / 'X_val.npy', allow_pickle=True)
        X_test = np.load(dataset_path / 'X_test.npy', allow_pickle=True)
        
        y_train = np.load(dataset_path / 'y_train.npy', allow_pickle=True)
        y_val = np.load(dataset_path / 'y_val.npy', allow_pickle=True)
        y_test = np.load(dataset_path / 'y_test.npy', allow_pickle=True)
        
        # Load metadata if available
        metadata_file = dataset_path / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            # Create basic metadata
            metadata = {
                'dataset': dataset_name,
                'samples': len(X_train) + len(X_val) + len(X_test),
                'features': X_train.shape[1],
                'classes': len(np.unique(np.concatenate([y_train, y_val, y_test])))
            }
        
        print(f"   Loaded from: {dataset_path}")
    
    except Exception as e:
        print(f"   Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Store original sizes
    original_sizes = {
        'train': len(X_train),
        'val': len(X_val),
        'test': len(X_test)
    }
    
    # Apply sampling if enabled
    sampled = False
    if ENABLE_SAMPLING:
        if len(X_train) > MAX_TRAIN_SAMPLES:
            print(f"  Large training set: {len(X_train):,} samples")
            X_train, y_train = sample_data_stratified(X_train, y_train, MAX_TRAIN_SAMPLES)
            sampled = True
        
        if len(X_val) > MAX_VAL_SAMPLES:
            X_val, y_val = sample_data_stratified(X_val, y_val, MAX_VAL_SAMPLES)
            sampled = True
        
        if len(X_test) > MAX_TEST_SAMPLES:
            X_test, y_test = sample_data_stratified(X_test, y_test, MAX_TEST_SAMPLES)
            sampled = True
    
    if sampled:
        print(f"   Sampled to: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'metadata': metadata,
        'original_sizes': original_sizes,
        'sampled': sampled
    }

# Training and Evaluation
def apply_scaling(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def calculate_metrics(y_true, y_pred, y_proba=None, average='weighted'):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        except:
            metrics['roc_auc'] = None
    
    # Sensitivity and Specificity
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivities = []
        specificities = []
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivities.append(sens)
            specificities.append(spec)
        metrics['sensitivity'] = np.mean(sensitivities)
        metrics['specificity'] = np.mean(specificities)
    
    return metrics

def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_proba, title, save_path):
    plt.figure(figsize=(8, 6))
    
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        auc = roc_auc_score(y_true, y_proba[:, 1])
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})', linewidth=2)
    else:
        from sklearn.preprocessing import label_binarize
        n_classes = len(np.unique(y_true))
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        for i in range(min(n_classes, 5)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def train_and_evaluate_model(model, model_name, data, dataset_name):
    print(f"Training: {model_name} on {dataset_name}")
    
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
    
    print(f"\nDataset Shapes:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
    
    if data['sampled']:
        print(f"  Note: Sampled (original: {data['original_sizes']['train']:,})")
    
    # Scale
    print(f"\n  Apply StandardScaler")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = apply_scaling(X_train, X_val, X_test)
    
    # Train
    print(f"\n  Training {model_name}")
    model.fit(X_train_scaled, y_train)
    print(f"   Model trained")
    
    # Validate
    print(f"\n  Validating")
    y_val_pred = model.predict(X_val_scaled)
    y_val_proba = model.predict_proba(X_val_scaled) if hasattr(model, 'predict_proba') else None
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
    print(f"    Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
    
    # Test
    print(f"\n  Testing")
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)
    print(f"    Test Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_score']:.4f}")
    
    cm_test = confusion_matrix(y_test, y_test_pred)
    class_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    
    # Save
    print(f"\n  Saving")
    model_save_dir = Path(MODELS_DIR) / dataset_name
    results_save_dir = Path(RESULTS_DIR) / dataset_name
    model_save_dir.mkdir(parents=True, exist_ok=True)
    results_save_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_save_dir / f"{model_name.lower()}.pkl")
    joblib.dump(scaler, model_save_dir / f"{model_name.lower()}_scaler.pkl")
    
    results = {
        'model': model_name,
        'dataset': dataset_name,
        'feature_type': 'advanced' if USE_ADVANCED_FEATURES else 'basic',
        'num_features': X_train.shape[1],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sampled': data['sampled'],
        'data_shapes': {'train': list(X_train.shape), 'val': list(X_val.shape), 'test': list(X_test.shape)},
        'original_sizes': data['original_sizes'],
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'confusion_matrix': cm_test.tolist(),
        'classification_report': class_report
    }
    
    with open(results_save_dir / f"{model_name.lower()}_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    viz_dir = results_save_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    cm_title = f'{model_name} - {dataset_name}\nConfusion Matrix'
    plot_confusion_matrix(cm_test, cm_title, viz_dir / f"{model_name.lower()}_confusion_matrix.png")
    
    if y_test_proba is not None:
        roc_title = f'{model_name} - {dataset_name}\nROC Curve'
        plot_roc_curve(y_test, y_test_proba, roc_title, viz_dir / f"{model_name.lower()}_roc_curve.png")
    
    print(f"   Saved to: {results_save_dir}")
    return results

# Run all models on all datasets
def run_all_experiments():
    all_results = {}
    
    for dataset_name in DATASETS:
        print(f"# DATASET: {dataset_name.upper()}")
        
        data = load_dataset(dataset_name)
        if data is None:
            print(f"   Skipping {dataset_name}")
            continue
        
        print(f"\n Loaded {dataset_name}")
        # Handle different metadata formats
        samples = data['metadata'].get('samples') or data['metadata'].get('total_samples', 'Unknown')
        features = data['metadata'].get('features') or data['metadata'].get('total_features', 'Unknown')
        classes = data['metadata'].get('classes', 'Unknown')
        
        print(f"  Samples: {samples:,}" if isinstance(samples, int) else f"  Samples: {samples}")
        print(f"  Features: {features}" if isinstance(features, int) else f"  Features: {features}")
        print(f"  Classes: {classes}")
        
        models = get_baseline_models()
        dataset_results = {}
        
        for model_name, model in models.items():
            try:
                results = train_and_evaluate_model(model, model_name, data, dataset_name)
                dataset_results[model_name] = results
                print(f"\n Completed {model_name}")
            except Exception as e:
                print(f"\n Error: {e}")
                import traceback
                traceback.print_exc()
        
        all_results[dataset_name] = dataset_results
    
    return all_results

def create_comparison_table(all_results):
    print("PERFORMANCE COMPARISON")
    
    comparison_data = []
    for dataset_name, dataset_results in all_results.items():
        for model_name, results in dataset_results.items():
            test_metrics = results['test_metrics']
            comparison_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Features': results['num_features'],
                'Type': results['feature_type'],
                'Accuracy': test_metrics['accuracy'],
                'Precision': test_metrics['precision'],
                'Recall': test_metrics['recall'],
                'F1-Score': test_metrics['f1_score'],
                'ROC-AUC': test_metrics.get('roc_auc', None)
            })
    
    df = pd.DataFrame(comparison_data)
    df.to_csv(f'{RESULTS_DIR}/comparison.csv', index=False)
    print(f"\n Saved: {RESULTS_DIR}/comparison.csv")
    print("\n" + df.to_string(index=False))
    
    print(f"\nBest per dataset:")
    for dataset in DATASETS:
        dataset_df = df[df['Dataset'] == dataset]
        if not dataset_df.empty:
            best = dataset_df.loc[dataset_df['Accuracy'].idxmax()]
            print(f"  {dataset}: {best['Model']} ({best['Accuracy']:.4f})")
    
    print(f"\nOverall best:")
    best = df.loc[df['Accuracy'].idxmax()]
    print(f"  {best['Model']} on {best['Dataset']}: {best['Accuracy']:.4f}")
    
    return df

# Main

if __name__ == '__main__':
    start_time = datetime.now()
    
    print(f"\nStart training")
    print(f"Timestamp: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    all_results = run_all_experiments()
    comparison_df = create_comparison_table(all_results)
    
    end_time = datetime.now()
    duration = (end_time - start_time).seconds
    
    print(f"Time: {duration//60}m {duration%60}s")
    print(f"Results: {RESULTS_DIR}")
