"""
Baseline ML Models for EEG Seizure Detection
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

print(" "*20 + "BASELINE ML MODELS - SYSTEMATIC APPROACH")
print(" "*25 + "WITH SMART SAMPLING")


BASE_DIR = '/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG'
PROCESSED_DIR = f'{BASE_DIR}/outputs/final_processed'
MODELS_DIR = f'{BASE_DIR}/outputs/baseline_models'
RESULTS_DIR = f'{BASE_DIR}/outputs/baseline_results'

# Create output directories
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

DATASETS = ['bonn_eeg', 'epilepsy_122mb', 'epilepsy_diagnosis', 'epileptic_seizure']
RANDOM_STATE = 42

# SAMPLING CONFIGURATION
ENABLE_SAMPLING = True  # Set to False to train on full datasets
MAX_TRAIN_SAMPLES = 50000  # Maximum training samples per dataset
MAX_VAL_SAMPLES = 10000    # Maximum validation samples
MAX_TEST_SAMPLES = 10000   # Maximum test samples


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
            n_estimators=100,  # Small number
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"
        )
    }
    return models

print("\n Select Baseline Models")
print("-" * 80)
for name in get_baseline_models().keys():
    print(f"  {name}")

def sample_data_stratified(X, y, max_samples, random_state=42):
    if len(X) <= max_samples:
        return X, y

    print(f"    Sampling from {len(X):,} to {max_samples:,} samples (stratified)...")

    # Stratified sampling
    X_sampled, _, y_sampled, _ = train_test_split(
        X, y,
        train_size=max_samples,
        stratify=y,
        random_state=random_state
    )

    return X_sampled, y_sampled

def load_dataset(dataset_name):
    print(f"\n  Load {dataset_name}...")

    # Handle different directory structures
    if dataset_name == 'epileptic_seizure':
        dataset_path = Path(BASE_DIR) / 'outputs/train_val_test_splits/Epileptic Seizure Dataset'

        # Load from the different naming convention
        X_train = np.load(dataset_path / 'X_train.npy', allow_pickle=True)
        X_val = np.load(dataset_path / 'X_val.npy', allow_pickle=True)
        X_test = np.load(dataset_path / 'X_test.npy', allow_pickle=True)

        y_train = np.load(dataset_path / 'y_train.npy', allow_pickle=True)
        y_val = np.load(dataset_path / 'y_val.npy', allow_pickle=True)
        y_test = np.load(dataset_path / 'y_test.npy', allow_pickle=True)

        # Create basic metadata if not available
        metadata = {
            'dataset': 'Epileptic Seizure Dataset',
            'samples': len(X_train) + len(X_val) + len(X_test),
            'features': X_train.shape[1],
            'classes': len(np.unique(np.concatenate([y_train, y_val, y_test])))
        }
    else:
        dataset_path = Path(PROCESSED_DIR) / dataset_name

        # Load feature matrices (already scaled during preprocessing)
        X_train = np.load(dataset_path / 'X_train.npy', allow_pickle=True)
        X_val = np.load(dataset_path / 'X_val.npy', allow_pickle=True)
        X_test = np.load(dataset_path / 'X_test.npy', allow_pickle=True)

        # Load labels
        y_train = np.load(dataset_path / 'y_train.npy', allow_pickle=True)
        y_val = np.load(dataset_path / 'y_val.npy', allow_pickle=True)
        y_test = np.load(dataset_path / 'y_test.npy', allow_pickle=True)

        # Load metadata
        with open(dataset_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

    # Store original sizes
    original_sizes = {
        'train': len(X_train),
        'val': len(X_val),
        'test': len(X_test)
    }

    # Apply sampling if enabled and dataset is large
    sampled = False
    if ENABLE_SAMPLING:
        if len(X_train) > MAX_TRAIN_SAMPLES:
            print(f"  Large training set detected: {len(X_train):,} samples")
            X_train, y_train = sample_data_stratified(X_train, y_train, MAX_TRAIN_SAMPLES)
            sampled = True

        if len(X_val) > MAX_VAL_SAMPLES:
            print(f"  Large validation set detected: {len(X_val):,} samples")
            X_val, y_val = sample_data_stratified(X_val, y_val, MAX_VAL_SAMPLES)
            sampled = True

        if len(X_test) > MAX_TEST_SAMPLES:
            print(f"  Large test set detected: {len(X_test):,} samples")
            X_test, y_test = sample_data_stratified(X_test, y_test, MAX_TEST_SAMPLES)
            sampled = True

    if sampled:
        print(f"  Sampled sizes: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")

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

print("\n Data Preparation")
if ENABLE_SAMPLING:
    print(f"  Sampling ENABLED: Max train={MAX_TRAIN_SAMPLES:,}, val={MAX_VAL_SAMPLES:,}, test={MAX_TEST_SAMPLES:,}")
else:
    print(f"  Sampling DISABLED: Using full datasets")

def apply_scaling(X_train, X_val, X_test):
    scaler = StandardScaler()

    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform validation and test using fitted scaler
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

    # ROC-AUC (only for binary or with probabilities)
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                # Multi-class (one-vs-rest)
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba,
                                                   multi_class='ovr', average='weighted')
        except Exception as e:
            metrics['roc_auc'] = None

    # Sensitivity (Recall) and Specificity
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] == 2:  # Binary classification
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        # Multi-class: compute per-class then average
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
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        auc = roc_auc_score(y_true, y_proba[:, 1])
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})', linewidth=2)
    else:
        # Multi-class: plot one curve per class
        from sklearn.preprocessing import label_binarize
        n_classes = len(np.unique(y_true))
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

        for i in range(min(n_classes, 5)):  # Plot first 5 classes
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

    # Extract data
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    print(f"\nDataset Shapes:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")

    if data['sampled']:
        print(f"  Note: Using sampled data (original train size: {data['original_sizes']['train']:,})")

    # Feature Scaling
    print(f"\n Applying StandardScaler")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = apply_scaling(
        X_train, X_val, X_test
    )

    # Train Model
    print(f"\nTrain {model_name}")
    model.fit(X_train_scaled, y_train)
    print(f"  Model trained")

    # Validate Model
    print(f"\n Validate on validation set")
    y_val_pred = model.predict(X_val_scaled)
    y_val_proba = model.predict_proba(X_val_scaled) if hasattr(model, 'predict_proba') else None
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)

    print(f"  Validation Metrics:")
    for metric, value in val_metrics.items():
        if value is not None:
            print(f"    {metric}: {value:.4f}")

    # Test Model
    print(f"\n Test on test set")
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

    print(f"  Test Metrics:")
    for metric, value in test_metrics.items():
        if value is not None:
            print(f"    {metric}: {value:.4f}")

    # Confusion Matrix
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Classification Report
    class_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

    # Save Models & Results
    print(f"\n Save model and results")

    # Create directories
    model_save_dir = Path(MODELS_DIR) / dataset_name
    results_save_dir = Path(RESULTS_DIR) / dataset_name
    model_save_dir.mkdir(parents=True, exist_ok=True)
    results_save_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_filename = f"{model_name.lower()}.pkl"
    joblib.dump(model, model_save_dir / model_filename)
    joblib.dump(scaler, model_save_dir / f"{model_name.lower()}_scaler.pkl")

    # Save results
    results = {
        'model': model_name,
        'dataset': dataset_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'sampled': data['sampled'],
        'data_shapes': {
            'train': list(X_train.shape),
            'val': list(X_val.shape),
            'test': list(X_test.shape)
        },
        'original_sizes': data['original_sizes'],
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'confusion_matrix': cm_test.tolist(),
        'classification_report': class_report
    }

    results_filename = f"{model_name.lower()}_results.json"
    with open(results_save_dir / results_filename, 'w') as f:
        json.dump(results, f, indent=2)

    # Save visualizations
    viz_dir = results_save_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    # Confusion Matrix
    cm_title = f'{model_name} - {dataset_name}\nConfusion Matrix (Test Set)'
    cm_path = viz_dir / f"{model_name.lower()}_confusion_matrix.png"
    plot_confusion_matrix(cm_test, cm_title, cm_path)

    # ROC Curve
    if y_test_proba is not None:
        roc_title = f'{model_name} - {dataset_name}\nROC Curve (Test Set)'
        roc_path = viz_dir / f"{model_name.lower()}_roc_curve.png"
        plot_roc_curve(y_test, y_test_proba, roc_title, roc_path)

    print(f"  Saved to: {results_save_dir}")

    return results

def run_all_experiments():
    all_results = {}

    for dataset_name in DATASETS:
        print(f"# DATASET: {dataset_name.upper()}")

        try:
            # Load data
            data = load_dataset(dataset_name)
            print(f"\n Load {dataset_name}")
            print(f"  Samples: {data['metadata']['samples']:,}")
            print(f"  Features: {data['metadata']['features']}")
            print(f"  Classes: {data['metadata']['classes']}")

            # Train-Val-Test already split
            print(f"\n Using pre-split data")
            print(f"  Train: {len(data['y_train']):,} samples")
            print(f"  Val:   {len(data['y_val']):,} samples")
            print(f"  Test:  {len(data['y_test']):,} samples")

            # Get models
            models = get_baseline_models()

            dataset_results = {}

            for model_name, model in models.items():
                try:
                    results = train_and_evaluate_model(
                        model, model_name, data, dataset_name
                    )
                    dataset_results[model_name] = results
                    print(f"\n Completed {model_name} on {dataset_name}")

                except Exception as e:
                    print(f"\n Error with {model_name} on {dataset_name}: {e}")
                    import traceback
                    traceback.print_exc()

            all_results[dataset_name] = dataset_results

        except Exception as e:
            print(f"\n Error loading {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    return all_results

def create_comparison_table(all_results):
    print("BASELINE PERFORMANCE COMPARISON")

    # Collect all metrics
    comparison_data = []

    for dataset_name, dataset_results in all_results.items():
        for model_name, results in dataset_results.items():
            test_metrics = results['test_metrics']
            comparison_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Sampled': results.get('sampled', False),
                'Train_Size': results['data_shapes']['train'][0],
                'Accuracy': test_metrics['accuracy'],
                'Precision': test_metrics['precision'],
                'Recall': test_metrics['recall'],
                'F1-Score': test_metrics['f1_score'],
                'Sensitivity': test_metrics.get('sensitivity', None),
                'Specificity': test_metrics.get('specificity', None),
                'ROC-AUC': test_metrics.get('roc_auc', None)
            })

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    # Save to CSV
    df.to_csv(f'{RESULTS_DIR}/baseline_comparison.csv', index=False)
    print(f"\n Saved comparison table to: {RESULTS_DIR}/baseline_comparison.csv")

    # Display table
    print("\nCOMPARISON TABLE:")
    print(df.to_string(index=False))

    # Identify best performers
    print(f"\n{'='*80}")
    print("KEY FINDINGS:")
    print(f"{'='*80}")

    # Best model per dataset
    print("\n Best Model per Dataset (by Accuracy):")
    for dataset in DATASETS:
        dataset_df = df[df['Dataset'] == dataset]
        if not dataset_df.empty:
            best = dataset_df.loc[dataset_df['Accuracy'].idxmax()]
            print(f"   {dataset}: {best['Model']} ({best['Accuracy']:.4f})")

    # Best dataset per model
    print("\n Best Dataset per Model (by Accuracy):")
    for model in df['Model'].unique():
        model_df = df[df['Model'] == model]
        if not model_df.empty:
            best = model_df.loc[model_df['Accuracy'].idxmax()]
            print(f"   {model}: {best['Dataset']} ({best['Accuracy']:.4f})")

    # Overall best
    overall_best = df.loc[df['Accuracy'].idxmax()]
    print(f"\n Overall Strongest Baseline:")
    print(f"   Model: {overall_best['Model']}")
    print(f"   Dataset: {overall_best['Dataset']}")
    print(f"   Accuracy: {overall_best['Accuracy']:.4f}")

    # Overall weakest
    overall_weakest = df.loc[df['Accuracy'].idxmin()]
    print(f"\n Weakest Performance:")
    print(f"   Model: {overall_weakest['Model']}")
    print(f"   Dataset: {overall_weakest['Dataset']}")
    print(f"   Accuracy: {overall_weakest['Accuracy']:.4f}")

    # Average performance
    print(f"\n Average Performance by Model:")
    model_avg = df.groupby('Model')['Accuracy'].mean().sort_values(ascending=False)
    for model, acc in model_avg.items():
        print(f"   {model}: {acc:.4f}")

    print(f"\n Average Performance by Dataset:")
    dataset_avg = df.groupby('Dataset')['Accuracy'].mean().sort_values(ascending=False)
    for dataset, acc in dataset_avg.items():
        print(f"   {dataset}: {acc:.4f}")

    # Create visualization
    create_comparison_visualizations(df)

    return df

def create_comparison_visualizations(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Baseline Model Performance Comparison', fontsize=16, fontweight='bold')

    # Accuracy heatmap
    ax = axes[0, 0]
    pivot = df.pivot(index='Model', columns='Dataset', values='Accuracy')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Accuracy'})
    ax.set_title('Accuracy: Model vs Dataset', fontsize=13, fontweight='bold')

    # F1-Score comparison
    ax = axes[0, 1]
    pivot = df.pivot(index='Model', columns='Dataset', values='F1-Score')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, cbar_kws={'label': 'F1-Score'})
    ax.set_title('F1-Score: Model vs Dataset', fontsize=13, fontweight='bold')

    # Bar plot: Average accuracy per model
    ax = axes[1, 0]
    model_avg = df.groupby('Model')['Accuracy'].mean().sort_values(ascending=False)
    model_avg.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_ylabel('Average Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Average Model Performance', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Bar plot: Average accuracy per dataset
    ax = axes[1, 1]
    dataset_avg = df.groupby('Dataset')['Accuracy'].mean().sort_values(ascending=False)
    dataset_avg.plot(kind='bar', ax=ax, color='lightcoral', edgecolor='black')
    ax.set_ylabel('Average Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Average Dataset Performance', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/baseline_comparison_visualization.png', dpi=300, bbox_inches='tight')
    print(f"\n Saved comparison visualization")
    plt.close()

if __name__ == '__main__':
    start_time = datetime.now()

    print("\nStarting baseline model training...")
    print(f"Timestamp: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all experiments
    all_results = run_all_experiments()

    # Create comparison
    comparison_df = create_comparison_table(all_results)
