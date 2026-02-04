"""
Feature Level Fusion Implementation
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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

print(" "*20 + "FEATURE-LEVEL FUSION FRAMEWORK")

BASE_DIR = '/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG'
FEATURES_DIR = f'{BASE_DIR}/outputs/advanced_features'
OUTPUT_DIR = f'{BASE_DIR}/outputs/feature_fusion'
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Focus on challenging datasets
DATASETS = ['epilepsy_122mb', 'epileptic_seizure']
RANDOM_STATE = 42

# Feature Selection Fusion: Combine multiple feature selection methods to identify robust features
class FeatureSelectionFusion:
    def __init__(self, n_features=50):
        self.n_features = n_features
        self.selected_features_ = None
        self.feature_scores_ = None
        
    def fit(self, X, y, feature_names=None):
        print(f"\n  Run Feature Selection Fusion")
        print(f"    Input: {X.shape[1]} features → Target: {self.n_features} features")
        
        feature_scores = pd.DataFrame()
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # ANOVA F-test
        print(f"  ANOVA F-test")
        selector_anova = SelectKBest(f_classif, k=min(self.n_features, X.shape[1]))
        selector_anova.fit(X, y)
        feature_scores['anova'] = selector_anova.scores_
        
        # Mutual Information
        print(f"  Mutual Information")
        mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
        feature_scores['mutual_info'] = mi_scores
        
        # Random Forest Importance
        print(f"  Random Forest Importance")
        rf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X, y)
        feature_scores['rf_importance'] = rf.feature_importances_
        
        # L1-based selection (Lasso)
        print(f"  L1-based Selection")
        lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=RANDOM_STATE, max_iter=1000)
        try:
            lasso.fit(X, y)
            feature_scores['lasso'] = np.abs(lasso.coef_).mean(axis=0)
        except:
            feature_scores['lasso'] = np.zeros(X.shape[1])
        
        # Normalize scores
        for col in feature_scores.columns:
            feature_scores[col] = (feature_scores[col] - feature_scores[col].min()) / \
                                  (feature_scores[col].max() - feature_scores[col].min() + 1e-10)
        
        # Combine scores (weighted average)
        feature_scores['combined'] = feature_scores.mean(axis=1)
        feature_scores['feature_name'] = feature_names
        
        # Select top features
        top_features_idx = feature_scores['combined'].nlargest(self.n_features).index.tolist()
        self.selected_features_ = top_features_idx
        self.feature_scores_ = feature_scores
        
        print(f"    Selected {len(self.selected_features_)} features")
        
        return self
    
    def transform(self, X):
        return X[:, self.selected_features_]
    
    def fit_transform(self, X, y, feature_names=None):
        self.fit(X, y, feature_names)
        return self.transform(X)

# PCA Fusion: Dimensionality reduction while preserving variance
class PCAFusion:
    def __init__(self, n_components=0.95):
        self.n_components = n_components  # Can be int or variance ratio
        self.pca = None
        
    def fit(self, X, y=None):
        print(f"\n  Run PCA Fusion")
        print(f"    Input: {X.shape[1]} features")
        
        self.pca = PCA(n_components=self.n_components, random_state=RANDOM_STATE)
        self.pca.fit(X)
        
        n_components = self.pca.n_components_
        variance = self.pca.explained_variance_ratio_.sum()
        
        print(f"    Reduced to {n_components} components")
        print(f"    Explained variance: {variance:.4f}")
        
        return self
    
    def transform(self, X):
        return self.pca.transform(X)
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

# Hybrid Fusion (Selection + PCA)
class HybridFusion:
    def __init__(self, n_select=60, n_components=0.95):
        self.n_select = n_select
        self.n_components = n_components
        self.selector = None
        self.pca = None
        
    def fit(self, X, y, feature_names=None):
        print(f"\n  Run Hybrid Fusion")
        
        # Feature Selection
        self.selector = FeatureSelectionFusion(n_features=self.n_select)
        X_selected = self.selector.fit_transform(X, y, feature_names)
        
        # PCA on selected features
        self.pca = PCAFusion(n_components=self.n_components)
        self.pca.fit(X_selected)
        
        return self
    
    def transform(self, X):
        X_selected = self.selector.transform(X)
        return self.pca.transform(X_selected)
    
    def fit_transform(self, X, y, feature_names=None):
        self.fit(X, y, feature_names)
        return self.transform(X)

# Load Data and Evalauate

def load_dataset(dataset_name):
    dataset_path = Path(FEATURES_DIR) / dataset_name
    
    if not dataset_path.exists():
        print(f"  Dataset not found: {dataset_path}")
        return None
    
    X_train = np.load(dataset_path / 'X_train.npy', allow_pickle=True)
    X_val = np.load(dataset_path / 'X_val.npy', allow_pickle=True)
    X_test = np.load(dataset_path / 'X_test.npy', allow_pickle=True)
    
    y_train = np.load(dataset_path / 'y_train.npy', allow_pickle=True)
    y_val = np.load(dataset_path / 'y_val.npy', allow_pickle=True)
    y_test = np.load(dataset_path / 'y_test.npy', allow_pickle=True)
    
    # Load feature names if available
    feature_names_file = dataset_path / 'feature_names.txt'
    if feature_names_file.exists():
        with open(feature_names_file, 'r') as f:
            feature_names = [line.strip() for line in f]
    else:
        feature_names = None
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_names
    }

def evaluate_fusion_method(X_train, X_val, X_test, y_train, y_val, y_test, 
                           fusion_method, dataset_name, method_name):
    print(f"Evaluating: {method_name} on {dataset_name}")
    
    results = {}
    
    # Define classifiers with optimized settings
    classifiers = {
        'Logistic_Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=100,  # Reduced from 200 for speed
            max_depth=20,      # Limit depth for speed
            random_state=RANDOM_STATE, 
            n_jobs=-1, 
            class_weight='balanced'
        ),
        'SVM_Linear': CalibratedClassifierCV(
            LinearSVC(max_iter=5000, random_state=RANDOM_STATE, class_weight='balanced'),
            method='sigmoid', cv=3
        )
    }
    
    # Train and evaluate each classifier
    for clf_name, clf in classifiers.items():
        print(f"\n  Train {clf_name}")
        
        try:
            clf.fit(X_train, y_train)
            
            # Evaluate
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # ROC-AUC
            try:
                if len(np.unique(y_test)) == 2:
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except:
                auc = None
            
            results[clf_name] = {
                'accuracy': acc,
                'f1_score': f1,
                'roc_auc': auc
            }
            
            auc_display = f"{auc:.4f}" if auc is not None else "N/A"
            print(f"    Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_display}")
            
        except Exception as e:
            print(f"    Error: {e}")
            results[clf_name] = {'accuracy': 0.0, 'f1_score': 0.0, 'roc_auc': None}
    
    return results

# Run all fusion methods on all datasets
def run_fusion_experiments():
    all_results = {}
    
    for dataset_name in DATASETS:
        print(f"# DATASET: {dataset_name.upper()}")
        
        # Load data
        data = load_dataset(dataset_name)
        if data is None:
            continue
        
        X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
        y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
        feature_names = data['feature_names']
        
        print(f"\n Loaded: {X_train.shape[0]:,} train, {X_test.shape[0]:,} test")
        print(f"  Original features: {X_train.shape[1]}")
        
        dataset_results = {}
        
        # BASELINE: No fusion (use all features)
        print(f"BASELINE: All Features (No Fusion)")
        baseline_results = evaluate_fusion_method(
            X_train, X_val, X_test, y_train, y_val, y_test,
            None, dataset_name, "Baseline"
        )
        dataset_results['baseline'] = baseline_results
        
        # Feature Selection Fusion
        fusion1 = FeatureSelectionFusion(n_features=50)
        X_train_f1 = fusion1.fit_transform(X_train, y_train, feature_names)
        X_val_f1 = fusion1.transform(X_val)
        X_test_f1 = fusion1.transform(X_test)
        
        fusion1_results = evaluate_fusion_method(
            X_train_f1, X_val_f1, X_test_f1, y_train, y_val, y_test,
            fusion1, dataset_name, "Feature_Selection_Fusion"
        )
        dataset_results['feature_selection'] = fusion1_results
        
        # PCA Fusion
        fusion2 = PCAFusion(n_components=0.95)
        X_train_f2 = fusion2.fit_transform(X_train)
        X_val_f2 = fusion2.transform(X_val)
        X_test_f2 = fusion2.transform(X_test)
        
        fusion2_results = evaluate_fusion_method(
            X_train_f2, X_val_f2, X_test_f2, y_train, y_val, y_test,
            fusion2, dataset_name, "PCA_Fusion"
        )
        dataset_results['pca'] = fusion2_results
        
        # Hybrid Fusion
        fusion3 = HybridFusion(n_select=60, n_components=0.95)
        X_train_f3 = fusion3.fit_transform(X_train, y_train, feature_names)
        X_val_f3 = fusion3.transform(X_val)
        X_test_f3 = fusion3.transform(X_test)
        
        fusion3_results = evaluate_fusion_method(
            X_train_f3, X_val_f3, X_test_f3, y_train, y_val, y_test,
            fusion3, dataset_name, "Hybrid_Fusion"
        )
        dataset_results['hybrid'] = fusion3_results
        
        # Save results
        all_results[dataset_name] = dataset_results
        
        # Save feature importance for this dataset
        if fusion1.feature_scores_ is not None:
            output_path = Path(OUTPUT_DIR) / dataset_name
            output_path.mkdir(exist_ok=True)
            fusion1.feature_scores_.to_csv(output_path / 'feature_importance.csv', index=False)
    
    return all_results

# Create Comparison and Visualization

def create_comparison(results):
    print("FUSION RESULTS COMPARISON")
    
    # Collect all results
    comparison_data = []
    
    for dataset_name, dataset_results in results.items():
        for fusion_method, method_results in dataset_results.items():
            for clf_name, metrics in method_results.items():
                comparison_data.append({
                    'Dataset': dataset_name,
                    'Fusion_Method': fusion_method,
                    'Classifier': clf_name,
                    'Accuracy': metrics['accuracy'],
                    'F1_Score': metrics['f1_score'],
                    'ROC_AUC': metrics.get('roc_auc', None)
                })
    
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    df.to_csv(f'{OUTPUT_DIR}/fusion_comparison.csv', index=False)
    print(f"\n Saved: {OUTPUT_DIR}/fusion_comparison.csv")
    
    # Print summary
    print("BEST RESULTS PER DATASET")
    
    for dataset in DATASETS:
        dataset_df = df[df['Dataset'] == dataset]
        if not dataset_df.empty:
            best = dataset_df.loc[dataset_df['Accuracy'].idxmax()]
            print(f"\n{dataset}:")
            print(f"  Best: {best['Fusion_Method']} + {best['Classifier']}")
            print(f"  Accuracy: {best['Accuracy']:.4f}")
            print(f"  F1: {best['F1_Score']:.4f}")
            if best['ROC_AUC']:
                print(f"  AUC: {best['ROC_AUC']:.4f}")
    
    # Create visualization
    create_fusion_visualizations(df)
    
    return df

def create_fusion_visualizations(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Feature Fusion Performance Comparison', fontsize=16, fontweight='bold')
    
    # Accuracy by fusion method
    ax = axes[0, 0]
    pivot = df.pivot_table(values='Accuracy', index='Fusion_Method', 
                           columns='Dataset', aggfunc='mean')
    pivot.plot(kind='bar', ax=ax, rot=45)
    ax.set_ylabel('Average Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy by Fusion Method', fontsize=13, fontweight='bold')
    ax.legend(title='Dataset')
    ax.grid(axis='y', alpha=0.3)
    
    # F1-Score comparison
    ax = axes[0, 1]
    pivot = df.pivot_table(values='F1_Score', index='Fusion_Method', 
                           columns='Dataset', aggfunc='mean')
    pivot.plot(kind='bar', ax=ax, rot=45)
    ax.set_ylabel('Average F1-Score', fontsize=11, fontweight='bold')
    ax.set_title('F1-Score by Fusion Method', fontsize=13, fontweight='bold')
    ax.legend(title='Dataset')
    ax.grid(axis='y', alpha=0.3)
    
    # Best method per dataset
    ax = axes[1, 0]
    best_per_dataset = df.loc[df.groupby('Dataset')['Accuracy'].idxmax()]
    ax.barh(best_per_dataset['Dataset'], best_per_dataset['Accuracy'], color='skyblue', edgecolor='black')
    ax.set_xlabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Best Accuracy per Dataset', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Improvement over baseline
    ax = axes[1, 1]
    for dataset in DATASETS:
        dataset_df = df[df['Dataset'] == dataset]
        baseline_acc = dataset_df[dataset_df['Fusion_Method'] == 'baseline']['Accuracy'].mean()
        
        improvements = []
        methods = []
        for method in ['feature_selection', 'pca', 'hybrid']:
            method_df = dataset_df[dataset_df['Fusion_Method'] == method]
            if not method_df.empty:
                method_acc = method_df['Accuracy'].mean()
                improvements.append((method_acc - baseline_acc) * 100)
                methods.append(method)
        
        x = np.arange(len(methods))
        ax.bar(x + DATASETS.index(dataset)*0.2, improvements, 0.2, 
               label=dataset, alpha=0.8)
    
    ax.set_ylabel('Improvement over Baseline (%)', fontsize=11, fontweight='bold')
    ax.set_title('Performance Improvement', fontsize=13, fontweight='bold')
    ax.set_xticks(x + 0.2)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fusion_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {OUTPUT_DIR}/fusion_comparison.png")
    plt.close()

# Main
if __name__ == '__main__':
    start_time = datetime.now()
    
    # Run experiments
    results = run_fusion_experiments()
    
    # Create comparison
    comparison_df = create_comparison(results)
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).seconds
    
    print("FEATURE FUSION COMPLETE!")
    print(f"Time: {duration//60}m {duration%60}s")
    print(f"Results: {OUTPUT_DIR}")
