"""
Freeze and Summarize ML Experiments
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
from datetime import datetime

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

print("="*80)
print(" "*20 + "FREEZE ML EXPERIMENTS - FINAL SUMMARY")
print("="*80)

BASE_DIR = '/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG'
BASELINE_RESULTS = f'{BASE_DIR}/outputs/baseline_results'
ADVANCED_RESULTS = f'{BASE_DIR}/outputs/advanced_results'
FUSION_RESULTS = f'{BASE_DIR}/outputs/feature_fusion'
FREEZE_DIR = f'{BASE_DIR}/outputs/frozen_experiments'

Path(FREEZE_DIR).mkdir(parents=True, exist_ok=True)

# Collect all results
def collect_all_results():
    all_results = []
    
    # Baseline results (basic features)
    baseline_file = Path(BASELINE_RESULTS) / 'baseline_comparison.csv'
    if baseline_file.exists():
        df_baseline = pd.read_csv(baseline_file)
        df_baseline['Experiment'] = 'Baseline (Basic Features)'
        df_baseline['Feature_Count'] = df_baseline.get('Train_Size', 'Unknown')
        all_results.append(df_baseline)
        print(" Loaded baseline results")
    
    # Advanced features results
    advanced_file = Path(ADVANCED_RESULTS) / 'comparison.csv'
    if advanced_file.exists():
        df_advanced = pd.read_csv(advanced_file)
        df_advanced['Experiment'] = 'Advanced Features'
        all_results.append(df_advanced)
        print(" Loaded advanced features results")
    
    # Fusion results
    fusion_file = Path(FUSION_RESULTS) / 'fusion_comparison.csv'
    if fusion_file.exists():
        df_fusion = pd.read_csv(fusion_file)
        df_fusion['Experiment'] = df_fusion['Fusion_Method'].apply(
            lambda x: f'Fusion: {x.replace("_", " ").title()}'
        )
        all_results.append(df_fusion)
        print(" Loaded fusion results")
    
    if not all_results:
        print(" No results found!")
        return None
    
    # Combine all results
    df_all = pd.concat(all_results, ignore_index=True)
    
    # Standardize column names
    df_all = df_all.rename(columns={
        'Accuracy': 'accuracy',
        'F1-Score': 'f1_score',
        'Precision': 'precision',
        'Recall': 'recall',
        'ROC-AUC': 'roc_auc'
    })
    
    return df_all

# Create Summary
def create_final_summary(df):
    print("FINAL ML EXPERIMENTS SUMMARY")
    
    # Best result per dataset
    print("\n BEST RESULTS PER DATASET:")
    
    for dataset in df['Dataset'].unique():
        dataset_df = df[df['Dataset'] == dataset]
        best = dataset_df.loc[dataset_df['accuracy'].idxmax()]
        
        print(f"\n{dataset.upper()}:")
        print(f"  Best Method: {best['Experiment']}")
        print(f"  Model: {best['Model'] if 'Model' in best else best.get('Classifier', 'N/A')}")
        print(f"  Accuracy: {best['accuracy']:.4f}")
        print(f"  F1-Score: {best.get('f1_score', 'N/A')}")
        if 'roc_auc' in best and pd.notna(best['roc_auc']):
            print(f"  ROC-AUC: {best['roc_auc']:.4f}")
    
    # Progression analysis
    print(" PROGRESSION ANALYSIS:")
    
    for dataset in df['Dataset'].unique():
        dataset_df = df[df['Dataset'] == dataset]
        
        # Get best from each experiment type
        experiments = ['Baseline (Basic Features)', 'Advanced Features', 'Fusion']
        
        print(f"\n{dataset.upper()}:")
        baseline_acc = None
        
        for exp_type in experiments:
            exp_df = dataset_df[dataset_df['Experiment'].str.contains(exp_type, case=False, na=False)]
            if not exp_df.empty:
                best_acc = exp_df['accuracy'].max()
                
                if baseline_acc is None:
                    baseline_acc = best_acc
                    improvement = 0
                else:
                    improvement = (best_acc - baseline_acc) * 100
                
                print(f"  {exp_type:30s}: {best_acc:.4f} ({improvement:+.2f}%)")
    
    return df

# Create Visualizations
def create_freeze_visualizations(df):
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Classical ML Experiments - Complete Summary', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Overall accuracy comparison
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Get best accuracy for each experiment type per dataset
    summary_data = []
    for dataset in df['Dataset'].unique():
        for exp in df['Experiment'].unique():
            subset = df[(df['Dataset'] == dataset) & (df['Experiment'] == exp)]
            if not subset.empty:
                summary_data.append({
                    'Dataset': dataset,
                    'Experiment': exp,
                    'Best_Accuracy': subset['accuracy'].max()
                })
    
    summary_df = pd.DataFrame(summary_data)
    pivot = summary_df.pivot(index='Experiment', columns='Dataset', values='Best_Accuracy')
    
    pivot.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Best Accuracy by Experiment Type', fontsize=14, fontweight='bold')
    ax1.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Model comparison
    ax2 = fig.add_subplot(gs[0, 2])
    
    model_col = 'Model' if 'Model' in df.columns else 'Classifier'
    if model_col in df.columns:
        model_perf = df.groupby(model_col)['accuracy'].mean().sort_values(ascending=False)
        model_perf.plot(kind='barh', ax=ax2, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Average Accuracy', fontsize=11, fontweight='bold')
        ax2.set_title('Model Performance', fontsize=13, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
    
    # Dataset difficulty
    ax3 = fig.add_subplot(gs[1, 0])
    
    dataset_best = df.groupby('Dataset')['accuracy'].max().sort_values()
    colors = ['red' if x < 0.7 else 'orange' if x < 0.8 else 'green' for x in dataset_best.values]
    dataset_best.plot(kind='barh', ax=ax3, color=colors, edgecolor='black')
    ax3.set_xlabel('Best Accuracy', fontsize=11, fontweight='bold')
    ax3.set_title('Dataset Difficulty\n(Best Achieved)', fontsize=13, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # Improvement over baseline
    ax4 = fig.add_subplot(gs[1, 1:])
    
    improvements = []
    for dataset in df['Dataset'].unique():
        dataset_df = df[df['Dataset'] == dataset]
        
        # Baseline
        baseline = dataset_df[dataset_df['Experiment'].str.contains('Baseline', na=False)]
        if baseline.empty:
            continue
        baseline_acc = baseline['accuracy'].max()
        
        # Advanced
        advanced = dataset_df[dataset_df['Experiment'].str.contains('Advanced', na=False)]
        if not advanced.empty:
            adv_acc = advanced['accuracy'].max()
            improvements.append({
                'Dataset': dataset,
                'Method': 'Advanced Features',
                'Improvement': (adv_acc - baseline_acc) * 100
            })
        
        # Fusion
        fusion = dataset_df[dataset_df['Experiment'].str.contains('Fusion', na=False)]
        if not fusion.empty:
            fusion_acc = fusion['accuracy'].max()
            improvements.append({
                'Dataset': dataset,
                'Method': 'Best Fusion',
                'Improvement': (fusion_acc - baseline_acc) * 100
            })
    
    if improvements:
        imp_df = pd.DataFrame(improvements)
        pivot_imp = imp_df.pivot(index='Dataset', columns='Method', values='Improvement')
        pivot_imp.plot(kind='bar', ax=ax4, width=0.7)
        ax4.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
        ax4.set_title('Improvement Over Baseline', fontsize=13, fontweight='bold')
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax4.legend(title='Method')
        ax4.grid(axis='y', alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # F1-Score distribution
    ax5 = fig.add_subplot(gs[2, :])
    
    if 'f1_score' in df.columns:
        f1_data = df.dropna(subset=['f1_score'])
        if not f1_data.empty:
            for dataset in f1_data['Dataset'].unique():
                dataset_f1 = f1_data[f1_data['Dataset'] == dataset]['f1_score']
                ax5.hist(dataset_f1, bins=20, alpha=0.5, label=dataset, edgecolor='black')
            
            ax5.set_xlabel('F1-Score', fontsize=11, fontweight='bold')
            ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax5.set_title('F1-Score Distribution Across All Experiments', fontsize=13, fontweight='bold')
            ax5.legend()
            ax5.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{FREEZE_DIR}/complete_ml_summary.png', dpi=300, bbox_inches='tight')
    print(f"\n Saved visualization: {FREEZE_DIR}/complete_ml_summary.png")
    plt.close()

# Save frozen results
def save_frozen_results(df):
    # Save complete results
    df.to_csv(f'{FREEZE_DIR}/all_ml_results.csv', index=False)
    print(f" Saved complete results: {FREEZE_DIR}/all_ml_results.csv")
    
    # Save best results only
    best_results = []
    for dataset in df['Dataset'].unique():
        dataset_df = df[df['Dataset'] == dataset]
        best = dataset_df.loc[dataset_df['accuracy'].idxmax()]
        best_results.append(best)
    
    best_df = pd.DataFrame(best_results)
    best_df.to_csv(f'{FREEZE_DIR}/best_results_per_dataset.csv', index=False)
    print(f" Saved best results: {FREEZE_DIR}/best_results_per_dataset.csv")
    
    # Create metadata
    metadata = {
        'freeze_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_experiments': len(df),
        'datasets': df['Dataset'].unique().tolist(),
        'experiment_types': df['Experiment'].unique().tolist(),
        'best_overall': {
            'dataset': best_df.loc[best_df['accuracy'].idxmax(), 'Dataset'],
            'experiment': best_df.loc[best_df['accuracy'].idxmax(), 'Experiment'],
            'accuracy': float(best_df['accuracy'].max()),
        },
        'summary': {
            'avg_accuracy': float(df['accuracy'].mean()),
            'max_accuracy': float(df['accuracy'].max()),
            'min_accuracy': float(df['accuracy'].min()),
        }
    }
    
    with open(f'{FREEZE_DIR}/experiment_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f" Saved metadata: {FREEZE_DIR}/experiment_metadata.json")

# Main
if __name__ == '__main__':
    print("\nCollecting all ML experiment results")
    
    df_all = collect_all_results()
    
    if df_all is not None:
        # Create summary
        df_summary = create_final_summary(df_all)
        
        # Create visualizations
        create_freeze_visualizations(df_summary)
        
        # Save frozen results
        save_frozen_results(df_summary)
        
        print(" ML EXPERIMENTS FROZEN!")
        print(f"\nResults saved to: {FREEZE_DIR}")
    else:
        print("\n Failed to collect results")
