"""
Freeze CNN Baseline Models
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
from datetime import datetime

print("FREEZING BASELINE MODELS")

BASE_DIR = "/content/drive/MyDrive/Final Year/Final Year Project (2025 - 2026)/NeuroFusion-EEG"
BASELINE_DIR = f"{BASE_DIR}/outputs/dl_optimized"  
OUTPUT_DIR = f"{BASE_DIR}/outputs/frozen_baselines"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Dataset Configuration
DATASETS_CONFIG = {
    "bonn_eeg": {
        "name": "Bonn EEG",
        "n_classes": 5,
        "label_map": {0: "Non-seizure", 1: "Seizure", 2: "Class2", 3: "Class3", 4: "Class4"}
    },
    "epileptic_seizure": {
        "name": "Epileptic Seizure",
        "n_classes": 5,
        "label_map": {
            0: "Seizure", 1: "Tumor", 2: "Healthy",
            3: "Eyes Closed", 4: "Eyes Open"
        }
    },
    "epilepsy_122mb": {
        "name": "Epilepsy 122MB",
        "n_classes": 2,
        "label_map": {0: "Non-seizure", 1: "Seizure"}
    },
    "merged_epileptic_bonn": {
        "name": "Merged Epileptic + Bonn",
        "n_classes": 2,
        "label_map": {0: "Non-seizure", 1: "Seizure"}
    }
}

def freeze_baseline(dataset_key, config):
    dataset_dir = Path(BASELINE_DIR) / dataset_key
    
    # Check if baseline exists
    model_path = dataset_dir / "model.pt"
    results_path = dataset_dir / "results.json"
    history_path = dataset_dir / "history.json"
    
    if not model_path.exists():
        print(f" Skipping {dataset_key}: model.pt not found at {model_path}")
        return None
    
    print(f"Processing: {config['name']}")
    
    # Load model weights
    try:
        model_state = torch.load(model_path, map_location='cpu')
        print(f"Loaded model weights from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Load metrics
    metrics = {}
    if results_path.exists():
        with open(results_path, 'r') as f:
            metrics = json.load(f)
        print(f"Loaded metrics: Accuracy={metrics.get('accuracy', 'N/A')}")
    else:
        print(f" No results.json found")
    
    # Load training history
    history = {}
    training_info = {}
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        training_info = {
            'epochs_trained': len(history.get('train', history.get('train_loss', []))),
            'final_train_loss': history.get('train', history.get('train_loss', [None]))[-1] if history.get('train', history.get('train_loss', [])) else None,
            'final_val_loss': history.get('val', history.get('val_loss', [None]))[-1] if history.get('val', history.get('val_loss', [])) else None,
            'best_val_loss': min(history.get('val', history.get('val_loss', [float('inf')]))) if history.get('val', history.get('val_loss', [])) else None
        }
        print(f"Loaded training history: {training_info['epochs_trained']} epochs")
    else:
        print(f"No history.json found")
    
    # Create frozen checkpoint
    frozen_checkpoint = {
        'model_state_dict': model_state,
        'model_config': {
            'n_classes': config['n_classes'],
            'architecture': 'EEG_CNN_Optimized', 
            'label_map': config['label_map']
        },
        'metrics': metrics,
        'training_config': training_info,
        'dataset': {
            'name': config['name'],
            'key': dataset_key,
            'n_classes': config['n_classes']
        },
        'freeze_info': {
            'frozen_date': datetime.now().isoformat(),
            'source_dir': str(dataset_dir),
            'baseline_version': 'v1.0'
        }
    }
    
    # Save frozen baseline
    frozen_name = f"baseline_{dataset_key}.pt"
    frozen_path = Path(OUTPUT_DIR) / frozen_name
    
    torch.save(frozen_checkpoint, frozen_path)
    print(f" FROZEN BASELINE SAVED: {frozen_name}")
    
    # Save human-readable summary
    summary = {
        'dataset': config['name'],
        'frozen_file': frozen_name,
        'metrics': metrics,
        'training': training_info,
        'frozen_date': datetime.now().isoformat()
    }
    
    summary_path = Path(OUTPUT_DIR) / f"baseline_{dataset_key}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f" Summary saved: baseline_{dataset_key}_summary.json")
    
    return {
        'dataset_key': dataset_key,
        'dataset_name': config['name'],
        'frozen_file': frozen_name,
        'metrics': metrics,
        'n_classes': config['n_classes']
    }

# Main execution
if __name__ == "__main__":
    frozen_baselines = []
    
    for key, config in DATASETS_CONFIG.items():
        result = freeze_baseline(key, config)
        if result:
            frozen_baselines.append(result)
    
    # Create master baseline registry
    if frozen_baselines:
        baseline_registry = {
            'creation_date': datetime.now().isoformat(),
            'total_baselines': len(frozen_baselines),
            'source_directory': BASELINE_DIR,
            'frozen_directory': OUTPUT_DIR,
            'baselines': {}
        }
        
        for baseline in frozen_baselines:
            baseline_registry['baselines'][baseline['dataset_key']] = {
                'dataset_name': baseline['dataset_name'],
                'frozen_file': baseline['frozen_file'],
                'n_classes': baseline['n_classes'],
                'metrics': baseline['metrics']
            }
        
        registry_path = Path(OUTPUT_DIR) / "baseline_registry.json"
        with open(registry_path, 'w') as f:
            json.dump(baseline_registry, f, indent=2)
        
        print(" BASELINE REGISTRY CREATED")
        print(f"Registry file: {registry_path}")
        print(f"\nFrozen Baseline Models ({len(frozen_baselines)}):")
        
        for baseline in frozen_baselines:
            acc = baseline['metrics'].get('accuracy', 'N/A')
            if acc != 'N/A':
                acc = f"{float(acc):.4f}"
            print(f"  • {baseline['frozen_file']:40s} → {baseline['dataset_name']:25s} (Acc: {acc})")
        
        # Create comparison table
        print("BASELINE PERFORMANCE SUMMARY")
        print(f"{'Dataset':<30s} {'Accuracy':<12s} {'F1':<12s} {'Precision':<12s} {'Recall':<12s}")
        
        for baseline in frozen_baselines:
            metrics = baseline['metrics']
            print(f"{baseline['dataset_name']:<30s} "
                  f"{metrics.get('accuracy', 'N/A')!s:<12s} "
                  f"{metrics.get('f1', 'N/A')!s:<12s} "
                  f"{metrics.get('precision', 'N/A')!s:<12s} "
                  f"{metrics.get('recall', 'N/A')!s:<12s}")
        
        print("ALL CNN BASELINES FROZEN SUCCESSFULLY")
        print(f"Frozen baselines saved to: {OUTPUT_DIR}")
        print(f"checkpoint = torch.load('{OUTPUT_DIR}/baseline_bonn_eeg.pt')")
    else:
        print(" No baselines found to freeze!")
        print(f"Please check that trained models exist in: {BASELINE_DIR}")
