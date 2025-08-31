#!/usr/bin/env python3
"""
Enhanced V2X Anomaly Detection Training Script
This script runs the enhanced training pipeline with different configurations
"""

import os
import json
import time
from v2x_training_lstm import run_training

def run_experiments():
    """Run multiple experiments with different configurations"""
    
    # Create results directory
    results_dir = "enhanced_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Experiment configurations
    experiments = [
        {
            "name": "baseline_single",
            "use_ensemble": False,
            "use_advanced_features": False,
            "sequence_length": 20,
            "epochs": 30,
            "lr": 1e-3
        },
        {
            "name": "baseline_ensemble", 
            "use_ensemble": True,
            "use_advanced_features": False,
            "sequence_length": 20,
            "epochs": 30,
            "lr": 1e-3
        },
        {
            "name": "advanced_single",
            "use_ensemble": False,
            "use_advanced_features": True,
            "sequence_length": 20,
            "epochs": 30,
            "lr": 1e-3
        },
        {
            "name": "advanced_ensemble",
            "use_ensemble": True,
            "use_advanced_features": True,
            "sequence_length": 20,
            "epochs": 30,
            "lr": 1e-3
        },
        {
            "name": "long_sequence_ensemble",
            "use_ensemble": True,
            "use_advanced_features": True,
            "sequence_length": 30,
            "epochs": 40,
            "lr": 5e-4
        }
    ]
    
    results_summary = []
    
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp['name']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Run training
            model, test_metrics = run_training(
                out_dir=f"artifacts_{exp['name']}",
                sequence_length=exp['sequence_length'],
                epochs=exp['epochs'],
                lr=exp['lr'],
                use_ensemble=exp['use_ensemble'],
                use_advanced_features=exp['use_advanced_features']
            )
            
            # Record results
            exp_time = time.time() - start_time
            result = {
                "experiment": exp['name'],
                "config": exp,
                "test_metrics": test_metrics,
                "training_time": exp_time
            }
            results_summary.append(result)
            
            print(f"Experiment {exp['name']} completed in {exp_time:.2f} seconds")
            print(f"Final AUC: {test_metrics['final_auc']:.4f}")
            
        except Exception as e:
            print(f"Error in experiment {exp['name']}: {str(e)}")
            continue
    
    # Save summary
    with open(os.path.join(results_dir, "experiment_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    # Print comparison
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*60}")
    
    for result in results_summary:
        exp_name = result['experiment']
        config = result['config']
        metrics = result['test_metrics']
        time_taken = result['training_time']
        
        print(f"\n{exp_name}:")
        print(f"  Config: Ensemble={config['use_ensemble']}, Advanced={config['use_advanced_features']}")
        print(f"  Final AUC: {metrics['final_auc']:.4f}")
        print(f"  Error AUC: {metrics['error_auc']:.4f}")
        print(f"  Rule AUC: {metrics['rule_auc']:.4f}")
        print(f"  Avg Precision: {metrics['avg_precision']:.4f}")
        print(f"  Training Time: {time_taken:.2f}s")
    
    # Find best configuration
    best_result = max(results_summary, key=lambda x: x['test_metrics']['final_auc'])
    print(f"\n{'='*60}")
    print(f"BEST CONFIGURATION: {best_result['experiment']}")
    print(f"Final AUC: {best_result['test_metrics']['final_auc']:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_experiments()
