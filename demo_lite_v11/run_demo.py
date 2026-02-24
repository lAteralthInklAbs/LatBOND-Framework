#!/usr/bin/env python3
"""
LatBOND Demo Lite v11 - Main Orchestrator
==========================================
Trains and evaluates all 6 models (3 architectures × 2 conditions).

v11 changes:
    - Triple-head output: onset + offset + velocity prediction
    - TripleHeadLoss combining focal + MSE losses
    - Models return dict with onset_logits, offset_logits, velocity_logits
    - Note-level evaluation ready (mir_eval)

All v10.1 innovations retained:
    - Asymmetric Focal Loss (Conservative preset: gamma_neg=3.0)
    - Cosine curriculum schedule
    - 25 epochs, patience=15
    - 5% MAESTRO data

Usage:
    python run_demo.py                    # Full run (download + train + eval)
    python run_demo.py --skip_download    # Use existing dataset
    python run_demo.py --eval_only        # Only evaluate (models must exist)
    python run_demo.py --arch streaming_cnn  # Train only one architecture
    python run_demo.py --batch_size 8     # Override batch size (for smaller GPUs)

Expected runtime: ~6-7 hours on T4 GPU (all 6 models)
"""

import argparse
import json
import os
import sys
import time
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from models import create_model, count_parameters
from data.download_maestro import download_maestro, create_subset
from data.maestro_dataset import create_dataloaders
from training.trainer import LatBONDTrainer
from evaluation.metrics import evaluate_model, generate_report
from evaluation.visualization import (
    plot_f1_comparison, plot_training_curves, plot_tolerance_comparison
)


def setup_device():
    """Setup compute device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    return device


def print_model_summary(model, arch_name):
    """Print model architecture summary."""
    n_params = count_parameters(model)
    print(f"\n  {arch_name}:")
    print(f"    Parameters: {n_params:,}")
    print(f"    Receptive field: {CONFIG.receptive_fields.get(arch_name, 'N/A')} frames")


def main():
    parser = argparse.ArgumentParser(description='LatBOND Demo Lite v11')
    parser.add_argument('--skip_download', action='store_true',
                       help='Skip dataset download (use existing)')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate existing models')
    parser.add_argument('--arch', type=str, default=None,
                       choices=['streaming_cnn', 'causal_transformer', 'tcn'],
                       help='Train only specific architecture')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override max epochs')
    # v10: Loss ablation arguments
    parser.add_argument('--loss_type', type=str, default=None,
                       choices=['bce', 'asymmetric_focal'],
                       help='Override loss function (default: asymmetric_focal)')
    parser.add_argument('--gamma_pos', type=float, default=None,
                       help='Focal loss gamma for positives (default: 1.0)')
    parser.add_argument('--gamma_neg', type=float, default=None,
                       help='Focal loss gamma for negatives (default: 3.0)')
    args = parser.parse_args()
    
    # Override config
    if args.batch_size:
        CONFIG.batch_size = args.batch_size
    if args.epochs:
        CONFIG.epochs = args.epochs
    if args.loss_type:
        CONFIG.loss_type = args.loss_type
    if args.gamma_pos is not None:
        CONFIG.focal_gamma_pos = args.gamma_pos
    if args.gamma_neg is not None:
        CONFIG.focal_gamma_neg = args.gamma_neg
    
    # Architectures to train
    archs = [args.arch] if args.arch else CONFIG.architectures
    
    print("=" * 70)
    print("LatBOND Demo Lite v11")
    print("=" * 70)
    print(f"Architectures: {archs}")
    print(f"Conditions: {CONFIG.conditions}")
    print(f"Budget: {CONFIG.budget_ms}ms ({CONFIG.budget_frames} frames)")
    print(f"Dataset: {CONFIG.subset_fraction*100:.0f}% MAESTRO")
    print(f"Epochs: {CONFIG.epochs} (patience={CONFIG.early_stopping_patience})")
    print(f"Loss: {CONFIG.loss_type}", end="")
    if CONFIG.loss_type == "asymmetric_focal":
        print(f" (gamma_pos={CONFIG.focal_gamma_pos}, gamma_neg={CONFIG.focal_gamma_neg})")
    else:
        print(f" (pos_weight={CONFIG.pos_weight})")
    print(f"Curriculum: cosine (500 → arch-specific target)")
    print(f"Models to train: {len(archs) * len(CONFIG.conditions)}")
    print("=" * 70)
    
    device = setup_device()
    os.makedirs(CONFIG.output_dir, exist_ok=True)
    
    # === Step 1: Data ===
    if not args.skip_download and not args.eval_only:
        print("\nDownloading and preparing dataset...")
        maestro_dir = download_maestro(CONFIG.data_dir)
        subset = create_subset(maestro_dir, CONFIG.subset_fraction)
    else:
        print("\nLoading existing dataset...")
        maestro_dir = os.path.join(CONFIG.data_dir, "maestro-v3.0.0")
        manifest_path = os.path.join(maestro_dir, "subset_manifest.json")
        with open(manifest_path) as f:
            subset = json.load(f)

    # Create dataloaders
    print("\nCreating DataLoaders...")
    loaders = create_dataloaders(subset, CONFIG)
    
    print(f"  Train: {len(loaders.get('train', []))} batches")
    print(f"  Val:   {len(loaders.get('validation', []))} batches")
    print(f"  Test:  {len(loaders.get('test', []))} batches")
    
    # === Step 2: Training ===
    all_histories = {}
    total_start = time.time()
    
    if not args.eval_only:
        print("\nTraining models...")
        
        for arch_name in archs:
            all_histories[arch_name] = {}
            
            for condition in CONFIG.conditions:
                model_id = f"{arch_name}_{condition}"
                print(f"\n{'#'*60}")
                print(f"# Training: {model_id}")
                print(f"{'#'*60}")
                
                # Create fresh model
                model = create_model(arch_name, CONFIG)
                print_model_summary(model, arch_name)
                
                # Create trainer
                trainer = LatBONDTrainer(
                    model, CONFIG, condition, arch_name, device
                )
                
                # Train
                history = trainer.train(
                    loaders['train'],
                    loaders['validation'],
                    CONFIG.output_dir
                )
                
                all_histories[arch_name][condition] = history
                
                # Free GPU memory
                del model, trainer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        train_time = time.time() - total_start
        print(f"\nTotal training time: {train_time/60:.1f} min")
    
    # === Step 3: Evaluation ===
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    all_results = {}
    
    for arch_name in archs:
        all_results[arch_name] = {}
        
        for condition in CONFIG.conditions:
            model_id = f"{arch_name}_{condition}"
            model_path = os.path.join(CONFIG.output_dir, f"{model_id}_best.pt")
            
            if not os.path.exists(model_path):
                print(f"  Skipping {model_id} - no checkpoint found")
                continue
            
            print(f"\n  Evaluating: {model_id}...")
            
            # Load best model
            model = create_model(arch_name, CONFIG).to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Evaluate
            results = evaluate_model(
                model, loaders['test'], CONFIG, condition, arch_name, device
            )
            all_results[arch_name][condition] = results
            
            # Print key metric
            f1_20ms = results.get('tol_20ms', {}).get('f1', 0)
            print(f"    F1 @ 20ms: {f1_20ms:.4f}")
            
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # === Step 4: Report & Visualization ===
    print("\nGenerating report and visualizations...")

    report_text, comparisons = generate_report(all_results, CONFIG.output_dir)

    # Visualizations
    try:
        plot_f1_comparison(all_results, CONFIG.output_dir)
        plot_training_curves(CONFIG.output_dir)
        plot_tolerance_comparison(all_results, CONFIG.output_dir)
    except Exception as e:
        print(f"Visualization error: {e}")
    
    # === Summary ===
    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print(f"Results saved to {CONFIG.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
