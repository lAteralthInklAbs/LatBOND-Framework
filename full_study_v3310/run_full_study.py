#!/usr/bin/env python3
"""
LatBOND Full Study - Main Orchestrator
=======================================
Trains and evaluates 42 models:
  3 architectures x 5 latency budgets x 3 conditions - 3 redundant (buffered@0ms)

Designed to run:
  - LOCALLY: Sequential on single GPU (~126 hours on T4)
  - ON VERTEX AI: One architecture per job (14 models each, ~42h per job)

Usage (local):
    python run_full_study.py --mode local

Usage (single architecture for Vertex AI):
    python run_full_study.py --mode single_arch --architecture streaming_cnn
"""

import os
import sys
import json
import time
import argparse
import glob as glob_module
import torch
import numpy as np
from datetime import datetime

# v2 #21: Enable cuDNN benchmark and TF32 for faster training
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LatBONDConfig
from data.download_maestro import download_and_prepare
from data.maestro_dataset import create_dataloaders
from models import create_model
from training.trainer import LatBONDTrainer
from evaluation.metrics import evaluate_model
from evaluation.statistical_analysis import run_full_analysis
from evaluation.visualization import generate_all_thesis_figures


def train_single_model(arch, budget_ms, condition, config,
                        train_loader, val_loader, test_loader, device,
                        use_amp=True, use_compile=True):
    """
    Train and evaluate one model configuration.

    v2 optimizations: AMP mixed precision (#20), torch.compile (#22).

    Returns:
        Dict with training history, test results, and metadata.
    """
    model_name = f"{arch}_{budget_ms}ms_{condition}"
    output_dir = config.get_output_dir(arch, budget_ms, condition)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"  Architecture: {arch}")
    print(f"  Latency budget: {budget_ms}ms ({config.get_budget_frames(budget_ms)} frames)")
    print(f"  Condition: {condition}")
    print(f"  Output: {output_dir}")
    print(f"  AMP: {use_amp and device == 'cuda'}")
    print(f"  torch.compile: {use_compile}")
    print(f"{'='*70}")

    # Configure for this specific model
    config.current_budget_ms = budget_ms
    config.current_condition = condition
    config.current_architecture = arch

    # Warn about TCN at low budgets
    if arch == 'tcn' and budget_ms < 40:
        print(f"  WARNING: TCN receptive field (65 frames=650ms) >> budget ({budget_ms}ms).")
        print(f"    Expect heavy zero-padding. Results should be interpreted with caution.")

    # v2 #20: Set use_amp on config so trainer can read it
    config.use_amp = use_amp and (device == 'cuda')

    # Create model
    model = create_model(arch, config).to(device)

    # v2 #22: torch.compile DISABLED - Vertex AI base image missing libcuda.so for Triton inductor
    # Savings were only ~3-5% for models this small, not worth the crash risk
    use_compile = False
    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print(f"  [INFO] Model compiled with torch.compile")
        except Exception as e:
            print(f"  [WARN] torch.compile failed: {e}")

    # Create trainer
    trainer = LatBONDTrainer(
        model=model,
        config=config,
        condition=condition,
        arch_name=arch,
        device=device,
    )

    # Train
    start_time = time.time()
    history = trainer.train(train_loader, val_loader, output_dir)
    train_time = time.time() - start_time

    print(f"  Training time: {train_time/3600:.1f} hours")
    print(f"  Best val F1: {trainer.best_val_f1:.4f} (threshold={trainer.best_threshold:.2f})")

    # Test evaluation
    # CRITICAL: Load best model checkpoint before test evaluation
    # Without this, test uses last-epoch weights (up to 30 epochs past best)
    best_model_path = os.path.join(output_dir, f"{arch}_{condition}_best.pt")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded best model from epoch {checkpoint.get('epoch', '?')}")
    else:
        print(f"  [WARN] No best checkpoint found, using last-epoch weights")

    test_results = trainer.evaluate_test(test_loader)

    # Compile results
    result = {
        'model_name': model_name,
        'architecture': arch,
        'budget_ms': budget_ms,
        'condition': condition,
        'training_time_hours': train_time / 3600,
        'best_val_f1': trainer.best_val_f1,
        'best_threshold': trainer.best_threshold,
        'epochs_trained': trainer.current_epoch,
        'aggregate_f1': test_results['aggregate_f1'],
        'file_f1_scores': test_results['file_f1_scores'],
        'file_f1_mean': test_results['file_f1_mean'],
        'file_f1_std': test_results['file_f1_std'],
        'training_history': history,
    }

    # Save individual results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json_safe = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in result.items()}
        json.dump(json_safe, f, indent=2, default=str)

    # v2 #19: Per-model GCS upload
    gcs_output = os.environ.get('LATBOND_GCS_OUTPUT', '')
    if gcs_output:
        try:
            from google.cloud import storage
            bucket_name = gcs_output.replace('gs://', '').split('/')[0]
            prefix = '/'.join(gcs_output.replace('gs://', '').split('/')[1:]).rstrip('/')
            client = storage.Client()
            bucket = client.bucket(bucket_name)

            # Upload all files in this model's output directory
            for local_file in glob_module.glob(f"{output_dir}/**", recursive=True):
                if os.path.isfile(local_file):
                    rel_path = os.path.relpath(local_file, config.output_base_dir)
                    blob_path = f"{prefix}/{rel_path}"
                    bucket.blob(blob_path).upload_from_filename(local_file)
            print(f"  -> Uploaded {model_name} to GCS")
        except Exception as e:
            print(f"  -> GCS upload failed: {e}")

    # Clean up GPU memory
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_architecture(arch, config, loaders, device, use_amp=True, use_compile=True,
                      job_index=None, total_jobs=None):
    """
    Run all models for one architecture (15 result entries: 5 budgets x 3 conditions).

    v2 #18: Support for job partitioning (6 jobs x 7 models each).
    v2 #19: Per-model GCS upload handled in train_single_model via LATBOND_GCS_OUTPUT env var.
    """
    # Get all training configs for this architecture
    all_configs = [(budget, cond) for budget in config.latency_budgets_ms
                   for cond in config.training_conditions
                   if config.is_valid_training(budget, cond)]

    # v2 #18: Partition configs across jobs
    if job_index is not None and total_jobs is not None:
        chunk_size = (len(all_configs) + total_jobs - 1) // total_jobs
        start_idx = job_index * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_configs))
        assigned_configs = all_configs[start_idx:end_idx]
        print(f"\n  Job {job_index+1}/{total_jobs}: configs {start_idx+1}-{end_idx} of {len(all_configs)}")
    else:
        assigned_configs = all_configs

    print(f"\n{'#'*70}")
    print(f"# Architecture: {arch}")
    print(f"# Models to train: {len(assigned_configs)}")
    print(f"{'#'*70}")

    results = {}

    for budget_ms, condition in assigned_configs:
        # v3.1 CHANGE 9: Condition sequencing fix - wrap each condition in try/except
        try:
            print(f"\n[START] {arch} @ {budget_ms}ms {condition}")
            result = train_single_model(
                arch, budget_ms, condition, config,
                loaders['train'], loaders['validation'], loaders['test'], device,
                use_amp=use_amp, use_compile=use_compile,
            )
            results[(arch, budget_ms, condition)] = result
            print(f"[COMPLETE] {arch} @ {budget_ms}ms {condition}")
        except Exception as e:
            print(f"[ERROR] {arch} @ {budget_ms}ms {condition}: {e}")
            import traceback
            traceback.print_exc()
            # Continue to next condition instead of crashing entire study
            continue

    # Fill in Buffered@0ms using the Buffered@20ms model as proxy
    if (arch, 20, 'buffered') in results:
        print(f"\n  Filling in Buffered@0ms using Buffered@20ms proxy...")
        proxy_result = results[(arch, 20, 'buffered')].copy()
        proxy_result['budget_ms'] = 0
        proxy_result['model_name'] = f"{arch}_0ms_buffered"

        # Save to disk
        proxy_dir = config.get_output_dir(arch, 0, 'buffered')
        os.makedirs(proxy_dir, exist_ok=True)
        with open(os.path.join(proxy_dir, 'results.json'), 'w') as f:
            json_safe = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in proxy_result.items()}
            json.dump(json_safe, f, indent=2, default=str)

        results[(arch, 0, 'buffered')] = proxy_result
        print(f"  Buffered@0ms result saved ({proxy_dir})")

    return results


def main():
    parser = argparse.ArgumentParser(description='LatBOND Full Study')
    parser.add_argument('--mode', choices=['local', 'single_arch', 'single_job'], default='local',
                        help='Run mode: local (all), single_arch (14 models), single_job (subset)')
    parser.add_argument('--architecture', default=None,
                        choices=['streaming_cnn', 'causal_transformer', 'tcn'],
                        help='Architecture to train (required for single_arch mode)')
    parser.add_argument('--output_dir', default='outputs',
                        help='Base output directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip dataset download')
    # v2 #18: Job partitioning for parallel runs
    parser.add_argument('--job_index', type=int, default=None,
                        help='Job index for partitioned runs (0-based)')
    parser.add_argument('--total_jobs', type=int, default=None,
                        help='Total number of parallel jobs')
    # v2 #20, #22: GPU optimizations
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable AMP mixed precision')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile')
    # v3: Pilot mode — filter to specific budgets
    parser.add_argument('--budgets', type=str, default=None,
                        help='Budget list. Use semicolons: "0;20;40" or single: "40". Default: all budgets')
    # v3.3.4: Single condition mode for 9-job parallel pilot
    parser.add_argument('--condition', type=str, default=None,
                        choices=['matched', 'truncated', 'buffered'],
                        help='Single condition to train (for parallel 9-job mode)')
    # v3.3.6: Epoch cap for smoke tests
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Override max epochs for all architectures (for smoke tests)')
    parser.add_argument('--matched_epochs', type=int, default=None,
                        help='Override epochs for matched condition (default: 100)')
    args = parser.parse_args()

    # Setup
    config = LatBONDConfig()
    config.output_base_dir = args.output_dir

    # v3: Pilot mode — override budget list if specified
    if args.budgets:
        # Support both semicolon-separated (Vertex AI safe) and comma-separated (local)
        sep = ';' if ';' in args.budgets else ','
        budget_list = [int(b.strip()) for b in args.budgets.split(sep)]
        config.latency_budgets_ms = budget_list
        print(f"[PILOT MODE] Budgets restricted to: {budget_list}")

    # v3.3.4: Single condition mode — override condition list if specified
    if args.condition:
        config.training_conditions = [args.condition]
        print(f"[PILOT MODE] Condition restricted to: {args.condition}")

    # v3.3.6: Epoch cap for smoke tests
    if args.max_epochs is not None:
        config._max_epochs_override = args.max_epochs
        print(f"[SMOKE TEST] Max epochs capped to: {args.max_epochs}")

    # v3.3.11: Matched condition epoch override
    if args.matched_epochs is not None:
        config.matched_epochs = args.matched_epochs
        print(f"[CONFIG] Matched epochs set to: {args.matched_epochs}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("LatBOND Full Study")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print(f"Training configs: {len(config.get_training_configurations())}")
    print(f"Evaluation configs: {len(config.get_evaluation_configurations())}")
    print("=" * 70)

    # Data
    if not args.skip_download:
        print("\nPreparing dataset...")
        subset = download_and_prepare(config)
    else:
        from data.download_maestro import load_existing_subset
        maestro_dir = os.path.join(config.data_dir, "maestro-v3.0.0")
        subset = load_existing_subset(maestro_dir, config)

    print("\nCreating DataLoaders...")
    loaders = create_dataloaders(subset, config)
    print(f"  Train: {len(loaders.get('train', []))} batches")
    print(f"  Val:   {len(loaders.get('validation', []))} batches")
    print(f"  Test:  {len(loaders.get('test', []))} batches")

    # v2 GPU optimization flags
    use_amp = not args.no_amp and device == 'cuda'
    use_compile = not args.no_compile

    print(f"  AMP: {use_amp}")
    print(f"  torch.compile: {use_compile}")
    print("=" * 70)

    # Training
    all_results = {}
    total_start = time.time()

    if args.mode == 'single_arch':
        assert args.architecture, "Must specify --architecture in single_arch mode"
        arch_results = run_architecture(
            args.architecture, config, loaders, device,
            use_amp=use_amp, use_compile=use_compile,
            job_index=args.job_index, total_jobs=args.total_jobs,
        )
        all_results.update(arch_results)
    elif args.mode == 'single_job':
        # v2 #18: Partition all 42 models across jobs
        assert args.architecture and args.job_index is not None, \
            "single_job mode requires --architecture and --job_index"
        arch_results = run_architecture(
            args.architecture, config, loaders, device,
            use_amp=use_amp, use_compile=use_compile,
            job_index=args.job_index, total_jobs=args.total_jobs or 6,
        )
        all_results.update(arch_results)
    else:
        # Local mode: run all 3 architectures sequentially
        for arch in ['streaming_cnn', 'causal_transformer', 'tcn']:
            arch_results = run_architecture(
                arch, config, loaders, device,
                use_amp=use_amp, use_compile=use_compile,
            )
            all_results.update(arch_results)

    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time/3600:.1f} hours")

    # Save aggregated results
    os.makedirs(args.output_dir, exist_ok=True)
    agg_path = os.path.join(args.output_dir, 'all_results.json')
    with open(agg_path, 'w') as f:
        json_safe = {}
        for key, val in all_results.items():
            str_key = f"{key[0]}_{key[1]}ms_{key[2]}"
            json_safe[str_key] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                   for k, v in val.items()}
        json.dump(json_safe, f, indent=2, default=str)

    # Run full statistical analysis (only if we have all results)
    if args.mode == 'local':
        print("\n" + "="*70)
        print("RUNNING FULL STATISTICAL ANALYSIS")
        print("="*70)

        analysis = run_full_analysis(all_results, config)

        analysis_path = os.path.join(args.output_dir, 'statistical_analysis.json')
        with open(analysis_path, 'w') as f:
            # Convert tuple keys to strings for JSON
            serializable = {}
            for k, v in analysis.items():
                if isinstance(v, dict):
                    serializable[k] = {}
                    for k2, v2 in v.items():
                        if isinstance(k2, tuple):
                            serializable[k][str(k2)] = v2
                        else:
                            serializable[k][k2] = v2
                else:
                    serializable[k] = v
            json.dump(serializable, f, indent=2, default=str)

        # Generate thesis figures
        generate_all_thesis_figures(all_results, analysis, args.output_dir)

        # Print summary
        print_summary(all_results, analysis, config)

    print(f"\nAll outputs saved to {args.output_dir}/")


def print_summary(all_results, analysis, config):
    """Print thesis-ready summary tables."""
    print("\n" + "="*70)
    print("LATBOND FULL STUDY - RESULTS SUMMARY")
    print("="*70)

    # Table 1: Per-budget matched vs truncated
    print("\n### Per-Budget Comparison (Matched vs Truncated)")
    print(f"{'Budget':>8} {'Matched F1':>12} {'Truncated F1':>13} {'Delta':>8} {'p-value':>10} {'Cohen d':>9} {'Sig?':>5}")
    print("-" * 70)

    for budget in config.latency_budgets_ms:
        ba = analysis.get('per_budget_comparisons', {}).get(budget, {})
        if not ba:
            continue
        m_mean = ba.get('matched_file_stats', {}).get('mean', 0)
        t_mean = ba.get('truncated_file_stats', {}).get('mean', 0)
        delta = m_mean - t_mean
        p = ba.get('paired_ttest', {}).get('p_value', 1.0)
        d = ba.get('cohens_d', {}).get('d', 0)
        sig = 'Yes*' if ba.get('paired_ttest', {}).get('significant', False) else 'No'
        print(f"{budget:>6}ms {m_mean:>12.4f} {t_mean:>13.4f} {delta:>+8.4f} {p:>10.2e} {d:>9.3f} {sig:>5}")

    # Table 2: Recovery calculations
    if analysis.get('recovery'):
        print("\n### Recovery Calculations")
        print(f"{'Arch':>20} {'Budget':>8} {'Recovery%':>10}")
        print("-" * 45)
        for key, rec in analysis['recovery'].items():
            arch, budget = key if isinstance(key, tuple) else eval(key)
            if rec.get('recovery_percentage') is not None:
                print(f"{arch:>20} {budget:>6}ms {rec['recovery_percentage']:>9.1f}%")


if __name__ == '__main__':
    main()
