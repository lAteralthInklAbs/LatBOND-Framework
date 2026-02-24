#!/usr/bin/env python3
"""
Launch LatBOND v3.3.4 Pilot with 12 GPUs.

GPU Allocation:
- CNN (3 jobs):         1 GPU each = 3 GPUs
- TCN (3 jobs):         1 GPU each = 3 GPUs
- Transformer (3 jobs): 2 GPUs each (DDP) = 6 GPUs
Total: 12 GPUs

Estimated time: ~7-8 hours (down from ~33 hours)
Estimated cost: ~$35-50 (Spot pricing)

Usage:
    python scripts/launch_pilot_12gpu.py              # Submit all 9 jobs
    python scripts/launch_pilot_12gpu.py --dry_run    # Preview without submitting
"""
import argparse
import sys
import time

try:
    from google.cloud import aiplatform
    HAS_AIPLATFORM = True
except ImportError:
    HAS_AIPLATFORM = False
    print("[WARN] google-cloud-aiplatform not installed")

PROJECT_ID = "YOUR_GCP_PROJECT"
REGION = "us-west1"
BUCKET = f"{PROJECT_ID}-data"
CONTAINER_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/latbond/training:v3.3.4"

# Job configurations with GPU allocation
PILOT_CONFIGS = [
    # CNN - 1 GPU each (3 total)
    {'arch': 'streaming_cnn', 'budget': 40, 'condition': 'matched', 'gpus': 1},
    {'arch': 'streaming_cnn', 'budget': 40, 'condition': 'truncated', 'gpus': 1},
    {'arch': 'streaming_cnn', 'budget': 40, 'condition': 'buffered', 'gpus': 1},
    # TCN - 1 GPU each (3 total)
    {'arch': 'tcn', 'budget': 40, 'condition': 'matched', 'gpus': 1},
    {'arch': 'tcn', 'budget': 40, 'condition': 'truncated', 'gpus': 1},
    {'arch': 'tcn', 'budget': 40, 'condition': 'buffered', 'gpus': 1},
    # Transformer - 2 GPUs each with DDP (6 total)
    {'arch': 'causal_transformer', 'budget': 40, 'condition': 'matched', 'gpus': 2},
    {'arch': 'causal_transformer', 'budget': 40, 'condition': 'truncated', 'gpus': 2},
    {'arch': 'causal_transformer', 'budget': 40, 'condition': 'buffered', 'gpus': 2},
]


def get_machine_spec(gpus):
    """Get machine type based on GPU count."""
    if gpus == 1:
        return "g2-standard-8"   # 8 vCPU, 32GB RAM, 1 L4
    elif gpus == 2:
        return "g2-standard-24"  # 24 vCPU, 96GB RAM, 2 L4s
    elif gpus == 4:
        return "g2-standard-48"  # 48 vCPU, 192GB RAM, 4 L4s
    else:
        return "g2-standard-8"


def launch_job(config, dry_run=False):
    """Launch a single Vertex AI training job."""
    arch = config['arch']
    budget = config['budget']
    condition = config['condition']
    gpus = config['gpus']

    # Short names for job ID
    arch_short = arch.replace('streaming_', 's').replace('causal_', 'c').replace('transformer', 'xfmr')
    cond_short = condition[:3]
    job_name = f"v334-{arch_short}-{budget}ms-{cond_short}-{gpus}gpu"

    machine_type = get_machine_spec(gpus)
    output_dir = f"gs://{BUCKET}/outputs/v334-pilot/{arch}/{budget}ms/{condition}/"

    # Container arguments
    container_args = [
        "--mode", "single_arch",
        "--architecture", arch,
        "--output_dir", output_dir,
        "--gcs_prefix", f"gs://{BUCKET}/datasets/maestro-v3/maestro-v3.0.0",
        "--seed", "42",
        "--budgets", str(budget),
        "--condition", condition,
    ]

    print(f"  {job_name}")
    print(f"    arch={arch}, budget={budget}ms, condition={condition}")
    print(f"    machine={machine_type}, gpus={gpus}")

    if dry_run:
        print(f"    [DRY RUN] Would submit")
        return True

    if not HAS_AIPLATFORM:
        print("    [ERROR] google-cloud-aiplatform required")
        return False

    try:
        # For multi-GPU, we need to use torchrun
        if gpus > 1:
            # Override container command for DDP
            container_command = [
                "torchrun",
                f"--nproc_per_node={gpus}",
                "scripts/vertex_entrypoint_ddp.py",
                f"--arch={arch}",
                f"--budget_ms={budget}",
                f"--condition={condition}",
                f"--output_dir=/tmp/outputs",
            ]
            container_spec = {
                "image_uri": CONTAINER_URI,
                "command": container_command,
            }
        else:
            # Single GPU - use standard entrypoint
            container_spec = {
                "image_uri": CONTAINER_URI,
                "args": container_args,
            }

        job = aiplatform.CustomJob(
            display_name=job_name,
            worker_pool_specs=[{
                "machine_spec": {
                    "machine_type": machine_type,
                    "accelerator_type": "NVIDIA_L4",
                    "accelerator_count": gpus,
                },
                "replica_count": 1,
                "disk_spec": {
                    "boot_disk_type": "pd-ssd",
                    "boot_disk_size_gb": 200,
                },
                "container_spec": container_spec,
            }],
        )
        job.submit(enable_web_access=False)
        print(f"    [SUBMITTED]")
        return True
    except Exception as e:
        print(f"    [ERROR] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Launch v3.3.4 pilot: 12 GPUs')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print job config without submitting')
    args = parser.parse_args()

    total_gpus = sum(c['gpus'] for c in PILOT_CONFIGS)

    print("=" * 70)
    print("LAUNCHING v3.3.4 PILOT - 12 GPUs")
    print("=" * 70)
    print("\nGPU Allocation:")
    print("  CNN:         3 jobs x 1 GPU  = 3 GPUs")
    print("  TCN:         3 jobs x 1 GPU  = 3 GPUs")
    print("  Transformer: 3 jobs x 2 GPUs = 6 GPUs (DDP)")
    print(f"  Total:       9 jobs         = {total_gpus} GPUs")
    print(f"\nContainer: {CONTAINER_URI}")
    print("\nEstimated time: ~7-8 hours")
    print("Estimated cost: ~$35-50 (Spot)")
    print("=" * 70)

    if not args.dry_run:
        if not HAS_AIPLATFORM:
            print("\n[ERROR] google-cloud-aiplatform required. Install with:")
            print("  pip install google-cloud-aiplatform")
            sys.exit(1)

        aiplatform.init(
            project=PROJECT_ID,
            location=REGION,
            staging_bucket=f"gs://{BUCKET}/staging"
        )

        confirm = input("\nPress Enter to launch, or Ctrl+C to cancel...")

    print("\nLaunching jobs:")
    success = 0
    for i, config in enumerate(PILOT_CONFIGS):
        print(f"\n[{i+1}/9]", end="")
        if launch_job(config, dry_run=args.dry_run):
            success += 1
        if not args.dry_run:
            time.sleep(3)  # Delay between launches

    print(f"\n{'=' * 70}")
    print(f"Launched {success}/9 jobs ({total_gpus} GPUs total)")
    print("=" * 70)

    if args.dry_run:
        print("\nTo submit for real, remove --dry_run")
    else:
        print("\nMonitor jobs:")
        print(f"  gcloud ai custom-jobs list --project={PROJECT_ID} --region={REGION}")
        print("\nCheck results when complete:")
        print(f"  gcloud storage ls gs://{BUCKET}/outputs/v334-pilot/")
        print("\nCompare matched vs truncated F1 for each architecture.")
        print("If matched > truncated by 3+pp consistently: thesis validated!")


if __name__ == "__main__":
    main()
