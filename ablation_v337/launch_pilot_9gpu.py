#!/usr/bin/env python3
"""
Launch LatBOND v3.3.7 Pilot: 9 jobs on 9 GPUs (Toronto).
3 architectures x 3 conditions @ 40ms budget.

Each job trains ONE model, allowing full parallelization.
v3.3.7: ~150s/epoch. Total time: ~2.5 hours with 9 parallel GPUs.

Usage:
    python scripts/launch_pilot_9gpu.py              # Submit all 9 jobs
    python scripts/launch_pilot_9gpu.py --dry_run    # Preview without submitting
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
REGION = "northamerica-northeast2"  # Toronto — v3.3.6 migrated here from us-west1
BUCKET = f"{PROJECT_ID}-data"
CONTAINER_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/latbond/training:v3.3.7"

PILOT_CONFIGS = [
    {'arch': 'streaming_cnn', 'budget': 40, 'condition': 'matched'},
    {'arch': 'streaming_cnn', 'budget': 40, 'condition': 'truncated'},
    {'arch': 'streaming_cnn', 'budget': 40, 'condition': 'buffered'},
    {'arch': 'causal_transformer', 'budget': 40, 'condition': 'matched'},
    {'arch': 'causal_transformer', 'budget': 40, 'condition': 'truncated'},
    {'arch': 'causal_transformer', 'budget': 40, 'condition': 'buffered'},
    {'arch': 'tcn', 'budget': 40, 'condition': 'matched'},
    {'arch': 'tcn', 'budget': 40, 'condition': 'truncated'},
    {'arch': 'tcn', 'budget': 40, 'condition': 'buffered'},
]


def launch_job(config, dry_run=False):
    """Launch a single Vertex AI job."""
    arch = config['arch']
    budget = config['budget']
    condition = config['condition']

    # Short names for job ID
    arch_short = arch.replace('streaming_', 's').replace('causal_', 'c').replace('transformer', 'xfmr')
    cond_short = condition[:3]
    job_name = f"latbond-v337-{arch_short}-{budget}ms-{cond_short}"

    # Container args
    container_args = [
        "--mode", "single_arch",
        "--architecture", arch,
        "--output_dir", f"gs://{BUCKET}/outputs/v337-pilot/{arch}/{budget}ms/{condition}/",
        "--gcs_prefix", f"gs://{BUCKET}/datasets/maestro-v3/maestro-v3.0.0",
        "--seed", "42",
        "--budgets", str(budget),
        "--condition", condition,
    ]

    print(f"  {job_name}")
    print(f"    arch={arch}, budget={budget}ms, condition={condition}")

    if dry_run:
        print(f"    [DRY RUN] Would submit with args: {' '.join(container_args)}")
        return True

    if not HAS_AIPLATFORM:
        print("    [ERROR] google-cloud-aiplatform required")
        return False

    try:
        job = aiplatform.CustomJob(
            display_name=job_name,
            worker_pool_specs=[{
                "machine_spec": {
                    "machine_type": "g2-standard-16",
                    "accelerator_type": "NVIDIA_L4",
                    "accelerator_count": 1,
                },
                "replica_count": 1,
                "disk_spec": {
                    "boot_disk_type": "pd-ssd",
                    "boot_disk_size_gb": 200,
                },
                "container_spec": {
                    "image_uri": CONTAINER_URI,
                    "args": container_args,
                },
            }],
        )
        job.submit(enable_web_access=False)
        print(f"    [SUBMITTED]")
        return True
    except Exception as e:
        print(f"    [ERROR] {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Launch v3.3.7 pilot: 9 parallel jobs')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print job config without submitting')
    args = parser.parse_args()

    print("=" * 60)
    print("LAUNCHING v3.3.7 PILOT (Toronto)")
    print("=" * 60)
    print(f"  Region:        {REGION}")
    print(f"  Jobs:          9 (3 arch x 3 conditions)")
    print(f"  Budget:        40ms")
    print(f"  GPU:           L4 (g2-standard-16)")
    print(f"  Container:     {CONTAINER_URI}")
    print(f"  Output:        gs://{BUCKET}/outputs/v337-pilot/")
    print(f"  Est. time:     ~2.5 hours (parallel)")
    print(f"  Est. cost:     ~$15")
    print("=" * 60)

    if not args.dry_run and not HAS_AIPLATFORM:
        print("\n[ERROR] google-cloud-aiplatform required. Install with:")
        print("  pip install google-cloud-aiplatform")
        sys.exit(1)

    if not args.dry_run:
        aiplatform.init(
            project=PROJECT_ID,
            location=REGION,
            staging_bucket="gs://YOUR_GCP_PROJECT-staging-toronto"
        )

    print("\nLaunching jobs:")
    success = 0
    for i, config in enumerate(PILOT_CONFIGS):
        print(f"\n[{i+1}/9]", end="")
        if launch_job(config, dry_run=args.dry_run):
            success += 1
        if not args.dry_run:
            time.sleep(2)  # Small delay between submissions

    print(f"\n{'=' * 60}")
    print(f"Launched {success}/9 jobs")
    print("=" * 60)

    if args.dry_run:
        print("\nTo submit for real, remove --dry_run")
    else:
        print("\nMonitor jobs:")
        print(f"  gcloud ai custom-jobs list --project={PROJECT_ID} --region={REGION} \\")
        print(f"    --filter=\"displayName:latbond-v337\" \\")
        print(f"    --format=\"table(displayName,state,createTime)\" --sort-by=createTime")
        print("\nCheck results when complete:")
        print(f"  gcloud storage ls gs://{BUCKET}/outputs/v337-pilot/")
        print("\nCompare matched vs truncated F1 for each architecture.")
        print("If matched > truncated by 3+pp consistently: thesis validated!")


if __name__ == '__main__':
    main()
