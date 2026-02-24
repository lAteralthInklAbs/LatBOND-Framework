#!/usr/bin/env python3
"""
LatBOND v3.3.6 SMOKE TEST: 1 job, 2 epochs, measure real timing.

Launches a single CNN buffered job (simplest path — no crops, no matched
complexity) with max_epochs=2. Purpose: measure actual per-phase timing
(train=Xs, val=Ys) so we can project total cost before committing 9 GPUs.

After epoch 1 completes, check logs for:
  Epoch 1/2 | ... | Time: Xs (train=As, val=Bs)

Then:
  - If train < 300s:   total pilot ~6 hrs, ~$80
  - If train 300-600s: total pilot ~12 hrs, ~$160
  - If train 600-1500s: total pilot ~25 hrs, ~$340
  - If train > 1500s:  need optimization before full pilot

Usage:
    python scripts/launch_smoke_test.py              # Submit 1 job
    python scripts/launch_smoke_test.py --dry_run    # Preview only
    python scripts/launch_smoke_test.py --matched    # Test matched path instead

Monitor:
    gcloud ai custom-jobs list --project=YOUR_GCP_PROJECT --region=us-west1 \\
        --filter="displayName~latbond-v336-smoke"
    
    gcloud ai custom-jobs stream-logs <JOB_ID> --project=YOUR_GCP_PROJECT --region=us-west1
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
CONTAINER_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/latbond/training:v3.3.6"


def main():
    parser = argparse.ArgumentParser(description='Launch v3.3.6 smoke test: 1 job, 2 epochs')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print job config without submitting')
    parser.add_argument('--matched', action='store_true',
                       help='Test matched condition instead of buffered (slower, tests crop loop)')
    parser.add_argument('--transformer', action='store_true',
                       help='Test transformer instead of CNN (slowest arch)')
    args = parser.parse_args()

    # Pick test configuration
    arch = 'causal_transformer' if args.transformer else 'streaming_cnn'
    condition = 'matched' if args.matched else 'buffered'
    arch_short = 'xfmr' if args.transformer else 'scnn'
    cond_short = condition[:3]
    
    job_name = f"latbond-v336-smoke-{arch_short}-{cond_short}"

    # Container args — note --max_epochs 2
    container_args = [
        "--mode", "single_arch",
        "--architecture", arch,
        "--output_dir", f"gs://{BUCKET}/outputs/v336-smoke/",
        "--gcs_prefix", f"gs://{BUCKET}/datasets/maestro-v3/maestro-v3.0.0",
        "--seed", "42",
        "--budgets", "40",
        "--condition", condition,
        "--max_epochs", "2",
    ]

    print("=" * 60)
    print("SMOKE TEST: v3.3.6 timing validation")
    print("=" * 60)
    print(f"  Job:           {job_name}")
    print(f"  Architecture:  {arch}")
    print(f"  Condition:     {condition}")
    print(f"  Budget:        40ms")
    print(f"  Max epochs:    2 (just measuring timing)")
    print(f"  Data:          30% MAESTRO (production scale)")
    print(f"  GPU:           L4 (g2-standard-16)")
    print(f"  Container:     {CONTAINER_URI}")
    print(f"  Est. cost:     < $5")
    print("=" * 60)
    print(f"\n  Args: {' '.join(container_args)}")

    if args.dry_run:
        print("\n  [DRY RUN] Would submit. Remove --dry_run to launch.")
        return

    if not HAS_AIPLATFORM:
        print("\n  [ERROR] google-cloud-aiplatform required.")
        print("  pip install google-cloud-aiplatform")
        sys.exit(1)

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET}/staging"
    )

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
        print(f"\n  [SUBMITTED] {job_name}")
        print(f"\n  Monitor with:")
        print(f"    gcloud ai custom-jobs list --project={PROJECT_ID} --region={REGION} --filter=\"displayName~smoke\"")
        print(f"\n  Stream logs:")
        print(f"    gcloud ai custom-jobs stream-logs <JOB_ID> --project={PROJECT_ID} --region={REGION}")
        print(f"\n  What to look for in logs:")
        print(f"    Epoch 1/2 | ... | Time: Xs (train=As, val=Bs)")
        print(f"                                ^^^^^      ^^^^")
        print(f"                          This is what we need")
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
