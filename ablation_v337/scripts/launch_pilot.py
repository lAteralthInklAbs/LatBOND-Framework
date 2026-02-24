#!/usr/bin/env python3
"""
Launch v3 PILOT: 1 L4 job, streaming_cnn at 40ms, 3 conditions.

Purpose:
    Validate v3 changes produce matched > truncated on real data
    before committing to the full 42-model $200 run.

What it trains:
    streaming_cnn @ 40ms matched   (fixed context = 98 frames)
    streaming_cnn @ 40ms truncated (full sequence training)
    streaming_cnn @ 40ms buffered  (80ms lookahead)

Expected results:
    matched F1 > truncated F1 by 3-8pp
    Absolute F1 in 0.45-0.55 range (vs v2's 0.39)
    Threshold != 0.30 for at least one condition
    No NaN, no curriculum decay in logs

Cost: ~$25 (1 L4 x ~15 hours)
Time: ~15 hours wall clock

After pilot succeeds, launch full run with:
    python scripts/launch_vertex_ai.py

Usage:
    python scripts/launch_pilot.py              # Submit job
    python scripts/launch_pilot.py --dry_run    # Preview without submitting
"""

import argparse
import sys

try:
    from google.cloud import aiplatform
    HAS_AIPLATFORM = True
except ImportError:
    HAS_AIPLATFORM = False
    print("[WARN] google-cloud-aiplatform not installed")

PROJECT_ID = "YOUR_GCP_PROJECT"
REGION = "us-west1"
BUCKET = f"{PROJECT_ID}-data"
CONTAINER_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/latbond/training:v3.3.3"


def main():
    parser = argparse.ArgumentParser(description='Launch v3 pilot: streaming_cnn @ 40ms')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print job config without submitting')
    parser.add_argument('--budget', type=int, default=40,
                       help='Budget to test (default: 40ms)')
    parser.add_argument('--architecture', default='streaming_cnn',
                       choices=['streaming_cnn', 'causal_transformer', 'tcn'],
                       help='Architecture to pilot (default: streaming_cnn)')
    parser.add_argument('--use_t4', action='store_true',
                       help='Use T4 instead of L4 (cheaper but slower)')
    args = parser.parse_args()

    arch = args.architecture
    budget = args.budget
    job_name = f"latbond-v3-pilot-{arch}-{budget}ms"

    # Container args: single_arch mode with budget filter
    container_args = [
        "--mode", "single_arch",
        "--architecture", arch,
        "--output_dir", f"gs://{BUCKET}/outputs/v3-pilot/{arch}/",
        "--gcs_prefix", f"gs://{BUCKET}/datasets/maestro-v3/maestro-v3.0.0",
        "--seed", "42",
        "--budgets", str(budget),
    ]

    # GPU config
    if args.use_t4:
        machine_spec = {
            "machine_type": "n1-standard-8",
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": 1,
        }
        gpu_name = "T4"
        est_hours = "~20"
        est_cost = "$20"
    else:
        machine_spec = {
            "machine_type": "g2-standard-16",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1,
        }
        gpu_name = "L4"
        est_hours = "~15"
        est_cost = "$25"

    print("=" * 60)
    print("LatBOND v3 PILOT JOB")
    print("=" * 60)
    print(f"  Job name:      {job_name}")
    print(f"  Architecture:  {arch}")
    print(f"  Budget:        {budget}ms")
    print(f"  Conditions:    matched, truncated, buffered")
    print(f"  Models:        3")
    print(f"  GPU:           {gpu_name}")
    print(f"  Container:     {CONTAINER_URI}")
    print(f"  Output:        gs://{BUCKET}/outputs/v3-pilot/{arch}/")
    print(f"  Est. time:     {est_hours}")
    print(f"  Est. cost:     {est_cost}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would submit with args:")
        print(f"  {' '.join(container_args)}")
        print("\nTo submit for real, remove --dry_run")
        return

    if not HAS_AIPLATFORM:
        print("\n[ERROR] google-cloud-aiplatform required. Install with:")
        print("  pip install google-cloud-aiplatform")
        sys.exit(1)

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET}/staging"
    )

    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=[{
            "machine_spec": machine_spec,
            "replica_count": 1,
            "disk_spec": {
                "boot_disk_type": "pd-ssd",
                "boot_disk_size_gb": 200,  # Default 100GB ran out with 30% MAESTRO
            },
            "container_spec": {
                "image_uri": CONTAINER_URI,
                "args": container_args,
            },
        }],
    )

    job.submit(enable_web_access=False)
    print(f"\nSubmitted: {job.display_name}")
    print(f"Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs")
    print(f"\nWhen complete, check results:")
    print(f"  gcloud storage ls gs://{BUCKET}/outputs/v3-pilot/{arch}/{budget}ms/")
    print(f"  gcloud storage cat gs://{BUCKET}/outputs/v3-pilot/{arch}/{budget}ms/matched/results.json")
    print(f"  gcloud storage cat gs://{BUCKET}/outputs/v3-pilot/{arch}/{budget}ms/truncated/results.json")
    print(f"\nCompare matched vs truncated F1. If matched > truncated by 3+pp:")
    print(f"  python scripts/launch_vertex_ai.py  # Launch full 42-model run")


if __name__ == '__main__':
    main()
