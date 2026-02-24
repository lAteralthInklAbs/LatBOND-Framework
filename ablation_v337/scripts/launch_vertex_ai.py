#!/usr/bin/env python3
"""
Launch 6 parallel Vertex AI training jobs.

v2 optimizations:
  - 6 jobs instead of 3 (halves wall clock time)
  - L4 GPU instead of T4 (faster memory bandwidth)
  - Each job trains ~7 models

Job distribution:
  - Jobs 0-1: streaming_cnn (7 models each)
  - Jobs 2-3: causal_transformer (7 models each)
  - Jobs 4-5: tcn (7 models each)

Prerequisites:
  - gcloud auth configured
  - Vertex AI API enabled
  - Training container built and pushed to Artifact Registry
  - MAESTRO dataset uploaded to GCS bucket

Usage:
    python scripts/launch_vertex_ai.py
    python scripts/launch_vertex_ai.py --dry_run  # Print jobs without submitting
    python scripts/launch_vertex_ai.py --jobs 3   # Launch only 3 jobs (14 models each)
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
CONTAINER_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/latbond/training:v3.3"

ARCHITECTURES = ["streaming_cnn", "causal_transformer", "tcn"]

# v2 #18: 6 jobs configuration (2 per architecture)
JOB_CONFIGS = [
    ("streaming_cnn", 0, 2),      # Jobs 0-1 for streaming_cnn
    ("streaming_cnn", 1, 2),
    ("causal_transformer", 0, 2), # Jobs 2-3 for causal_transformer
    ("causal_transformer", 1, 2),
    ("tcn", 0, 2),                # Jobs 4-5 for tcn
    ("tcn", 1, 2),
]


def create_job(architecture: str, job_index: int = None, total_jobs: int = None,
               use_l4: bool = True):
    """
    Create a single Vertex AI training job.

    v2 changes:
      - L4 GPU instead of T4 (#17)
      - Job partitioning support (#18)
    """
    if not HAS_AIPLATFORM:
        raise ImportError("google-cloud-aiplatform required")

    if job_index is not None:
        job_name = f"YOUR_GCP_PROJECT-{architecture}-job{job_index}"
    else:
        job_name = f"YOUR_GCP_PROJECT-{architecture}"

    args = [
        "--mode", "single_arch",
        "--architecture", architecture,
        "--output_dir", f"gs://{BUCKET}/outputs/{architecture}/",
        "--gcs_prefix", f"gs://{BUCKET}/datasets/maestro-v3/maestro-v3.0.0",
        "--seed", "42",
    ]

    # v2 #18: Add job partitioning args
    if job_index is not None and total_jobs is not None:
        args.extend(["--job_index", str(job_index), "--total_jobs", str(total_jobs)])

    # v2 #17: L4 GPU with g2 machine type
    if use_l4:
        machine_spec = {
            "machine_type": "g2-standard-16",
            "accelerator_type": "NVIDIA_L4",
            "accelerator_count": 1,
        }
    else:
        machine_spec = {
            "machine_type": "n1-standard-8",
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": 1,
        }

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
                "args": args,
            },
        }],
    )

    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true',
                       help='Print jobs without submitting')
    parser.add_argument('--project', default=PROJECT_ID,
                       help='GCP project ID')
    parser.add_argument('--region', default=REGION,
                       help='GCP region')
    # v2 #18: Configurable number of jobs
    parser.add_argument('--jobs', type=int, default=6, choices=[3, 6],
                       help='Number of parallel jobs (3 or 6)')
    # v2 #17: GPU selection
    parser.add_argument('--use_t4', action='store_true',
                       help='Use T4 GPU instead of L4')
    args = parser.parse_args()

    if not HAS_AIPLATFORM and not args.dry_run:
        print("[ERROR] google-cloud-aiplatform required. Install with:")
        print("  pip install google-cloud-aiplatform")
        sys.exit(1)

    if HAS_AIPLATFORM:
        aiplatform.init(
            project=args.project,
            location=args.region,
            staging_bucket=f"gs://{BUCKET}/staging"
        )

    use_l4 = not args.use_t4
    gpu_name = "L4" if use_l4 else "T4"

    jobs = []

    if args.jobs == 6:
        # v2: 6 jobs (2 per architecture, ~7 models each)
        for arch, job_idx, total in JOB_CONFIGS:
            if args.dry_run:
                print(f"[DRY RUN] Would submit: YOUR_GCP_PROJECT-{arch}-job{job_idx}")
                print(f"  Architecture: {arch}")
                print(f"  Job partition: {job_idx+1}/{total}")
                print(f"  Models: ~7")
                print(f"  GPU: {gpu_name}")
                print(f"  Estimated time: ~22-35 hours on {gpu_name}")
                print()
            else:
                job = create_job(arch, job_idx, total, use_l4=use_l4)
                job.submit(enable_web_access=False)
                jobs.append(job)
                print(f"Submitted: {job.display_name}")
    else:
        # Legacy: 3 jobs (1 per architecture, 14 models each)
        for arch in ARCHITECTURES:
            if args.dry_run:
                print(f"[DRY RUN] Would submit: YOUR_GCP_PROJECT-{arch}")
                print(f"  Architecture: {arch}")
                print(f"  Models: 14")
                print(f"  GPU: {gpu_name}")
                print(f"  Estimated time: ~42 hours on T4, ~30 hours on L4")
                print()
            else:
                job = create_job(arch, use_l4=use_l4)
                job.submit(enable_web_access=False)
                jobs.append(job)
                print(f"Submitted: {job.display_name}")

    if not args.dry_run and jobs:
        print(f"\nSubmitted {len(jobs)} parallel jobs")
        if args.jobs == 6:
            print(f"  Estimated total wall time: ~35-40 hours")
            print(f"  Estimated total cost: ~$200")
        else:
            print(f"  Estimated total wall time: ~50-60 hours")
            print(f"  Estimated total cost: ~$200")
        print(f"\nMonitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs")


if __name__ == '__main__':
    main()
