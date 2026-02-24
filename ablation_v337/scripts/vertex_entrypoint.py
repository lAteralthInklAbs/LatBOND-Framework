#!/usr/bin/env python3
"""
Vertex AI Entrypoint Script.

Parses Vertex AI arguments, sets environment variables,
runs training locally within the container, and uploads results to GCS.

CRITICAL: Standard Python cannot write to gs:// URIs directly.
This script redirects output to /tmp/outputs and uploads to GCS after
training completes.

Called by: Dockerfile ENTRYPOINT
Calls: run_full_study.py
"""

import os
import sys
import argparse
import subprocess
import glob

try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False


def upload_folder_to_gcs(local_folder: str, gcs_uri: str):
    """
    Upload an entire local folder to a GCS URI.

    Args:
        local_folder: Local directory path (e.g., /tmp/outputs)
        gcs_uri: GCS destination (e.g., gs://YOUR_GCP_PROJECT-data/outputs/streaming_cnn/)
    """
    if not gcs_uri.startswith('gs://'):
        print(f"Skipping upload: {gcs_uri} is not a gs:// path")
        return

    if not HAS_GCS:
        print("[WARN] google-cloud-storage not installed, skipping upload")
        return

    parts = gcs_uri[5:].split('/', 1)
    bucket_name = parts[0]
    prefix = parts[1].rstrip('/') if len(parts) > 1 else ""

    print(f"\nUploading {local_folder} -> gs://{bucket_name}/{prefix}/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    uploaded = 0
    for local_file in glob.glob(f"{local_folder}/**", recursive=True):
        if os.path.isfile(local_file):
            rel_path = os.path.relpath(local_file, local_folder)
            blob_path = f"{prefix}/{rel_path}" if prefix else rel_path

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file)
            uploaded += 1
            print(f"  Uploaded: {rel_path}")

    print(f"Uploaded {uploaded} files to gs://{bucket_name}/{prefix}/")


def main():
    parser = argparse.ArgumentParser(description='LatBOND Vertex AI Entrypoint')
    parser.add_argument('--mode', required=True,
                        choices=['local', 'single_arch', 'single_job'])
    parser.add_argument('--architecture', required=True,
                        choices=['streaming_cnn', 'causal_transformer', 'tcn'])
    parser.add_argument('--output_dir', required=True,
                        help='GCS output path (e.g., gs://YOUR_GCP_PROJECT-data/outputs/streaming_cnn/)')
    parser.add_argument('--gcs_prefix', required=True,
                        help='GCS URI for MAESTRO dataset')
    parser.add_argument('--seed', default='42')
    # v2 #18: Job partitioning arguments
    parser.add_argument('--job_index', type=int, default=None,
                        help='Job index for partitioned runs (0-based)')
    parser.add_argument('--total_jobs', type=int, default=None,
                        help='Total number of parallel jobs')
    # v3: Pilot budget filter
    parser.add_argument('--budgets', type=str, default=None,
                        help='Comma-separated budget list for pilot runs')
    # v3.3.4: Single condition mode for 9-job parallel pilot
    parser.add_argument('--condition', type=str, default=None,
                        choices=['matched', 'truncated', 'buffered'],
                        help='Single condition to train (for parallel 9-job mode)')

    args = parser.parse_args()

    # CRITICAL: Python cannot write to gs:// paths directly.
    # Train to local /tmp/outputs, then upload to GCS after completion.
    gcs_output_dir = args.output_dir
    local_output_dir = "/tmp/outputs"

    print("=" * 60)
    print("LatBOND Vertex AI Entrypoint (v2)")
    print("=" * 60)
    print(f"  Architecture:    {args.architecture}")
    print(f"  Mode:            {args.mode}")
    if args.job_index is not None:
        print(f"  Job partition:   {args.job_index + 1}/{args.total_jobs}")
    print(f"  Local output:    {local_output_dir}")
    print(f"  GCS destination: {gcs_output_dir}")
    print(f"  GCS data:        {args.gcs_prefix}")
    print(f"  Seed:            {args.seed}")
    print("=" * 60)

    # Set env var for dataset access
    os.environ['MAESTRO_GCS_PREFIX'] = args.gcs_prefix

    # v2 #19: Set env var for per-model upload callback
    os.environ['LATBOND_GCS_OUTPUT'] = gcs_output_dir

    # Build command - pass LOCAL path, not gs://
    cmd = [
        sys.executable, "run_full_study.py",
        "--mode", args.mode,
        "--architecture", args.architecture,
        "--output_dir", local_output_dir,
        "--seed", args.seed,
    ]

    # v2 #18: Add job partitioning args
    if args.job_index is not None:
        cmd.extend(["--job_index", str(args.job_index)])
    if args.total_jobs is not None:
        cmd.extend(["--total_jobs", str(args.total_jobs)])
    # v3: Budget filter passthrough
    if args.budgets is not None:
        cmd.extend(["--budgets", args.budgets])
    # v3.3.4: Condition filter passthrough
    if args.condition is not None:
        cmd.extend(["--condition", args.condition])

    print(f"\nLaunching: {' '.join(cmd)}")
    sys.stdout.flush()

    # Run training
    process = subprocess.run(cmd)

    # Upload results to GCS (even on partial failure - saves whatever completed)
    print(f"\nTraining finished (exit code {process.returncode}). Uploading artifacts...")
    try:
        upload_folder_to_gcs(local_output_dir, gcs_output_dir)
        print("Upload to GCS complete")
    except Exception as e:
        print(f"Upload failed: {e}")

    sys.exit(process.returncode)


if __name__ == '__main__':
    main()
