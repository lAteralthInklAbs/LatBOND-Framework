#!/usr/bin/env python3
"""
Vertex AI Entrypoint for LatBOND v3.3.10
==========================================
Bridges Vertex AI container args → run_full_study.py

Handles:
  1. --gcs_prefix → MAESTRO_GCS_PREFIX env var (for dataset download)
  2. --output_dir gs://... → local /app/outputs + LATBOND_GCS_OUTPUT env var
  3. Passes remaining args to run_full_study.py
"""

import os
import sys
import argparse

def main():
    # Parse our args, pass unknown to run_full_study.py
    parser = argparse.ArgumentParser()
    parser.add_argument('--gcs_prefix', type=str, default=None,
                        help='GCS path to MAESTRO dataset')
    parser.add_argument('--output_dir', type=str, default='/app/outputs')
    args, remaining = parser.parse_known_args()

    # Set GCS prefix for dataset download
    if args.gcs_prefix:
        os.environ['MAESTRO_GCS_PREFIX'] = args.gcs_prefix
        print(f"[ENTRYPOINT] MAESTRO_GCS_PREFIX={args.gcs_prefix}")

    # Handle GCS output directory
    if args.output_dir.startswith('gs://'):
        os.environ['LATBOND_GCS_OUTPUT'] = args.output_dir
        local_output = '/app/outputs'
        print(f"[ENTRYPOINT] GCS output: {args.output_dir}")
        print(f"[ENTRYPOINT] Local output: {local_output}")
    else:
        local_output = args.output_dir

    os.makedirs(local_output, exist_ok=True)

    # Build args for run_full_study.py
    study_args = ['--output_dir', local_output] + remaining

    print(f"[ENTRYPOINT] Delegating to run_full_study.py {' '.join(study_args)}")
    print(f"[ENTRYPOINT] Python: {sys.version}")
    print(f"[ENTRYPOINT] CUDA: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    # Import and run
    sys.argv = ['run_full_study.py'] + study_args
    sys.path.insert(0, '/app')

    from run_full_study import main as study_main
    study_main()


if __name__ == '__main__':
    main()
