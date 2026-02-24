"""
MAESTRO Dataset Download + Subset Creation - Full Version
==========================================================
Downloads MAESTRO v3.0.0 from GCP and creates a stratified subset.

Supports two modes:
1. Internet download (default): Download from Google's public hosting
2. GCS copy (Vertex AI): Copy from private GCS bucket
"""

import os
import json
import random
import subprocess
import csv
from pathlib import Path

# Try to import GCS client
try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False


def download_maestro(data_dir="data"):
    """Download MAESTRO v3.0.0 from Google Cloud Storage."""
    maestro_dir = os.path.join(data_dir, "maestro-v3.0.0")

    if os.path.exists(maestro_dir):
        print(f"[INFO] MAESTRO already exists at {maestro_dir}")
        return maestro_dir

    os.makedirs(data_dir, exist_ok=True)

    print("[INFO] Downloading MAESTRO v3.0.0 from GCS...")
    url = "gs://magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
    zip_path = os.path.join(data_dir, "maestro-v3.0.0.zip")

    # Try gsutil first (faster on GCP), fall back to wget
    try:
        subprocess.run(
            ["gsutil", "-m", "cp", url, zip_path],
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[INFO] gsutil failed, trying wget...")
        wget_url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
        subprocess.run(
            ["wget", "-O", zip_path, wget_url],
            check=True
        )

    print("[INFO] Extracting...")
    subprocess.run(
        ["unzip", "-q", zip_path, "-d", data_dir],
        check=True
    )

    # Clean up zip
    os.remove(zip_path)
    print(f"[INFO] MAESTRO extracted to {maestro_dir}")
    return maestro_dir


def create_subset(maestro_dir, subset_fraction=0.05, seed=42, skip_file_check=False):
    """
    Create a stratified subset of MAESTRO.

    Ensures train/val/test split proportions are maintained.
    Returns dict of {split: [file_paths]}.
    
    Args:
        skip_file_check: If True, skip os.path.exists check (for GCS mode
                        where files are lazily downloaded during training)
    """
    random.seed(seed)

    metadata_path = os.path.join(maestro_dir, "maestro-v3.0.0.csv")

    # Parse metadata
    splits = {"train": [], "validation": [], "test": []}

    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            audio_path = os.path.join(maestro_dir, row["audio_filename"])
            midi_path = os.path.join(maestro_dir, row["midi_filename"])

            # In GCS mode, files don't exist locally yet — they're downloaded
            # lazily by resolve_audio_path() during training
            if skip_file_check or (os.path.exists(audio_path) and os.path.exists(midi_path)):
                splits[split].append({
                    "audio": audio_path,
                    "midi": midi_path,
                    "duration": float(row["duration"]),
                    "canonical_composer": row.get("canonical_composer", ""),
                })

    # Stratified sampling
    subset = {}
    for split_name, files in splits.items():
        n_select = max(2, int(len(files) * subset_fraction))
        selected = random.sample(files, min(n_select, len(files)))
        subset[split_name] = selected
        print(f"[INFO] {split_name}: {len(selected)}/{len(files)} files selected")

    # Save subset manifest
    manifest_path = os.path.join(maestro_dir, "subset_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(subset, f, indent=2)

    total_duration = sum(
        f["duration"] for files in subset.values() for f in files
    )
    print(f"[INFO] Total subset duration: {total_duration/60:.1f} minutes")

    return subset


def download_and_prepare(config, gcs_prefix=None):
    """
    Download or copy MAESTRO dataset.

    If gcs_prefix is set (Vertex AI mode), copies from GCS bucket.
    Otherwise downloads from Google's public MAESTRO hosting.
    """
    # Fallback to env var if arg is None
    if gcs_prefix is None:
        gcs_prefix = os.environ.get('MAESTRO_GCS_PREFIX')

    maestro_dir = getattr(config, 'maestro_root', os.path.join(config.data_dir, "maestro-v3.0.0"))

    if os.path.exists(os.path.join(maestro_dir, 'maestro-v3.0.0.csv')):
        print(f"[INFO] MAESTRO already exists at {maestro_dir}")
        return load_existing_subset(maestro_dir, config)

    if gcs_prefix and HAS_GCS:
        # Vertex AI mode: copy CSV from GCS
        print(f"[INFO] Copying MAESTRO metadata from {gcs_prefix}...")
        os.makedirs(maestro_dir, exist_ok=True)

        client = storage.Client()
        bucket_name = gcs_prefix.replace('gs://', '').split('/')[0]
        prefix = '/'.join(gcs_prefix.replace('gs://', '').split('/')[1:])

        bucket = client.bucket(bucket_name)

        # Download CSV
        csv_blob = bucket.blob(f"{prefix}/maestro-v3.0.0.csv")
        csv_path = os.path.join(maestro_dir, "maestro-v3.0.0.csv")
        csv_blob.download_to_filename(csv_path)
        print(f"[INFO] Downloaded {csv_path}")

        return create_subset(maestro_dir, config.subset_fraction, skip_file_check=True)
    else:
        # Original download logic
        maestro_dir = download_maestro(config.data_dir)
        return create_subset(maestro_dir, config.subset_fraction)


def load_existing_subset(maestro_dir, config):
    """Load existing subset manifest."""
    manifest_path = os.path.join(maestro_dir, "subset_manifest.json")

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            subset = json.load(f)
        print(f"[INFO] Loaded existing subset manifest")
        return subset
    else:
        # Create new subset
        return create_subset(maestro_dir, config.subset_fraction)


if __name__ == "__main__":
    from config import CONFIG

    maestro_dir = download_maestro(CONFIG.data_dir)
    subset = create_subset(maestro_dir, CONFIG.subset_fraction)
