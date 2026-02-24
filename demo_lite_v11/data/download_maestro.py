"""
MAESTRO Dataset Download + Subset Creation
===========================================
Downloads MAESTRO v3.0.0 from GCP and creates a stratified subset.
"""

import os
import json
import random
import subprocess
import csv
from pathlib import Path


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


def create_subset(maestro_dir, subset_fraction=0.03, seed=42):
    """
    Create a stratified subset of MAESTRO.
    
    Ensures train/val/test split proportions are maintained.
    Returns dict of {split: [file_paths]}.
    """
    random.seed(seed)
    
    metadata_path = os.path.join(maestro_dir, "maestro-v3.0.0.csv")
    
    # Parse metadata
    splits = {"train": [], "validation": [], "test": []}
    
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            audio_path = os.path.join(maestro_dir, row["audio_filename"])
            midi_path = os.path.join(maestro_dir, row["midi_filename"])
            
            if os.path.exists(audio_path) and os.path.exists(midi_path):
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


if __name__ == "__main__":
    from config import CONFIG
    
    maestro_dir = download_maestro(CONFIG.data_dir)
    subset = create_subset(maestro_dir, CONFIG.subset_fraction)
