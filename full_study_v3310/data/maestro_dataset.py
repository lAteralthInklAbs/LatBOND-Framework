"""
MAESTRO PyTorch Dataset - Full Version
=======================================
Loads audio + MIDI, computes causal mel spectrogram,
creates frame-level onset, offset, and velocity labels.

Updates for Full Version:
  - GCS bucket support for Vertex AI
  - Thread-safe file downloading with filelock
"""

import os
import json
import hashlib
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torchaudio

# Try to import optional dependencies
try:
    import pretty_midi
except ImportError:
    pretty_midi = None
    print("[WARN] pretty_midi not installed. Run: pip install pretty_midi")

try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False

try:
    from filelock import FileLock
    HAS_FILELOCK = True
except ImportError:
    HAS_FILELOCK = False


def resolve_audio_path(local_path: str, gcs_prefix: str = None) -> str:
    """
    Resolve audio file path -- use local if exists, otherwise download from GCS.

    THREAD SAFETY: Uses filelock to prevent race conditions when num_workers > 0
    in the DataLoader.

    Args:
        local_path: Original local path from MAESTRO CSV
        gcs_prefix: GCS bucket prefix (e.g., "gs://YOUR_GCP_PROJECT-data/datasets/maestro-v3")

    Returns:
        Resolved local path (cached if downloaded from GCS)
    """
    if os.path.exists(local_path):
        return local_path

    if gcs_prefix is None:
        gcs_prefix = os.environ.get('MAESTRO_GCS_PREFIX', '')

    if not gcs_prefix:
        raise FileNotFoundError(f"Audio file not found locally and no GCS prefix set: {local_path}")

    if not HAS_GCS:
        raise ImportError("google-cloud-storage required for GCS access")

    # Cache downloaded files to /tmp/maestro-cache/
    cache_dir = '/tmp/maestro-cache'
    # Extract relative path from the full path
    relative_path = os.path.basename(os.path.dirname(local_path)) + '/' + os.path.basename(local_path)
    cached_path = os.path.join(cache_dir, relative_path)

    os.makedirs(os.path.dirname(cached_path), exist_ok=True)

    # Use file lock if available
    if HAS_FILELOCK:
        lock_path = f"{cached_path}.lock"
        with FileLock(lock_path):
            if not os.path.exists(cached_path):
                _download_from_gcs(gcs_prefix, relative_path, cached_path)
    else:
        if not os.path.exists(cached_path):
            _download_from_gcs(gcs_prefix, relative_path, cached_path)

    return cached_path


def _download_from_gcs(gcs_prefix: str, relative_path: str, local_path: str):
    """Download a file from GCS."""
    gcs_uri = f"{gcs_prefix.rstrip('/')}/{relative_path}"
    bucket_name = gcs_uri.replace('gs://', '').split('/')[0]
    blob_path = '/'.join(gcs_uri.replace('gs://', '').split('/')[1:])

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)


class MAESTRODataset(Dataset):
    """
    PyTorch Dataset for MAESTRO piano note transcription.

    Preprocessing:
    - Causal STFT (center=False, left-padding)
    - Log-mel spectrogram (229 bins)
    - Frame-level labels: onset, offset, velocity (88 pitches each)

    v2 #25: Optional SpecAugment (freq + time masking) for training.
    """

    def __init__(self, file_list, config, max_frames=500, gcs_prefix=None, use_specaugment=False):
        """
        Args:
            file_list: List of dicts with 'audio' and 'midi' paths
            config: Config object
            max_frames: Max frames per sample (truncate longer)
            gcs_prefix: GCS bucket prefix for Vertex AI
            use_specaugment: Enable SpecAugment for data augmentation
        """
        self.files = file_list
        self.config = config
        self.max_frames = max_frames
        self.gcs_prefix = gcs_prefix or os.environ.get('MAESTRO_GCS_PREFIX')
        self.use_specaugment = use_specaugment

        # v2 #25: SpecAugment parameters
        self.freq_mask_param = 27  # Max frequency bands to mask
        self.time_mask_param = 50  # Max time frames to mask
        self.num_freq_masks = 2
        self.num_time_masks = 2

        # Mel spectrogram transform (causal: center=False)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            center=False,  # CRITICAL: causal processing
        )

        # Preload and cache
        self._cache = []
        self._preprocess_all()

    def _get_cache_path(self):
        """Path for pre-cached tensor file."""
        # Create a hash of the file list + config to detect changes
        key = str(len(self.files)) + str(self.max_frames) + str(getattr(self.config, 'subset_fraction', 0.05))
        cache_hash = hashlib.md5(key.encode()).hexdigest()[:8]
        cache_dir = '/tmp/latbond-tensor-cache'
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f'maestro_cache_{cache_hash}.pt'), cache_hash

    def _preprocess_all(self):
        """Preprocess all files into mel + labels, with tensor caching."""
        cache_path, cache_hash = self._get_cache_path()

        # v2: Try downloading from GCS first
        if not os.path.exists(cache_path) and self.gcs_prefix:
            try:
                if HAS_GCS:
                    bucket_name = self.gcs_prefix.replace('gs://', '').split('/')[0]
                    client = storage.Client()
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(f'tensor-cache/maestro_cache_{cache_hash}.pt')
                    if blob.exists():
                        blob.download_to_filename(cache_path)
                        print(f"[INFO] Downloaded tensor cache from GCS")
            except Exception:
                pass

        # v2: Try loading from local cache first
        if os.path.exists(cache_path):
            print(f"[INFO] Loading pre-cached tensors from {cache_path}")
            try:
                self._cache = torch.load(cache_path, weights_only=False)
                print(f"[INFO] Loaded {len(self._cache)} cached chunks")
                return
            except Exception as e:
                print(f"[WARN] Cache load failed: {e}, reprocessing...")

        print(f"[INFO] Preprocessing {len(self.files)} files...")

        for i, file_info in enumerate(self.files):
            try:
                mel, labels = self._process_file(file_info)
                if mel is not None and mel.shape[-1] >= 10:
                    # Split into chunks of max_frames
                    n_frames = mel.shape[-1]
                    for start in range(0, n_frames - self.max_frames + 1, self.max_frames // 2):
                        end = start + self.max_frames
                        if end <= n_frames:
                            self._cache.append({
                                'mel': mel[:, start:end],
                                'onsets': labels['onsets'][:, start:end],
                                'offsets': labels['offsets'][:, start:end],
                                'velocities': labels['velocities'][:, start:end],
                            })
            except Exception as e:
                print(f"[WARN] Failed to process {file_info['audio']}: {e}")
                continue

            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(self.files)} files, "
                      f"{len(self._cache)} chunks cached")

        print(f"[INFO] Dataset ready: {len(self._cache)} chunks")

        # v2: Save cache for future runs
        try:
            torch.save(self._cache, cache_path)
            print(f"[INFO] Saved tensor cache to {cache_path} ({len(self._cache)} chunks)")
        except Exception as e:
            print(f"[WARN] Failed to save cache: {e}")

        # v2: Upload to GCS for other jobs
        if self.gcs_prefix:
            try:
                if HAS_GCS:
                    bucket_name = self.gcs_prefix.replace('gs://', '').split('/')[0]
                    client = storage.Client()
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(f'tensor-cache/maestro_cache_{cache_hash}.pt')
                    if not blob.exists():
                        blob.upload_from_filename(cache_path)
                        print(f"[INFO] Uploaded tensor cache to GCS")
            except Exception as e:
                print(f"[WARN] GCS cache upload failed: {e}")

    def _process_file(self, file_info):
        """Process single audio + MIDI file pair."""
        # Resolve audio path (handles GCS if needed)
        audio_path = resolve_audio_path(file_info['audio'], self.gcs_prefix)

        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Left-pad for causal STFT
        pad_length = self.config.n_fft // 2
        waveform = torch.nn.functional.pad(waveform, (pad_length, 0))

        # Compute mel spectrogram
        mel = self.mel_transform(waveform)
        mel = torch.log(mel.clamp(min=1e-8))
        mel = mel.squeeze(0)

        n_frames = mel.shape[-1]

        # Create labels from MIDI
        midi_path = file_info['midi']
        if self.gcs_prefix and not os.path.exists(midi_path):
            midi_path = resolve_audio_path(midi_path, self.gcs_prefix)

        labels = self._midi_to_labels(midi_path, n_frames)

        return mel, labels

    def _midi_to_labels(self, midi_path, n_frames):
        """Extract frame-level onset, offset, and velocity labels from MIDI."""
        n_pitches = self.config.n_pitches
        onsets = torch.zeros(n_pitches, n_frames)
        offsets = torch.zeros(n_pitches, n_frames)
        velocities = torch.zeros(n_pitches, n_frames)

        if pretty_midi is None:
            return {'onsets': onsets, 'offsets': offsets, 'velocities': velocities}

        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
        except Exception:
            return {'onsets': onsets, 'offsets': offsets, 'velocities': velocities}

        frame_duration = self.config.hop_length / self.config.sample_rate
        velocity_window = getattr(self.config, 'velocity_window_frames', 3)
        velocity_scale = getattr(self.config, 'velocity_scale', 127.0)

        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                pitch_idx = note.pitch - 21
                if not (0 <= pitch_idx < n_pitches):
                    continue

                onset_frame = int(note.start / frame_duration)
                if 0 <= onset_frame < n_frames:
                    onsets[pitch_idx, onset_frame] = 1.0

                    vel_normalized = note.velocity / velocity_scale
                    for w in range(velocity_window):
                        vf = onset_frame + w
                        if 0 <= vf < n_frames:
                            velocities[pitch_idx, vf] = vel_normalized

                offset_frame = int(note.end / frame_duration)
                if 0 <= offset_frame < n_frames:
                    offsets[pitch_idx, offset_frame] = 1.0

        return {'onsets': onsets, 'offsets': offsets, 'velocities': velocities}

    def __len__(self):
        return len(self._cache)

    def _apply_specaugment(self, mel):
        """
        v2 #25: Apply SpecAugment (frequency and time masking).
        Operates on [n_mels, T] tensor.
        """
        mel = mel.clone()
        n_mels, n_frames = mel.shape

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = torch.randint(0, min(self.freq_mask_param, n_mels), (1,)).item()
            f0 = torch.randint(0, max(1, n_mels - f), (1,)).item()
            mel[f0:f0 + f, :] = 0.0

        # Time masking
        for _ in range(self.num_time_masks):
            t = torch.randint(0, min(self.time_mask_param, n_frames), (1,)).item()
            t0 = torch.randint(0, max(1, n_frames - t), (1,)).item()
            mel[:, t0:t0 + t] = 0.0

        return mel

    def __getitem__(self, idx):
        item = self._cache[idx]
        mel = item['mel']

        # v2 #25: Apply SpecAugment during training
        if self.use_specaugment:
            mel = self._apply_specaugment(mel)

        return mel, {
            'onsets': item['onsets'],
            'offsets': item['offsets'],
            'velocities': item['velocities'],
        }


def create_dataloaders(subset, config, batch_size=None, gcs_prefix=None):
    """
    Create train/val/test DataLoaders.

    v2 optimizations: configurable num_workers, pin_memory, persistent_workers.
    v3.3.4: DDP sampler support for multi-GPU training.

    Returns: dict of {split_name: DataLoader}
    """
    if batch_size is None:
        batch_size = config.batch_size

    # v2 #13, #23: DataLoader optimizations from config
    num_workers = getattr(config, 'num_workers', 4)
    pin_memory = getattr(config, 'pin_memory', True)
    persistent_workers = getattr(config, 'persistent_workers', True) and num_workers > 0

    # v3.3.4: Check for distributed training
    is_distributed = dist.is_initialized()
    if is_distributed:
        print(f"[INFO] DDP enabled: creating DistributedSamplers")

    loaders = {}
    samplers = {}  # v3.3.4: Track samplers for epoch sync
    for split_name in ['train', 'validation', 'test']:
        if split_name not in subset or len(subset[split_name]) == 0:
            continue

        # v2 #25: SpecAugment for training only
        use_augment = (split_name == 'train')

        dataset = MAESTRODataset(
            subset[split_name], config,
            max_frames=500,
            gcs_prefix=gcs_prefix,
            use_specaugment=use_augment,
        )

        # v3.3.4: DDP sampler support
        sampler = None
        shuffle = (split_name == 'train')
        if is_distributed and split_name == 'train':
            sampler = DistributedSampler(dataset, shuffle=True)
            shuffle = False  # Sampler handles shuffling
            samplers[split_name] = sampler

        loaders[split_name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=(split_name == 'train'),
        )

    # v3.3.4: Attach samplers to loaders dict for epoch sync
    loaders['_samplers'] = samplers

    return loaders
