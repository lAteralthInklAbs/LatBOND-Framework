"""
MAESTRO PyTorch Dataset v11
============================
Loads audio + MIDI, computes causal mel spectrogram,
creates frame-level onset, offset, and velocity labels.

v11: Returns dict with 'onsets', 'offsets', 'velocities' instead of single tensor.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio

# Try to import pretty_midi, fall back gracefully
try:
    import pretty_midi
except ImportError:
    pretty_midi = None
    print("[WARN] pretty_midi not installed. Run: pip install pretty_midi")


class MAESTRODataset(Dataset):
    """
    PyTorch Dataset for MAESTRO piano note transcription (v11).

    Preprocessing:
    - Causal STFT (center=False, left-padding)
    - Log-mel spectrogram (229 bins)
    - Frame-level labels: onset, offset, velocity (88 pitches each)

    Returns:
        mel: [n_mels, T] tensor
        labels: dict with 'onsets', 'offsets', 'velocities' each [88, T]
    """
    
    def __init__(self, file_list, config, max_frames=500):
        """
        Args:
            file_list: List of dicts with 'audio' and 'midi' paths
            config: Config object
            max_frames: Max frames per sample (truncate longer)
        """
        self.files = file_list
        self.config = config
        self.max_frames = max_frames
        
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
    
    def _preprocess_all(self):
        """Preprocess all files into mel + labels dict."""
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
    
    def _process_file(self, file_info):
        """Process single audio + MIDI file pair."""
        # Load audio
        waveform, sr = torchaudio.load(file_info['audio'])

        # Resample if needed
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            waveform = resampler(waveform)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Left-pad for causal STFT (equivalent to center=True but causal)
        pad_length = self.config.n_fft // 2
        waveform = torch.nn.functional.pad(waveform, (pad_length, 0))

        # Compute mel spectrogram
        mel = self.mel_transform(waveform)  # [1, n_mels, T]
        mel = torch.log(mel.clamp(min=1e-8))  # Log-mel
        mel = mel.squeeze(0)  # [n_mels, T]

        n_frames = mel.shape[-1]

        # Create onset/offset/velocity labels from MIDI (v11)
        labels = self._midi_to_labels(file_info['midi'], n_frames)

        return mel, labels

    def _midi_to_labels(self, midi_path, n_frames):
        """
        Extract frame-level onset, offset, and velocity labels from MIDI.

        Returns dict with:
            'onsets':     [88, n_frames] binary — 1.0 at onset frames
            'offsets':    [88, n_frames] binary — 1.0 at offset frames
            'velocities': [88, n_frames] float  — normalized velocity (0–1) at onset frames,
                          spread over a small window (velocity_window_frames)
        """
        n_pitches = self.config.n_pitches
        onsets = torch.zeros(n_pitches, n_frames)
        offsets = torch.zeros(n_pitches, n_frames)
        velocities = torch.zeros(n_pitches, n_frames)

        if pretty_midi is None:
            return {'onsets': onsets, 'offsets': offsets, 'velocities': velocities}

        midi = pretty_midi.PrettyMIDI(midi_path)
        frame_duration = self.config.hop_length / self.config.sample_rate
        velocity_window = getattr(self.config, 'velocity_window_frames', 3)
        velocity_scale = getattr(self.config, 'velocity_scale', 127.0)

        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                pitch_idx = note.pitch - 21  # MIDI 21 = A0
                if not (0 <= pitch_idx < n_pitches):
                    continue

                # Onset
                onset_frame = int(note.start / frame_duration)
                if 0 <= onset_frame < n_frames:
                    onsets[pitch_idx, onset_frame] = 1.0

                    # Velocity: normalized to [0, 1], spread over window around onset
                    vel_normalized = note.velocity / velocity_scale
                    for w in range(velocity_window):
                        vf = onset_frame + w
                        if 0 <= vf < n_frames:
                            velocities[pitch_idx, vf] = vel_normalized

                # Offset
                offset_frame = int(note.end / frame_duration)
                if 0 <= offset_frame < n_frames:
                    offsets[pitch_idx, offset_frame] = 1.0

        return {'onsets': onsets, 'offsets': offsets, 'velocities': velocities}
    
    def __len__(self):
        return len(self._cache)

    def __getitem__(self, idx):
        item = self._cache[idx]
        return item['mel'], {
            'onsets': item['onsets'],
            'offsets': item['offsets'],
            'velocities': item['velocities'],
        }


def create_dataloaders(subset, config, batch_size=None):
    """
    Create train/val/test DataLoaders.
    
    Returns: dict of {split_name: DataLoader}
    """
    if batch_size is None:
        batch_size = config.batch_size
    
    loaders = {}
    for split_name in ['train', 'validation', 'test']:
        if split_name not in subset or len(subset[split_name]) == 0:
            continue
        
        dataset = MAESTRODataset(
            subset[split_name], config,
            max_frames=500
        )
        
        loaders[split_name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            num_workers=2,
            pin_memory=True,
            drop_last=(split_name == 'train'),
        )
    
    return loaders
