#!/usr/bin/env python3
"""
LatBOND Demo Pipeline
=====================
End-to-end demonstration: audio -> model -> MIDI -> comparison visualization.

Takes MAESTRO test audio and produces:
1. Piano roll visualization with TP/FP/FN coloring
2. Predicted MIDI files from matched and truncated models
3. Side-by-side comparison metrics

Usage:
    python demo_pipeline.py \\
        --matched_checkpoint outputs/tcn/20ms/matched/tcn_matched_best.pt \\
        --truncated_checkpoint outputs/tcn/20ms/truncated/tcn_truncated_best.pt \\
        --test_audio /path/to/maestro/test/file.wav \\
        --test_midi /path/to/maestro/test/file.midi \\
        --output_dir demo_output/ \\
        --architecture tcn \\
        --budget_ms 20
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LatBONDConfig
from models import create_model

# Optional imports
try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False
    print("[WARN] torchaudio not installed")

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False
    print("[WARN] pretty_midi not installed")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARN] matplotlib not installed")


def load_audio_and_preprocess(audio_path, config):
    """Load audio file, compute mel spectrogram with causal STFT."""
    if not HAS_TORCHAUDIO:
        raise ImportError("torchaudio required")

    waveform, sr = torchaudio.load(audio_path)
    if sr != config.sample_rate:
        waveform = torchaudio.transforms.Resample(sr, config.sample_rate)(waveform)
    waveform = waveform.mean(dim=0)  # Mono

    # Causal STFT
    pad_length = config.n_fft - config.hop_length  # 1888
    waveform_padded = F.pad(waveform, (pad_length, 0))

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        center=False,
    )

    mel = mel_transform(waveform_padded.unsqueeze(0))
    mel = torch.log(mel.clamp(min=1e-5))

    return mel  # [1, n_mels, T]


def load_ground_truth(midi_path, n_frames, config):
    """Load MIDI and extract ground truth note events."""
    if not HAS_PRETTY_MIDI:
        return []

    midi = pretty_midi.PrettyMIDI(midi_path)
    frame_duration = config.hop_length / config.sample_rate

    notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            pitch_idx = note.pitch - 21
            if 0 <= pitch_idx < 88:
                notes.append({
                    'onset_sec': note.start,
                    'offset_sec': note.end,
                    'pitch_midi': note.pitch,
                    'pitch_hz': 440.0 * (2.0 ** ((note.pitch - 69) / 12.0)),
                    'velocity': note.velocity,
                })
    return notes


def run_inference(model, mel, config, condition, device='cuda'):
    """Run inference under the specified condition."""
    model.eval()
    mel = mel.to(device)

    with torch.no_grad():
        if condition == 'buffered':
            outputs = model(mel)
        else:
            outputs = _sliding_window_inference(model, mel, config, device)

    onset_probs = torch.sigmoid(outputs['onset_logits']).cpu().numpy()[0]
    offset_probs = torch.sigmoid(outputs['offset_logits']).cpu().numpy()[0]
    velocity_preds = torch.sigmoid(outputs['velocity_logits']).cpu().numpy()[0]

    return onset_probs, offset_probs, velocity_preds


def _sliding_window_inference(model, mel, config, device):
    """Sliding window inference for matched/truncated."""
    B, C, T = mel.shape
    ctx = config.get_budget_frames() * config.context_multiplier
    stride = config.eval_stride

    all_onset_logits = torch.zeros(B, config.n_pitches, T, device=device)
    all_offset_logits = torch.zeros(B, config.n_pitches, T, device=device)
    all_velocity_logits = torch.zeros(B, config.n_pitches, T, device=device)
    counts = torch.zeros(T, device=device)

    positions = list(range(0, T, stride))

    for pos in positions:
        end = min(pos + ctx, T)
        actual_start = max(0, end - ctx)

        chunk = mel[:, :, actual_start:end]
        if chunk.shape[-1] < ctx:
            chunk = F.pad(chunk, (ctx - chunk.shape[-1], 0))

        out = model(chunk)
        pred_len = end - actual_start
        out_start = max(0, out['onset_logits'].shape[-1] - pred_len)

        all_onset_logits[:, :, actual_start:end] += out['onset_logits'][:, :, out_start:]
        all_offset_logits[:, :, actual_start:end] += out['offset_logits'][:, :, out_start:]
        all_velocity_logits[:, :, actual_start:end] += out['velocity_logits'][:, :, out_start:]
        counts[actual_start:end] += 1

    counts = counts.clamp(min=1).unsqueeze(0).unsqueeze(0)
    return {
        'onset_logits': all_onset_logits / counts,
        'offset_logits': all_offset_logits / counts,
        'velocity_logits': all_velocity_logits / counts,
    }


def assemble_notes(onset_probs, offset_probs, velocity_preds, config, threshold=0.5):
    """Convert frame-level predictions to note events."""
    frame_duration = config.hop_length / config.sample_rate
    n_pitches, n_frames = onset_probs.shape
    notes = []

    for pitch in range(n_pitches):
        onset_frames = np.where(onset_probs[pitch] > threshold)[0]

        for onset_frame in onset_frames:
            # Find corresponding offset
            offset_candidates = np.where(offset_probs[pitch, onset_frame:] > threshold)[0]

            if len(offset_candidates) > 0:
                offset_frame = onset_frame + offset_candidates[0]
            else:
                offset_frame = min(onset_frame + 50, n_frames - 1)

            velocity = velocity_preds[pitch, onset_frame]

            onset_sec = onset_frame * frame_duration
            offset_sec = offset_frame * frame_duration
            pitch_hz = 440.0 * (2.0 ** ((pitch + 21 - 69) / 12.0))

            notes.append((onset_sec, offset_sec, pitch_hz, velocity))

    return notes


def predictions_to_midi(notes, output_path):
    """Convert predicted notes to a MIDI file."""
    if not HAS_PRETTY_MIDI:
        print("[WARN] Cannot save MIDI - pretty_midi not installed")
        return

    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0, name='Piano')

    for note in notes:
        onset, offset, pitch_hz, velocity = note
        midi_pitch = int(round(69 + 12 * np.log2(pitch_hz / 440.0)))
        midi_pitch = np.clip(midi_pitch, 21, 108)
        vel = int(np.clip(velocity * 127, 1, 127))

        piano.notes.append(pretty_midi.Note(
            velocity=vel, pitch=midi_pitch,
            start=onset, end=offset
        ))

    midi.instruments.append(piano)
    midi.write(output_path)


def create_piano_roll_comparison(gt_notes, matched_notes, truncated_notes,
                                  time_range, output_path, config, tolerance_sec=0.03):
    """Create piano roll comparison figure."""
    if not HAS_MATPLOTLIB:
        print("[WARN] Cannot create plot - matplotlib not installed")
        return

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    start_time, end_time = time_range

    # Panel 1: Ground truth
    axes[0].set_title('Ground Truth', fontsize=12)
    for note in gt_notes:
        if start_time <= note['onset_sec'] <= end_time:
            pitch = note['pitch_midi'] - 21
            axes[0].barh(pitch, note['offset_sec'] - note['onset_sec'],
                        left=note['onset_sec'], height=0.8, color='blue', alpha=0.7)
    axes[0].set_ylabel('Piano Key')
    axes[0].set_ylim(0, 88)

    # Panel 2: Matched predictions
    axes[1].set_title('Matched Predictions', fontsize=12)
    for onset, offset, pitch_hz, vel in matched_notes:
        if start_time <= onset <= end_time:
            pitch = int(round(69 + 12 * np.log2(pitch_hz / 440.0))) - 21
            axes[1].barh(pitch, offset - onset, left=onset, height=0.8, color='green', alpha=0.7)
    axes[1].set_ylabel('Piano Key')
    axes[1].set_ylim(0, 88)

    # Panel 3: Truncated predictions
    axes[2].set_title('Truncated Predictions', fontsize=12)
    for onset, offset, pitch_hz, vel in truncated_notes:
        if start_time <= onset <= end_time:
            pitch = int(round(69 + 12 * np.log2(pitch_hz / 440.0))) - 21
            axes[2].barh(pitch, offset - onset, left=onset, height=0.8, color='orange', alpha=0.7)
    axes[2].set_ylabel('Piano Key')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylim(0, 88)
    axes[2].set_xlim(start_time, end_time)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_demo_pipeline(args):
    """Main demo pipeline entry point."""
    config = LatBONDConfig()
    config.current_budget_ms = args.budget_ms
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"LatBOND Demo Pipeline")
    print(f"Architecture: {args.architecture}")
    print(f"Budget: {args.budget_ms}ms")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Load models
    print("\nLoading models...")
    matched_model = create_model(args.architecture, config).to(device)
    truncated_model = create_model(args.architecture, config).to(device)

    matched_ckpt = torch.load(args.matched_checkpoint, map_location=device, weights_only=False)
    truncated_ckpt = torch.load(args.truncated_checkpoint, map_location=device, weights_only=False)

    if 'model_state_dict' in matched_ckpt:
        matched_model.load_state_dict(matched_ckpt['model_state_dict'])
    else:
        matched_model.load_state_dict(matched_ckpt)

    if 'model_state_dict' in truncated_ckpt:
        truncated_model.load_state_dict(truncated_ckpt['model_state_dict'])
    else:
        truncated_model.load_state_dict(truncated_ckpt)

    matched_model.eval()
    truncated_model.eval()

    # Load and preprocess audio
    print("Loading audio...")
    mel = load_audio_and_preprocess(args.test_audio, config)
    n_frames = mel.shape[-1]
    print(f"  Mel shape: {mel.shape}")

    # Ground truth
    print("Loading ground truth MIDI...")
    gt_notes = load_ground_truth(args.test_midi, n_frames, config)
    print(f"  Ground truth notes: {len(gt_notes)}")

    # Run inference
    print("\nRunning matched inference...")
    m_onset, m_offset, m_vel = run_inference(matched_model, mel, config, 'matched', device)

    print("Running truncated inference...")
    t_onset, t_offset, t_vel = run_inference(truncated_model, mel, config, 'truncated', device)

    # Assemble notes
    threshold = matched_ckpt.get('best_threshold', 0.3)
    print(f"\nAssembling notes (threshold={threshold:.2f})...")
    matched_notes = assemble_notes(m_onset, m_offset, m_vel, config, threshold)
    truncated_notes = assemble_notes(t_onset, t_offset, t_vel, config, threshold)

    print(f"  Matched predictions: {len(matched_notes)}")
    print(f"  Truncated predictions: {len(truncated_notes)}")

    # Save predicted MIDI
    print("\nSaving predicted MIDI files...")
    predictions_to_midi(matched_notes,
                        os.path.join(args.output_dir, 'matched_prediction.mid'))
    predictions_to_midi(truncated_notes,
                        os.path.join(args.output_dir, 'truncated_prediction.mid'))

    # Create comparison visualization
    print("Creating piano roll comparison...")
    create_piano_roll_comparison(
        gt_notes, matched_notes, truncated_notes,
        time_range=(0, 10),
        output_path=os.path.join(args.output_dir, 'piano_roll_comparison.png'),
        config=config
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Demo Pipeline Results: {args.architecture} @ {args.budget_ms}ms")
    print(f"{'='*60}")
    print(f"Ground truth notes: {len(gt_notes)}")
    print(f"Matched predictions: {len(matched_notes)}")
    print(f"Truncated predictions: {len(truncated_notes)}")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LatBOND Demo Pipeline')
    parser.add_argument('--matched_checkpoint', required=True)
    parser.add_argument('--truncated_checkpoint', required=True)
    parser.add_argument('--test_audio', required=True)
    parser.add_argument('--test_midi', required=True)
    parser.add_argument('--output_dir', default='demo_output')
    parser.add_argument('--architecture', default='tcn',
                        choices=['streaming_cnn', 'causal_transformer', 'tcn'])
    parser.add_argument('--budget_ms', type=int, default=20)
    args = parser.parse_args()
    run_demo_pipeline(args)
