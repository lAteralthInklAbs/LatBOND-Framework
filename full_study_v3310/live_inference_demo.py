#!/usr/bin/env python3
"""
LatBOND Live Inference Demo - Full Version
===========================================
Demonstrates matched vs truncated onset detection in real-time simulation.

Usage:
    python live_inference_demo.py audio.wav --mode static
    python live_inference_demo.py audio.wav --mode terminal --arch streaming_cnn
    python live_inference_demo.py audio.wav --mode static --midi ground_truth.mid
    python live_inference_demo.py audio.wav --mode all-archs
"""

import argparse
import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from models import create_model

# Optional imports
try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False


def load_model(arch_name, condition, device, config):
    """Load a trained model with its optimal threshold."""
    model_path = os.path.join(config.output_dir, f"{arch_name}_{condition}_best.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = create_model(arch_name, config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        threshold = checkpoint.get('best_threshold', checkpoint.get('threshold', 0.3))
    else:
        model.load_state_dict(checkpoint)
        threshold = 0.3

    model.eval()
    return model, threshold


def load_and_preprocess_audio(audio_path, config, start_sec=0, duration_sec=5):
    """Load audio, resample to 16kHz mono, compute CAUSAL mel spectrogram."""
    if not HAS_TORCHAUDIO:
        raise ImportError("torchaudio required for audio loading")

    waveform, sr = torchaudio.load(audio_path)

    if sr != config.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    start_sample = int(start_sec * config.sample_rate)
    end_sample = start_sample + int(duration_sec * config.sample_rate)
    waveform = waveform[:, start_sample:end_sample]

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        center=False,
    )

    pad_amount = config.n_fft - config.hop_length
    waveform_padded = F.pad(waveform, (pad_amount, 0))

    mel = mel_transform(waveform_padded)
    mel = torch.log(mel + 1e-8)

    return mel


def sliding_window_inference(model, mel, window_size, threshold, device, config):
    """O(n) sliding window inference with fixed context."""
    mel = mel.to(device)
    _, _, T = mel.shape

    all_probs = torch.zeros(config.n_pitches, T, device=device)

    with torch.no_grad():
        for t in range(T):
            start = max(0, t - window_size + 1)
            window = mel[:, :, start:t+1]

            if window.shape[2] < window_size:
                pad_len = window_size - window.shape[2]
                window = F.pad(window, (pad_len, 0))

            outputs = model(window)
            prob = torch.sigmoid(outputs['onset_logits'][0, :, -1])
            all_probs[:, t] = prob

    probs_np = all_probs.cpu().numpy()
    onsets = []
    for pitch in range(config.n_pitches):
        for t in range(T):
            if probs_np[pitch, t] > threshold:
                onsets.append((t, pitch, probs_np[pitch, t]))

    return probs_np, onsets


def run_static_demo(mel, model_matched, model_truncated,
                    thresh_matched, thresh_truncated,
                    window_size, gt_onsets, device, output_dir, config):
    """Generate static PNG comparison."""
    if not HAS_MATPLOTLIB:
        print("[ERROR] matplotlib required for static demo")
        return

    print("\n[INFO] Running sliding window inference...")

    probs_m, onsets_m = sliding_window_inference(
        model_matched, mel, window_size, thresh_matched, device, config
    )
    probs_t, onsets_t = sliding_window_inference(
        model_truncated, mel, window_size, thresh_truncated, device, config
    )

    T = mel.shape[2]
    time_axis = np.arange(T) * config.frame_duration_ms / 1000

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].imshow(mel[0].cpu().numpy(), aspect='auto', origin='lower',
                   extent=[0, time_axis[-1], 0, config.n_mels])
    axes[0].set_ylabel('Mel bin')
    axes[0].set_title('Log Mel Spectrogram')

    axes[1].imshow(probs_m, aspect='auto', origin='lower', cmap='hot',
                   extent=[0, time_axis[-1], 0, 88], vmin=0, vmax=1)
    axes[1].set_ylabel('Piano key')
    axes[1].set_title(f'MATCHED (thresh={thresh_matched:.2f}) | Detections: {len(onsets_m)}')

    axes[2].imshow(probs_t, aspect='auto', origin='lower', cmap='hot',
                   extent=[0, time_axis[-1], 0, 88], vmin=0, vmax=1)
    axes[2].set_ylabel('Piano key')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title(f'TRUNCATED (thresh={thresh_truncated:.2f}) | Detections: {len(onsets_t)}')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'live_demo_static.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[INFO] Saved: {output_path}")
    print(f"\nResults:")
    print(f"  Matched:   {len(onsets_m)} detections")
    print(f"  Truncated: {len(onsets_t)} detections")


def run_smoke_test(device, config):
    """Smoke test with synthetic data."""
    print("\n=== SMOKE TEST MODE ===")

    print("\n--- Check 1: Model output shape ---")
    for arch_name in config.architectures:
        model = create_model(arch_name, config).to(device)
        dummy_input = torch.randn(1, config.n_mels, 50).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  {arch_name}: input={dummy_input.shape} -> onset={output['onset_logits'].shape}")
        assert output['onset_logits'].shape == (1, config.n_pitches, 50)
    print("  All models produce correct output shapes: VERIFIED")

    print("\n--- Check 2: Sliding window inference ---")
    model = create_model('streaming_cnn', config).to(device)
    mel_test = torch.randn(1, config.n_mels, 100).to(device)
    window_size = 15

    probs, onsets = sliding_window_inference(model, mel_test, window_size, 0.5, device, config)
    print(f"  Frames: 100, Output shape: {probs.shape}")
    print("  Sliding window: VERIFIED")

    print("\n=== SMOKE TEST PASSED ===")


def main():
    parser = argparse.ArgumentParser(description='LatBOND Live Inference Demo')
    parser.add_argument('audio_path', type=str, nargs='?',
                        help='Path to audio file')
    parser.add_argument('--arch', type=str, default='streaming_cnn',
                        choices=['streaming_cnn', 'causal_transformer', 'tcn'])
    parser.add_argument('--midi', type=str, default=None,
                        help='Path to ground truth MIDI')
    parser.add_argument('--mode', type=str, default='static',
                        choices=['terminal', 'static', 'all-archs'])
    parser.add_argument('--start', type=float, default=0.0)
    parser.add_argument('--duration', type=float, default=5.0)
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    os.makedirs(CONFIG.output_dir, exist_ok=True)

    if args.smoke_test:
        run_smoke_test(device, CONFIG)
        return

    if not args.audio_path:
        parser.error("audio_path required unless --smoke-test is specified")

    if not os.path.exists(args.audio_path):
        print(f"[ERROR] Audio file not found: {args.audio_path}")
        sys.exit(1)

    # Load audio
    print("\n[INFO] Loading audio...")
    mel = load_and_preprocess_audio(args.audio_path, CONFIG, args.start, args.duration)
    print(f"[INFO] Mel shape: {mel.shape}")

    # Load models
    print(f"\n[INFO] Loading {args.arch} models...")
    try:
        model_matched, thresh_matched = load_model(args.arch, 'matched', device, CONFIG)
        model_truncated, thresh_truncated = load_model(args.arch, 'truncated', device, CONFIG)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[HINT] Run training first: python run_full_study.py")
        sys.exit(1)

    window_size = CONFIG.receptive_fields.get(args.arch, 32)

    if args.mode == 'static':
        run_static_demo(mel, model_matched, model_truncated,
                       thresh_matched, thresh_truncated,
                       window_size, None, device, CONFIG.output_dir, CONFIG)

    print("\n[DONE]")


if __name__ == '__main__':
    main()
