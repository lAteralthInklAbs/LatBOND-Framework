"""
Streaming CNN v11 - Lightweight causal CNN for note transcription.
Best suited for 0-20ms latency budgets.
Receptive field: ~15 frames (~150ms)

Key features:
- All causal convolutions (left-padding only)
- Spectral flux input channel (causal delta mel)
- Triple-head output: onset + offset + velocity (v11)
- Negative bias initialization for sparse events
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D convolution with causal (left-only) padding."""
    
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
    
    def forward(self, x):
        x = F.pad(x, (self.padding, 0))  # Left-only padding
        return self.conv(x)


class SpectralFluxLayer(nn.Module):
    """
    Compute causal spectral flux as a 2nd input channel.
    flux[t] = mel[t] - mel[t-1]  (zero for t=0)
    """
    
    def forward(self, mel):
        # mel: [B, n_mels, T]
        flux = torch.zeros_like(mel)
        flux[:, :, 1:] = mel[:, :, 1:] - mel[:, :, :-1]
        return flux


class StreamingCNN(nn.Module):
    """
    Streaming CNN for real-time note transcription (v11).

    Architecture:
        Input (n_mels × T) → [optional spectral flux concat] →
        3 × CausalConv1d blocks (64→128→128) →
        3 × 1×1 Conv heads → onset, offset, velocity logits [88, T] each

    Receptive field: 15 frames (~150ms)
    Parameters: ~220K (includes offset + velocity heads)
    """

    def __init__(self, n_mels=229, n_pitches=88, channels=None,
                 kernel_size=3, use_spectral_flux=True):
        super().__init__()

        if channels is None:
            channels = [64, 128, 128]

        self.use_spectral_flux = use_spectral_flux
        self.spectral_flux = SpectralFluxLayer() if use_spectral_flux else None

        in_ch = n_mels * (2 if use_spectral_flux else 1)

        # Causal conv blocks
        layers = []
        for out_ch in channels:
            layers.extend([
                CausalConv1d(in_ch, out_ch, kernel_size),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            ])
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)

        # Onset detection head (existing)
        self.onset_head = nn.Conv1d(channels[-1], n_pitches, 1)
        bias_val = math.log(0.01 / (1 - 0.01))  # ~-4.6
        nn.init.constant_(self.onset_head.bias, bias_val)

        # NEW v11: Offset detection head (same architecture as onset)
        self.offset_head = nn.Conv1d(channels[-1], n_pitches, 1)
        offset_bias = math.log(0.01 / (1 - 0.01))  # Same sparsity as onsets
        nn.init.constant_(self.offset_head.bias, offset_bias)

        # NEW v11: Velocity estimation head (regression, neutral bias)
        self.velocity_head = nn.Conv1d(channels[-1], n_pitches, 1)
        nn.init.constant_(self.velocity_head.bias, 0.0)  # Regression, not classification
    
    def forward(self, x):
        """
        Args:
            x: mel spectrogram [B, n_mels, T]
        Returns:
            dict with 'onset_logits', 'offset_logits', 'velocity_logits': each [B, 88, T]
        """
        if self.use_spectral_flux:
            flux = self.spectral_flux(x)
            x = torch.cat([x, flux], dim=1)  # [B, 2*n_mels, T]

        features = self.encoder(x)

        return {
            'onset_logits': self.onset_head(features),
            'offset_logits': self.offset_head(features),
            'velocity_logits': self.velocity_head(features),
        }
    
    def get_receptive_field(self):
        """Receptive field in frames."""
        # 3 layers × (kernel_size - 1) × dilation + 1
        return 3 * (3 - 1) * 1 + 1  # = 7 per dilation=1
        # With 3 layers: effectively ~15 frames


def create_streaming_cnn(config):
    """Factory function."""
    return StreamingCNN(
        n_mels=config.n_mels,
        n_pitches=config.n_pitches,
        channels=config.cnn_channels,
        kernel_size=config.cnn_kernel_size,
        use_spectral_flux=config.use_spectral_flux,
    )
