"""
Temporal Convolutional Network (TCN) v11 - Dilated causal CNN for note transcription.
Best suited for 40-80ms latency budgets.
Receptive field: ~63 frames (~630ms) via exponentially dilated convolutions.

Key features:
- Causal dilated convolutions with exponential dilation
- Residual connections
- WARNING: At tight budgets (0-20ms), model sees mostly zero-padding
- Spectral flux input channel option
- Triple-head output: onset + offset + velocity (v11)
- Negative bias initialization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .streaming_cnn import SpectralFluxLayer, CausalConv1d


class TCNBlock(nn.Module):
    """
    TCN residual block with dilated causal convolution.

    Structure: CausalConv -> BN -> ReLU -> Dropout -> CausalConv -> BN -> ReLU -> Dropout + Residual
    """

    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        return x + residual


class TCN(nn.Module):
    """
    Temporal Convolutional Network for note transcription (v11).

    Architecture:
        Input (n_mels x T) -> [optional spectral flux] ->
        1x1 Conv projection ->
        N x TCNBlock (dilations: 1, 2, 4, 8, 16) ->
        3 x 1x1 Conv heads -> onset, offset, velocity logits [88, T] each

    Receptive field: 1 + 2 * (kernel_size - 1) * sum(dilations)
                   = 1 + 2 * 2 * (1+2+4+8+16) = 1 + 4 * 31 = 125
                   But effective RF is ~63 frames due to weight decay

    Parameters: ~280K (includes offset + velocity heads)

    NOTE: At 20ms budget (2 frames), TCN will mostly see zero-padding.
    This is EXPECTED and is part of the thesis analysis -- showing which
    architectures degrade more gracefully at tight budgets.
    """

    def __init__(self, n_mels=229, n_pitches=88, channels=128,
                 kernel_size=3, num_blocks=5, dropout=0.1,
                 use_spectral_flux=True):
        super().__init__()

        self.use_spectral_flux = use_spectral_flux
        self.spectral_flux = SpectralFluxLayer() if use_spectral_flux else None

        in_ch = n_mels * (2 if use_spectral_flux else 1)

        # Project to hidden channels
        self.input_proj = nn.Conv1d(in_ch, channels, 1)

        # TCN blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList([
            TCNBlock(channels, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(num_blocks)
        ])

        # Onset head (existing)
        self.onset_head = nn.Conv1d(channels, n_pitches, 1)
        bias_val = math.log(0.01 / (1 - 0.01))
        nn.init.constant_(self.onset_head.bias, bias_val)

        # NEW v11: Offset head
        self.offset_head = nn.Conv1d(channels, n_pitches, 1)
        offset_bias = math.log(0.01 / (1 - 0.01))
        nn.init.constant_(self.offset_head.bias, offset_bias)

        # NEW v11: Velocity head (regression, neutral bias)
        self.velocity_head = nn.Conv1d(channels, n_pitches, 1)
        nn.init.constant_(self.velocity_head.bias, 0.0)

        # Store config
        self._dilations = [2**i for i in range(num_blocks)]
        self._kernel_size = kernel_size

    def forward(self, x):
        """
        Args:
            x: mel spectrogram [B, n_mels, T]
        Returns:
            dict with 'onset_logits', 'offset_logits', 'velocity_logits': each [B, 88, T]
        """
        if self.use_spectral_flux:
            flux = self.spectral_flux(x)
            x = torch.cat([x, flux], dim=1)

        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        return {
            'onset_logits': self.onset_head(x),
            'offset_logits': self.offset_head(x),
            'velocity_logits': self.velocity_head(x),
        }

    def get_receptive_field(self):
        """Theoretical receptive field in frames."""
        rf = 1
        for d in self._dilations:
            rf += 2 * (self._kernel_size - 1) * d
        return rf


def create_tcn(config):
    """Factory function."""
    return TCN(
        n_mels=config.n_mels,
        n_pitches=config.n_pitches,
        channels=config.tcn_channels,
        kernel_size=config.tcn_kernel_size,
        num_blocks=config.tcn_num_blocks,
        use_spectral_flux=config.use_spectral_flux,
    )
