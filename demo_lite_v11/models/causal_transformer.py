"""
Causal Transformer v11 - Attention-based model for note transcription.
Best suited for 20-40ms latency budgets.
Receptive field: ~32 frames (~320ms) via attention window.

Key features:
- Causal attention mask (no future leakage)
- Sinusoidal positional encoding
- Spectral flux input channel option
- Triple-head output: onset + offset + velocity (v11)
- Negative bias initialization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .streaming_cnn import SpectralFluxLayer


class CausalMultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal mask."""
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Causal mask: prevent attending to future positions
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with causal attention + FFN."""
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attn = CausalMultiHeadAttention(d_model, nhead, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (no learned parameters)."""
    
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x):
        """x: [B, T, d_model]"""
        return x + self.pe[:, :x.size(1), :]


class CausalTransformer(nn.Module):
    """
    Causal Transformer for note transcription (v11).

    Architecture:
        Input (n_mels × T) → [optional spectral flux] →
        Linear projection → Positional encoding →
        N × TransformerBlock (causal attention) →
        3 × Linear heads → onset, offset, velocity logits [88, T] each

    NOTE: For Lite demo, use num_layers=2. Transformers are data-hungry
    and deeper models risk non-convergence at 3% MAESTRO. The shallow
    config still captures attention patterns while remaining trainable.

    Receptive field: Theoretically unlimited (attention),
                     practically ~32 frames via attention patterns.
    Parameters: ~280K (at num_layers=2, includes offset + velocity heads)
    """

    def __init__(self, n_mels=229, n_pitches=88, d_model=128,
                 nhead=4, num_layers=2, dim_feedforward=256,
                 dropout=0.1, use_spectral_flux=True):
        super().__init__()

        self.use_spectral_flux = use_spectral_flux
        self.spectral_flux = SpectralFluxLayer() if use_spectral_flux else None

        in_dim = n_mels * (2 if use_spectral_flux else 1)

        # Project mel features to model dimension
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Onset head (existing)
        self.onset_head = nn.Linear(d_model, n_pitches)
        bias_val = math.log(0.01 / (1 - 0.01))
        nn.init.constant_(self.onset_head.bias, bias_val)

        # NEW v11: Offset head
        self.offset_head = nn.Linear(d_model, n_pitches)
        offset_bias = math.log(0.01 / (1 - 0.01))
        nn.init.constant_(self.offset_head.bias, offset_bias)

        # NEW v11: Velocity head (regression, neutral bias)
        self.velocity_head = nn.Linear(d_model, n_pitches)
        nn.init.constant_(self.velocity_head.bias, 0.0)
    
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

        # Transpose to [B, T, features] for transformer
        x = x.transpose(1, 2)

        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.input_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        onset_logits = self.onset_head(x)      # [B, T, 88]
        offset_logits = self.offset_head(x)    # [B, T, 88]
        velocity_logits = self.velocity_head(x) # [B, T, 88]

        return {
            'onset_logits': onset_logits.transpose(1, 2),      # [B, 88, T]
            'offset_logits': offset_logits.transpose(1, 2),    # [B, 88, T]
            'velocity_logits': velocity_logits.transpose(1, 2), # [B, 88, T]
        }


def create_causal_transformer(config):
    """Factory function."""
    return CausalTransformer(
        n_mels=config.n_mels,
        n_pitches=config.n_pitches,
        d_model=config.transformer_d_model,
        nhead=config.transformer_nhead,
        num_layers=config.transformer_num_layers,
        dim_feedforward=config.transformer_dim_feedforward,
        dropout=config.transformer_dropout,
        use_spectral_flux=config.use_spectral_flux,
    )
