"""
LatBOND Demo Lite v11 - Configuration
======================================
3 Architectures × 2 Conditions × 1 Budget = 6 models
Target: ~6-7 hours total on single T4 GPU

v11 CHANGES:
  - Triple-head output: onset + offset + velocity prediction
  - TripleHeadLoss combining AsymmetricFocalLoss + MSE
  - Note-level evaluation via mir_eval (Hu et al. comparable)
  - Models return dict with onset_logits, offset_logits, velocity_logits

All v10.1 innovations retained:
  - Asymmetric Focal Loss (Conservative preset: gamma_neg=3.0)
  - Cosine curriculum schedule
  - Architecture-aware context windows
  - Batched parallel crop training
  - Full-window loss
  - Sliding window evaluation
  - Spectral flux input channel
  - Negative bias initialization
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # === Dataset ===
    subset_fraction: float = 0.05          # 5% MAESTRO (Run 1 proven baseline)
    maestro_version: str = "v3.0.0"
    sample_rate: int = 16000
    
    # === Audio / STFT ===
    n_fft: int = 2048
    hop_length: int = 160                  # 10ms frames at 16kHz
    n_mels: int = 229                      # Matching Hu et al.
    frame_duration_ms: float = 10.0        # hop_length / sample_rate * 1000
    
    # === Task ===
    n_pitches: int = 88                    # Piano keys
    onset_prior: float = 0.01             # ~1% of frames are onsets

    # === NEW in v11: Triple-head note detection ===
    midi_offset: int = 21                  # MIDI note number of lowest piano key (A0)

    # Loss weights for triple-head combination
    onset_loss_weight: float = 1.0         # Primary task — drives early stopping
    offset_loss_weight: float = 0.5        # Secondary — offsets are less temporally precise
    velocity_loss_weight: float = 0.1      # Tertiary — regression task, lower weight

    # Velocity normalization
    velocity_scale: float = 127.0          # MIDI velocity range for normalization to [0, 1]

    # Per-head priors for bias initialization
    # onset_prior already exists (0.01) — keep it
    offset_prior: float = 0.01            # ~1% of frames are offsets (same sparsity as onsets)

    # Velocity supervision window
    velocity_window_frames: int = 3        # Supervise velocity on 3-frame window around onsets
    
    # === Latency Budget ===
    budget_ms: int = 20                    # Single budget for lite demo
    budget_frames: int = 2                 # 20ms / 10ms = 2 frames
    
    # === Architectures ===
    architectures: List[str] = field(default_factory=lambda: [
        "streaming_cnn",
        "causal_transformer",
        "tcn"
    ])
    
    # --- Streaming CNN ---
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 128])
    cnn_kernel_size: int = 3
    
    # --- Causal Transformer ---
    transformer_d_model: int = 128
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 256
    transformer_dropout: float = 0.1
    
    # --- TCN ---
    tcn_channels: int = 128
    tcn_kernel_size: int = 3
    tcn_num_blocks: int = 5                # Dilations: 1, 2, 4, 8, 16
    
    # === Training ===
    conditions: List[str] = field(default_factory=lambda: [
        "matched",
        "truncated"
    ])
    epochs: int = 25                       # SOLUTION 2: was 12
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 15      # SOLUTION 2: was 4
    warmup_epochs: int = 2                 # Linear LR warmup (stabilizes Transformer)
    
    # --- Matched training specifics ---
    context_multiplier: int = 4
    num_crops_per_sample: int = 8
    onset_crop_ratio: float = 0.50
    
    # --- Loss (SOLUTION 3) ---
    loss_type: str = "asymmetric_focal"    # "bce" or "asymmetric_focal"
    pos_weight: float = 10.0
    # Asymmetric Focal Loss gamma tuning (v10.1: Conservative is new default):
    #   Conservative: gamma_pos=1.0, gamma_neg=3.0 (default — best precision/recall balance)
    #   Balanced:     gamma_pos=1.0, gamma_neg=4.0 (more recall, higher FP risk)
    #   Aggressive:   gamma_pos=0.5, gamma_neg=5.0 (max recall, causes FP explosion)
    focal_gamma_pos: float = 1.0
    focal_gamma_neg: float = 3.0           # v10.1: was 4.0, ablation showed FP risk
    
    # --- Spectral Flux ---
    use_spectral_flux: bool = True
    
    # --- Curriculum (SOLUTION 1) ---
    curriculum_type: str = "cosine"        # "cosine" (v10) or "linear" (v9)
    
    # === Evaluation ===
    eval_tolerance_frames: int = 2         # ±2 frames = ±20ms tolerance
    eval_stride: int = 2
    eval_batch_size: int = 128
    threshold_sweep: List[float] = field(default_factory=lambda: [
        0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5
    ])
    
    # === Paths ===
    data_dir: str = "data"
    output_dir: str = "outputs"
    
    # === Receptive Fields (frames) ===
    @property
    def receptive_fields(self):
        """Receptive field in frames for each architecture."""
        return {
            "streaming_cnn": 15,
            "causal_transformer": 32,
            "tcn": 63,
        }

    @property
    def context_frames_per_arch(self):
        """Context window = receptive_field + budget_frames for each architecture."""
        return {
            "streaming_cnn":      self.receptive_fields["streaming_cnn"] + self.budget_frames,
            "causal_transformer": self.receptive_fields["causal_transformer"] + self.budget_frames,
            "tcn":                self.receptive_fields["tcn"] + self.budget_frames,
        }

    @property
    def input_channels(self):
        """Number of input channels (mel + optional spectral flux)."""
        return 2 if self.use_spectral_flux else 1


CONFIG = Config()
