"""
LatBOND Full Version - Configuration
=====================================
42 Models: 3 Architectures x 5 Latency Budgets x 3 Conditions - 3 Redundant

Full upgrade from v11 Demo Lite:
  - 5 latency budgets: 0, 20, 40, 60, 80ms
  - 3 conditions: matched, truncated, buffered
  - Multi-GPU Vertex AI support
  - Full statistical analysis suite

All v11 innovations retained:
  - Triple-head output: onset + offset + velocity
  - TripleHeadLoss (focal + MSE)
  - Asymmetric Focal Loss (gamma_neg=3.0)
  - Cosine curriculum schedule
  - Architecture-aware context windows
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LatBONDConfig:
    # === Dataset ===
    subset_fraction: float = 0.30          # v3.3.5: 30% MAESTRO for pilot study
    maestro_version: str = "v3.0.0"
    maestro_root: str = "data/maestro-v3.0.0"
    sample_rate: int = 16000

    # === Audio / STFT (CRITICAL: DO NOT CHANGE) ===
    n_fft: int = 2048
    hop_length: int = 160                  # 10ms frames at 16kHz
    n_mels: int = 229                      # Matching Hu et al.
    f_min: float = 30.0
    f_max: float = 8000.0
    frame_duration_ms: float = 10.0        # hop_length / sample_rate * 1000

    # === Task ===
    n_pitches: int = 88                    # Piano keys
    onset_prior: float = 0.01              # ~1% of frames are onsets

    # === Triple-head note detection ===
    midi_offset: int = 21                  # MIDI note number of lowest piano key (A0)

    # Loss weights for triple-head combination
    onset_loss_weight: float = 1.0         # Primary task - drives early stopping
    offset_loss_weight: float = 0.5        # Secondary
    velocity_loss_weight: float = 0.1      # Tertiary - regression

    # Velocity normalization
    velocity_scale: float = 127.0          # MIDI velocity range

    # Per-head priors
    offset_prior: float = 0.01             # ~1% of frames are offsets

    # Velocity supervision window
    velocity_window_frames: int = 3        # Supervise velocity on 3-frame window

    # === NEW in Full Version: Multi-budget support ===
    latency_budgets_ms: List[int] = field(default_factory=lambda: [0, 20, 40, 60, 80])
    training_conditions: List[str] = field(default_factory=lambda: ['matched', 'truncated', 'buffered'])

    # Current run configuration (set per model)
    current_budget_ms: int = 20            # Active latency budget for this run
    current_condition: str = 'matched'     # Active training condition
    current_architecture: str = 'streaming_cnn'  # Current architecture being trained

    # Legacy fields for v11 compatibility
    budget_ms: int = 20
    budget_frames: int = 3

    # === Architectures ===
    architectures: List[str] = field(default_factory=lambda: [
        "streaming_cnn",
        "causal_transformer",
        "tcn"
    ])

    # --- Streaming CNN ---
    cnn_channels: List[int] = field(default_factory=lambda: [128, 256, 256, 256])
    cnn_kernel_size: int = 3
    cnn_dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])

    # --- Causal Transformer ---
    transformer_d_model: int = 192
    transformer_nhead: int = 6
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 384
    transformer_dropout: float = 0.1

    # --- TCN ---
    tcn_channels: int = 160
    tcn_kernel_size: int = 3
    tcn_num_blocks: int = 4                # v3: 4 blocks, dilations: 1, 2, 4, 8

    # === Training ===
    conditions: List[str] = field(default_factory=lambda: ["matched", "truncated"])
    epochs: int = 50                       # v3.3.10: compressed (was 100)
    matched_epochs: int = 100              # v3.3.11: matched gets 2x epochs (still improving at 50)
    batch_size: int = 128                  # v3.3.4: L4 has 24GB, models use <600MB at 128
    learning_rate: float = 2.8e-3          # v3.3.4: scaled by sqrt(4)=2 for batch 128
    weight_decay: float = 1e-4             # v2: was 1e-5, prevents overfitting at 15%/100ep
    early_stopping_patience: int = 15      # v3.3: restored from v11 (was 10)
    warmup_epochs: int = 2
    optimizer_type: str = "adamw"          # v2: AdamW for properly decoupled weight decay
    label_smoothing: float = 0.01          # v2: better threshold stability

    # v2 #20: AMP mixed precision training
    use_amp: bool = True                   # v2: enable automatic mixed precision

    # v2 #25: SpecAugment data augmentation
    use_specaugment: bool = True           # v2: enable SpecAugment
    specaug_freq_masks: int = 2            # Number of frequency masks
    specaug_freq_width: int = 15           # Max frequency mask width
    specaug_time_masks: int = 2            # Number of time masks
    specaug_time_width: int = 25           # Max time mask width

    # --- Matched training specifics ---
    context_multiplier: int = 4
    num_crops_per_sample: int = 8
    onset_crop_ratio: float = 0.50

    # --- Loss ---
    loss_type: str = "asymmetric_focal"
    pos_weight: float = 10.0
    focal_gamma_pos: float = 1.0
    focal_gamma_neg: float = 3.0           # Conservative preset

    # --- Spectral Flux ---
    use_spectral_flux: bool = True

    # --- Curriculum ---
    curriculum_type: str = "cosine"

    # === Evaluation ===
    eval_tolerance_frames: int = 2         # +/-2 frames = +/-20ms tolerance
    eval_stride: int = 2                  # v3.3.7 FIX #5: Fixed stride=2 for all archs (was ctx-RF variable)
    eval_batch_size: int = 64             # v3.3.7 FIX #6: Reduced from 128 to prevent OOM on CausalXfmr
    threshold_sweep: List[float] = field(default_factory=lambda: [
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80
    ])  # v3.3.7 FIX #3: Extended to 0.80 (was 0.50) — logs showed Val F1 rising at 0.50

    # v3.3.7 FIX #4: Per-arch AMP disable (CausalXfmr crashes under float16)
    amp_disabled_archs: List[str] = field(default_factory=lambda: ["causal_transformer"])
    
    # v3.3.3 FIX #4: Peak picking post-processing
    use_peak_picking: bool = False         # v3.3.4: Disabled for clean pilot, not in validated v11
    peak_pick_min_distance: int = 2        # Minimum frames between peaks (20ms)

    # === Paths ===
    data_dir: str = "data"
    output_dir: str = "outputs"

    # === Statistical analysis parameters ===
    bonferroni_alpha: float = 0.01         # 0.05 / 5 budgets
    bootstrap_n_resamples: int = 1000      # Bootstrap iterations for CIs
    bootstrap_ci_level: float = 0.95       # 95% confidence intervals
    random_seed: int = 42                  # Fixed seed for reproducibility

    # === Vertex AI configuration ===
    gcp_project_id: str = "YOUR_GCP_PROJECT"
    gcp_region: str = "northamerica-northeast2"  # Toronto - where Docker image is pushed
    gcp_bucket: str = "YOUR_GCP_PROJECT-data"
    vertex_machine_type: str = "g2-standard-16"  # v2: g2 for L4 GPU (16 vCPU, 64GB RAM)
    vertex_accelerator: str = "NVIDIA_L4"        # v2: was T4, faster memory bandwidth
    num_parallel_jobs: int = 6                   # v2: was 3, halves wall clock

    # === DataLoader configuration (v2 optimizations) ===
    num_workers: int = 4                   # v3.3.5: Reduced from 8 to prevent DataLoader deadlock in Docker
    pin_memory: bool = True                # v2: page-locked memory for faster GPU transfer
    persistent_workers: bool = True        # v2: workers survive between epochs

    # === Output paths ===
    output_base_dir: str = "outputs"       # Base directory for all outputs

    # === Demo pipeline ===
    demo_n_examples: int = 5               # Number of demo examples to generate
    demo_tolerance_ms: float = 30.0        # Primary evaluation tolerance for demos

    # === Receptive Fields (frames) ===
    @property
    def receptive_fields(self):
        """v3.3.3: Empirical receptive field in frames (verified via gradient backprop)."""
        return {
            "streaming_cnn": 32,       # Empirical (was 31 analytical)
            "causal_transformer": 32,  # Design choice (actual RF = unbounded)
            "tcn": 62,                 # Empirical (was 61 analytical)
        }

    @property
    def arch_overrides(self):
        """v3.3.3: Per-architecture hyperparameters with budget-specific LR scaling."""
        return {
            'streaming_cnn': {
                'learning_rate': 4e-3,     # v3.3.6 FIX #9: 2x increase for faster convergence
                'warmup_epochs': 3,        # v3.3.7 FIX #7: Increased from 2 for stability
                'epochs': 50,              # v3.3.7 FIX #8: Unified to 50 (was 40)
                'receptive_field': 32,     # Empirical RF (was 31)
                'context_override': {0: 33, 20: 35, 40: 37, 60: 39, 80: 41},  # RF(32)+budget
                # v3.3.3 FIX #3: Per-budget LR scaling (tighter budget = lower LR)
                'lr_budget_scale': {0: 0.7, 20: 0.85, 40: 1.0, 60: 1.1, 80: 1.2},
            },
            'causal_transformer': {
                'learning_rate': 2e-3,     # v3.3.6 FIX #9: 2x increase for faster convergence
                'warmup_epochs': 3,        # v3.3.7 FIX #7: Reduced from 8 for consistency
                'epochs': 50,              # v3.3.7 FIX #8: Unified to 50
                'receptive_field': 32,
                'context_override': {0: 33, 20: 35, 40: 37, 60: 39, 80: 41},  # v3.3.3: Fixed RF(32)+budget
                # v3.3.3 FIX #3: Per-budget LR scaling
                'lr_budget_scale': {0: 0.7, 20: 0.85, 40: 1.0, 60: 1.1, 80: 1.2},
            },
            'tcn': {
                'learning_rate': 4e-3,     # v3.3.6 FIX #9: 2x increase for faster convergence
                'warmup_epochs': 3,        # v3.3.7 FIX #7: Increased from 2 for stability
                'epochs': 50,              # v3.3.7 FIX #8: Unified to 50 (was 40)
                'receptive_field': 62,     # Empirical RF (was 61)
                'context_override': {0: 63, 20: 65, 40: 67, 60: 69, 80: 71},  # RF(62)+budget
                # v3.3.3 FIX #3: Per-budget LR scaling
                'lr_budget_scale': {0: 0.7, 20: 0.85, 40: 1.0, 60: 1.1, 80: 1.2},
            },
        }

    @property
    def context_frames_per_arch(self):
        """v3: Context window from arch_overrides or fallback to receptive_field + budget_frames."""
        budget_frames = self.get_budget_frames()
        budget_ms = self.current_budget_ms
        result = {}
        for arch in self.architectures:
            overrides = self.arch_overrides.get(arch, {})
            ctx_map = overrides.get('context_override', {})
            if ctx_map and budget_ms in ctx_map:
                result[arch] = ctx_map[budget_ms]
            else:
                result[arch] = self.receptive_fields[arch] + budget_frames
        return result

    @property
    def input_channels(self):
        """Number of input channels (mel + optional spectral flux)."""
        return 2 if self.use_spectral_flux else 1

    def get_budget_frames(self, budget_ms: Optional[int] = None) -> int:
        """Convert ms budget to frame count. 0ms = 1 frame (current only)."""
        bms = budget_ms if budget_ms is not None else self.current_budget_ms
        # Formula: max(1, budget_ms // 10 + 1)
        # 0ms -> 1 frame, 20ms -> 3 frames, 40ms -> 5 frames, etc.
        return max(1, int(bms / (self.hop_length / self.sample_rate * 1000)) + 1)

    def get_output_dir(self, arch: str, budget_ms: int, condition: str) -> str:
        """Structured output directory for each model."""
        return f"{self.output_base_dir}/{arch}/{budget_ms}ms/{condition}"

    def is_valid_training(self, budget_ms: int, condition: str) -> bool:
        """Check if this (budget, condition) combination should be TRAINED."""
        # Buffered@0ms: skip training (no lookahead possible at 0ms)
        # But we still EVALUATE the buffered model at 0ms for recovery %
        if condition == 'buffered' and budget_ms == 0:
            return False
        return True

    def is_valid_evaluation(self, budget_ms: int, condition: str) -> bool:
        """Check if this (budget, condition) combination should be EVALUATED.
        Always True - we need buffered@0ms for recovery % calculation."""
        return True

    def get_training_configurations(self) -> list:
        """Generate all (arch, budget, condition) triples that need TRAINING."""
        configs = []
        for arch in self.architectures:
            for budget in self.latency_budgets_ms:
                for condition in self.training_conditions:
                    if self.is_valid_training(budget, condition):
                        configs.append((arch, budget, condition))
        return configs  # 42 training configurations

    def get_evaluation_configurations(self) -> list:
        """Generate all (arch, budget, condition) triples that need EVALUATION.
        Includes buffered@0ms (uses buffered model from 20ms budget)."""
        configs = []
        for arch in self.architectures:
            for budget in self.latency_budgets_ms:
                for condition in self.training_conditions:
                    configs.append((arch, budget, condition))
        return configs  # 45 evaluation configurations


# Default config instance
CONFIG = LatBONDConfig()
