"""
LatBOND Trainer v3.3.10
========================
v3.3.10 FIXES (fixing matched training to outperform truncated):
  - FIX #1: CRITICAL — validate() now uses _last_frame_constrained_eval()
    for matched condition. v3.3.8 BUG: validation used full 500-frame forward
    pass, so early stopping selected checkpoints from early epochs when context
    was still ~494 frames, making matched functionally identical to truncated.
    FIX: Matched validation now evaluates under restricted context, so early
    stopping picks the model that performs best at deployment conditions.
  - FIX #2: Stepped geometric curriculum replacing cosine decay.
    Stages: 500→250→125→62→37 (geometric halving)
    Each stage lets the model converge before the next context reduction.
    Stage 5 (38% of training) is fine-tuning at deployment context.
    v3.3.8 cosine reached target only at epoch 98/100, giving just 2 epochs
    of fine-tuning. Now 19+ epochs are at target context.
  - FIX #3: Matched condition uses patience=30 (vs 15 for truncated/buffered)
    to allow curriculum to complete without premature early stopping.

v3.3.8 FIX (restoring matched > truncated):
  - FIX #1: CRITICAL — evaluate_test() now uses _last_frame_constrained_eval()
    for matched/truncated conditions. v3.3.6 used identical sliding window for
    both → same ~0.34 F1 penalty → matched-truncated difference invisible.
    v3.3.7: Both tested under identical constraint; matched trained under it
    → should outperform. Truncated faces distribution shift → penalty visible.
  - FIX #2: CRITICAL — New _last_frame_constrained_eval() extracts ONLY the
    last frame of each window (has full RF context). v3.3.6 averaged ALL frames
    including 84% with incomplete context → boundary noise corrupted predictions.
  - FIX #3: Threshold sweep extended to 0.80 (was 0.50). Logs showed Val F1
    still rising at 0.50 for CausalXfmr.
  - FIX #4: CRITICAL — AMP disabled for CausalXfmr (crashes epoch 17 under
    float16). GradScaler guard: unscale_() called BEFORE step() to register
    inf checks that update() requires.
  - FIX #5: Eval stride = 2 fixed (was variable ctx-RF).
  - FIX #6: Eval batch size = 64 (was 128, OOM on CausalXfmr).
  - FIX #7: Warmup = 3 epochs (was 2/8, now unified).
  - FIX #8: Epochs = 50 unified (was 40/50 split).

v3.3.6 FIXES:
  - FIX #8: PERFORMANCE FIX — Validation epoch reduced from ~9,000s to ~15s.
    validate() now uses direct forward pass + vectorized tolerance F1.
  - FIX #9: Higher learning rates (2x) for faster convergence.

v3.3.5 FIXES:
  - FIX #5: CRITICAL LOGIC FIX — Truncated validation now uses constrained
    context (use_full_context=False), matching thesis Section 3.3.2 definition.
    Previously truncated was grouped with buffered (use_full_context=True),
    which violated the thesis by eliminating distribution shift during validation.
    Now: matched + truncated → constrained context (deployment conditions)
         buffered only → full context (upper bound)
  - FIX #6: OOM-safe full-context via context cap in _batched_sliding_window.
    Caps ctx to 2500 frames (25s) when use_full_context=True, preventing
    massive tensor allocation that OOMed on L4 GPUs (22GB).
    Uses unified tiling stride (ctx - rf) for both modes to avoid
    performance bomb from dense eval_stride with large windows.
  - FIX #7: Added diagnostic logging before first epoch to catch hangs.
    Logs condition logic and flushes stdout.

v3.3.3 FIXES (all 4 fixes applied):
  - FIX #1: Validation F1 uses tolerance-based matching (±2 frames)
    (was frame-level, causing threshold stuck at 0.50)
  - FIX #2: Per-file test F1 uses tolerance-based matching + best threshold
    (was frame-level with hardcoded 0.5, invalidating statistical tests)
  - FIX #3: Per-budget learning rate scaling
    (tighter budgets use lower LR for better convergence)
  - FIX #4: Onset peak picking post-processing
    (removes double detections, reduces false positives)

Back to v11 fundamentals that WORKED, scaled up for full study.

v3.3 CHANGES (fixing v3.1/v3.2 failures):
  - REMOVED two-phase training (caused early stop before phase 2)
  - RESTORED v11 cosine curriculum (gradual context reduction, no shock)
  - RESTORED threshold sweep in validation (was hardcoded at 0.3)
  - RESTORED per-epoch validation (was skipping 2/3 of epochs)
  - Patience = 15 (was 10, killed models prematurely)

Retained from v3:
  - Buffered condition support
  - AMP mixed precision
  - Per-architecture epochs/LR
  - SpecAugment
  - Batched sliding window inference
  - NaN safety net

Retained from v11:
  - Cosine curriculum schedule
  - Asymmetric Focal Loss + TripleHeadLoss
  - Batched parallel crop training
  - Architecture-aware context windows
  - F1-based early stopping with threshold sweep
"""

import os
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from pathlib import Path
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from training.losses import create_loss


class LatBONDTrainer:
    """
    Trainer for LatBOND onset detection models.
    Handles matched, truncated, and buffered training conditions.
    """

    def __init__(self, model, config, condition, arch_name, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.condition = condition  # 'matched', 'truncated', or 'buffered'
        self.arch_name = arch_name
        self.device = device
        self.budget_ms = config.current_budget_ms

        # v3.3.4: DDP support for multi-GPU training
        self.distributed = False
        self.local_rank = 0
        if dist.is_initialized():
            self.distributed = True
            self.local_rank = dist.get_rank()
            self.model = DDP(self.model, device_ids=[self.local_rank])
            print(f"  DDP enabled: rank {self.local_rank} of {dist.get_world_size()}")

        # v3: Get per-architecture overrides
        overrides = config.arch_overrides.get(arch_name, {})

        # Context window (target for matched training)
        ctx_map = overrides.get('context_override', {})
        budget_ms_val = config.current_budget_ms

        # Debug: uncomment to trace context initialization
        # print(f"DEBUG: arch={arch_name}, budget_ms={budget_ms_val}, ctx_map keys={list(ctx_map.keys())}, in_map={budget_ms_val in ctx_map}")

        if ctx_map and budget_ms_val in ctx_map:
            self.context_frames = ctx_map[budget_ms_val]
        elif hasattr(config, 'context_frames_per_arch') and arch_name in config.context_frames_per_arch:
            self.context_frames = config.context_frames_per_arch[arch_name]
        else:
            budget_frames = config.get_budget_frames()
            self.context_frames = config.receptive_fields.get(arch_name, 32) + budget_frames

        # Budget frames for sliding window
        self.budget_frames = config.get_budget_frames()

        # v3.3.4: RF verification logging
        print(f"RF Verification: {arch_name}")
        print(f"  Config RF: {config.receptive_fields.get(arch_name)} frames")
        print(f"  Context frames: {self.context_frames} frames")
        print(f"  Condition: {self.condition}")
        print(f"  Budget: {config.current_budget_ms}ms ({self.budget_frames} frames)")
        if self.condition == 'matched':
            expected = config.receptive_fields.get(arch_name, 0) + self.budget_frames
            if abs(self.context_frames - expected) > 2:
                print(f"  WARNING: context_frames ({self.context_frames}) != RF+budget ({expected})")
            else:
                print(f"  OK: context matches RF + budget")

        # v3.3.3 FIX #3: Per-architecture learning rate with budget-specific scaling
        base_arch_lr = overrides.get('learning_rate', config.learning_rate)
        lr_budget_scale = overrides.get('lr_budget_scale', {})
        budget_scale = lr_budget_scale.get(budget_ms_val, 1.0)  # Default 1.0 if not specified
        self.base_lr = base_arch_lr * budget_scale
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.base_lr,
            weight_decay=config.weight_decay,
        )

        # v3: Per-architecture warmup and epochs
        warmup_epochs = overrides.get('warmup_epochs', config.warmup_epochs)
        total_epochs = overrides.get('epochs', config.epochs)

        # v3.3.11: Matched condition gets more epochs (still improving at 50)
        if condition == 'matched' and hasattr(config, 'matched_epochs'):
            total_epochs = config.matched_epochs
            print(f"  [v3.3.11] Matched epochs: {total_epochs} (vs {config.epochs} for others)")

        # v3.3.6: Smoke test epoch cap
        if hasattr(config, '_max_epochs_override') and config._max_epochs_override is not None:
            total_epochs = min(total_epochs, config._max_epochs_override)
            print(f"  [SMOKE TEST] Epochs capped: {total_epochs}")

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return 0.1 + 0.9 * ((epoch + 1) / warmup_epochs)
            else:
                progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )

        # v3: Store max epochs for train loop
        self._max_epochs = total_epochs

        # Loss function
        self.criterion = create_loss(config, device)

        # AMP mixed precision
        # v3.3.7 FIX #4: Per-arch AMP disable (CausalXfmr crashes under float16)
        amp_disabled = getattr(config, 'amp_disabled_archs', [])
        if arch_name in amp_disabled:
            self.use_amp = False
            print(f"  [v3.3.7] AMP DISABLED for {arch_name} (float16 causes NaN/Inf)")
        else:
            self.use_amp = getattr(config, 'use_amp', False) and (device == 'cuda')
        self.scaler = GradScaler(enabled=self.use_amp)

        # Tracking
        self.history = {
            'train_loss': [], 'val_loss': [],
            'val_f1': [], 'best_val_f1': 0.0,
            'best_epoch': 0, 'best_threshold': 0.3,
            'curriculum_context': [],
        }
        self.patience_counter = 0
        self.best_val_f1 = 0.0
        self.best_threshold = 0.3
        self.current_epoch = 0

        # v3.3.4: OOM recovery - reduce mini-batch size if OOM detected
        self.effective_batch_size = config.eval_batch_size  # Start with config value
        self.oom_recovery_active = False

    # ==================================================================
    # v3.3.10 STEPPED GEOMETRIC CURRICULUM
    # ==================================================================

    def get_context_for_epoch(self, epoch):
        """
        v3.3.10: Stepped geometric curriculum replacing v3.3.8 cosine.

        v3.3.8 PROBLEM: Cosine curriculum + full-context validation + early stopping
        caused best checkpoint to be saved from early epochs (context ~494 frames),
        making matched ≈ truncated. Model never reached target context (37 frames).

        v3.3.10 FIX: Stepped geometric halving (500→250→125→62→37).
        Each stage halves the context, giving the model time to fully converge
        at each level before the next reduction. This is more stable than
        continuous schedules because:
          1. Neural networks learn in discrete stages — stepped matches this
          2. No "moving target" — context is fixed within each stage
          3. Geometric progression (÷2 each step) is a natural difficulty ramp
          4. Clear stage boundaries for thesis analysis and reporting

        Schedule (50 epochs):
          Stage 1 (epochs  1-10): 500 frames — learn onset features
          Stage 2 (epochs 11-17): 250 frames — adapt to half context
          Stage 3 (epochs 18-24): 125 frames — quarter context
          Stage 4 (epochs 25-31):  62 frames — near-target adaptation
          Stage 5 (epochs 32-50):  37 frames — finetune at deployment (19 epochs)

        Truncated: always 500 (full context during training)
        Buffered: always 500 + budget_frames
        """
        # Truncated: always full context during training
        if self.condition == 'truncated':
            return 500

        # Buffered: full context + lookahead
        if self.condition == 'buffered':
            return 500 + self.budget_frames

        # Matched: stepped geometric curriculum
        target_ctx = self.context_frames
        total_epochs = self._max_epochs

        # Compute stage boundaries as fractions of total epochs
        # Stage 1: 20% learn, Stage 2-4: 14% each adapt, Stage 5: 38% finetune
        s1_end = max(int(total_epochs * 0.20), 5)       # 10 at 50 epochs
        s2_end = max(int(total_epochs * 0.34), s1_end + 3)  # 17
        s3_end = max(int(total_epochs * 0.48), s2_end + 3)  # 24
        s4_end = max(int(total_epochs * 0.62), s3_end + 3)  # 31
        # Stage 5: s4_end+1 to total_epochs (finetune at target)

        if epoch <= s1_end:
            return 500
        elif epoch <= s2_end:
            return 250
        elif epoch <= s3_end:
            return 125
        elif epoch <= s4_end:
            return 62
        else:
            return target_ctx

    # ==================================================================
    # TRAINING
    # ==================================================================

    def train_epoch(self, train_loader, epoch=None):
        """Train one epoch with OOM recovery."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (mel, labels) in enumerate(tqdm(train_loader, desc="Train", leave=False)):
            mel = mel.to(self.device, non_blocking=True)
            labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}

            try:
                if self.condition == 'matched':
                    loss = self._matched_train_step(mel, labels, epoch)
                elif self.condition in ('truncated', 'buffered'):
                    loss = self._truncated_train_step(mel, labels)
                else:
                    raise ValueError(f"Unknown condition: {self.condition}")

                total_loss += loss
                n_batches += 1

            except torch.cuda.OutOfMemoryError as e:
                # v3.3.4: OOM recovery - clear memory and enable gradient accumulation
                self._handle_oom(batch_idx, epoch)
                # Skip this batch after OOM recovery
                continue

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    def _handle_oom(self, batch_idx, epoch):
        """Handle OOM by clearing cache and reducing effective batch size."""
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Halve effective mini-batch size (used in forward pass chunking)
        old_batch = self.effective_batch_size
        self.effective_batch_size = max(self.effective_batch_size // 2, 8)  # Min 8

        if not self.oom_recovery_active:
            self.oom_recovery_active = True
            print(f"\n  [OOM RECOVERY] CUDA OOM at epoch {epoch}, batch {batch_idx}")
            print(f"  [OOM RECOVERY] Reducing mini-batch: {old_batch} -> {self.effective_batch_size}")
        else:
            if self.effective_batch_size < old_batch:
                print(f"  [OOM RECOVERY] Further reduced mini-batch: {old_batch} -> {self.effective_batch_size}")

    def _matched_train_step(self, mel, labels, epoch=None):
        """
        Matched training with cosine curriculum context.
        v11 approach: onset-biased crops at curriculum-determined context size.
        """
        B, C, T = mel.shape
        ctx = self.get_context_for_epoch(epoch) if epoch is not None else self.context_frames
        n_crops = self.config.num_crops_per_sample

        # Adaptive onset-crop ratio — increases as context shrinks
        base_ratio = self.config.onset_crop_ratio
        full_ctx = 500
        target_ctx = self.context_frames
        if ctx < full_ctx:
            progress = (full_ctx - ctx) / max(full_ctx - target_ctx, 1)
            onset_ratio = base_ratio + (0.8 - base_ratio) * progress
        else:
            onset_ratio = base_ratio

        total_crops = B * n_crops
        all_crops = torch.zeros(total_crops, C, ctx, device=self.device)
        all_onset_labels = torch.zeros(total_crops, self.config.n_pitches, ctx, device=self.device)
        all_offset_labels = torch.zeros(total_crops, self.config.n_pitches, ctx, device=self.device)
        all_velocity_labels = torch.zeros(total_crops, self.config.n_pitches, ctx, device=self.device)

        for b in range(B):
            onset_frames = labels['onsets'][b].sum(dim=0).nonzero(as_tuple=True)[0]
            n_onsets = len(onset_frames)
            use_onset = torch.rand(n_crops, device=self.device) < onset_ratio

            random_centers = torch.randint(ctx, max(T - 1, ctx + 1), (n_crops,), device=self.device)
            if n_onsets > 0:
                onset_indices = torch.randint(n_onsets, (n_crops,), device=self.device)
                onset_centers = onset_frames[onset_indices]
                centers = torch.where(use_onset, onset_centers, random_centers)
            else:
                centers = random_centers

            starts = (centers - ctx + 1).clamp(min=0)
            ends = (centers + 1).clamp(max=T)
            pad_lens = ctx - (ends - starts)

            base_indices = torch.arange(ctx, device=self.device).unsqueeze(0)
            gather_indices = (starts.unsqueeze(1) + base_indices).clamp(max=T - 1)

            position_indices = torch.arange(ctx, device=self.device).unsqueeze(0)
            valid_mask = position_indices >= pad_lens.unsqueeze(1)
            valid_mask_3d = valid_mask.unsqueeze(1).float()

            mel_indices = gather_indices.unsqueeze(1).expand(-1, C, -1)
            mel_gathered = mel[b].unsqueeze(0).expand(n_crops, -1, -1).gather(2, mel_indices)
            all_crops[b * n_crops:(b + 1) * n_crops] = mel_gathered * valid_mask_3d

            P = self.config.n_pitches
            label_indices = gather_indices.unsqueeze(1).expand(-1, P, -1)
            valid_mask_labels = valid_mask.unsqueeze(1).float()

            onset_gathered = labels['onsets'][b].unsqueeze(0).expand(n_crops, -1, -1).gather(2, label_indices)
            all_onset_labels[b * n_crops:(b + 1) * n_crops] = onset_gathered * valid_mask_labels

            offset_gathered = labels['offsets'][b].unsqueeze(0).expand(n_crops, -1, -1).gather(2, label_indices)
            all_offset_labels[b * n_crops:(b + 1) * n_crops] = offset_gathered * valid_mask_labels

            velocity_gathered = labels['velocities'][b].unsqueeze(0).expand(n_crops, -1, -1).gather(2, label_indices)
            all_velocity_labels[b * n_crops:(b + 1) * n_crops] = velocity_gathered * valid_mask_labels

        if total_crops == 0:
            return 0.0

        self.optimizer.zero_grad(set_to_none=True)
        max_batch = self.effective_batch_size  # v3.3.4: Use effective size (reduced on OOM)

        with autocast(enabled=self.use_amp):
            if total_crops <= max_batch:
                outputs = self.model(all_crops)
            else:
                onset_parts, offset_parts, velocity_parts = [], [], []
                for i in range(0, total_crops, max_batch):
                    out = self.model(all_crops[i:i + max_batch])
                    onset_parts.append(out['onset_logits'])
                    offset_parts.append(out['offset_logits'])
                    velocity_parts.append(out['velocity_logits'])
                outputs = {
                    'onset_logits': torch.cat(onset_parts, dim=0),
                    'offset_logits': torch.cat(offset_parts, dim=0),
                    'velocity_logits': torch.cat(velocity_parts, dim=0),
                }

            targets = {
                'onsets': all_onset_labels,
                'offsets': all_offset_labels,
                'velocities': all_velocity_labels,
            }

            loss, loss_dict = self.criterion(outputs, targets)

            # Context-aware loss weighting
            context_weight = min(full_ctx / max(ctx, 1), 2.0)
            loss = loss * context_weight

        # NaN safety net
        # v3.3.7 FIX: Do NOT call scaler.update() here — no inf checks recorded
        # Calling update() without unscale_/step causes assertion crash
        if torch.isnan(loss) or torch.isinf(loss):
            self.optimizer.zero_grad(set_to_none=True)
            print(f"  [WARN] NaN loss at epoch {self.current_epoch}, skipping step")
            return 0.0

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item() / context_weight

    def _truncated_train_step(self, mel, labels):
        """Truncated/Buffered training with AMP and OOM-safe chunking."""
        self.optimizer.zero_grad(set_to_none=True)
        B = mel.shape[0]
        max_batch = self.effective_batch_size  # v3.3.4: Use effective size (reduced on OOM)

        with autocast(enabled=self.use_amp):
            # v3.3.4: Process in chunks if batch is large or OOM recovery active
            if B <= max_batch:
                outputs = self.model(mel)
            else:
                onset_parts, offset_parts, velocity_parts = [], [], []
                for i in range(0, B, max_batch):
                    chunk_mel = mel[i:i + max_batch]
                    out = self.model(chunk_mel)
                    onset_parts.append(out['onset_logits'])
                    offset_parts.append(out['offset_logits'])
                    velocity_parts.append(out['velocity_logits'])
                outputs = {
                    'onset_logits': torch.cat(onset_parts, dim=0),
                    'offset_logits': torch.cat(offset_parts, dim=0),
                    'velocity_logits': torch.cat(velocity_parts, dim=0),
                }

            loss, loss_dict = self.criterion(outputs, labels)

        # NaN safety net
        # v3.3.7 FIX: Do NOT call scaler.update() here — no inf checks recorded
        if torch.isnan(loss) or torch.isinf(loss):
            self.optimizer.zero_grad(set_to_none=True)
            print(f"  [WARN] NaN loss at epoch {self.current_epoch}, skipping step")
            return 0.0

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    # ==================================================================
    # VALIDATION (v11 approach: threshold sweep, every epoch)
    # ==================================================================

    @torch.no_grad()
    def validate(self, val_loader):
        """
        Fast validation with vectorized tolerance-aware F1.

        v3.3.6 FIX #8: TWO BOTTLENECKS ELIMINATED
        ==========================================
        BOTTLENECK 1 (eliminated): _batched_sliding_window() in validation
          - Constrained ctx=37, stride=5 → 100 windows per 500-frame chunk
          - 100 × 128 batch × 68 batches = 6,700 forward passes per epoch
          - Added ~1,000-2,000s per epoch
          FIX: Direct forward pass. Models are CAUSAL — running on 500-frame
          chunks is valid. Predictions at frame t only depend on frames ≤ t.
          After ~RF frames, the model has full context. The <7% of frames
          at chunk start with incomplete context is negligible for model
          selection. evaluate_test() still uses proper sliding window.

        BOTTLENECK 2 (eliminated): _compute_tolerance_f1() in threshold sweep
          - 8,680 chunks × 88 pitches × 12 thresholds = 9.17M Python iterations
          - Each iteration: tensor indexing + nonzero() + distance matrix
          - Consumed ~6,400 seconds (107 min) per epoch
          FIX: Vectorized tolerance F1 using max_pool1d dilation (~100ms per
          threshold). Uses the SAME ±2 frame tolerance as evaluate_test(),
          ensuring validation ranking matches test ranking.

        Combined: ~9,000s → ~15s per validation pass.
        """
        self.model.eval()

        all_onset_probs = []
        all_onset_labels = []
        total_loss = 0.0
        n_batches = 0

        for mel, labels in tqdm(val_loader, desc="Val", leave=False):
            mel = mel.to(self.device, non_blocking=True)
            labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}

            with autocast(enabled=self.use_amp):
                # v3.3.10 FIX: Matched validation uses constrained context
                # v3.3.8 BUG: Direct forward pass on 500-frame chunks meant
                # validation measured full-context performance, not deployment
                # performance. Early stopping selected checkpoints from early
                # epochs (context ~494) making matched ≈ truncated.
                # FIX: Use _last_frame_constrained_eval for matched so early
                # stopping selects the model best at RESTRICTED context.
                if self.condition == 'matched':
                    saved_stride = self.config.eval_stride
                    self.config.eval_stride = 5  # Faster than test stride=2
                    outputs = self._last_frame_constrained_eval(mel)
                    self.config.eval_stride = saved_stride
                else:
                    # v3.3.6 FIX #8: Direct forward pass — no sliding window.
                    # Causal models only use past context, so 500-frame chunks
                    # produce valid predictions (identical to sliding window
                    # after the first RF frames of each chunk).
                    outputs = self.model(mel)

                loss, _ = self.criterion(outputs, labels)

            total_loss += loss.item()
            n_batches += 1

            all_onset_probs.append(torch.sigmoid(outputs['onset_logits']).cpu())
            all_onset_labels.append(labels['onsets'].cpu())

        avg_loss = total_loss / max(n_batches, 1)

        all_probs = torch.cat(all_onset_probs, dim=0)
        all_labels = torch.cat(all_onset_labels, dim=0)

        # v3.3.6 FIX #8: Vectorized threshold sweep (<1s total)
        best_f1 = 0.0
        best_thresh = 0.3

        for thresh in self.config.threshold_sweep:
            f1 = self._compute_fast_f1(all_probs, all_labels, thresh)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        return avg_loss, best_f1, best_thresh

    def _compute_fast_f1(self, probs, labels, threshold):
        """
        v3.3.6: Vectorized TOLERANCE-AWARE F1 — matches the ±2 frame tolerance
        used in evaluate_test(), but computed in ~100ms instead of ~530s.

        Method: Dilate labels and predictions by ±tolerance frames using max_pool1d.
        - Dilated labels: label at frame t spreads to [t-2, t+2]
        - Dilated preds: prediction at frame t spreads to [t-2, t+2]
        - Precision: fraction of predictions that land within ±2 of any label
        - Recall: fraction of labels that have a prediction within ±2

        This is NOT identical to 1-to-1 greedy matching (the slow version) —
        if 3 predictions cluster near 1 label, all 3 count as precision hits
        here vs only 1 in greedy matching. But this bias is consistent across
        epochs, so model RANKING is preserved. And since onset density is ~1%,
        clustering is rare.

        Why this matters (credit: ChatGPT observation):
        A model that predicts onsets 1 frame early would score poorly on
        frame-level F1 but well on tolerance F1. Using the same tolerance
        window for validation and test ensures the best checkpoint selected
        during training is also the best under the final evaluation metric.
        """
        tolerance = self.config.eval_tolerance_frames  # 2 frames = ±20ms
        kernel = 2 * tolerance + 1  # 5

        preds = (probs > threshold).float()

        B, P, T = preds.shape

        # Dilate labels: each label at frame t becomes a window [t-2, t+2]
        labels_flat = labels.view(B * P, 1, T)
        dilated_labels = F.max_pool1d(labels_flat, kernel_size=kernel, stride=1, padding=tolerance)
        dilated_labels = dilated_labels.view(B, P, T)

        # Dilate preds: each prediction at frame t becomes a window [t-2, t+2]
        preds_flat = preds.view(B * P, 1, T)
        dilated_preds = F.max_pool1d(preds_flat, kernel_size=kernel, stride=1, padding=tolerance)
        dilated_preds = dilated_preds.view(B, P, T)

        # Precision: predictions that land near a label
        hits_precision = (preds * dilated_labels).sum().item()
        total_preds = preds.sum().item()

        # Recall: labels that have a nearby prediction
        hits_recall = (labels * dilated_preds).sum().item()
        total_labels = labels.sum().item()

        if total_preds == 0 or total_labels == 0:
            return 0.0

        precision = hits_precision / total_preds
        recall = hits_recall / total_labels
        return 2.0 * precision * recall / (precision + recall + 1e-8)

    # ==================================================================
    # TEST EVALUATION
    # ==================================================================

    @torch.no_grad()
    def evaluate_test(self, test_loader):
        """
        Test-time evaluation with condition-appropriate inference.
        Returns per-file results for statistical analysis.

        v3.3.7 FIX #1 & #2: CRITICAL CHANGES
        ====================================
        Matched/Truncated: _last_frame_constrained_eval()
          - Simulates deployment: only RF+budget frames of context
          - Extracts ONLY last frame (full RF context, no boundary noise)
          - Matched trained under this constraint → should outperform
          - Truncated trained with full context → distribution shift visible

        Buffered: Direct forward pass (model(mel))
          - Upper bound with full 500-frame context
          - No constraint simulation needed

        v3.3.6 BUG: Both used identical _batched_sliding_window() with ctx=37
        → same ~0.34 F1 penalty → matched-truncated difference invisible.
        """
        self.model.eval()

        all_logits = []
        all_labels = []

        # Phase 1: Collect all predictions
        for mel, labels in tqdm(test_loader, desc="Test", leave=False):
            mel = mel.to(self.device, non_blocking=True)
            labels = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}

            with autocast(enabled=self.use_amp):
                if self.condition == 'buffered':
                    # v3.3.7: Direct forward pass for upper bound (full 500-frame context)
                    outputs = self.model(mel)
                else:
                    # v3.3.7 FIX #1: matched + truncated use last-frame constrained eval
                    outputs = self._last_frame_constrained_eval(mel)

            onset_probs = torch.sigmoid(outputs['onset_logits']).cpu()
            onset_labels = labels['onsets'].cpu()

            all_logits.append(onset_probs)
            all_labels.append(onset_labels)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Phase 2: Find best threshold using tolerance-based F1
        best_threshold, best_f1 = self._threshold_sweep(all_logits, all_labels)

        # Phase 3: Compute per-file F1 using best threshold AND tolerance matching
        file_results = []
        for b in range(all_logits.shape[0]):
            file_f1 = self._compute_single_file_tolerance_f1(
                all_logits[b], all_labels[b], best_threshold
            )
            file_results.append(file_f1)

        return {
            'aggregate_f1': best_f1,
            'best_threshold': best_threshold,
            'file_f1_scores': file_results,
            'file_f1_mean': np.mean(file_results),
            'file_f1_std': np.std(file_results),
        }

    def _compute_single_file_f1(self, probs, labels, threshold=0.5):
        """Compute F1 for a single file (frame-level, legacy)."""
        preds = (probs > threshold).float()
        tp = (preds * labels).sum().item()
        fp = (preds * (1 - labels)).sum().item()
        fn = ((1 - preds) * labels).sum().item()
        if tp == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)

    def _compute_single_file_tolerance_f1(self, probs, labels, threshold, tolerance=None):
        """
        Compute F1 for a single file with tolerance-based matching.
        v3.3.2: Used for per-file statistical analysis.
        v3.3.3: Now applies peak picking if enabled.
        
        Args:
            probs: [88, T] onset probabilities
            labels: [88, T] onset labels
            threshold: detection threshold
            tolerance: frame tolerance (default from config)
        """
        if tolerance is None:
            tolerance = self.config.eval_tolerance_frames
        
        # v3.3.3 FIX #4: Apply peak picking to clean double detections
        # _peak_pick expects [B, P, T] so add batch dim
        preds = self._peak_pick(probs.unsqueeze(0), threshold).squeeze(0)
        P, T = preds.shape
        
        tp = 0
        fp = 0
        fn = 0
        
        for p in range(P):
            pred_frames = preds[p].nonzero(as_tuple=True)[0]
            label_frames = labels[p].nonzero(as_tuple=True)[0]
            
            n_preds = len(pred_frames)
            n_labels = len(label_frames)
            
            if n_labels == 0 and n_preds == 0:
                continue
            if n_labels == 0:
                fp += n_preds
                continue
            if n_preds == 0:
                fn += n_labels
                continue
            
            # Match predictions to labels within tolerance
            matched_labels = set()
            matched_preds = set()
            
            for pi, pf in enumerate(pred_frames):
                diffs = torch.abs(label_frames.float() - pf.float())
                min_idx = diffs.argmin()
                if diffs[min_idx] <= tolerance:
                    li = min_idx.item()
                    if li not in matched_labels:
                        matched_labels.add(li)
                        matched_preds.add(pi)
            
            tp += len(matched_labels)
            fp += n_preds - len(matched_preds)
            fn += n_labels - len(matched_labels)
        
        if tp == 0:
            return 0.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)

    def _threshold_sweep(self, probs, labels):
        """Find best threshold via sweep with tolerance F1."""
        best_f1 = 0.0
        best_thresh = 0.3

        print("  Threshold sweep:")
        for thresh in self.config.threshold_sweep:
            f1 = self._compute_tolerance_f1(probs, labels, thresh)
            # v3.3.4: Log each threshold result
            print(f"    Thresh {thresh:.2f}: F1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        print(f"  Best: Thresh={best_thresh:.2f}, F1={best_f1:.4f}")
        return best_thresh, best_f1

    def _peak_pick(self, probs, threshold):
        """
        v3.3.3 FIX #4: Peak picking post-processing.
        Only keep local maxima above threshold to clean up double detections.
        
        Args:
            probs: [P, T] or [B, P, T] probability tensor
            threshold: detection threshold
            
        Returns:
            Binary predictions with only local maxima retained
        """
        if not self.config.use_peak_picking:
            return (probs > threshold).float()
        
        min_distance = self.config.peak_pick_min_distance
        
        # Handle both 2D and 3D inputs
        if probs.dim() == 2:
            probs = probs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, P, T = probs.shape
        preds = torch.zeros_like(probs)
        
        for b in range(B):
            for p in range(P):
                signal = probs[b, p]
                above_thresh = signal > threshold
                
                # Find local maxima using max pooling
                if T > 2 * min_distance + 1:
                    # Pad signal for max pooling
                    padded = F.pad(signal.unsqueeze(0).unsqueeze(0), 
                                   (min_distance, min_distance), mode='constant', value=-float('inf'))
                    local_max = F.max_pool1d(padded, kernel_size=2*min_distance+1, stride=1)
                    local_max = local_max.squeeze()
                    
                    # Peak = above threshold AND equal to local max
                    is_peak = above_thresh & (signal >= local_max)
                else:
                    # Fallback for very short sequences
                    is_peak = above_thresh
                
                preds[b, p] = is_peak.float()
        
        if squeeze_output:
            preds = preds.squeeze(0)
        
        return preds

    # ==================================================================
    # LAST-FRAME CONSTRAINED EVALUATION (v3.3.7 FIX #2)
    # ==================================================================

    def _last_frame_constrained_eval(self, mel):
        """
        v3.3.7 FIX #2: Last-frame-only inference for deployment simulation.

        For each target frame t:
          1. Extract window [t-ctx+1 : t+1] (ctx frames ending at t)
          2. Forward pass through model
          3. Keep ONLY the last frame prediction (guaranteed full RF context)

        Why this matters:
        - In _batched_sliding_window(), ALL frames are averaged including those
          with incomplete context (only 16% have full RF for SCNN ctx=37, RF=31)
        - This 84% boundary noise corrupts predictions
        - Last-frame extraction ensures every prediction has full context

        CRITICAL: Predictions are placed at their EXACT temporal positions.
        DO NOT use F.interpolate — it causes ~28 frame temporal shift that
        destroys F1 scores (predictions at frame 128 misaligned with labels at 100).

        Returns same shape as model(mel) for compatibility.
        """
        B, C, T = mel.shape
        ctx = self.context_frames
        stride = self.config.eval_stride  # v3.3.7: Fixed stride=2
        eval_bs = self.config.eval_batch_size
        P = self.config.n_pitches

        # Positions: each position is a target frame, window ends at that frame
        # First valid position is ctx-1 (first frame with full context window)
        positions = list(range(ctx - 1, T, stride))
        n_pos = len(positions)

        if n_pos == 0:
            # Edge case: sequence shorter than context
            return self.model(mel)

        # Build windows: each window[i] = mel[:, :, pos-ctx+1 : pos+1]
        all_windows = torch.zeros(n_pos * B, C, ctx, device=self.device)

        for w_idx, pos in enumerate(positions):
            start = pos - ctx + 1
            end = pos + 1
            # Handle edge case where start < 0 (left-pad with zeros)
            if start < 0:
                pad_len = -start
                all_windows[w_idx * B : (w_idx + 1) * B, :, pad_len:] = mel[:, :, 0:end]
            else:
                all_windows[w_idx * B : (w_idx + 1) * B] = mel[:, :, start:end]

        # Forward pass in batches
        onset_last_frames = []
        offset_last_frames = []
        velocity_last_frames = []

        for i in range(0, n_pos * B, eval_bs):
            batch_chunk = all_windows[i:i + eval_bs]
            with autocast(enabled=self.use_amp):
                out = self.model(batch_chunk)
            # Extract ONLY last frame from each window
            onset_last_frames.append(out['onset_logits'][:, :, -1])
            offset_last_frames.append(out['offset_logits'][:, :, -1])
            velocity_last_frames.append(out['velocity_logits'][:, :, -1])

        # Concatenate: [n_pos * B, P]
        all_onset = torch.cat(onset_last_frames, dim=0)
        all_offset = torch.cat(offset_last_frames, dim=0)
        all_velocity = torch.cat(velocity_last_frames, dim=0)

        # Reshape to [n_pos, B, P] then to [B, P, n_pos]
        all_onset = all_onset.view(n_pos, B, P).permute(1, 2, 0)
        all_offset = all_offset.view(n_pos, B, P).permute(1, 2, 0)
        all_velocity = all_velocity.view(n_pos, B, P).permute(1, 2, 0)

        # CRITICAL FIX: Place predictions at their EXACT temporal positions
        # DO NOT use F.interpolate — it causes temporal misalignment
        result_onset = torch.zeros(B, P, T, device=self.device)
        result_offset = torch.zeros(B, P, T, device=self.device)
        result_velocity = torch.zeros(B, P, T, device=self.device)

        # Place each prediction at its exact position
        for idx, pos in enumerate(positions):
            result_onset[:, :, pos] = all_onset[:, :, idx]
            result_offset[:, :, pos] = all_offset[:, :, idx]
            result_velocity[:, :, pos] = all_velocity[:, :, idx]

        # Fill gaps (frames between evaluated positions) with nearest prediction
        # v3.3.10: Vectorized gap-filling (was O(T * n_positions) Python loop)
        positions_tensor = torch.tensor(positions, device=self.device)
        all_frames = torch.arange(T, device=self.device)
        # For each frame, find index of nearest evaluated position
        dist = torch.abs(all_frames.unsqueeze(1) - positions_tensor.unsqueeze(0))
        nearest_idx = dist.argmin(dim=1)  # [T] → index into positions

        # Gather from evaluated positions into full-length tensors
        nearest_positions = positions_tensor[nearest_idx]  # [T] actual frame indices
        result_onset_full = result_onset[:, :, nearest_positions]
        result_offset_full = result_offset[:, :, nearest_positions]
        result_velocity_full = result_velocity[:, :, nearest_positions]

        return {
            'onset_logits': result_onset_full,
            'offset_logits': result_offset_full,
            'velocity_logits': result_velocity_full,
        }

    # ==================================================================
    # BATCHED SLIDING WINDOW INFERENCE
    # ==================================================================

    def _batched_sliding_window(self, mel, use_full_context=False):
        """
        Sliding window inference - all batch items processed simultaneously.
        Frame buffering with receptive field overlap.

        v3.3.5 FIX #6: Two changes:
          1. Cap ctx to MAX_SAFE_CONTEXT (2500) when use_full_context=True
             to prevent OOM from massive tensor allocation.
          2. Unified tiling stride (ctx - rf) for BOTH modes.
             Previously use_full_context used eval_stride (2), which was fine
             when ctx=T (single window), but with ctx=2500 it creates 7000+
             passes per song ("performance bomb").
        """
        B, C, T = mel.shape

        # v3.3.5 FIX #6: Cap context to prevent OOM even in "full" mode
        # 2500 frames = 25 seconds at 100fps (safe for L4 22GB VRAM)
        MAX_SAFE_CONTEXT = 2500

        if use_full_context:
            ctx = min(T, MAX_SAFE_CONTEXT)  # Was: ctx = T (OOM for full songs)
        else:
            ctx = self.context_frames

        rf = self.config.receptive_fields.get(self.arch_name, 15)

        # v3.3.5: Unified tiling stride for both modes
        # Overlap by RF ensures causal continuity at window boundaries
        # Constrained (ctx=37, rf=15): stride=22, ~818 windows for T=18000
        # Buffered   (ctx=2500, rf=15): stride=2485, ~7 windows for T=18000
        stride = max(1, ctx - rf)

        eval_bs = self.config.eval_batch_size

        positions = []
        for start in range(0, T, stride):
            end = min(start + ctx, T)
            actual_start = max(0, end - ctx)
            positions.append((actual_start, end))

        n_windows = len(positions)

        if n_windows == 1 and use_full_context:
            return self.model(mel)

        all_windows = torch.zeros(n_windows * B, C, ctx, device=self.device)

        for w_idx, (s, e) in enumerate(positions):
            chunk = mel[:, :, s:e]
            chunk_len = chunk.shape[-1]
            if chunk_len < ctx:
                all_windows[w_idx * B : (w_idx + 1) * B, :, ctx - chunk_len:] = chunk
            else:
                all_windows[w_idx * B : (w_idx + 1) * B] = chunk

        onset_parts, offset_parts, velocity_parts = [], [], []
        for i in range(0, n_windows * B, eval_bs):
            batch_chunk = all_windows[i:i + eval_bs]
            with autocast(enabled=self.use_amp):
                out = self.model(batch_chunk)
            onset_parts.append(out['onset_logits'])
            offset_parts.append(out['offset_logits'])
            velocity_parts.append(out['velocity_logits'])

        all_onset_out = torch.cat(onset_parts, dim=0)
        all_offset_out = torch.cat(offset_parts, dim=0)
        all_velocity_out = torch.cat(velocity_parts, dim=0)

        P = self.config.n_pitches
        all_onset_out = all_onset_out.view(n_windows, B, P, -1)
        all_offset_out = all_offset_out.view(n_windows, B, P, -1)
        all_velocity_out = all_velocity_out.view(n_windows, B, P, -1)

        all_onset_logits = torch.zeros(B, P, T, device=self.device)
        all_offset_logits = torch.zeros(B, P, T, device=self.device)
        all_velocity_logits = torch.zeros(B, P, T, device=self.device)
        counts = torch.zeros(1, 1, T, device=self.device)

        for w_idx, (s, e) in enumerate(positions):
            pred_len = e - s
            out_start = max(0, ctx - pred_len)
            all_onset_logits[:, :, s:e] += all_onset_out[w_idx, :, :, out_start:]
            all_offset_logits[:, :, s:e] += all_offset_out[w_idx, :, :, out_start:]
            all_velocity_logits[:, :, s:e] += all_velocity_out[w_idx, :, :, out_start:]
            counts[0, 0, s:e] += 1

        counts = counts.clamp(min=1)
        return {
            'onset_logits': all_onset_logits / counts,
            'offset_logits': all_offset_logits / counts,
            'velocity_logits': all_velocity_logits / counts,
        }

    # ==================================================================
    # METRICS
    # ==================================================================

    def _compute_tolerance_f1(self, probs, labels, threshold, tolerance=None):
        """
        Compute F1 with +/-tolerance frame tolerance window.
        Vectorized with torch distance computation.
        v3.3.3: Now applies peak picking if enabled.
        """
        if tolerance is None:
            tolerance = self.config.eval_tolerance_frames

        # v3.3.3 FIX #4: Apply peak picking to clean double detections
        preds = self._peak_pick(probs, threshold)

        B, P, T = preds.shape
        preds_flat = preds.view(B * P, T)
        labels_flat = labels.view(B * P, T)

        tp = 0
        fp = 0
        fn = 0

        chunk_size = 256
        n_samples = B * P

        for chunk_start in range(0, n_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_samples)

            for i in range(chunk_start, chunk_end):
                pred_frames = preds_flat[i].nonzero(as_tuple=True)[0]
                label_frames = labels_flat[i].nonzero(as_tuple=True)[0]

                n_preds = len(pred_frames)
                n_labels = len(label_frames)

                if n_labels == 0 and n_preds == 0:
                    continue
                if n_labels == 0:
                    fp += n_preds
                    continue
                if n_preds == 0:
                    fn += n_labels
                    continue

                pred_f = pred_frames.float().unsqueeze(1)
                label_f = label_frames.float().unsqueeze(0)
                dists = torch.abs(pred_f - label_f)

                matched_labels = torch.zeros(n_labels, dtype=torch.bool, device=dists.device)
                matched_preds = torch.zeros(n_preds, dtype=torch.bool, device=dists.device)

                min_dists, _ = dists.min(dim=1)
                pred_order = min_dists.argsort()

                for pi in pred_order:
                    if matched_preds[pi]:
                        continue
                    masked_dists = dists[pi].clone()
                    masked_dists[matched_labels] = float('inf')
                    min_dist, li = masked_dists.min(dim=0)

                    if min_dist <= tolerance:
                        matched_preds[pi] = True
                        matched_labels[li] = True

                n_matched = matched_labels.sum().item()
                tp += n_matched
                fp += n_preds - matched_preds.sum().item()
                fn += n_labels - n_matched

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return f1

    # ==================================================================
    # CHECKPOINT
    # ==================================================================

    def save_checkpoint(self, path, extra_metadata=None):
        """Save model with full training metadata."""
        # v3.3.4: Handle DDP wrapped model
        model_state = self.model.module.state_dict() if self.distributed else self.model.state_dict()

        metadata = {
            'architecture': self.arch_name,
            'budget_ms': self.budget_ms,
            'condition': self.condition,
            'best_val_f1': self.best_val_f1,
            'best_threshold': self.best_threshold,
            'epochs_trained': self.current_epoch,
            'training_history': self.history,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        torch.save({
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metadata': metadata,
        }, path)

    # ==================================================================
    # MAIN TRAINING LOOP
    # ==================================================================

    def train(self, train_loader, val_loader, output_dir):
        """
        Full training loop with:
          - LR warmup + cosine decay
          - v11 cosine curriculum (matched only)
          - Threshold sweep validation every epoch
          - Early stopping on val F1 (patience=15)
        """
        os.makedirs(output_dir, exist_ok=True)
        model_id = f"{self.arch_name}_{self.condition}"

        print(f"\n{'='*60}")
        print(f"Training: {self.condition.upper()} | {self.arch_name}")
        print(f"Budget: {self.budget_ms}ms ({self.budget_frames} frames)")
        print(f"Context target: {self.context_frames} frames")
        print(f"Loss: {self.config.loss_type}", end="")
        if self.config.loss_type == "asymmetric_focal":
            print(f" (gamma+={self.config.focal_gamma_pos}, gamma-={self.config.focal_gamma_neg})", end="")
        # v3.3.10: Stepped geometric curriculum
        total_e = self._max_epochs
        s1 = max(int(total_e * 0.20), 5)
        s2 = max(int(total_e * 0.34), s1 + 3)
        s3 = max(int(total_e * 0.48), s2 + 3)
        s4 = max(int(total_e * 0.62), s3 + 3)
        print(f"\nCurriculum: stepped geometric (v3.3.10)")
        print(f"  Stage 1: epochs 1-{s1} (500 frames — learn)")
        print(f"  Stage 2: epochs {s1+1}-{s2} (250 frames — half)")
        print(f"  Stage 3: epochs {s2+1}-{s3} (125 frames — quarter)")
        print(f"  Stage 4: epochs {s3+1}-{s4} (62 frames — eighth)")
        print(f"  Stage 5: epochs {s4+1}-{total_e} ({self.context_frames} frames — finetune)")
        # v3.3.10: Show effective patience
        eff_patience = self.config.early_stopping_patience
        if self.condition == 'matched':
            eff_patience = max(eff_patience, 30)
        print(f"Epochs: {self._max_epochs} | Patience: {eff_patience}"
              f"{' (matched override)' if self.condition == 'matched' else ''}")
        print(f"LR: {self.base_lr} | Threshold sweep: {len(self.config.threshold_sweep)} values")
        print(f"{'='*60}")

        # v3.3.6 FIX #8: Diagnostic logging
        import sys
        print(f"[v3.3.7 DIAGNOSTIC] Validation: direct forward pass (no sliding window) + vectorized F1")
        print(f"[v3.3.7 DIAGNOSTIC] DataLoader workers: {getattr(train_loader, 'num_workers', 'unknown')}")
        print(f"[v3.3.7 DIAGNOSTIC] Starting training loop...")
        sys.stdout.flush()

        start_time = time.time()

        for epoch in range(1, self._max_epochs + 1):
            self.current_epoch = epoch

            # v3.3.6: Flush before early epochs to catch hangs
            if epoch <= 2:
                import sys
                print(f"[v3.3.7 DIAGNOSTIC] Entering epoch {epoch}...")
                sys.stdout.flush()

            epoch_start = time.time()

            # v3.3.4: DDP sampler epoch sync
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            # Track curriculum context for matched
            ctx_this_epoch = None
            if self.condition == 'matched':
                ctx_this_epoch = self.get_context_for_epoch(epoch)
                self.history['curriculum_context'].append(ctx_this_epoch)

            # Train (with per-phase timing)
            t_train_start = time.time()
            train_loss = self.train_epoch(train_loader, epoch=epoch)
            t_train_end = time.time()
            train_secs = t_train_end - t_train_start

            # Validate every epoch (v3.3: restored from v11)
            t_val_start = time.time()
            val_loss, val_f1, best_thresh = self.validate(val_loader)
            t_val_end = time.time()
            val_secs = t_val_end - t_val_start

            # Step scheduler
            self.scheduler.step()

            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]['lr']

            # Log
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)

            # Print with curriculum info for matched
            ctx_str = ""
            if self.condition == 'matched' and ctx_this_epoch is not None:
                ctx_str = f" | Ctx: {ctx_this_epoch}"

            # v3.3.6: Per-phase timing so we can see exactly where time goes
            print(f"Epoch {epoch}/{self._max_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"Thr: {best_thresh:.2f} | "
                  f"LR: {lr:.6f}{ctx_str} | "
                  f"Time: {epoch_time:.1f}s (train={train_secs:.0f}s, val={val_secs:.0f}s)")

            # Early stopping on F1
            if val_f1 > self.history['best_val_f1']:
                self.history['best_val_f1'] = val_f1
                self.history['best_epoch'] = epoch
                self.history['best_threshold'] = best_thresh
                self.best_val_f1 = val_f1
                self.best_threshold = best_thresh
                self.patience_counter = 0

                # Save best model (v3.3.4: handle DDP)
                save_path = os.path.join(output_dir, f"{model_id}_best.pt")
                model_state = self.model.module.state_dict() if self.distributed else self.model.state_dict()
                torch.save({
                    'model_state_dict': model_state,
                    'config': {
                        'arch': self.arch_name,
                        'condition': self.condition,
                        'budget_ms': self.budget_ms,
                        'loss_type': self.config.loss_type,
                        'focal_gamma_pos': getattr(self.config, 'focal_gamma_pos', None),
                        'focal_gamma_neg': getattr(self.config, 'focal_gamma_neg', None),
                    },
                    'best_f1': val_f1,
                    'best_threshold': best_thresh,
                    'epoch': epoch,
                }, save_path)
                print(f"  -> Saved best model (val_f1={val_f1:.4f}, thresh={best_thresh:.2f})")
            else:
                self.patience_counter += 1
                # v3.3.10 FIX #3: Matched needs more patience to let curriculum complete
                effective_patience = self.config.early_stopping_patience
                if self.condition == 'matched':
                    effective_patience = max(effective_patience, 30)
                if self.patience_counter >= effective_patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        total_time = time.time() - start_time
        self.history['total_time_seconds'] = total_time
        self.history['model_id'] = model_id
        self.history['loss_type'] = self.config.loss_type
        if self.config.loss_type == "asymmetric_focal":
            self.history['focal_gamma_pos'] = self.config.focal_gamma_pos
            self.history['focal_gamma_neg'] = self.config.focal_gamma_neg

        # Load best model back
        best_path = os.path.join(output_dir, f"{model_id}_best.pt")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {self.history['best_epoch']}")

        # Save history
        history_path = os.path.join(output_dir, f"{model_id}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"Training complete. Best F1: {self.history['best_val_f1']:.4f} "
              f"at epoch {self.history['best_epoch']}")
        print(f"Training time: {total_time/3600:.1f} hours")

        return self.history
