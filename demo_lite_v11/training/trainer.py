"""
LatBOND Trainer v11
====================
v11 changes:
  - Models return dict with onset/offset/velocity logits
  - Loss is TripleHeadLoss returning (total, loss_dict) tuple
  - Labels are dict with 'onsets', 'offsets', 'velocities'
  - Early stopping uses onset F1 (primary metric)

Retained from v10.1:
  - Cosine curriculum schedule (Solution 1)
  - Asymmetric Focal Loss for onset/offset (Solution 3)
  - Batched parallel crop training (Algorithm 1)
  - Architecture-aware context windows (Algorithm 2)
  - Full-window loss computation (Algorithm 3)
  - Linear LR warmup + cosine decay
  - Batched sliding window inference
  - F1-based early stopping with threshold sweep
"""

import os
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

from training.losses import create_loss


class LatBONDTrainer:
    """
    Trainer for LatBOND onset detection models.
    Handles both matched and truncated training conditions.
    """
    
    def __init__(self, model, config, condition, arch_name, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.condition = condition  # 'matched' or 'truncated'
        self.arch_name = arch_name
        self.device = device
        
        # Context window for matched training
        if hasattr(config, 'context_frames_per_arch') and arch_name in config.context_frames_per_arch:
            self.context_frames = config.context_frames_per_arch[arch_name]
        else:
            self.context_frames = config.budget_frames * config.context_multiplier
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # LR scheduler: Linear warmup + Cosine decay
        warmup_epochs = config.warmup_epochs
        total_epochs = config.epochs
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return 0.1 + 0.9 * ((epoch + 1) / warmup_epochs)
            else:
                progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )
        
        # ============================================================
        # SOLUTION 3: Use configurable loss function
        # ============================================================
        self.criterion = create_loss(config, device)
        
        # Tracking
        self.history = {
            'train_loss': [], 'val_loss': [],
            'val_f1': [], 'best_val_f1': 0.0,
            'best_epoch': 0, 'best_threshold': 0.3,
            'curriculum_context': [],  # v10: track context per epoch
        }
        self.patience_counter = 0
    
    # ==================================================================
    # SOLUTION 1: COSINE CURRICULUM SCHEDULE
    # ==================================================================

    def get_context_for_epoch(self, epoch):
        """
        Curriculum: context shrinks from full sequence to target across training.

        v10 COSINE schedule (Solution 1):
          Uses cosine annealing — stays near full context longer, then
          gradually tightens. Eliminates the sharp "shock" at epoch 6
          that destabilized matched training in v9.

          Cosine: context = min + (max - min) * 0.5 * (1 + cos(π * progress))
            Epoch  1: 100% context (warmup)
            Epoch  5: ~93% context  (slow start)
            Epoch 10: ~65% context  (accelerating)
            Epoch 15: ~35% context  (decelerating)
            Epoch 20: ~7% context   (slow finish)
            Epoch 25: target context

        v9 LINEAR schedule (fallback):
            Epoch  1: 100% (warmup)
            Epoch  6: ~50%  ← SHOCK POINT
            Epoch 12: target

        Both warmup phases use full context (matches LR warmup period).
        """
        warmup = self.config.warmup_epochs
        full_ctx = 500
        target_ctx = self.context_frames

        if epoch <= warmup:
            return full_ctx

        curriculum_type = getattr(self.config, 'curriculum_type', 'cosine')

        if curriculum_type == "cosine":
            # Cosine annealing: smooth transition, no shock
            progress = (epoch - warmup) / max(self.config.epochs - warmup, 1)
            # cos goes from 1 (progress=0) to -1 (progress=1)
            # ratio goes from 1.0 (full context) to 0.0 (target context)
            ratio = 0.5 * (1 + math.cos(math.pi * progress))
            return int(target_ctx + (full_ctx - target_ctx) * ratio)
        else:
            # v9 linear fallback
            transition_end = self.config.epochs // 2
            if epoch <= transition_end:
                progress = (epoch - warmup) / (transition_end - warmup)
                return int(full_ctx - progress * (full_ctx - target_ctx))
            else:
                return target_ctx

    def train_epoch(self, train_loader, epoch=None):
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (mel, labels) in enumerate(tqdm(train_loader, desc="Train", leave=False)):
            mel = mel.to(self.device)
            # v11: labels is now a dict — move each tensor to device
            labels = {k: v.to(self.device) for k, v in labels.items()}

            if self.condition == 'matched':
                loss = self._matched_train_step(mel, labels, epoch)
            else:
                loss = self._truncated_train_step(mel, labels)

            total_loss += loss
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss
    
    def _matched_train_step(self, mel, labels, epoch=None):
        """
        Matched training v11: Batched parallel crops with cosine curriculum
        context and TripleHeadLoss.
        """
        B, C, T = mel.shape

        # Curriculum: dynamic context per epoch (now cosine)
        if epoch is not None:
            ctx = self.get_context_for_epoch(epoch)
        else:
            ctx = self.context_frames

        n_crops = self.config.num_crops_per_sample
        onset_ratio = self.config.onset_crop_ratio

        # === Phase 1: Collect all crop positions ===
        crop_mels = []
        crop_onsets_list = []
        crop_offsets_list = []
        crop_velocities_list = []

        for b in range(B):
            # v11: onset-centered sampling uses onsets only
            onset_frames = labels['onsets'][b].sum(dim=0).nonzero(as_tuple=True)[0]

            for _ in range(n_crops):
                if len(onset_frames) > 0 and torch.rand(1).item() < onset_ratio:
                    center = onset_frames[torch.randint(len(onset_frames), (1,))].item()
                else:
                    center = torch.randint(ctx, max(T - 1, ctx + 1), (1,)).item()

                start = max(0, center - ctx + 1)
                end = min(T, center + 1)

                crop_mel = mel[b, :, start:end]
                crop_onsets = labels['onsets'][b, :, start:end]
                crop_offsets = labels['offsets'][b, :, start:end]
                crop_velocities = labels['velocities'][b, :, start:end]

                # Left-pad to ctx if shorter (maintains causality)
                if crop_mel.shape[-1] < ctx:
                    pad_len = ctx - crop_mel.shape[-1]
                    crop_mel = F.pad(crop_mel, (pad_len, 0))
                    crop_onsets = F.pad(crop_onsets, (pad_len, 0))
                    crop_offsets = F.pad(crop_offsets, (pad_len, 0))
                    crop_velocities = F.pad(crop_velocities, (pad_len, 0))

                crop_mels.append(crop_mel)
                crop_onsets_list.append(crop_onsets)
                crop_offsets_list.append(crop_offsets)
                crop_velocities_list.append(crop_velocities)

        if len(crop_mels) == 0:
            return 0.0

        # === Phase 2: Stack into single tensor ===
        all_crops = torch.stack(crop_mels, dim=0)
        all_onset_labels = torch.stack(crop_onsets_list, dim=0)
        all_offset_labels = torch.stack(crop_offsets_list, dim=0)
        all_velocity_labels = torch.stack(crop_velocities_list, dim=0)

        # === Phase 3: Single batched forward pass ===
        self.optimizer.zero_grad()

        max_batch = self.config.eval_batch_size
        total_crops = all_crops.shape[0]

        if total_crops <= max_batch:
            outputs = self.model(all_crops)  # v11: dict
        else:
            # Batched forward for large crop counts
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

        # v11: Build targets dict
        targets = {
            'onsets': all_onset_labels,
            'offsets': all_offset_labels,
            'velocities': all_velocity_labels,
        }

        # Full-window loss: all frames contribute
        loss, loss_dict = self.criterion(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()
    
    def _truncated_train_step(self, mel, labels):
        """
        Truncated training: standard full-context training.
        Model sees entire sequence during training (no crops).
        At inference, context will be truncated → distribution shift.
        """
        self.optimizer.zero_grad()

        outputs = self.model(mel)  # v11: dict
        loss, loss_dict = self.criterion(outputs, labels)  # v11: labels is dict

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()
    
    # ==================================================================
    # VALIDATION
    # ==================================================================
    
    @torch.no_grad()
    def validate(self, val_loader):
        """
        Validate with tolerance-based F1 and optimal threshold sweep.
        Uses standard sequential data (no onset resampling).
        For matched: batched sliding window evaluation.
        For truncated: full-sequence forward pass.

        v11: Uses onset F1 as primary metric for early stopping.
        """
        self.model.eval()

        all_onset_probs = []
        all_onset_labels = []
        total_loss = 0.0
        n_batches = 0

        for mel, labels in tqdm(val_loader, desc="Val", leave=False):
            mel = mel.to(self.device)
            # v11: labels is dict
            labels = {k: v.to(self.device) for k, v in labels.items()}

            if self.condition == 'matched':
                outputs = self._batched_sliding_window(mel)
            else:
                outputs = self.model(mel)

            loss, _ = self.criterion(outputs, labels)
            total_loss += loss.item()
            n_batches += 1

            # v11: For F1 threshold sweep, use ONSET probabilities only (primary metric)
            all_onset_probs.append(torch.sigmoid(outputs['onset_logits']).cpu())
            all_onset_labels.append(labels['onsets'].cpu())

        avg_loss = total_loss / max(n_batches, 1)

        # Concatenate
        all_probs = torch.cat(all_onset_probs, dim=0)
        all_labels = torch.cat(all_onset_labels, dim=0)

        # Threshold sweep for optimal F1
        best_f1 = 0.0
        best_thresh = 0.3

        for thresh in self.config.threshold_sweep:
            f1 = self._compute_tolerance_f1(all_probs, all_labels, thresh)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        return avg_loss, best_f1, best_thresh
    
    # ==================================================================
    # BATCHED SLIDING WINDOW INFERENCE
    # ==================================================================
    
    def _batched_sliding_window(self, mel):
        """
        Batched sliding window inference for matched evaluation.
        Pre-extracts all windows → batched forward → scatter-average.

        v11: Returns dict with all three head logits.
        """
        B, C, T = mel.shape
        ctx = self.context_frames
        stride = self.config.eval_stride
        eval_bs = self.config.eval_batch_size

        positions = []
        for start in range(0, T, stride):
            end = min(start + ctx, T)
            actual_start = max(0, end - ctx)
            positions.append((actual_start, end))

        n_windows = len(positions)
        all_onset_logits = torch.zeros(B, self.config.n_pitches, T, device=self.device)
        all_offset_logits = torch.zeros(B, self.config.n_pitches, T, device=self.device)
        all_velocity_logits = torch.zeros(B, self.config.n_pitches, T, device=self.device)
        counts = torch.zeros(1, 1, T, device=self.device)

        for b in range(B):
            windows = torch.zeros(n_windows, C, ctx, device=self.device)

            for w_idx, (s, e) in enumerate(positions):
                chunk = mel[b, :, s:e]
                chunk_len = chunk.shape[-1]
                if chunk_len < ctx:
                    windows[w_idx, :, ctx - chunk_len:] = chunk
                else:
                    windows[w_idx] = chunk

            # Batched forward
            onset_parts, offset_parts, velocity_parts = [], [], []
            for i in range(0, n_windows, eval_bs):
                batch_chunk = windows[i:i + eval_bs]
                out = self.model(batch_chunk)  # v11: dict
                onset_parts.append(out['onset_logits'])
                offset_parts.append(out['offset_logits'])
                velocity_parts.append(out['velocity_logits'])

            window_onset = torch.cat(onset_parts, dim=0)
            window_offset = torch.cat(offset_parts, dim=0)
            window_velocity = torch.cat(velocity_parts, dim=0)

            for w_idx, (s, e) in enumerate(positions):
                pred_len = e - s
                out_start = max(0, ctx - pred_len)
                all_onset_logits[b, :, s:e] += window_onset[w_idx, :, out_start:]
                all_offset_logits[b, :, s:e] += window_offset[w_idx, :, out_start:]
                all_velocity_logits[b, :, s:e] += window_velocity[w_idx, :, out_start:]
                if b == 0:
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
        Compute F1 with ±tolerance frame tolerance window.
        This is the authoritative metric for onset detection research.
        """
        if tolerance is None:
            tolerance = self.config.eval_tolerance_frames
        
        preds = (probs > threshold).float()
        
        B, P, T = preds.shape
        preds_flat = preds.view(B * P, T)
        labels_flat = labels.view(B * P, T)
        
        tp = 0
        fp = 0
        fn = 0
        
        for i in range(B * P):
            pred_frames = preds_flat[i].nonzero(as_tuple=True)[0]
            label_frames = labels_flat[i].nonzero(as_tuple=True)[0]
            
            if len(label_frames) == 0 and len(pred_frames) == 0:
                continue
            if len(label_frames) == 0:
                fp += len(pred_frames)
                continue
            if len(pred_frames) == 0:
                fn += len(label_frames)
                continue
            
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
            fp += len(pred_frames) - len(matched_preds)
            fn += len(label_frames) - len(matched_labels)
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        return f1
    
    # ==================================================================
    # MAIN TRAINING LOOP
    # ==================================================================
    
    def train(self, train_loader, val_loader, output_dir):
        """
        Full training loop with:
          - LR warmup + cosine decay
          - Cosine curriculum schedule (Solution 1)
          - Asymmetric Focal Loss (Solution 3)
          - Early stopping on val F1 (patience=15)
        """
        os.makedirs(output_dir, exist_ok=True)
        model_id = f"{self.arch_name}_{self.condition}"
        
        print(f"\n{'='*60}")
        print(f"Training: {self.condition.upper()} | {self.arch_name}")
        print(f"Budget: {self.config.budget_ms}ms ({self.config.budget_frames} frames)")
        print(f"Context target: {self.context_frames} frames")
        print(f"Loss: {self.config.loss_type}", end="")
        if self.config.loss_type == "asymmetric_focal":
            print(f" (γ+={self.config.focal_gamma_pos}, γ-={self.config.focal_gamma_neg})", end="")
        print(f"\nCurriculum: {getattr(self.config, 'curriculum_type', 'cosine')}")
        print(f"Epochs: {self.config.epochs} | Patience: {self.config.early_stopping_patience}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            
            # Track curriculum context for matched
            if self.condition == 'matched':
                ctx_this_epoch = self.get_context_for_epoch(epoch)
                self.history['curriculum_context'].append(ctx_this_epoch)
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch=epoch)
            
            # Validate
            val_loss, val_f1, best_thresh = self.validate(val_loader)
            
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
            if self.condition == 'matched':
                ctx_str = f" | Ctx: {ctx_this_epoch}"
            
            print(f"Epoch {epoch}/{self.config.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"LR: {lr:.6f}{ctx_str} | "
                  f"Time: {epoch_time:.1f}s")

            # Early stopping on F1
            if val_f1 > self.history['best_val_f1']:
                self.history['best_val_f1'] = val_f1
                self.history['best_epoch'] = epoch
                self.history['best_threshold'] = best_thresh
                self.patience_counter = 0
                
                # Save best model
                save_path = os.path.join(output_dir, f"{model_id}_best.pt")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': {
                        'arch': self.arch_name,
                        'condition': self.condition,
                        'budget_ms': self.config.budget_ms,
                        'loss_type': self.config.loss_type,
                        'focal_gamma_pos': getattr(self.config, 'focal_gamma_pos', None),
                        'focal_gamma_neg': getattr(self.config, 'focal_gamma_neg', None),
                    },
                    'best_f1': val_f1,
                    'best_threshold': best_thresh,
                    'epoch': epoch,
                }, save_path)
                print(f"  -> Saved best model (val_f1={val_f1:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        
        total_time = time.time() - start_time
        self.history['total_time_seconds'] = total_time
        self.history['model_id'] = model_id
        self.history['loss_type'] = self.config.loss_type
        if self.config.loss_type == "asymmetric_focal":
            self.history['focal_gamma_pos'] = self.config.focal_gamma_pos
            self.history['focal_gamma_neg'] = self.config.focal_gamma_neg
        
        # Save history
        history_path = os.path.join(output_dir, f"{model_id}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training complete. Best F1: {self.history['best_val_f1']:.4f} "
              f"at epoch {self.history['best_epoch']}")
        
        return self.history
