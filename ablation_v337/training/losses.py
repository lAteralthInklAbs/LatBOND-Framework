"""
LatBOND v11 - Loss Functions
=============================
Triple-head loss for onset + offset + velocity prediction.

v11 adds TripleHeadLoss combining:
  - AsymmetricFocalLoss for onsets (primary task)
  - AsymmetricFocalLoss for offsets (secondary task)
  - MSE for velocity, masked to onset regions only

Tuning guide (Conservative preset from v10.1 ablation):
  Config          gamma_pos   gamma_neg   Expected behavior
  --------------  ---------   ---------   ---------------------
  Conservative    1.0         3.0         Best balance (default)
  Balanced        1.0         4.0         Higher recall, +95K FP risk
  Aggressive      0.5         5.0         Max recall, +240K FP explosion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricFocalLoss(nn.Module):
    """
    Focal loss with separate gamma for positives and negatives.

    Standard focal loss: FL(p_t) = -(1-p_t)^gamma * log(p_t)

    Asymmetric version:
      - Positive samples: gamma = gamma_pos (small -> don't suppress easy positives)
      - Negative samples: gamma = gamma_neg (large -> suppress easy negatives hard)

    Combined with pos_weight to handle class imbalance (onsets are ~1% of frames).

    v2 #8: Label smoothing for better threshold stability.

    Args:
        gamma_pos: Focal gamma for positive (onset) samples. Default 1.0.
        gamma_neg: Focal gamma for negative (non-onset) samples. Default 4.0.
        pos_weight: Weight multiplier for positive samples. Default 10.0.
        label_smoothing: Epsilon for label smoothing. Default 0.0 (disabled).
    """

    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, pos_weight=10.0, label_smoothing=0.0):
        super().__init__()
        # register_buffer ensures these tensors auto-move to GPU
        self.register_buffer('gamma_pos', torch.tensor(gamma_pos))
        self.register_buffer('gamma_neg', torch.tensor(gamma_neg))
        self.register_buffer('pos_weight', torch.tensor(pos_weight))
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model output (before sigmoid), shape [B, 88, T]
            targets: Binary onset labels, shape [B, 88, T]

        Returns:
            Scalar loss value
        """
        # v2 #8: Apply label smoothing for binary labels
        # targets=1 -> 1-eps/2, targets=0 -> eps/2 (keeps midpoint at 0.5)
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5

        p = torch.sigmoid(logits)

        # Standard BCE per-element (no reduction)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # p_t = probability of being correct class
        p_t = p * targets + (1 - p) * (1 - targets)

        # Per-element gamma: gamma_pos for onsets, gamma_neg for non-onsets
        # Use original binary targets for gamma selection (before smoothing)
        binary_targets = (targets > 0.5).float()
        gamma = self.gamma_pos * binary_targets + self.gamma_neg * (1 - binary_targets)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** gamma

        # Class weight: pos_weight for positives, 1.0 for negatives
        class_weight = self.pos_weight * binary_targets + (1 - binary_targets)

        # Combined loss
        loss = focal_weight * class_weight * ce

        return loss.mean()


class TripleHeadLoss(nn.Module):
    """
    Combined loss for onset + offset + velocity prediction (v11).

    - Onset: AsymmetricFocalLoss (primary task)
    - Offset: AsymmetricFocalLoss (same sparsity characteristics)
    - Velocity: MSE loss, masked to onset regions only
      (velocity is only meaningful where notes actually start)

    Total = w_onset * L_onset + w_offset * L_offset + w_velocity * L_velocity

    Returns:
        (total_loss, loss_dict) tuple for backward and logging
    """

    def __init__(self, config, device='cuda'):
        super().__init__()

        # v2 #8: Get label smoothing from config
        label_smoothing = getattr(config, 'label_smoothing', 0.0)

        # Onset loss (existing asymmetric focal)
        self.onset_loss = AsymmetricFocalLoss(
            gamma_pos=config.focal_gamma_pos,
            gamma_neg=config.focal_gamma_neg,
            pos_weight=config.pos_weight,
            label_smoothing=label_smoothing,
        )

        # Offset loss (same class, same gamma - offsets have similar sparsity)
        self.offset_loss = AsymmetricFocalLoss(
            gamma_pos=config.focal_gamma_pos,
            gamma_neg=config.focal_gamma_neg,
            pos_weight=config.pos_weight,
            label_smoothing=label_smoothing,
        )

        # Velocity loss: MSE, applied only at onset frames
        self.velocity_loss_fn = nn.MSELoss(reduction='none')

        # Loss weights
        self.onset_weight = getattr(config, 'onset_loss_weight', 1.0)
        self.offset_weight = getattr(config, 'offset_loss_weight', 0.5)
        self.velocity_weight = getattr(config, 'velocity_loss_weight', 0.1)

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'onset_logits', 'offset_logits', 'velocity_logits' [B, 88, T]
            targets: dict with 'onsets', 'offsets', 'velocities' [B, 88, T]
        Returns:
            total_loss: scalar tensor (for backward)
            loss_dict: dict with individual loss components (for logging)
        """
        # Onset loss
        l_onset = self.onset_loss(outputs['onset_logits'], targets['onsets'])

        # Offset loss
        l_offset = self.offset_loss(outputs['offset_logits'], targets['offsets'])

        # Velocity loss: MSE masked to onset regions
        vel_pred = torch.sigmoid(outputs['velocity_logits'])
        vel_target = targets['velocities']

        # Mask: only compute velocity loss where velocity labels > 0
        vel_mask = (vel_target > 0).float()
        vel_mse = self.velocity_loss_fn(vel_pred, vel_target)

        if vel_mask.sum() > 0:
            l_velocity = (vel_mse * vel_mask).sum() / vel_mask.sum()
        else:
            l_velocity = torch.tensor(0.0, device=vel_pred.device)

        # Weighted combination
        total = (self.onset_weight * l_onset +
                 self.offset_weight * l_offset +
                 self.velocity_weight * l_velocity)

        loss_dict = {
            'onset_loss': l_onset.item(),
            'offset_loss': l_offset.item(),
            'velocity_loss': l_velocity.item() if isinstance(l_velocity, torch.Tensor) else l_velocity,
            'total_loss': total.item(),
        }

        return total, loss_dict


def create_loss(config, device='cuda'):
    """
    Factory function to create the loss function.

    v11: Returns TripleHeadLoss by default (onset + offset + velocity).

    Args:
        config: Config dataclass with loss_type and related params
        device: Target device

    Returns:
        nn.Module loss function (TripleHeadLoss in v11)
    """
    return TripleHeadLoss(config, device).to(device)
