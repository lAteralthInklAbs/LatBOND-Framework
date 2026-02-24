#!/usr/bin/env python3
"""
Empirical receptive field verification for LatBOND models.
Uses gradient backprop: feeds random input, backprops from the LAST output frame,
and checks how many input frames have non-zero gradient.
"""
import sys
sys.path.insert(0, '.')

import torch
from models.streaming_cnn import StreamingCNN
from models.causal_transformer import CausalTransformer
from models.tcn import TCN

# Use the actual config defaults from config.py
MODELS = {
    'StreamingCNN': lambda: StreamingCNN(
        n_mels=229, n_pitches=88,
        channels=[128, 256, 256, 256],
        kernel_size=3,
        dilations=[1, 2, 4, 8],
        use_spectral_flux=True,
    ),
    'CausalTransformer': lambda: CausalTransformer(
        n_mels=229, n_pitches=88,
        d_model=192, nhead=6,
        num_layers=2, dim_feedforward=384,
        dropout=0.0,
        use_spectral_flux=True,
    ),
    'TCN': lambda: TCN(
        n_mels=229, n_pitches=88,
        channels=160,
        kernel_size=3,
        num_blocks=4,
        dropout=0.0,
        use_spectral_flux=True,
    ),
}

EXPECTED_RF = {
    'StreamingCNN': 31,
    'CausalTransformer': 500,  # Unbounded causal attention — will equal input length
    'TCN': 61,
}

CURRENT_OVERRIDES = {
    'StreamingCNN':       {0: 93, 20: 96, 40: 98, 60: 100, 80: 102},
    'CausalTransformer':  {0: 96, 20: 99, 40: 101, 60: 103, 80: 105},
    'TCN':                {0: 61, 20: 64, 40: 66, 60: 68, 80: 70},
}

print("=" * 70)
print("LATBOND RECEPTIVE FIELD VERIFICATION")
print("=" * 70)

for name, model_fn in MODELS.items():
    print(f"\n--- {name} ---")

    model = model_fn()
    model.eval()

    T = 500
    x = torch.randn(1, 229, T, requires_grad=True)

    with torch.enable_grad():
        out = model(x)
        onset_logits = out['onset_logits']
        target = onset_logits[0, :, -1].sum()
        target.backward()

    grad = x.grad[0].abs().sum(dim=0)
    nonzero_mask = grad > 1e-10
    rf_empirical = nonzero_mask.sum().item()

    if nonzero_mask.any():
        first_active = nonzero_mask.nonzero()[0].item()
        last_active = nonzero_mask.nonzero()[-1].item()
        rf_from_end = T - first_active
    else:
        rf_from_end = 0

    expected = EXPECTED_RF[name]
    match = "PASS" if rf_empirical == expected else "MISMATCH"

    print(f"  Empirical RF:  {rf_empirical} frames ({rf_empirical * 10}ms)")
    print(f"  Expected RF:   {expected} frames")
    print(f"  Status:        {match}")
    print(f"  Active range:  frames {first_active} to {last_active} (of 0-{T-1})")
    print(f"  RF from end:   {rf_from_end} frames")

    print(f"\n  CORRECT context_override values (RF + budget_frames):")
    for budget_ms in [0, 20, 40, 60, 80]:
        budget_frames = max(1, budget_ms // 10 + 1)
        correct_ctx = rf_empirical + budget_frames
        current_ctx = CURRENT_OVERRIDES[name].get(budget_ms, '?')
        delta = current_ctx - correct_ctx if isinstance(current_ctx, int) else '?'
        status = "OK" if delta == 0 else f"ERROR: off by {delta}"
        print(f"    {budget_ms}ms: RF({rf_empirical}) + budget({budget_frames}) = {correct_ctx}  |  config has: {current_ctx}  |  {status}")

print("\n" + "=" * 70)
print("INTERPRETATION:")
print("=" * 70)
print("""
IF CNN RF = 31 and TCN RF = 61:
  -> CNN context_override is WRONG (has ~93-102, should be ~32-40)
  -> TCN context_override is CORRECT
  -> Fix CNN overrides in TASK 2

IF CNN RF = 93 (or similar):
  -> CNN context_override is CORRECT
  -> The receptive_fields property in config.py is WRONG (says 31)
  -> Keep current overrides, fix the receptive_fields property instead

FOR TRANSFORMER:
  -> RF will be 500 (full sequence) because causal attention is unbounded
  -> The "RF=32" in config is a PRACTICAL design choice, not architectural
  -> This MISMATCH is expected and can be IGNORED
  -> Use RF=32 + budget for matched training context (design decision)
""")
