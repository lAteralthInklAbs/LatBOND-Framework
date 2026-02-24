# LatBOND Version History

Detailed changelog of the LatBOND codebase from Demo Lite through the full 42-model study.

---

## Demo Lite (Phase 1) — Local Development

### v1–v10: Iterative Prototyping

Eleven experimental iterations on a reduced dataset (5% MAESTRO, single T4 GPU) exploring:
- Architecture design (Streaming CNN, Causal Transformer, TCN)
- Loss function tuning (BCE → Focal Loss → AsymmetricFocalLoss)
- Training methodology (curriculum schedules, onset-biased crop sampling)
- Context window discovery (architecture-aware RF + budget)

### v11: Phase 1 Final Release ⭐

**Location:** `demo_lite_v11/`

The definitive Phase 1 codebase. All thesis Phase 1 results (Table 6.1) derive from this version.

**Features:**
- Triple-head output: onset + offset + velocity prediction
- TripleHeadLoss = AsymmetricFocalLoss(γ⁺=1.0, γ⁻=3.0) + OffsetLoss + VelocityMSE
- Cosine curriculum schedule (500 → target frames over training)
- Architecture-aware context windows (CNN/Transformer=32, TCN=62)
- Batched parallel crop training with onset-biased sampling
- Spectral flux input channel (230 total input channels)
- Tolerance-based F1 validation (±2 frames)
- Negative bias initialization for sparse event detection

**Configuration:** 25 epochs, patience=15, 5% MAESTRO, single GPU

**Results at 20ms budget:**
| Architecture | Matched F1 | Truncated F1 | Δ (pp) | *p*-value |
|---|:---:|:---:|:---:|:---:|
| Streaming CNN | 0.3349 | 0.3261 | +0.88 | < 0.001 |
| Causal Transformer | 0.2844 | 0.2781 | +0.63 | 0.00047 |
| TCN | 0.4271 | 0.4062 | +2.09 | < 0.001 |

---

## Full Study (Phase 2) — GCP Vertex AI Deployment

### v3.3 (v3.3.0–v3.3.2): Initial GCP Port

Ported v11 features to Docker/Vertex AI. Fixed critical issues:
- Removed two-phase training (caused early stop before phase 2)
- Restored v11 cosine curriculum
- Restored threshold sweep in validation (was hardcoded at 0.3)
- Restored per-epoch validation
- Patience increased to 15

### v3.3.3: Tolerance Fixes

- Tolerance-based matching in both validation and test (±2 frames)
- Per-budget learning rate scaling
- Onset peak picking post-processing
- Per-threshold F1 logging (all 12 thresholds)

### v3.3.4: Receptive Field Verification

- Empirical RF verification via gradient backpropagation (CNN=32, TCN=62, Transformer=32 design)
- **Curriculum removed** — matched models trained at fixed target context from epoch 1
- Context override contamination fixed
- *Note: Curriculum removal was later identified as counterproductive (v3.3.7)*

### v3.3.5: Validation Logic Fix

- **Critical:** Truncated validation corrected to use constrained context (`use_full_context=False`)
- Previously truncated used full context during validation, eliminating distribution shift and violating thesis definition
- OOM-safe full-context inference for buffered condition (caps at 2,500 frames)

### v3.3.6: Performance Optimization

- Validation time: **~9,000s → ~6s per epoch**
- Replaced sliding window validation (6,700+ forward passes) with direct forward pass (68 passes)
- Replaced Python-loop tolerance F1 with vectorized `max_pool1d` dilation
- Same ±2 frame tolerance, 4 tensor operations in ~100ms instead of ~6,400s

### v3.3.7: Curriculum Ablation ⭐

**Location:** `ablation_v337/`

9-model pilot (3 architectures × 3 conditions @ 40ms budget) — **without curriculum**.

**Fixes applied:**
- Condition-differentiated test evaluation (last-frame constrained inference)
- Threshold sweep extended to 0.80 (previously capped at 0.50)
- AMP disabled for Causal Transformer (GradScaler crash at epoch 17)
- Unified training parameters (50 epochs, 3 warmup epochs)

**Critical finding:** Matched training **underperformed** truncated (avg F1 0.459 vs 0.477). Root cause: without curriculum, matched models trained at fixed minimal context from epoch 1, suffering training shock. This proved the curriculum is not optional — it is **necessary**.

### v3.3.8–v3.3.9: Curriculum Restoration

- v3.3.8: Restored cosine curriculum from v11
- v3.3.9: Identified three critical issues:
  1. **Validation-curriculum mismatch:** Full-context validation caused early stopping to select early-epoch checkpoints
  2. **Insufficient fine-tuning at target:** Cosine decay spent too little time at target context
  3. **Premature early stopping:** Standard patience insufficient for curriculum transitions
- Switched from cosine to **stepped geometric** curriculum
- Matched validation changed to deployment-aligned constrained evaluation

### v3.3.10: Phase 2 Final Release ⭐

**Location:** `full_study_v3310/`

The definitive Phase 2 codebase. All thesis Phase 2 results (Table 6.4, 6.5, 6.6) derive from this version.

**Three critical fixes from v3.3.7:**
1. **Stepped geometric curriculum:** 500→250→125→62→target (discrete halving stages)
2. **Deployment-aligned validation:** Matched models validated under constrained conditions
3. **Extended training:** Matched 100 epochs / patience=30 (vs 50/15 for truncated/buffered)

**Deployment:**
- Matched conditions: A100 GPUs (us-central1)
- Truncated/Buffered: L4 GPUs (northamerica-northeast2)
- Docker container via Vertex AI
- 30% MAESTRO dataset

**Results (42 models, grand average):**
| Condition | Test F1 |
|---|:---:|
| Matched | **0.5295** |
| Truncated | 0.5208 |
| Buffered | 0.8341 |
| **Δ (Matched − Truncated)** | **+0.87 pp** |

---

## Version Relationship Diagram

```
Demo Lite v1 → v2 → ... → v10 → v11 (Phase 1 results)
                                   ↓
                               v3.3.0 (GCP port)
                                   ↓
                           v3.3.3 → v3.3.4 (RF verification + curriculum removed)
                                   ↓
                           v3.3.5 → v3.3.6 (validation fixes + performance)
                                   ↓
                               v3.3.7 (pilot: matched < truncated ← NO CURRICULUM)
                                   ↓
                           v3.3.8 → v3.3.9 (curriculum restored: cosine → stepped)
                                   ↓
                               v3.3.10 (Phase 2 results) ⭐
```
