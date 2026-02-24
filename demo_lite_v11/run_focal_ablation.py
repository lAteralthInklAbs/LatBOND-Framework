#!/usr/bin/env python3
"""
LatBOND v10 — Focal Loss Ablation Test
========================================
Runs on the VM with real MAESTRO data to validate gamma presets
before committing to the full 6-7 hour training run.

Tests 4 configurations on CNN matched+truncated (3 epochs each):
  1. BCE baseline (reference)
  2. Conservative (γ+=1.0, γ-=3.0)
  3. Balanced     (γ+=1.0, γ-=4.0) ← current v10 default
  4. Aggressive   (γ+=0.5, γ-=5.0)

Generates a report with:
  - Per-preset F1, precision, recall at 20ms tolerance
  - Safety verdicts (overprediction, underprediction, loss stability)
  - Recommended gamma for the full v10 run
  - Exact config.py changes if the default needs updating

Usage:
    python run_focal_ablation.py                   # Full run
    python run_focal_ablation.py --skip_download   # Use existing MAESTRO

Runtime: ~20-30 minutes on T4 GPU (4 presets × ~5-7 min each)
"""

import argparse
import json
import os
import sys
import time
import copy
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from models import create_model, count_parameters
from data.download_maestro import download_maestro, create_subset
from data.maestro_dataset import create_dataloaders
from training.trainer import LatBONDTrainer
from evaluation.metrics import evaluate_model


# ============================================================
# PRESETS
# ============================================================

PRESETS = [
    {
        "name": "bce_baseline",
        "label": "BCE Baseline",
        "loss_type": "bce",
        "gamma_pos": None,
        "gamma_neg": None,
    },
    {
        "name": "conservative",
        "label": "Conservative (γ+=1.0, γ-=3.0)",
        "loss_type": "asymmetric_focal",
        "gamma_pos": 1.0,
        "gamma_neg": 3.0,
    },
    {
        "name": "balanced",
        "label": "Balanced (γ+=1.0, γ-=4.0)",
        "loss_type": "asymmetric_focal",
        "gamma_pos": 1.0,
        "gamma_neg": 4.0,
    },
    {
        "name": "aggressive",
        "label": "Aggressive (γ+=0.5, γ-=5.0)",
        "loss_type": "asymmetric_focal",
        "gamma_pos": 0.5,
        "gamma_neg": 5.0,
    },
]

ABLATION_EPOCHS = 3
ABLATION_PATIENCE = 10
ARCH = "streaming_cnn"


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LatBOND v10 Focal Ablation")
    parser.add_argument("--skip_download", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ablation_dir = os.path.join(CONFIG.output_dir, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    # === Data ===
    if not args.skip_download:
        print("\n[Downloading MAESTRO dataset...]")
        maestro_dir = download_maestro(CONFIG.data_dir)
        subset = create_subset(maestro_dir, CONFIG.subset_fraction)
    else:
        print("\n[Loading existing dataset...]")
        maestro_dir = os.path.join(CONFIG.data_dir, "maestro-v3.0.0")
        manifest_path = os.path.join(maestro_dir, "subset_manifest.json")
        with open(manifest_path) as f:
            subset = json.load(f)

    loaders = create_dataloaders(subset, CONFIG)
    print(f"  Train: {len(loaders['train'])} batches")
    print(f"  Val:   {len(loaders['validation'])} batches")
    print(f"  Test:  {len(loaders['test'])} batches")

    # === Run each preset ===
    all_results = []
    total_start = time.time()

    for preset in PRESETS:
        print(f"\n{'#' * 60}")
        print(f"# ABLATION: {preset['label']}")
        print(f"{'#' * 60}")

        # Configure
        CONFIG.loss_type = preset["loss_type"]
        if preset["gamma_pos"] is not None:
            CONFIG.focal_gamma_pos = preset["gamma_pos"]
        if preset["gamma_neg"] is not None:
            CONFIG.focal_gamma_neg = preset["gamma_neg"]
        CONFIG.epochs = ABLATION_EPOCHS
        CONFIG.early_stopping_patience = ABLATION_PATIENCE

        preset_results = {"preset": preset, "conditions": {}}
        preset_start = time.time()

        for condition in ["matched", "truncated"]:
            print(f"\n--- {condition.upper()} ---")

            # Fresh model, same seed for fair comparison
            torch.manual_seed(42)
            model = create_model(ARCH, CONFIG)
            trainer = LatBONDTrainer(model, CONFIG, condition, ARCH, device)

            # Train
            history = trainer.train(
                loaders["train"],
                loaders["validation"],
                ablation_dir,
            )

            # Evaluate on test set
            model_path = os.path.join(ablation_dir, f"{ARCH}_{condition}_best.pt")
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])

            model.eval()
            eval_results = evaluate_model(
                model, loaders["test"], CONFIG, condition, ARCH, device
            )

            # Extract 20ms tolerance metrics
            tol20 = eval_results.get("tol_20ms", {})

            preset_results["conditions"][condition] = {
                "f1": tol20.get("f1", 0),
                "precision": tol20.get("precision", 0),
                "recall": tol20.get("recall", 0),
                "threshold": tol20.get("threshold", 0),
                "tp": tol20.get("tp", 0),
                "fp": tol20.get("fp", 0),
                "fn": tol20.get("fn", 0),
                "best_val_f1": history.get("best_val_f1", 0),
                "best_epoch": history.get("best_epoch", 0),
                "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0,
            }

            del model, trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        preset_results["time_seconds"] = time.time() - preset_start
        all_results.append(preset_results)

    total_time = time.time() - total_start

    # === Save raw results ===
    raw_path = os.path.join(ablation_dir, "ablation_raw.json")
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # === Generate report ===
    report = generate_report(all_results, total_time)

    report_path = os.path.join(ablation_dir, "ablation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to: {report_path}")
    print(f"Raw results saved to: {raw_path}")


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(all_results, total_time):
    lines = []
    lines.append("=" * 72)
    lines.append("  LatBOND v10 — FOCAL LOSS ABLATION REPORT")
    lines.append("=" * 72)
    lines.append(f"  Architecture: {ARCH}")
    lines.append(f"  Epochs per preset: {ABLATION_EPOCHS}")
    lines.append(f"  Total runtime: {total_time / 60:.1f} min")
    lines.append("")

    # --- Table ---
    lines.append("-" * 72)
    header = (
        f"{'Preset':<16} {'γ+':<5} {'γ-':<5} "
        f"{'M_F1':>7} {'M_Prec':>7} {'M_Rec':>7} "
        f"{'T_F1':>7} {'Δ F1':>7}"
    )
    lines.append(header)
    lines.append("-" * 72)

    parsed = []
    for r in all_results:
        p = r["preset"]
        m = r["conditions"].get("matched", {})
        t = r["conditions"].get("truncated", {})
        gp = f"{p['gamma_pos']}" if p["gamma_pos"] is not None else "N/A"
        gn = f"{p['gamma_neg']}" if p["gamma_neg"] is not None else "N/A"
        improvement = m.get("f1", 0) - t.get("f1", 0)

        row = {
            "name": p["name"],
            "label": p["label"],
            "gamma_pos": p["gamma_pos"],
            "gamma_neg": p["gamma_neg"],
            "loss_type": p["loss_type"],
            "m_f1": m.get("f1", 0),
            "m_prec": m.get("precision", 0),
            "m_rec": m.get("recall", 0),
            "m_thresh": m.get("threshold", 0),
            "m_tp": m.get("tp", 0),
            "m_fp": m.get("fp", 0),
            "m_fn": m.get("fn", 0),
            "m_train_loss": m.get("final_train_loss", 0),
            "t_f1": t.get("f1", 0),
            "t_prec": t.get("precision", 0),
            "t_rec": t.get("recall", 0),
            "improvement": improvement,
            "time": r.get("time_seconds", 0),
        }
        parsed.append(row)

        lines.append(
            f"{p['name']:<16} {gp:<5} {gn:<5} "
            f"{row['m_f1']:>7.4f} {row['m_prec']:>7.4f} {row['m_rec']:>7.4f} "
            f"{row['t_f1']:>7.4f} {row['improvement']:>+7.4f}"
        )

    lines.append("-" * 72)
    lines.append("")

    # --- Detailed per-preset breakdown ---
    lines.append("DETAILED BREAKDOWN")
    lines.append("-" * 72)

    for row in parsed:
        lines.append(f"\n  {row['label']}:")
        lines.append(f"    Matched:   F1={row['m_f1']:.4f}  P={row['m_prec']:.4f}  R={row['m_rec']:.4f}  thresh={row['m_thresh']}")
        lines.append(f"    Truncated: F1={row['t_f1']:.4f}  P={row['t_prec']:.4f}  R={row['t_rec']:.4f}")
        lines.append(f"    Δ F1: {row['improvement']:+.4f}  |  TP={row['m_tp']}  FP={row['m_fp']}  FN={row['m_fn']}")
        lines.append(f"    Train loss: {row['m_train_loss']:.4f}  |  Time: {row['time']:.0f}s")

    lines.append("")

    # --- Safety verdicts ---
    lines.append("=" * 72)
    lines.append("  SAFETY VERDICTS")
    lines.append("=" * 72)

    bce = parsed[0]  # BCE baseline is always first
    failures = []
    warnings = []

    for row in parsed:
        name = row["name"]
        verdict_parts = []

        # Check 1: Precision collapse (overprediction)
        if row["m_prec"] < 0.05 and (row["m_tp"] + row["m_fp"]) > 0:
            msg = f"FAIL: {name} — precision collapse ({row['m_prec']:.4f}). Model overpredicting."
            failures.append(msg)
            verdict_parts.append("❌ PRECISION COLLAPSE")

        # Check 2: Recall collapse (underprediction)
        if row["m_rec"] < 0.05 and row["m_fn"] > 10:
            msg = f"FAIL: {name} — recall collapse ({row['m_rec']:.4f}). Model not detecting onsets."
            failures.append(msg)
            verdict_parts.append("❌ RECALL COLLAPSE")

        # Check 3: Loss instability
        if row["m_train_loss"] > 50 or row["m_train_loss"] != row["m_train_loss"]:
            msg = f"FAIL: {name} — loss unstable ({row['m_train_loss']:.4f})."
            failures.append(msg)
            verdict_parts.append("❌ LOSS UNSTABLE")

        # Check 4: Precision dropped >70% relative to BCE baseline
        if name != "bce_baseline" and bce["m_prec"] > 0:
            ratio = row["m_prec"] / bce["m_prec"]
            if ratio < 0.3:
                msg = (f"WARN: {name} — precision is {ratio:.0%} of BCE baseline "
                       f"({row['m_prec']:.4f} vs {bce['m_prec']:.4f}). "
                       f"Possible overprediction.")
                warnings.append(msg)
                verdict_parts.append(f"⚠ PRECISION {ratio:.0%} OF BCE")

        # Check 5: Recall improvement check (the whole point)
        if name != "bce_baseline":
            rec_gain = row["m_rec"] - bce["m_rec"]
            if rec_gain > 0.02:
                verdict_parts.append(f"✓ RECALL +{rec_gain:.4f} vs BCE")
            elif rec_gain > -0.02:
                verdict_parts.append(f"~ RECALL NEUTRAL ({rec_gain:+.4f})")
            else:
                verdict_parts.append(f"⚠ RECALL WORSE ({rec_gain:+.4f})")

        # Check 6: Does matched beat truncated?
        if row["improvement"] > 0:
            verdict_parts.append(f"✓ MATCHED > TRUNCATED (+{row['improvement']:.4f})")
        else:
            verdict_parts.append(f"✗ TRUNCATED STILL AHEAD ({row['improvement']:+.4f})")

        verdict_str = " | ".join(verdict_parts) if verdict_parts else "✓ OK"
        lines.append(f"\n  {row['label']}:")
        lines.append(f"    {verdict_str}")

    lines.append("")

    if failures:
        lines.append("⚠ FAILURES:")
        for f in failures:
            lines.append(f"  {f}")
        lines.append("")

    if warnings:
        lines.append("⚠ WARNINGS:")
        for w in warnings:
            lines.append(f"  {w}")
        lines.append("")

    if not failures and not warnings:
        lines.append("✓ All presets passed safety checks.")
        lines.append("")

    # --- Recommendation ---
    lines.append("=" * 72)
    lines.append("  RECOMMENDATION")
    lines.append("=" * 72)

    # Among focal presets that didn't fail, pick highest matched F1
    focal = [r for r in parsed if r["name"] != "bce_baseline"]
    failed_names = set()
    for f in failures:
        for r in focal:
            if r["name"] in f:
                failed_names.add(r["name"])

    safe_focal = [r for r in focal if r["name"] not in failed_names]

    if safe_focal:
        # Primary: highest matched F1
        winner = max(safe_focal, key=lambda r: r["m_f1"])

        # If multiple are close (within 0.01 F1), prefer higher recall
        close = [r for r in safe_focal if abs(r["m_f1"] - winner["m_f1"]) < 0.01]
        if len(close) > 1:
            winner = max(close, key=lambda r: r["m_rec"])

        lines.append(f"\n  ★ USE: {winner['label']}")
        lines.append(f"    Matched F1:    {winner['m_f1']:.4f}")
        lines.append(f"    Precision:     {winner['m_prec']:.4f}")
        lines.append(f"    Recall:        {winner['m_rec']:.4f}")
        lines.append(f"    vs Truncated:  {winner['improvement']:+.4f}")

        lines.append(f"\n  config.py settings for full v10 run:")
        lines.append(f"    loss_type: str = \"{winner['loss_type']}\"")
        if winner["gamma_pos"] is not None:
            lines.append(f"    focal_gamma_pos: float = {winner['gamma_pos']}")
            lines.append(f"    focal_gamma_neg: float = {winner['gamma_neg']}")

        lines.append(f"\n  CLI for Phase 2:")
        if winner["gamma_pos"] is not None:
            lines.append(
                f"    bash run_all.sh --skip_download "
                f"--gamma_pos {winner['gamma_pos']} --gamma_neg {winner['gamma_neg']}"
            )
        else:
            lines.append(f"    bash run_all.sh --skip_download --loss_type bce")

        # Warn if winner didn't beat truncated
        if winner["improvement"] <= 0:
            lines.append(f"\n  NOTE: Matched did not beat truncated in 3 epochs.")
            lines.append(f"  This is expected — the full 25-epoch run with cosine")
            lines.append(f"  curriculum gives matched time to converge. The ablation")
            lines.append(f"  confirms the loss function is SAFE, not that it WINS.")

    else:
        lines.append(f"\n  ⚠ No safe focal preset found. Fall back to BCE.")
        lines.append(f"\n  config.py change:")
        lines.append(f'    loss_type: str = "bce"')
        lines.append(f"\n  CLI for Phase 2:")
        lines.append(f"    bash run_all.sh --skip_download --loss_type bce")

    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)


if __name__ == "__main__":
    main()
