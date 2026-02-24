#!/usr/bin/env python3
"""
Check v3 pilot results and decide whether to launch full run.

Usage:
    python scripts/check_pilot.py
    python scripts/check_pilot.py --bucket YOUR_GCP_PROJECT-data
"""

import argparse
import json
import sys

try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False


def main():
    parser = argparse.ArgumentParser(description='Check v3 pilot results')
    parser.add_argument('--bucket', default='YOUR_GCP_PROJECT-data')
    parser.add_argument('--arch', default='streaming_cnn')
    parser.add_argument('--budget', type=int, default=40)
    args = parser.parse_args()

    if not HAS_GCS:
        print("[ERROR] google-cloud-storage required")
        sys.exit(1)

    client = storage.Client()
    bucket = client.bucket(args.bucket)
    base = f"outputs/v3-pilot/{args.arch}/{args.budget}ms"

    print("=" * 60)
    print("LatBOND v3 PILOT RESULTS")
    print("=" * 60)

    results = {}
    for cond in ['matched', 'truncated', 'buffered']:
        blob_path = f"{base}/{cond}/results.json"
        blob = bucket.blob(blob_path)
        if not blob.exists():
            print(f"  {cond}: NOT FOUND at gs://{args.bucket}/{blob_path}")
            continue

        data = json.loads(blob.download_as_text())
        results[cond] = data

        f1 = data.get('aggregate_f1', 0)
        thresh = data.get('best_threshold', 0)
        epochs = data.get('epochs_trained', 0)
        val_f1 = data.get('best_val_f1', 0)
        time_hrs = data.get('training_time_hours', 0)

        # Check training history for NaN and curriculum
        history = data.get('training_history', {})
        train_losses = history.get('train_loss', [])
        ctx_history = history.get('curriculum_context', [])
        has_nan = any(str(l) == 'nan' or (isinstance(l, float) and l != l) for l in train_losses)
        ctx_fixed = len(set(ctx_history)) <= 1 if ctx_history else True

        print(f"\n  {cond.upper()}:")
        print(f"    Test F1:        {f1:.4f}")
        print(f"    Val F1 (best):  {val_f1:.4f}")
        print(f"    Threshold:      {thresh:.2f}")
        print(f"    Epochs:         {epochs}")
        print(f"    Training time:  {time_hrs:.1f} hours")
        print(f"    NaN detected:   {'YES ⚠️' if has_nan else 'No ✓'}")
        print(f"    Fixed context:  {'Yes ✓' if ctx_fixed else 'NO ⚠️  (curriculum still active!)'}")

    # Verdict
    if 'matched' in results and 'truncated' in results:
        m_f1 = results['matched'].get('aggregate_f1', 0)
        t_f1 = results['truncated'].get('aggregate_f1', 0)
        delta = m_f1 - t_f1
        m_thresh = results['matched'].get('best_threshold', 0)
        t_thresh = results['truncated'].get('best_threshold', 0)

        print(f"\n{'=' * 60}")
        print(f"VERDICT")
        print(f"{'=' * 60}")
        print(f"  Matched F1:    {m_f1:.4f}")
        print(f"  Truncated F1:  {t_f1:.4f}")
        print(f"  Delta:         {delta:+.4f}")
        print(f"  Thresholds:    matched={m_thresh:.2f}, truncated={t_thresh:.2f}")

        checks = []

        # Check 1: Matched > Truncated
        if delta > 0.03:
            checks.append(('Matched > Truncated by 3+pp', 'PASS ✓'))
        elif delta > 0:
            checks.append(('Matched > Truncated (marginal)', 'WEAK ⚠️'))
        else:
            checks.append(('Matched > Truncated', f'FAIL ✗ (delta={delta:+.4f})'))

        # Check 2: Absolute F1 improved
        if m_f1 > 0.45:
            checks.append(('F1 above 0.45', 'PASS ✓'))
        elif m_f1 > 0.40:
            checks.append(('F1 above 0.40', 'WEAK ⚠️'))
        else:
            checks.append(('F1 above 0.40', f'FAIL ✗ (F1={m_f1:.4f})'))

        # Check 3: Thresholds differ
        if m_thresh != t_thresh:
            checks.append(('Thresholds differ', 'PASS ✓'))
        else:
            checks.append(('Thresholds differ', f'SAME ({m_thresh:.2f})'))

        # Check 4: No NaN
        m_history = results['matched'].get('training_history', {}).get('train_loss', [])
        m_nan = any(str(l) == 'nan' or (isinstance(l, float) and l != l) for l in m_history)
        if not m_nan:
            checks.append(('No NaN in matched', 'PASS ✓'))
        else:
            checks.append(('No NaN in matched', 'FAIL ✗'))

        print()
        all_pass = True
        for check_name, status in checks:
            print(f"  {check_name}: {status}")
            if 'FAIL' in status:
                all_pass = False

        print()
        if all_pass:
            print("  ✅ PILOT PASSED — launch full run:")
            print("     python scripts/launch_vertex_ai.py")
        else:
            print("  ❌ PILOT HAS ISSUES — investigate before full run")
            print("     Review training histories and consider adjustments")

    else:
        print("\n  ⚠️  Not all conditions found. Job may still be running.")
        print(f"  Check: https://console.cloud.google.com/vertex-ai/training/custom-jobs")


if __name__ == '__main__':
    main()
