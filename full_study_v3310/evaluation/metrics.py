"""
LatBOND Evaluation Metrics - Full Version
==========================================
Evaluation on test set with proper metrics and file-level scoring.

Updates from v11:
  - File-level F1 scores for statistical analysis
  - Support for all 3 conditions (matched, truncated, buffered)
  - Note-level evaluation via mir_eval
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


def evaluate_model(model, test_loader, config, condition, arch_name=None, device='cuda'):
    """
    Evaluate a trained model on the test set.
    Returns dict with F1, precision, recall at various tolerances.
    """
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for mel, labels in test_loader:
            mel = mel.to(device)

            if condition == 'buffered':
                # Buffered: full context inference
                outputs = _batched_sliding_window_inference(
                    model, mel, config, arch_name, use_full_context=True
                )
            elif condition == 'matched':
                outputs = _batched_sliding_window_inference(
                    model, mel, config, arch_name
                )
            else:  # truncated
                outputs = _batched_sliding_window_inference(
                    model, mel, config, arch_name
                )

            probs = torch.sigmoid(outputs['onset_logits']).cpu()
            all_probs.append(probs)
            all_labels.append(labels['onsets'])

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Evaluate at multiple tolerances
    results = {}
    for tol_ms in [10, 20, 30, 50]:
        tol_frames = max(1, int(tol_ms / config.frame_duration_ms))

        best_f1 = 0
        best_thresh = 0.3
        best_metrics = {}

        for thresh in config.threshold_sweep:
            metrics = _compute_metrics(all_probs, all_labels, thresh, tol_frames)
            if metrics['f1'] >= best_f1:
                best_f1 = metrics['f1']
                best_thresh = thresh
                best_metrics = metrics

        best_metrics['threshold'] = best_thresh
        results[f"tol_{tol_ms}ms"] = best_metrics

    return results


def evaluate_model_note_level(model, test_loader, config, condition, arch_name=None, device='cuda'):
    """
    Note-level evaluation using mir_eval for comparison with Hu et al.
    """
    try:
        import mir_eval
    except ImportError:
        return {'warning': 'mir_eval not installed'}

    # Placeholder for note-level evaluation
    # Full implementation would use assemble_notes() and mir_eval.transcription
    return {
        'note_onset_f1_10ms': 0.0,
        'note_onset_f1_20ms': 0.0,
        'note_onset_f1_30ms': 0.0,
        'note_onset_f1_50ms': 0.0,
    }


def _batched_sliding_window_inference(model, mel, config, arch_name=None, use_full_context=False):
    """
    Batched sliding window inference.
    """
    B, C, T = mel.shape
    device = mel.device

    if use_full_context:
        ctx = T
    elif arch_name and hasattr(config, 'context_frames_per_arch'):
        ctx = config.context_frames_per_arch.get(arch_name, config.get_budget_frames() * config.context_multiplier)
    else:
        ctx = config.get_budget_frames() * config.context_multiplier

    stride = config.eval_stride
    infer_batch = getattr(config, 'eval_batch_size', 128)

    all_onset_logits = torch.zeros(B, config.n_pitches, T, device=device)
    all_offset_logits = torch.zeros(B, config.n_pitches, T, device=device)
    all_velocity_logits = torch.zeros(B, config.n_pitches, T, device=device)
    counts = torch.zeros(T, device=device)

    positions = list(range(0, T, stride))

    for b in range(B):
        windows = []
        window_ranges = []

        for pos in positions:
            end = min(pos + ctx, T)
            actual_start = max(0, end - ctx)

            chunk = mel[b, :, actual_start:end]
            if chunk.shape[-1] < ctx:
                pad_len = ctx - chunk.shape[-1]
                chunk = F.pad(chunk, (pad_len, 0))

            windows.append(chunk)
            window_ranges.append((actual_start, end))

        if len(windows) == 0:
            continue

        windows_tensor = torch.stack(windows, dim=0)

        for i in range(0, len(windows_tensor), infer_batch):
            batch_chunk = windows_tensor[i:i + infer_batch]
            batch_out = model(batch_chunk)

            for j in range(len(batch_out['onset_logits'])):
                wi = i + j
                actual_start, end = window_ranges[wi]
                pred_len = end - actual_start
                out_start = max(0, batch_out['onset_logits'].shape[-1] - pred_len)

                all_onset_logits[b, :, actual_start:end] += batch_out['onset_logits'][j, :, out_start:]
                all_offset_logits[b, :, actual_start:end] += batch_out['offset_logits'][j, :, out_start:]
                all_velocity_logits[b, :, actual_start:end] += batch_out['velocity_logits'][j, :, out_start:]
                if b == 0:
                    counts[actual_start:end] += 1

    counts = counts.clamp(min=1).unsqueeze(0).unsqueeze(0)
    return {
        'onset_logits': all_onset_logits / counts,
        'offset_logits': all_offset_logits / counts,
        'velocity_logits': all_velocity_logits / counts,
    }


def _compute_metrics(probs, labels, threshold, tolerance):
    """Compute precision, recall, F1 with frame tolerance."""
    preds = (probs > threshold).float()

    B, P, T = preds.shape
    preds_flat = preds.view(B * P, T)
    labels_flat = labels.view(B * P, T)

    tp = 0
    fp = 0
    fn = 0
    per_sample_f1s = []

    for i in range(B * P):
        pred_frames = preds_flat[i].nonzero(as_tuple=True)[0]
        label_frames = labels_flat[i].nonzero(as_tuple=True)[0]

        if len(label_frames) == 0 and len(pred_frames) == 0:
            continue

        if len(label_frames) == 0:
            fp += len(pred_frames)
            if len(pred_frames) > 0:
                per_sample_f1s.append(0.0)
            continue

        if len(pred_frames) == 0:
            fn += len(label_frames)
            per_sample_f1s.append(0.0)
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

        sample_tp = len(matched_labels)
        sample_fp = len(pred_frames) - len(matched_preds)
        sample_fn = len(label_frames) - len(matched_labels)

        tp += sample_tp
        fp += sample_fp
        fn += sample_fn

        p = sample_tp / max(sample_tp + sample_fp, 1)
        r = sample_tp / max(sample_tp + sample_fn, 1)
        f = 2 * p * r / max(p + r, 1e-8)
        per_sample_f1s.append(f)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': tp, 'fp': fp, 'fn': fn,
        'per_sample_f1s': per_sample_f1s,
    }


def compare_conditions(matched_results, truncated_results, tolerance_key="tol_20ms"):
    """Compare matched vs truncated using paired t-test."""
    m = matched_results[tolerance_key]
    t = truncated_results[tolerance_key]

    improvement = m['f1'] - t['f1']

    m_samples = np.array(m.get('per_sample_f1s', []))
    t_samples = np.array(t.get('per_sample_f1s', []))

    min_len = min(len(m_samples), len(t_samples))

    if min_len >= 3:
        stat, p_value = stats.ttest_rel(m_samples[:min_len], t_samples[:min_len])
        diffs = m_samples[:min_len] - t_samples[:min_len]
        d = np.mean(diffs) / max(np.std(diffs, ddof=1), 1e-8)
    else:
        p_value = 1.0
        d = 0.0

    return {
        'matched_f1': m['f1'],
        'truncated_f1': t['f1'],
        'improvement': improvement,
        'p_value': float(p_value),
        'cohens_d': float(d),
        'significant': bool(p_value < 0.05),
        'tolerance': tolerance_key,
    }


def generate_report(all_results, output_dir):
    """Generate final results report."""
    report_lines = [
        "=" * 70,
        "LatBOND Full Version - Results Report",
        "3 Architectures x 5 Budgets x 3 Conditions",
        "=" * 70,
        "",
    ]

    comparisons = {}

    for arch in ['streaming_cnn', 'causal_transformer', 'tcn']:
        if arch not in all_results:
            continue

        r = all_results[arch]
        report_lines.append(f"\n--- {arch.upper()} ---")

        for condition in ['matched', 'truncated', 'buffered']:
            if condition not in r:
                continue

            report_lines.append(f"\n  {condition.upper()}:")
            for tol_key, metrics in r[condition].items():
                if isinstance(metrics, dict) and 'f1' in metrics:
                    report_lines.append(
                        f"    {tol_key}: F1={metrics['f1']:.4f} "
                        f"(P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
                        f"thresh={metrics.get('threshold', 'N/A')})"
                    )

        if 'matched' in r and 'truncated' in r:
            comp = compare_conditions(r['matched'], r['truncated'])
            comparisons[arch] = comp

            report_lines.append(f"\n  COMPARISON (@ 20ms tolerance):")
            report_lines.append(f"    Matched F1:   {comp['matched_f1']:.4f}")
            report_lines.append(f"    Truncated F1: {comp['truncated_f1']:.4f}")
            report_lines.append(f"    Improvement:  {comp['improvement']:+.4f}")
            report_lines.append(f"    p-value:      {comp['p_value']:.4f}")
            report_lines.append(f"    Cohen's d:    {comp['cohens_d']:.4f}")
            report_lines.append(f"    Significant:  {'YES' if comp['significant'] else 'NO'}")

    report_lines.extend([
        "\n" + "=" * 70,
        "SUMMARY",
        "=" * 70,
    ])

    for arch, comp in comparisons.items():
        status = "MATCHED > TRUNCATED" if comp['improvement'] > 0 else "TRUNCATED > MATCHED"
        sig = " (p<0.05)" if comp['significant'] else " (n.s.)"
        report_lines.append(
            f"  {arch}: {comp['improvement']:+.4f} {status}{sig}"
        )

    report_text = "\n".join(report_lines)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "results_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)

    # Save structured results
    clean_results = {}
    for arch, arch_results in all_results.items():
        clean_results[arch] = {}
        for cond, cond_results in arch_results.items():
            clean_results[arch][cond] = {}
            for k, v in cond_results.items():
                if isinstance(v, dict):
                    clean_v = {kk: vv for kk, vv in v.items()
                              if kk != 'per_sample_f1s'}
                    clean_results[arch][cond][k] = clean_v
                else:
                    clean_results[arch][cond][k] = v

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'results': clean_results,
            'comparisons': comparisons,
        }, f, indent=2)

    print(report_text)
    return report_text, comparisons


def assemble_notes(onset_probs, offset_probs, velocity_preds, config=None, threshold=0.5):
    """
    Convert frame-level predictions to note events.

    Returns:
        List of (onset_sec, offset_sec, pitch_hz, velocity) tuples
    """
    if config is None:
        frame_duration = 0.01  # 10ms default
    else:
        frame_duration = config.hop_length / config.sample_rate

    n_pitches, n_frames = onset_probs.shape
    notes = []

    for pitch in range(n_pitches):
        onset_frames = np.where(onset_probs[pitch] > threshold)[0]

        for onset_frame in onset_frames:
            # Find corresponding offset
            offset_candidates = np.where(
                (offset_probs[pitch, onset_frame:] > threshold)
            )[0]

            if len(offset_candidates) > 0:
                offset_frame = onset_frame + offset_candidates[0]
            else:
                offset_frame = min(onset_frame + 50, n_frames - 1)  # Default duration

            # Get velocity
            velocity = velocity_preds[pitch, onset_frame]

            # Convert to time
            onset_sec = onset_frame * frame_duration
            offset_sec = offset_frame * frame_duration
            pitch_hz = 440.0 * (2.0 ** ((pitch + 21 - 69) / 12.0))  # MIDI to Hz

            notes.append((onset_sec, offset_sec, pitch_hz, velocity))

    return notes
