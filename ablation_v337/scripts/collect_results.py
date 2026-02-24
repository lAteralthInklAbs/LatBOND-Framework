#!/usr/bin/env python3
"""
Collect results from all 42 models (possibly across 3 Vertex AI jobs)
and run the full statistical analysis suite.

Usage:
    python scripts/collect_results.py --results_dir outputs/

    # Or from GCS after Vertex AI jobs complete:
    python scripts/collect_results.py --gcs_prefix gs://YOUR_GCP_PROJECT-data/outputs/
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LatBONDConfig
from evaluation.statistical_analysis import run_full_analysis
from evaluation.visualization import generate_all_thesis_figures


def collect_local_results(results_dir):
    """Collect results.json from each model subdirectory."""
    all_results = {}

    for arch_dir in os.listdir(results_dir):
        arch_path = os.path.join(results_dir, arch_dir)
        if not os.path.isdir(arch_path):
            continue

        for budget_dir in os.listdir(arch_path):
            budget_path = os.path.join(arch_path, budget_dir)
            if not os.path.isdir(budget_path):
                continue

            for cond_dir in os.listdir(budget_path):
                results_file = os.path.join(budget_path, cond_dir, 'results.json')
                if os.path.exists(results_file):
                    with open(results_file) as f:
                        result = json.load(f)

                    arch = result['architecture']
                    budget = result['budget_ms']
                    condition = result['condition']
                    all_results[(arch, budget, condition)] = result
                    print(f"  Loaded: {arch}_{budget}ms_{condition}")

    return all_results


def generate_thesis_tables(all_results, analysis, config, output_dir):
    """Generate thesis-ready tables in markdown and LaTeX format."""
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    # Main results table (Markdown)
    md_lines = [
        "# LatBOND Full Study - Main Results",
        "",
        "## Per-Budget Comparison (Matched vs Truncated)",
        "",
        "| Budget | Matched F1 | Truncated F1 | Delta | p-value | Cohen's d | Significant |",
        "|--------|------------|--------------|-------|---------|-----------|-------------|",
    ]

    for budget in config.latency_budgets_ms:
        ba = analysis.get('per_budget_comparisons', {}).get(budget, {})
        if not ba:
            continue
        m_mean = ba.get('matched_file_stats', {}).get('mean', 0)
        t_mean = ba.get('truncated_file_stats', {}).get('mean', 0)
        delta = m_mean - t_mean
        p = ba.get('paired_ttest', {}).get('p_value', 1.0)
        d = ba.get('cohens_d', {}).get('d', 0)
        sig = 'Yes' if ba.get('paired_ttest', {}).get('significant', False) else 'No'
        md_lines.append(f"| {budget}ms | {m_mean:.4f} | {t_mean:.4f} | {delta:+.4f} | {p:.2e} | {d:.3f} | {sig} |")

    with open(os.path.join(tables_dir, 'main_results.md'), 'w') as f:
        f.write('\n'.join(md_lines))

    # Recovery table
    if analysis.get('recovery'):
        recovery_lines = [
            "# Recovery Calculations",
            "",
            "| Architecture | Budget | Recovery % |",
            "|--------------|--------|------------|",
        ]
        for key, rec in analysis['recovery'].items():
            arch, budget = key if isinstance(key, tuple) else eval(key)
            pct = rec.get('recovery_percentage')
            if pct is not None:
                recovery_lines.append(f"| {arch} | {budget}ms | {pct:.1f}% |")

        with open(os.path.join(tables_dir, 'recovery_table.md'), 'w') as f:
            f.write('\n'.join(recovery_lines))

    print(f"  Tables saved to {tables_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='outputs')
    parser.add_argument('--output_dir', default='outputs/analysis')
    args = parser.parse_args()

    config = LatBONDConfig()

    print("Collecting results...")
    all_results = collect_local_results(args.results_dir)
    print(f"\nFound {len(all_results)} model results (expected: 45 including buffered@0ms)")

    # Validate completeness
    expected = config.get_evaluation_configurations()
    missing = [c for c in expected if c not in all_results]
    if missing:
        print(f"\nWARNING: Missing {len(missing)} models:")
        for m in missing[:10]:
            print(f"  - {m[0]}_{m[1]}ms_{m[2]}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    # Run full statistical analysis
    print("\nRunning statistical analysis...")
    os.makedirs(args.output_dir, exist_ok=True)

    analysis = run_full_analysis(all_results, config)

    # Save analysis
    analysis_path = os.path.join(args.output_dir, 'statistical_analysis.json')
    with open(analysis_path, 'w') as f:
        # Convert tuple keys to strings for JSON
        serializable = {}
        for k, v in analysis.items():
            if isinstance(v, dict):
                serializable[k] = {}
                for k2, v2 in v.items():
                    if isinstance(k2, tuple):
                        serializable[k][str(k2)] = v2
                    else:
                        serializable[k][k2] = v2
            else:
                serializable[k] = v
        json.dump(serializable, f, indent=2, default=str)
    print(f"Analysis saved to {analysis_path}")

    # Generate figures
    generate_all_thesis_figures(all_results, analysis, args.output_dir)

    # Generate thesis tables
    generate_thesis_tables(all_results, analysis, config, args.output_dir)

    print(f"\nAll analysis outputs saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
