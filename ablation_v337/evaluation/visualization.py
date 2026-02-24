"""
LatBOND Visualization - Full Version
=====================================
Generate comparison charts and thesis figures.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Color scheme
COLORS = {
    'matched': '#2196F3',     # Blue
    'truncated': '#F44336',   # Red
    'buffered': '#4CAF50',    # Green
}

MARKERS = {
    'matched': 'o',
    'truncated': '^',
    'buffered': 's',
}


def plot_f1_comparison(all_results, output_dir):
    """Bar chart: Matched vs Truncated F1 for all 3 architectures."""
    archs = ['streaming_cnn', 'causal_transformer', 'tcn']
    arch_labels = ['Streaming CNN', 'Causal Transformer', 'TCN']

    matched_f1s = []
    truncated_f1s = []

    for arch in archs:
        if arch in all_results:
            m_f1 = all_results[arch].get('matched', {}).get('tol_20ms', {}).get('f1', 0)
            t_f1 = all_results[arch].get('truncated', {}).get('tol_20ms', {}).get('f1', 0)
        else:
            m_f1 = t_f1 = 0
        matched_f1s.append(m_f1)
        truncated_f1s.append(t_f1)

    x = np.arange(len(archs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, matched_f1s, width, label='Matched (LatBOND)',
                   color=COLORS['matched'], edgecolor='white')
    bars2 = ax.bar(x + width/2, truncated_f1s, width, label='Truncated (Baseline)',
                   color=COLORS['truncated'], edgecolor='white')

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('F1 Score (+/-20ms tolerance)', fontsize=12)
    ax.set_title('LatBOND: Matched vs Truncated Training', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(matched_f1s), max(truncated_f1s)) * 1.2 + 0.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'f1_comparison.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'f1_comparison.pdf'))
    plt.close()
    print("[INFO] Saved f1_comparison.png/pdf")


def plot_training_curves(output_dir):
    """Plot training curves for all models."""
    archs = ['streaming_cnn', 'causal_transformer', 'tcn']
    conditions = ['matched', 'truncated']
    colors = {
        ('streaming_cnn', 'matched'): '#1565C0',
        ('streaming_cnn', 'truncated'): '#64B5F6',
        ('causal_transformer', 'matched'): '#2E7D32',
        ('causal_transformer', 'truncated'): '#81C784',
        ('tcn', 'matched'): '#E65100',
        ('tcn', 'truncated'): '#FFB74D',
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for arch in archs:
        for cond in conditions:
            history_path = os.path.join(output_dir, f"{arch}_{cond}_history.json")
            if not os.path.exists(history_path):
                continue

            with open(history_path) as f:
                history = json.load(f)

            label = f"{arch.replace('_', ' ').title()} ({cond})"
            color = colors.get((arch, cond), 'gray')
            linestyle = '-' if cond == 'matched' else '--'

            epochs = range(1, len(history.get('val_loss', [])) + 1)

            if 'val_loss' in history:
                axes[0].plot(epochs, history['val_loss'],
                            label=label, color=color, linestyle=linestyle)

            if 'val_f1' in history:
                axes[1].plot(epochs, history['val_f1'],
                            label=label, color=color, linestyle=linestyle)

    axes[0].set_title('Validation Loss', fontsize=12)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].set_title('Validation F1 Score', fontsize=12)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1')
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    plt.suptitle('LatBOND - Training Curves', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print("[INFO] Saved training_curves.png")


def plot_tolerance_comparison(all_results, output_dir):
    """Line chart: F1 across different tolerances for each architecture."""
    archs = ['streaming_cnn', 'causal_transformer', 'tcn']
    arch_labels = ['Streaming CNN', 'Causal Transformer', 'TCN']
    tolerances = [10, 20, 30, 50]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (arch, label) in enumerate(zip(archs, arch_labels)):
        ax = axes[idx]

        if arch not in all_results:
            ax.set_title(f'{label}\n(No data)')
            continue

        for cond, color, marker in [('matched', COLORS['matched'], 'o'),
                                     ('truncated', COLORS['truncated'], 's')]:
            if cond not in all_results[arch]:
                continue

            f1s = []
            for tol in tolerances:
                key = f'tol_{tol}ms'
                f1 = all_results[arch][cond].get(key, {}).get('f1', 0)
                f1s.append(f1)

            ax.plot(tolerances, f1s, color=color, marker=marker,
                   label=cond.title(), linewidth=2)

        ax.set_title(label, fontsize=12)
        ax.set_xlabel('Tolerance (ms)')
        ax.set_ylabel('F1 Score')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xticks(tolerances)

    plt.suptitle('F1 vs Tolerance Window - All Architectures', fontsize=14)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'tolerance_comparison.png'), dpi=150)
    plt.close()
    print("[INFO] Saved tolerance_comparison.png")


def plot_cross_budget_comparison(all_results, output_dir):
    """
    Main thesis figure: F1 vs latency budget for matched/truncated/buffered.
    """
    budgets = [0, 20, 40, 60, 80]
    archs = ['streaming_cnn', 'causal_transformer', 'tcn']
    arch_labels = ['Streaming CNN', 'Causal Transformer', 'TCN', 'Average']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Plot each architecture
    for idx, (arch, label) in enumerate(zip(archs + ['average'], arch_labels)):
        ax = axes[idx]

        for condition in ['matched', 'truncated', 'buffered']:
            f1s = []
            stds = []

            for budget in budgets:
                if arch == 'average':
                    # Average across architectures
                    vals = []
                    for a in archs:
                        key = (a, budget, condition)
                        if key in all_results:
                            vals.append(all_results[key].get('aggregate_f1', 0))
                    f1 = np.mean(vals) if vals else 0
                    std = np.std(vals) if vals else 0
                else:
                    key = (arch, budget, condition)
                    if key in all_results:
                        f1 = all_results[key].get('aggregate_f1', 0)
                        std = all_results[key].get('file_f1_std', 0)
                    else:
                        f1 = 0
                        std = 0

                f1s.append(f1)
                stds.append(std)

            ax.errorbar(budgets, f1s, yerr=stds,
                       color=COLORS[condition],
                       marker=MARKERS[condition],
                       label=condition.title(),
                       linewidth=2, capsize=3)

        ax.set_title(label, fontsize=12)
        ax.set_xlabel('Latency Budget (ms)')
        ax.set_ylabel('Note Onset F1')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xticks(budgets)

    plt.suptitle('LatBOND: F1 vs Latency Budget (All Conditions)', fontsize=14)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'cross_budget_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'cross_budget_comparison.pdf'))
    plt.close()
    print("[INFO] Saved cross_budget_comparison.png/pdf")


def plot_pareto_curves(all_results, output_dir):
    """Accuracy-latency Pareto trade-off curves."""
    budgets = [0, 20, 40, 60, 80]
    archs = ['streaming_cnn', 'causal_transformer', 'tcn']

    fig, ax = plt.subplots(figsize=(10, 6))

    for arch in archs:
        for condition in ['matched', 'truncated']:
            f1s = []
            for budget in budgets:
                key = (arch, budget, condition)
                if key in all_results:
                    f1s.append(all_results[key].get('aggregate_f1', 0))
                else:
                    f1s.append(0)

            linestyle = '-' if condition == 'matched' else '--'
            label = f"{arch.replace('_', ' ').title()} ({condition})"
            ax.plot(budgets, f1s, marker='o', linestyle=linestyle, label=label)

    ax.set_xlabel('Latency Budget (ms)', fontsize=12)
    ax.set_ylabel('Note Onset F1', fontsize=12)
    ax.set_title('Accuracy-Latency Pareto Curves', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'pareto_curves.png'), dpi=150)
    plt.close()
    print("[INFO] Saved pareto_curves.png")


def plot_recovery_heatmap(recovery_results, output_dir):
    """Heatmap showing recovery percentage: arch x budget."""
    archs = ['streaming_cnn', 'causal_transformer', 'tcn']
    budgets = [0, 20, 40, 60, 80]

    data = np.zeros((len(archs), len(budgets)))

    for i, arch in enumerate(archs):
        for j, budget in enumerate(budgets):
            key = (arch, budget)
            if key in recovery_results:
                rec = recovery_results[key].get('recovery_percentage')
                data[i, j] = rec if rec is not None else np.nan
            else:
                data[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(budgets)))
    ax.set_yticks(np.arange(len(archs)))
    ax.set_xticklabels([f'{b}ms' for b in budgets])
    ax.set_yticklabels([a.replace('_', ' ').title() for a in archs])

    # Add text annotations
    for i in range(len(archs)):
        for j in range(len(budgets)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=10)

    ax.set_xlabel('Latency Budget')
    ax.set_ylabel('Architecture')
    ax.set_title('Recovery Percentage: (Matched - Truncated) / (Buffered - Truncated)', fontsize=12)

    cbar = plt.colorbar(im)
    cbar.set_label('Recovery %')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'recovery_heatmap.png'), dpi=150)
    plt.close()
    print("[INFO] Saved recovery_heatmap.png")


def plot_effect_size_by_budget(analysis, output_dir):
    """Cohen's d effect sizes by budget."""
    if 'per_budget_comparisons' not in analysis:
        return

    budgets = sorted(analysis['per_budget_comparisons'].keys())
    effect_sizes = []

    for budget in budgets:
        d = analysis['per_budget_comparisons'][budget].get('cohens_d', {}).get('d', 0)
        effect_sizes.append(d)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar([str(b) for b in budgets], effect_sizes, color=COLORS['matched'])

    # Reference lines
    ax.axhline(0.2, color='gray', linestyle='--', alpha=0.5, label='Small (0.2)')
    ax.axhline(0.5, color='gray', linestyle='-.', alpha=0.5, label='Medium (0.5)')
    ax.axhline(0.8, color='gray', linestyle=':', alpha=0.5, label='Large (0.8)')

    ax.set_xlabel('Latency Budget (ms)', fontsize=12)
    ax.set_ylabel("Cohen's d", fontsize=12)
    ax.set_title('Effect Size (Matched vs Truncated) by Budget', fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'effect_sizes.png'), dpi=150)
    plt.close()
    print("[INFO] Saved effect_sizes.png")


def plot_file_level_distribution(all_results, output_dir):
    """Box plots showing per-file F1 distribution for matched vs truncated."""
    budgets = [0, 20, 40, 60, 80]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    for idx, budget in enumerate(budgets):
        ax = axes[idx]

        data = []
        labels = []

        for condition in ['matched', 'truncated']:
            scores = []
            for arch in ['streaming_cnn', 'causal_transformer', 'tcn']:
                key = (arch, budget, condition)
                if key in all_results:
                    scores.extend(all_results[key].get('file_f1_scores', []))

            if scores:
                data.append(scores)
                labels.append(condition.title())

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, cond in zip(bp['boxes'], ['matched', 'truncated']):
                patch.set_facecolor(COLORS[cond])
                patch.set_alpha(0.7)

        ax.set_title(f'{budget}ms Budget', fontsize=11)
        ax.set_ylabel('File F1' if idx == 0 else '')
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Per-File F1 Distribution: Matched vs Truncated', fontsize=14)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'file_level_distributions.png'), dpi=150)
    plt.close()
    print("[INFO] Saved file_level_distributions.png")


def generate_all_thesis_figures(all_results, analysis, output_dir):
    """Generate all figures needed for the thesis."""
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    try:
        plot_cross_budget_comparison(all_results, fig_dir)
    except Exception as e:
        print(f"[WARN] cross_budget_comparison failed: {e}")

    try:
        plot_pareto_curves(all_results, fig_dir)
    except Exception as e:
        print(f"[WARN] pareto_curves failed: {e}")

    try:
        if analysis and 'recovery' in analysis:
            plot_recovery_heatmap(analysis.get('recovery', {}), fig_dir)
    except Exception as e:
        print(f"[WARN] recovery_heatmap failed: {e}")

    try:
        if analysis:
            plot_effect_size_by_budget(analysis, fig_dir)
    except Exception as e:
        print(f"[WARN] effect_sizes failed: {e}")

    try:
        plot_file_level_distribution(all_results, fig_dir)
    except Exception as e:
        print(f"[WARN] file_level_distributions failed: {e}")

    print(f"All thesis figures saved to {fig_dir}/")
