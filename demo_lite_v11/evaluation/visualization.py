"""
LatBOND Visualization v8
=========================
Generate comparison charts for all 3 architectures.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_f1_comparison(all_results, output_dir):
    """
    Bar chart: Matched vs Truncated F1 for all 3 architectures.
    """
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
                   color='#2196F3', edgecolor='white')
    bars2 = ax.bar(x + width/2, truncated_f1s, width, label='Truncated (Baseline)',
                   color='#FF5722', edgecolor='white')
    
    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f'{h:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('F1 Score (±20ms tolerance)', fontsize=12)
    ax.set_title('LatBOND Lite v8: Matched vs Truncated Training\n'
                 f'Budget: 20ms | Dataset: 3% MAESTRO', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(matched_f1s), max(truncated_f1s)) * 1.2 + 0.05)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_comparison.png'), dpi=150)
    plt.close()
    print("[INFO] Saved f1_comparison.png")


def plot_training_curves(output_dir):
    """
    Plot training curves for all models.
    """
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
            
            # Loss
            if 'val_loss' in history:
                axes[0].plot(epochs, history['val_loss'],
                            label=label, color=color, linestyle=linestyle)
            
            # F1
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
    
    plt.suptitle('LatBOND Lite v8 - Training Curves', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print("[INFO] Saved training_curves.png")


def plot_tolerance_comparison(all_results, output_dir):
    """
    Line chart: F1 across different tolerances for each architecture.
    Shows how matched vs truncated gap changes with tolerance.
    """
    archs = ['streaming_cnn', 'causal_transformer', 'tcn']
    arch_labels = ['Streaming CNN', 'Causal Transformer', 'TCN']
    tolerances = [10, 20, 30, 50]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (arch, label) in enumerate(zip(archs, arch_labels)):
        ax = axes[idx]
        
        if arch not in all_results:
            ax.set_title(f'{label}\n(No data)')
            continue
        
        for cond, color, marker in [('matched', '#2196F3', 'o'), 
                                     ('truncated', '#FF5722', 's')]:
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
    plt.savefig(os.path.join(output_dir, 'tolerance_comparison.png'), dpi=150)
    plt.close()
    print("[INFO] Saved tolerance_comparison.png")
