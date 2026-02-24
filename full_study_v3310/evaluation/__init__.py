"""
LatBOND Evaluation Module
"""

from .metrics import evaluate_model, compare_conditions, generate_report
from .statistical_analysis import (
    paired_ttest, cohens_d, wilcoxon_test, two_way_anova,
    bootstrap_confidence_interval, bootstrap_difference_ci,
    file_level_variance, recovery_calculation, bonferroni_correction,
    run_full_analysis
)
from .visualization import (
    plot_f1_comparison, plot_training_curves, plot_tolerance_comparison,
    plot_cross_budget_comparison, plot_pareto_curves, plot_recovery_heatmap,
    generate_all_thesis_figures
)
