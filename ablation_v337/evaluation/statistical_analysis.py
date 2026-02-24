"""
LatBOND Statistical Analysis Module

Implements the full statistical framework from thesis Section 4.5:
1. Paired t-test (primary comparison)
2. Cohen's d (effect size)
3. Wilcoxon signed-rank test (non-parametric alternative)
4. Two-way ANOVA (budget x condition interaction) with eta-squared
5. Bonferroni correction (alpha = 0.05/5 = 0.01)
6. Bootstrap resampling (confidence intervals)
7. File-level variance reporting

All tests operate on per-file F1 scores or aggregate predictions.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
import warnings


def paired_ttest(matched_scores: np.ndarray, truncated_scores: np.ndarray,
                 alpha: float = 0.01) -> Dict:
    """
    Paired t-test comparing matched vs truncated per-file F1 scores.

    Args:
        matched_scores: Array of per-file F1 scores from matched model
        truncated_scores: Array of per-file F1 scores from truncated model
        alpha: Significance threshold (Bonferroni-corrected: 0.05/5 = 0.01)

    Returns:
        Dict with t-statistic, p-value, significant flag, improvement
    """
    assert len(matched_scores) == len(truncated_scores), \
        "Matched and truncated must have same number of files"

    t_stat, p_value = stats.ttest_rel(matched_scores, truncated_scores)

    return {
        'test': 'paired_t_test',
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < alpha),
        'alpha': alpha,
        'n_files': len(matched_scores),
        'mean_improvement': float(np.mean(matched_scores) - np.mean(truncated_scores)),
    }


def cohens_d(matched_scores: np.ndarray, truncated_scores: np.ndarray) -> Dict:
    """
    Cohen's d effect size for matched vs truncated.

    Interpretation:
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    diff = matched_scores - truncated_scores
    std = np.std(diff, ddof=1)
    if std == 0:
        d = 0.0
    else:
        d = np.mean(diff) / std

    magnitude = 'negligible'
    if abs(d) >= 0.8:
        magnitude = 'large'
    elif abs(d) >= 0.5:
        magnitude = 'medium'
    elif abs(d) >= 0.2:
        magnitude = 'small'

    return {
        'test': 'cohens_d',
        'd': float(d),
        'magnitude': magnitude,
    }


def wilcoxon_test(matched_scores: np.ndarray, truncated_scores: np.ndarray,
                  alpha: float = 0.01) -> Dict:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    Does not assume normal distribution of differences.
    """
    diff = matched_scores - truncated_scores

    # Remove zero differences (Wilcoxon cannot handle them)
    nonzero_mask = diff != 0
    if nonzero_mask.sum() < 5:
        return {
            'test': 'wilcoxon_signed_rank',
            'W_statistic': None,
            'p_value': None,
            'significant': None,
            'alpha': alpha,
            'warning': 'Too few non-zero differences for Wilcoxon test',
        }

    try:
        W_stat, p_value = stats.wilcoxon(
            matched_scores[nonzero_mask],
            truncated_scores[nonzero_mask],
            alternative='greater'  # One-sided: matched > truncated
        )
    except Exception as e:
        return {
            'test': 'wilcoxon_signed_rank',
            'W_statistic': None,
            'p_value': None,
            'significant': None,
            'alpha': alpha,
            'warning': f'Wilcoxon test failed: {str(e)}',
        }

    return {
        'test': 'wilcoxon_signed_rank',
        'W_statistic': float(W_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < alpha),
        'alpha': alpha,
        'n_nonzero_pairs': int(nonzero_mask.sum()),
    }


def two_way_anova(results_by_budget_condition: Dict[Tuple[int, str], np.ndarray],
                  budgets: List[int],
                  conditions: List[str]) -> Dict:
    """
    Two-way ANOVA testing interaction effects between budget and condition.

    Args:
        results_by_budget_condition: Dict mapping (budget_ms, condition) -> per-file F1 array
        budgets: List of latency budgets [0, 20, 40, 60, 80]
        conditions: List of conditions ['matched', 'truncated', 'buffered']

    Returns:
        Dict with F-statistics, p-values, and eta-squared for each factor
    """
    try:
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
    except ImportError:
        return {
            'test': 'two_way_anova',
            'warning': 'statsmodels not installed. Run: pip install statsmodels',
        }

    # Build a long-format DataFrame for statsmodels
    rows = []
    for budget in budgets:
        for condition in conditions:
            key = (budget, condition)
            if key not in results_by_budget_condition:
                continue
            scores = results_by_budget_condition[key]
            for s in scores:
                rows.append({'f1': float(s), 'budget': str(budget), 'condition': condition})

    df = pd.DataFrame(rows)

    if len(df) < 10:
        return {
            'test': 'two_way_anova',
            'warning': 'Insufficient data for ANOVA',
        }

    # Fit OLS model with interaction term
    model = ols('f1 ~ C(budget) * C(condition)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Extract results
    SS_total = anova_table['sum_sq'].sum()

    def extract_factor(name):
        if name in anova_table.index:
            row = anova_table.loc[name]
            return {
                'F_statistic': float(row['F']) if not pd.isna(row['F']) else 0.0,
                'p_value': float(row['PR(>F)']) if not pd.isna(row['PR(>F)']) else 1.0,
                'eta_squared': float(row['sum_sq'] / SS_total) if SS_total > 0 else 0.0,
                'df': int(row['df']),
                'significant': bool(row['PR(>F)'] < 0.05) if not pd.isna(row['PR(>F)']) else False,
            }
        return {'F_statistic': 0, 'p_value': 1.0, 'eta_squared': 0, 'df': 0, 'significant': False}

    return {
        'test': 'two_way_anova',
        'budget_factor': extract_factor('C(budget)'),
        'condition_factor': extract_factor('C(condition)'),
        'interaction': extract_factor('C(budget):C(condition)'),
        'residual': {
            'df': int(anova_table.loc['Residual', 'df']),
            'SS': float(anova_table.loc['Residual', 'sum_sq']),
        },
    }


def bootstrap_confidence_interval(scores: np.ndarray,
                                   n_resamples: int = 1000,
                                   ci_level: float = 0.95,
                                   statistic: str = 'mean',
                                   seed: int = 42) -> Dict:
    """
    Bootstrap resampling for confidence intervals on test predictions.
    Resamples per-file F1 scores to estimate CI of the mean.
    """
    rng = np.random.RandomState(seed)

    stat_fn = np.mean if statistic == 'mean' else np.median

    bootstrap_stats = []
    n = len(scores)

    for _ in range(n_resamples):
        resample_idx = rng.choice(n, size=n, replace=True)
        bootstrap_stats.append(stat_fn(scores[resample_idx]))

    bootstrap_stats = np.array(bootstrap_stats)

    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return {
        'test': 'bootstrap_ci',
        'statistic': statistic,
        'point_estimate': float(stat_fn(scores)),
        'ci_lower': float(lower),
        'ci_upper': float(upper),
        'ci_level': ci_level,
        'n_resamples': n_resamples,
        'se': float(np.std(bootstrap_stats)),
    }


def bootstrap_difference_ci(matched_scores: np.ndarray,
                             truncated_scores: np.ndarray,
                             n_resamples: int = 1000,
                             ci_level: float = 0.95,
                             seed: int = 42) -> Dict:
    """
    Bootstrap CI for the DIFFERENCE (matched - truncated).
    If CI excludes zero, the improvement is significant.
    """
    rng = np.random.RandomState(seed)

    diffs = matched_scores - truncated_scores
    n = len(diffs)

    bootstrap_diffs = []
    for _ in range(n_resamples):
        resample_idx = rng.choice(n, size=n, replace=True)
        bootstrap_diffs.append(np.mean(diffs[resample_idx]))

    bootstrap_diffs = np.array(bootstrap_diffs)

    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return {
        'test': 'bootstrap_difference_ci',
        'mean_difference': float(np.mean(diffs)),
        'ci_lower': float(lower),
        'ci_upper': float(upper),
        'ci_level': ci_level,
        'excludes_zero': bool(lower > 0 or upper < 0),
        'n_resamples': n_resamples,
    }


def file_level_variance(file_scores: np.ndarray) -> Dict:
    """
    Report file-level F1 variance as required by thesis Section 4.5.
    """
    return {
        'n_files': len(file_scores),
        'mean': float(np.mean(file_scores)),
        'std': float(np.std(file_scores, ddof=1)) if len(file_scores) > 1 else 0.0,
        'variance': float(np.var(file_scores, ddof=1)) if len(file_scores) > 1 else 0.0,
        'min': float(np.min(file_scores)),
        'max': float(np.max(file_scores)),
        'median': float(np.median(file_scores)),
        'iqr': float(np.percentile(file_scores, 75) - np.percentile(file_scores, 25)),
    }


def recovery_calculation(matched_f1: float, truncated_f1: float,
                          buffered_f1: float) -> Dict:
    """
    Calculate how much of the buffered-truncated gap matched recovers.

    recovery% = (matched - truncated) / (buffered - truncated) x 100

    This shows what fraction of accuracy lost to real-time constraints
    is recovered by matched training.
    """
    gap = buffered_f1 - truncated_f1
    improvement = matched_f1 - truncated_f1

    if gap <= 0:
        recovery_pct = None
        warning = "Buffered <= truncated (no gap to recover)"
    else:
        recovery_pct = (improvement / gap) * 100
        warning = None

    return {
        'matched_f1': float(matched_f1),
        'truncated_f1': float(truncated_f1),
        'buffered_f1': float(buffered_f1),
        'improvement_over_truncated': float(improvement),
        'buffered_truncated_gap': float(gap),
        'recovery_percentage': float(recovery_pct) if recovery_pct is not None else None,
        'warning': warning,
    }


def bonferroni_correction(p_values: List[float], n_comparisons: int = 5,
                           family_alpha: float = 0.05) -> Dict:
    """
    Bonferroni correction for multiple latency budget comparisons.
    alpha_corrected = 0.05 / 5 = 0.01 per budget
    """
    alpha_corrected = family_alpha / n_comparisons

    corrected_results = []
    for i, p in enumerate(p_values):
        corrected_results.append({
            'budget_index': i,
            'raw_p_value': float(p),
            'corrected_alpha': float(alpha_corrected),
            'significant_after_correction': bool(p < alpha_corrected),
        })

    return {
        'test': 'bonferroni_correction',
        'family_alpha': family_alpha,
        'n_comparisons': n_comparisons,
        'corrected_alpha': float(alpha_corrected),
        'results': corrected_results,
        'n_significant': sum(1 for r in corrected_results if r['significant_after_correction']),
    }


def run_full_analysis(all_results: Dict, config) -> Dict:
    """
    Run the complete statistical analysis suite on all 42 model results.

    Args:
        all_results: Dict mapping (arch, budget, condition) -> test results dict
                     Each test results dict has 'file_f1_scores', 'aggregate_f1', etc.
        config: LatBONDConfig

    Returns:
        Comprehensive analysis dict ready for thesis reporting
    """
    analysis = {
        'per_budget_comparisons': {},
        'per_architecture_comparisons': {},
        'anova': None,
        'recovery': {},
        'bonferroni': None,
    }

    architectures = ['streaming_cnn', 'causal_transformer', 'tcn']
    budgets = config.latency_budgets_ms

    # === Per-budget analysis (matched vs truncated, averaged across architectures) ===
    budget_p_values = []

    for budget in budgets:
        matched_all = []
        truncated_all = []

        for arch in architectures:
            m_key = (arch, budget, 'matched')
            t_key = (arch, budget, 'truncated')

            if m_key in all_results and t_key in all_results:
                matched_all.extend(all_results[m_key].get('file_f1_scores', []))
                truncated_all.extend(all_results[t_key].get('file_f1_scores', []))

        if len(matched_all) == 0 or len(truncated_all) == 0:
            continue

        matched_arr = np.array(matched_all)
        truncated_arr = np.array(truncated_all)

        # Ensure equal length for paired tests
        min_len = min(len(matched_arr), len(truncated_arr))
        matched_arr = matched_arr[:min_len]
        truncated_arr = truncated_arr[:min_len]

        budget_analysis = {
            'budget_ms': budget,
            'n_files': len(matched_arr),
            'matched_file_stats': file_level_variance(matched_arr),
            'truncated_file_stats': file_level_variance(truncated_arr),
            'paired_ttest': paired_ttest(matched_arr, truncated_arr,
                                          alpha=config.bonferroni_alpha),
            'cohens_d': cohens_d(matched_arr, truncated_arr),
            'wilcoxon': wilcoxon_test(matched_arr, truncated_arr,
                                       alpha=config.bonferroni_alpha),
            'bootstrap_diff': bootstrap_difference_ci(
                matched_arr, truncated_arr,
                n_resamples=config.bootstrap_n_resamples,
                ci_level=config.bootstrap_ci_level,
                seed=config.random_seed,
            ),
            'matched_bootstrap': bootstrap_confidence_interval(
                matched_arr,
                n_resamples=config.bootstrap_n_resamples,
                ci_level=config.bootstrap_ci_level,
                seed=config.random_seed,
            ),
            'truncated_bootstrap': bootstrap_confidence_interval(
                truncated_arr,
                n_resamples=config.bootstrap_n_resamples,
                ci_level=config.bootstrap_ci_level,
                seed=config.random_seed,
            ),
        }

        budget_p_values.append(budget_analysis['paired_ttest']['p_value'])
        analysis['per_budget_comparisons'][budget] = budget_analysis

    # === Bonferroni correction across budgets ===
    if budget_p_values:
        analysis['bonferroni'] = bonferroni_correction(
            budget_p_values,
            n_comparisons=len(budgets),
            family_alpha=0.05,
        )

    # === Per-architecture analysis ===
    for arch in architectures:
        arch_analysis = {}
        for budget in budgets:
            m_key = (arch, budget, 'matched')
            t_key = (arch, budget, 'truncated')

            if m_key in all_results and t_key in all_results:
                m_scores = np.array(all_results[m_key].get('file_f1_scores', []))
                t_scores = np.array(all_results[t_key].get('file_f1_scores', []))

                if len(m_scores) > 0 and len(t_scores) > 0:
                    min_len = min(len(m_scores), len(t_scores))
                    arch_analysis[budget] = {
                        'matched_f1': all_results[m_key].get('aggregate_f1', 0),
                        'truncated_f1': all_results[t_key].get('aggregate_f1', 0),
                        'improvement': all_results[m_key].get('aggregate_f1', 0) - all_results[t_key].get('aggregate_f1', 0),
                        'paired_ttest': paired_ttest(m_scores[:min_len], t_scores[:min_len], alpha=config.bonferroni_alpha),
                        'cohens_d': cohens_d(m_scores[:min_len], t_scores[:min_len]),
                    }

        analysis['per_architecture_comparisons'][arch] = arch_analysis

    # === Recovery calculations (where buffered is available) ===
    for arch in architectures:
        for budget in budgets:
            m_key = (arch, budget, 'matched')
            t_key = (arch, budget, 'truncated')
            b_key = (arch, budget, 'buffered')

            if all([k in all_results for k in (m_key, t_key, b_key)]):
                analysis['recovery'][(arch, budget)] = recovery_calculation(
                    matched_f1=all_results[m_key].get('aggregate_f1', 0),
                    truncated_f1=all_results[t_key].get('aggregate_f1', 0),
                    buffered_f1=all_results[b_key].get('aggregate_f1', 0),
                )

    # === Two-way ANOVA ===
    anova_data = {}
    for budget in budgets:
        for condition in config.training_conditions:
            if not config.is_valid_training(budget, condition):
                continue
            scores = []
            for arch in architectures:
                key = (arch, budget, condition)
                if key in all_results:
                    scores.extend(all_results[key].get('file_f1_scores', []))
            if scores:
                anova_data[(budget, condition)] = np.array(scores)

    if len(anova_data) > 3:  # Need enough groups
        analysis['anova'] = two_way_anova(
            anova_data, budgets, config.training_conditions
        )

    return analysis
