"""Statistical significance testing for FJSSP-W experiments."""

import numpy as np
from scipy.stats import mannwhitneyu, friedmanchisquare, wilcoxon


def mann_whitney(results_a, results_b, alpha=0.05):
    """Pairwise Mann-Whitney U test."""
    stat, p = mannwhitneyu(results_a, results_b, alternative='two-sided')
    return {
        'stat': float(stat),
        'p_value': float(p),
        'significant': bool(p < alpha),
        'n_a': len(results_a),
        'n_b': len(results_b),
        'mean_a': float(np.mean(results_a)),
        'mean_b': float(np.mean(results_b)),
    }


def friedman(results_dict, alpha=0.05):
    """Omnibus Friedman test for >=3 algorithms."""
    arrays = list(results_dict.values())
    if len(arrays) < 3:
        raise ValueError("Friedman requires >= 3 algorithms")
    if len(set(len(a) for a in arrays)) != 1:
        raise ValueError("All result lists must have equal length")
    stat, p = friedmanchisquare(*arrays)
    return {
        'stat': float(stat),
        'p_value': float(p),
        'significant': bool(p < alpha),
        'n_algorithms': len(arrays),
        'n_samples': len(arrays[0]),
    }


def wilcoxon_posthoc(results_dict, alpha=0.05):
    """Pairwise Wilcoxon signed-rank with Bonferroni correction."""
    names = list(results_dict.keys())
    k = len(names)
    n_comparisons = k * (k - 1) // 2
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            stat, p = wilcoxon(results_dict[names[i]], results_dict[names[j]])
            p_adj = min(p * n_comparisons, 1.0)
            pairs.append({
                'pair': (names[i], names[j]),
                'stat': float(stat),
                'p_value': float(p),
                'p_adjusted': float(p_adj),
                'significant': bool(p_adj < alpha),
            })
    return pairs


def vargha_delaney_a(results_a, results_b):
    """Effect size A. >0.71 large, 0.29-0.71 negligible, <0.29 reverse."""
    a = np.array(results_a)
    b = np.array(results_b)
    n = len(a) * len(b)
    wins = sum(1 for x in a for y in b if x < y)
    ties = sum(1 for x in a for y in b if x == y)
    return (wins + 0.5 * ties) / n


def summarise(results):
    """Descriptive statistics."""
    arr = np.array(results, dtype=float)
    return {
        'n': len(arr),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
        'q1': float(np.percentile(arr, 25)),
        'q3': float(np.percentile(arr, 75)),
    }