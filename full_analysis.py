"""Combined Statistical Analyzer (Analyser A + Analyser B)

Exports all experimental results, statistical tests, family breakdowns, and
uncertainty summaries into CSV files.
"""

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon
from pathlib import Path


def vargha_delaney_A(x, y):
  """Calculates A_xy: probability that Proposed (x) < Baseline (y)."""
  m, n = len(x), len(y)
  r = sum(1 for xi in x for yi in y if xi < yi) + 0.5 * sum(
      1 for xi in x for yi in y if xi == yi
  )
  return r / (m * n)

# Create summary folder if it does not exist
results_dir_path = Path(__file__).parent.resolve()/"results"/"summary"
results_dir_path.mkdir(parents=True, exist_ok=True)

print("Analysis started...")
# Load dataset
df_raw = pd.read_csv('results/baselines_raw.csv')

family_map = {
    '0': 'Behnke-Geiger',
    '1': 'Brandimarte',
    '2a': 'Hurink sdata',
    '2b': 'Hurink edata',
    '2c': 'Hurink rdata',
    '2d': 'Hurink vdata',
    '3': 'DPpaulli',
    '4': 'Chambers-Barnes',
    '5': 'Kacem',
    '6': 'Fattahi',
}

df_raw['family_code'] = df_raw['instance'].apply(lambda x: x.split('_')[0])
df_raw['family_name'] = df_raw['family_code'].map(family_map)

# Statistics summary per Algorithm
pvt_mean = df_raw.groupby(['instance', 'algorithm'])['makespan'].mean().unstack()
pvt_best = df_raw.groupby(['instance', 'algorithm'])['makespan'].min().unstack()
pvt_time = (
    df_raw.groupby(['instance', 'algorithm'])['runtime_s'].mean().unstack()
)

algos = ['Proposed_FPC_GA', 'Standard_GA', 'Hybrid_GA', 'PSO']
mean_ranks = pvt_mean.rank(axis=1, method='average').mean(axis=0)
wins_mean = pvt_mean.idxmin(axis=1).value_counts()
wins_best = pvt_best.idxmin(axis=1).value_counts()

summary_rows = []
for algo in algos:
  v_raw = df_raw[df_raw['algorithm'] == algo]['makespan']
  summary_rows.append({
      'Algorithm': algo,
      'Mean Makespan': round(v_raw.mean(), 1),
      'Median Makespan': round(v_raw.median(), 1),
      'Std Dev': round(v_raw.std(), 1),
      '25th Percentile': round(v_raw.quantile(0.25), 1),
      '75th Percentile': round(v_raw.quantile(0.75), 1),
      'Min Makespan': round(v_raw.min(), 0),
      'Max Makespan': round(v_raw.max(), 0),
      'Mean Rank': round(mean_ranks[algo], 2),
      'Wins (Mean Makespan)': int(wins_mean.get(algo, 0)),
      'Wins (Best Run)': int(wins_best.get(algo, 0)),
      'Mean Runtime (s)': round(pvt_time[algo].mean(), 2),
  })

pd.DataFrame(summary_rows).to_csv(
    'results/summary/summary_statistics_by_algorithm.csv', index=False
)

# Pairwise Tests & Effect Sizes (FPC-GA vs Baselines)
m_comparisons = 3
pairwise_rows = []

for baseline in ['Standard_GA', 'Hybrid_GA', 'PSO']:
  wins, ties, losses = 0, 0, 0
  a_list = []

  for inst in df_raw['instance'].unique():
    fpc_runs = df_raw[
        (df_raw['instance'] == inst) & (df_raw['algorithm'] == 'Proposed_FPC_GA')
    ]['makespan'].values
    base_runs = df_raw[
        (df_raw['instance'] == inst) & (df_raw['algorithm'] == baseline)
    ]['makespan'].values

    stat_u, p_u = mannwhitneyu(fpc_runs, base_runs, alternative='two-sided')
    if p_u < 0.05:
      if fpc_runs.mean() < base_runs.mean():
        wins += 1
      else:
        losses += 1
    else:
      ties += 1

    a_list.append(vargha_delaney_A(fpc_runs, base_runs))

  w_stat, w_p = wilcoxon(pvt_mean['Proposed_FPC_GA'], pvt_mean[baseline])
  w_p_bonf = min(1.0, w_p * m_comparisons)
  mean_a = np.mean(a_list)
  effect_str = (
      'large'
      if mean_a > 0.71
      else 'medium'
      if mean_a > 0.64
      else 'small'
      if mean_a > 0.56
      else 'negligible'
  )

  pairwise_rows.append({
      'Baseline Algorithm': baseline,
      'Proposed Mean': round(pvt_mean['Proposed_FPC_GA'].mean(), 1),
      'Baseline Mean': round(pvt_mean[baseline].mean(), 1),
      'Per-Instance W/T/L (U test)': f'{wins} / {ties} / {losses}',
      'Wilcoxon W Stat': round(w_stat, 1),
      'Wilcoxon Raw p-value': (
          round(w_p, 4) if w_p >= 0.0001 else '< 0.0001'
      ),
      'Wilcoxon Bonferroni p-value': (
          round(w_p_bonf, 4) if w_p_bonf >= 0.0001 else '< 0.0001'
      ),
      'Significance (alpha=0.05)': (
          'Significant' if w_p_bonf < 0.05 else 'Not Significant'
      ),
      'Vargha-Delaney A': round(mean_a, 3),
      'Effect Size': effect_str,
  })

pd.DataFrame(pairwise_rows).to_csv(
    'results/summary/pairwise_statistical_tests_and_effect_sizes.csv', index=False
)

# Performance by benchmark family
fam_stats = (
    df_raw.groupby(['family_code', 'family_name', 'algorithm'])['makespan']
    .mean()
    .unstack()
    .reset_index()
)
fam_stats['Best Algorithm'] = fam_stats[algos].idxmin(axis=1)
fam_stats['FPC-GA Improvement vs Standard GA (%)'] = (
    (fam_stats['Standard_GA'] - fam_stats['Proposed_FPC_GA'])
    / fam_stats['Standard_GA']
    * 100
).round(2)

df_family = fam_stats.sort_values(
    'FPC-GA Improvement vs Standard GA (%)', ascending=False
)
df_family.to_csv('results/summary/performance_by_benchmark_family.csv', index=False)

# Friedman omnibus test
f_stat, f_p = friedmanchisquare(
    pvt_mean['Proposed_FPC_GA'],
    pvt_mean['Standard_GA'],
    pvt_mean['Hybrid_GA'],
    pvt_mean['PSO'],
)

pd.DataFrame([{
    'Test Name': 'Friedman Omnibus Test',
    'Instances Count (N)': len(pvt_mean),
    'Algorithms Compared (k)': len(algos),
    'Chi-Square Statistic': round(f_stat, 2),
    'Degrees of Freedom': len(algos) - 1,
    'p-value': round(f_p, 6) if f_p >= 0.000001 else '< 0.000001',
    'Significant Difference': 'Yes' if f_p < 0.05 else 'No',
}]).to_csv('friedman_test_results.csv', index=False)

# Scenario 2 robustness summary
try:
  df_s2 = pd.read_csv('results/scenario2_uncertainty.csv')
  s2_summary = pd.DataFrame([
      {'Metric': 'Total Test Instances', 'Value': len(df_s2)},
      {
          'Metric': 'Mean Nominal Makespan',
          'Value': round(df_s2['mean_nominal'].mean(), 1),
      },
      {
          'Metric': 'Mean Robust Makespan',
          'Value': round(df_s2['mean_robust'].mean(), 1),
      },
      {
          'Metric': 'Mean Deterioration Ratio (R)',
          'Value': round(df_s2['mean_deterioration_ratio'].mean(), 4),
      },
      {
          'Metric': 'Min Deterioration Ratio',
          'Value': round(df_s2['mean_deterioration_ratio'].min(), 4),
      },
      {
          'Metric': 'Max Deterioration Ratio',
          'Value': round(df_s2['mean_deterioration_ratio'].max(), 4),
      },
  ])
  s2_summary.to_csv('results/summary/scenario2_robustness_summary.csv', index=False)
except Exception as e:
  print('Scenario 2 CSV not loaded:', e)

print(f"Operation complete. Results saved to {results_dir_path}")