"""Generate report-ready tables from results."""
import pandas as pd
import numpy as np

# Load baseline results
df = pd.read_csv('results/baselines_raw.csv')

print("=" * 70)
print("TABLE 1: Baseline Comparison (Deterministic)")
print("=" * 70)
print(f"{'Algorithm':<20} {'Mean':>8} {'Median':>8} {'Std':>8} {'Best':>8} {'Wins':>6}")
print("-" * 70)

# Win counts
from collections import Counter
winner_rows = []
for inst in df['instance'].unique():
    inst_df = df[df['instance'] == inst]
    best_mk = inst_df['makespan'].min()
    for w in inst_df[inst_df['makespan'] == best_mk]['algorithm']:
        winner_rows.append(w)
win_counts = Counter(winner_rows)

for algo in ['Proposed_FPC_GA', 'Hybrid_GA', 'Standard_GA', 'PSO']:
    v = df[df['algorithm'] == algo]['makespan']
    print(f"{algo:<20} {v.mean():>8.1f} {v.median():>8.1f} {v.std():>8.1f} "
          f"{v.min():>8.0f} {win_counts.get(algo, 0):>6}")

print()
print("=" * 70)
print("TABLE 2: Scenario 2 — Robustness Under Uncertainty")
print("=" * 70)
s2 = pd.read_csv('results/scenario2_uncertainty.csv')
print(f"Instances: {len(s2)}")
print(f"Mean nominal makespan:   {s2['mean_nominal'].mean():>10.1f}")
print(f"Mean robust makespan:    {s2['mean_robust'].mean():>10.1f}")
print(f"Mean deterioration (R):  {s2['mean_deterioration_ratio'].mean():>10.4f}")
print(f"Min deterioration:       {s2['mean_deterioration_ratio'].min():>10.4f}")
print(f"Max deterioration:       {s2['mean_deterioration_ratio'].max():>10.4f}")

print()
print("=" * 70)
print("TABLE 3: Statistical Tests")
print("=" * 70)
from scipy.stats import mannwhitneyu, friedmanchisquare
proposed = df[df['algorithm'] == 'Proposed_FPC_GA']['makespan'].values

print(f"{'Comparison':<30} {'p-value':>10} {'Significant':>12}")
print("-" * 70)
for baseline in ['Standard_GA', 'Hybrid_GA', 'PSO']:
    b = df[df['algorithm'] == baseline]['makespan'].values
    _, p = mannwhitneyu(proposed, b, alternative='two-sided')
    sig = 'Yes' if p < 0.05 else 'No'
    print(f"Proposed vs {baseline:<18} {p:>10.4f} {sig:>12}")

groups = [df[df['algorithm'] == a]['makespan'].values for a in
          ['Proposed_FPC_GA', 'Standard_GA', 'Hybrid_GA', 'PSO']]
_, p_friedman = friedmanchisquare(*groups)
print(f"Friedman (all 4 algorithms)    {p_friedman:>10.4f} {'Yes' if p_friedman < 0.05 else 'No':>12}")

# Save tables
s2_summary = pd.DataFrame({
    'Metric': ['Instances', 'Mean Nominal', 'Mean Robust', 'Mean Deterioration',
               'Min Deterioration', 'Max Deterioration'],
    'Value': [len(s2), s2['mean_nominal'].mean(), s2['mean_robust'].mean(),
              s2['mean_deterioration_ratio'].mean(), s2['mean_deterioration_ratio'].min(),
              s2['mean_deterioration_ratio'].max()]
})
s2_summary.to_csv('results/report_table_scenario2_summary.csv', index=False)

print()
print("Saved supplementary table to results/report_table_scenario2_summary.csv")