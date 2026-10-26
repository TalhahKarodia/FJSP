"""Statistical analysis of baselines vs proposed algorithm."""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, friedmanchisquare

df = pd.read_csv('results/baselines_raw.csv')
proposed = df[df['algorithm'] == 'Proposed_FPC_GA']['makespan'].values

print("=== MANN-WHITNEY U TESTS (Proposed vs each baseline) ===")
print(f"Proposed mean: {proposed.mean():.1f}")
print()

for baseline in ['Standard_GA', 'Hybrid_GA', 'PSO']:
    b = df[df['algorithm'] == baseline]['makespan'].values
    stat, p = mannwhitneyu(proposed, b, alternative='two-sided')
    sig = 'SIGNIFICANT' if p < 0.05 else 'not significant'
    print(f"  vs {baseline:15s}: p = {p:.4f}  ({sig})  mean = {b.mean():.1f}")

print()
print("=== FRIEDMAN TEST (all 4 algorithms) ===")
groups = [df[df['algorithm'] == a]['makespan'].values for a in
          ['Proposed_FPC_GA', 'Standard_GA', 'Hybrid_GA', 'PSO']]
stat, p = friedmanchisquare(*groups)
print(f"  Friedman stat = {stat:.2f}, p = {p:.4f}")
if p < 0.05:
    print("  -> Significant differences across algorithms")
else:
    print("  -> No significant differences")

print()
print("=== MEAN MAKESPAN PER ALGORITHM ===")
for algo in ['Proposed_FPC_GA', 'Standard_GA', 'Hybrid_GA', 'PSO']:
    v = df[df['algorithm'] == algo]['makespan']
    print(f"  {algo:20s}: mean={v.mean():.1f}, std={v.std():.1f}, min={v.min():.0f}, max={v.max():.0f}")

print()
print("=== WIN COUNT PER ALGORITHM (best per instance) ===")
best_per_instance = df.groupby(['instance'])['makespan'].min().reset_index()
best_per_instance.columns = ['instance', 'best_overall']
winner_rows = []
for inst in df['instance'].unique():
    inst_df = df[df['instance'] == inst]
    best_mk = inst_df['makespan'].min()
    winners = inst_df[inst_df['makespan'] == best_mk]['algorithm'].tolist()
    for w in winners:
        winner_rows.append(w)

from collections import Counter
win_counts = Counter(winner_rows)
for algo in ['Proposed_FPC_GA', 'Standard_GA', 'Hybrid_GA', 'PSO']:
    print(f"  {algo:20s}: {win_counts.get(algo, 0)} wins")