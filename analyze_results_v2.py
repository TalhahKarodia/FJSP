"""Extended statistical analysis with medians and effect sizes."""
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

df = pd.read_csv('results/baselines_raw.csv')

print("=== MEDIAN MAKESPAN PER ALGORITHM ===")
for algo in ['Proposed_FPC_GA', 'Standard_GA', 'Hybrid_GA', 'PSO']:
    v = df[df['algorithm'] == algo]['makespan']
    print(f"  {algo:20s}: median={v.median():.1f}, mean={v.mean():.1f}, "
          f"25th={v.quantile(0.25):.0f}, 75th={v.quantile(0.75):.0f}")

print()
print("=== EFFECT SIZE (Vargha-Delaney A) - Proposed vs baselines ===")
proposed = df[df['algorithm'] == 'Proposed_FPC_GA']['makespan'].values
for baseline in ['Standard_GA', 'Hybrid_GA', 'PSO']:
    b = df[df['algorithm'] == baseline]['makespan'].values
    n = len(proposed) * len(b)
    wins = sum(1 for x in proposed for y in b if x < y)
    ties = sum(1 for x in proposed for y in b if x == y)
    a = (wins + 0.5 * ties) / n
    effect = "large" if a > 0.71 else "medium" if a > 0.64 else "negligible"
    print(f"  vs {baseline:15s}: A = {a:.3f}  ({effect})")

print()
print("=== BEST MAKESPAN PER ALGORITHM PER INSTANCE TYPE ===")
# Group instances by prefix
df['family'] = df['instance'].apply(lambda x: x.split('_')[0])
family_stats = df.groupby(['family', 'algorithm'])['makespan'].mean().reset_index()
pivot = family_stats.pivot(index='family', columns='algorithm', values='makespan')
print(pivot.round(0).to_string())

print()
print("=== SCENARIO 2 SUMMARY (from uncertainty CSV) ===")
try:
    s2 = pd.read_csv('results/scenario2_uncertainty.csv')
    print(f"  Instances: {len(s2)}")
    print(f"  Mean robust makespan: {s2['mean_robust'].mean():.1f}")
    print(f"  Mean nominal makespan: {s2['mean_nominal'].mean():.1f}")
    print(f"  Mean deterioration ratio: {s2['mean_deterioration_ratio'].mean():.4f}")
    print(f"  Min deterioration: {s2['mean_deterioration_ratio'].min():.4f}")
    print(f"  Max deterioration: {s2['mean_deterioration_ratio'].max():.4f}")
except Exception as e:
    print(f"  Could not read S2: {e}")