"""Test impact of balance_weight on makespan, robustness, and worker balance."""
import sys
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
from ga.fjssp_w.uncertainty_sim import (
    create_seeded_uncertainty_vector,
    evaluate_robustness,
)
import glob, io, contextlib
from collections import defaultdict
import numpy as np


def compute_worker_cv(chrom, decoder, d):
    """Coefficient of variation of worker loads (as fraction, not percent)."""
    loads = defaultdict(float)
    for op_idx in range(len(chrom['machine_assignment'])):
        m = chrom['machine_assignment'][op_idx]
        w = chrom['worker_assignment'][op_idx]
        loads[w] += d[op_idx][m][w]
    vals = [loads.get(i, 0.0) for i in range(decoder.n_workers)]
    mean = sum(vals) / len(vals)
    if mean == 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return (var ** 0.5) / mean


# Load instance
path = glob.glob("src/FJSSP-W-Competition/instances/fjssp-w/*Fattahi_20*")[0]
instance = parse_competition_instance(path)
print(f"Instance: {path}")
print(f"Jobs={instance['n_jobs']}, Machines={instance['n_machines']}, "
      f"Workers={instance['n_workers']}, Ops={instance['n_operations']}\n")

# Fixed evaluation uncertainty vector (same for all configs)
eval_uv = create_seeded_uncertainty_vector(instance['n_workers'], seed=9999)

# Test different balance weights
balance_weights = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5]

print("=" * 75)
print(f"{'λ':<8} {'Nominal':<10} {'Robust':<10} {'R':<8} {'Worker CV':<12} {'FEV':<8}")
print("=" * 75)

results = []
for bw in balance_weights:
    ga = FJSSPW_GA(
        instance,
        population_size=30,
        generations=30,
        random_seed=0,
        fitness_mode='uncertainty',
        n_scenarios=10,
        uncertainty_seed=42,
        scenario_seed=42,
        balance_weight=bw,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        res = ga.evolve()

    # Nominal makespan of the best (robust) chromosome
    nominal = ga.decoder.decode(res['best_solution'])['makespan']

    # Evaluate robustness on a fixed test set
    metrics = evaluate_robustness(
        res['best_solution'], instance, ga.decoder,
        eval_uv, n_scenarios=50, seed=1000
    )

    # Worker balance CV
    cv = compute_worker_cv(res['best_solution'], ga.decoder, instance['durations'])

    print(f"{bw:<8} {nominal:<10} {metrics['mean_makespan']:<10.1f} "
          f"{metrics['deterioration_ratio']:<8.4f} {cv*100:<12.2f} {ga.fev_count:<8}")

    results.append({
        'bw': bw,
        'nominal': nominal,
        'robust': metrics['mean_makespan'],
        'R': metrics['deterioration_ratio'],
        'cv': cv,
    })

# Interpretation
print("\n" + "=" * 75)
print("Interpretation")
print("=" * 75)

baseline = results[0]  # λ=0.0
print(f"Baseline (λ=0.0): robust={baseline['robust']:.1f}, CV={baseline['cv']*100:.1f}%\n")

for r in results[1:]:
    robust_pct = (r['robust'] - baseline['robust']) / baseline['robust'] * 100
    cv_pct = (r['cv'] - baseline['cv']) / baseline['cv'] * 100
    print(f"  λ={r['bw']:<5}: robust {robust_pct:+.2f}%  CV {cv_pct:+.1f}%")

# Find sweet spot: robust within +0.5% of baseline, CV minimized
print()
good = [r for r in results if (r['robust'] - baseline['robust']) / baseline['robust'] < 0.005]
if good:
    best = min(good, key=lambda r: r['cv'])
    print(f"Sweet spot: λ={best['bw']}  (robust={best['robust']:.1f}, CV={best['cv']*100:.1f}%)")
else:
    print("No config within +0.5% robust. Consider accepting λ=0.")