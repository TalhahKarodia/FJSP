"""
Scenario 1: Deterministic FJSSP-W.

Runs deterministic GA on 30 competition instances × 30 runs.
Saves makespan, worker balance, runtime, FEV counts.
"""
import sys
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
import glob, io, contextlib, time
from collections import defaultdict
import numpy as np
import pandas as pd
import os

INSTANCES = sorted(glob.glob("src/FJSSP-W-Competition/instances/fjssp-w/*.fjs"))
N_RUNS = 30

print(f"Running Scenario 1 (deterministic) on {len(INSTANCES)} instances × {N_RUNS} runs")
print("=" * 70)

def worker_cv(chrom, decoder, d):
    loads = defaultdict(float)
    for i in range(len(chrom['machine_assignment'])):
        m = chrom['machine_assignment'][i]
        w = chrom['worker_assignment'][i]
        loads[w] += d[i][m][w]
    vals = [loads.get(i, 0.0) for i in range(decoder.n_workers)]
    mean = sum(vals) / len(vals)
    if mean == 0: return 0.0
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return (var ** 0.5) / mean

all_rows = []
os.makedirs("results", exist_ok=True)

for idx, path in enumerate(INSTANCES):
    name = path.replace("\\", "/").split("/")[-1].replace(".fjs", "")
    instance = parse_competition_instance(path)

    makespans = []
    cvs = []
    fevs = []
    runtimes = []

    for seed in range(N_RUNS):
        t0 = time.time()
        ga = FJSSPW_GA(
            instance,
            population_size=30,
            generations=50,
            random_seed=seed,
            fitness_mode='deterministic',
            balance_weight=0.1,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            res = ga.evolve()
        elapsed = time.time() - t0

        makespans.append(res['best_makespan'])
        cvs.append(worker_cv(res['best_solution'], ga.decoder, instance['durations']))
        fevs.append(res['fev_count'])
        runtimes.append(elapsed)

    arr = np.array(makespans)
    all_rows.append({
        'instance': name,
        'n_runs': N_RUNS,
        'best': int(arr.min()),
        'worst': int(arr.max()),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'median': float(np.median(arr)),
        'mean_worker_cv': float(np.mean(cvs)),
        'mean_fev': float(np.mean(fevs)),
        'mean_runtime_s': float(np.mean(runtimes)),
    })
    print(f"[{idx+1:>2}/{len(INSTANCES)}] {name:<40} "
          f"best={arr.min():<8} mean={arr.mean():.0f} std={arr.std():.1f}")

df = pd.DataFrame(all_rows)
df.to_csv("results/scenario1_deterministic.csv", index=False)
print(f"\nSaved to results/scenario1_deterministic.csv")
