"""
Scenario 1: Deterministic FJSSP-W.

Runs deterministic GA on all competition instances x 30 runs each.
Saves makespan, worker balance, runtime, and FEV counts per run.
"""
import sys, io, contextlib, time, glob, os
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
from collections import defaultdict
import numpy as np
import pandas as pd


# --- Configuration ---
N_RUNS = 30
POP_SIZE = 30
GENERATIONS = 50
BALANCE_WEIGHT = 0.0

INSTANCES_DIR = "../FJSSP-W-Competition/instances/fjssp-w"
RESULTS_DIR = "results"


def worker_cv(chrom, decoder, d):
    """Coefficient of variation of worker loads."""
    loads = defaultdict(float)
    for i in range(len(chrom['machine_assignment'])):
        m = chrom['machine_assignment'][i]
        w = chrom['worker_assignment'][i]
        loads[w] += d[i][m][w]
    vals = [loads.get(i, 0.0) for i in range(decoder.n_workers)]
    mean = sum(vals) / len(vals)
    if mean == 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return (var ** 0.5) / mean


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    instances = sorted(glob.glob(f"{INSTANCES_DIR}/*.fjs"))
    print(f"Running Scenario 1 (deterministic) on {len(instances)} instances x {N_RUNS} runs")
    print("=" * 75)

    all_rows = []

    for idx, path in enumerate(instances):
        name = os.path.basename(path).replace(".fjs", "")
        try:
            instance = parse_competition_instance(path)
        except Exception as e:
            print(f"[{idx+1:>2}/{len(instances)}] {name}: PARSE ERROR - {e}")
            continue

        makespans = []
        cvs = []
        fevs = []
        runtimes = []

        for seed in range(N_RUNS):
            t0 = time.time()
            ga = FJSSPW_GA(
                instance,
                population_size=POP_SIZE,
                generations=GENERATIONS,
                random_seed=seed,
                fitness_mode='deterministic',
                balance_weight=BALANCE_WEIGHT,
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

        print(f"[{idx+1:>2}/{len(instances)}] {name:<40} "
              f"best={arr.min():<8} mean={arr.mean():.0f} std={arr.std():.1f} "
              f"CV={np.mean(cvs)*100:.1f}%")

    df = pd.DataFrame(all_rows)
    out_path = f"{RESULTS_DIR}/scenario1_deterministic.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")

    # Summary
    if all_rows:
        print(f"\nSummary:")
        print(f"  Mean best makespan:   {df['best'].mean():.1f}")
        print(f"  Mean std across runs: {df['std'].mean():.2f}")
        print(f"  Mean worker CV:       {df['mean_worker_cv'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()