"""
Scenario 2: FJSSP-W under uncertainty.

Runs uncertainty-aware GA on all competition instances x 30 runs each.
Reports robust makespan and deterioration ratio.
"""
import sys, io, contextlib, time, glob, os
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
from ga.fjssp_w.uncertainty_sim import (
    create_seeded_uncertainty_vector,
    evaluate_robustness,
)
from collections import defaultdict
import numpy as np
import pandas as pd


# --- Configuration ---
N_RUNS = 30
POP_SIZE = 30
GENERATIONS = 50
S_EVOLVE = 10            # scenarios per fitness eval during GA
S_FINAL = 50             # competition-mandated 50 for final evaluation
BALANCE_WEIGHT = 0.0

INSTANCES_DIR = "../FJSSP-W-Competition/instances/fjssp-w"
RESULTS_DIR = "results"


def worker_cv(chrom, decoder, d):
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
    print(f"Running Scenario 2 (uncertainty) on {len(instances)} instances x {N_RUNS} runs")
    print(f"Evolution S={S_EVOLVE}, final evaluation S={S_FINAL}")
    print("=" * 75)

    all_rows = []
    all_raw = []

    for idx, path in enumerate(instances):
        name = os.path.basename(path).replace(".fjs", "")
        try:
            instance = parse_competition_instance(path)
        except Exception as e:
            print(f"[{idx+1:>2}/{len(instances)}] {name}: PARSE ERROR - {e}")
            continue

        # Fixed evaluation uncertainty vector for this instance
        eval_uv = create_seeded_uncertainty_vector(
            instance['n_workers'], seed=9999
        )

        robust_means = []
        nominal_makespans = []
        deterioration_ratios = []
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
                fitness_mode='uncertainty',
                n_scenarios=S_EVOLVE,
                uncertainty_seed=seed + 1000,
                scenario_seed=seed + 2000,
                balance_weight=BALANCE_WEIGHT,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                res = ga.evolve()
            elapsed = time.time() - t0

            # Final robustness evaluation (S_FINAL = 50, per competition rules)
            metrics = evaluate_robustness(
                res['best_solution'], instance, ga.decoder,
                eval_uv, n_scenarios=S_FINAL, seed=1000 + seed
            )

            robust_means.append(metrics['mean_makespan'])
            nominal_makespans.append(metrics['nominal_makespan'])
            deterioration_ratios.append(metrics['deterioration_ratio'])
            cvs.append(worker_cv(res['best_solution'], ga.decoder,
                                 instance['durations']))
            fevs.append(res['fev_count'])
            runtimes.append(elapsed)

            all_raw.append({
                'instance': name,
                'run': seed,
                'robust_mean': metrics['mean_makespan'],
                'robust_std': metrics['std_makespan'],
                'nominal': metrics['nominal_makespan'],
                'deterioration_ratio': metrics['deterioration_ratio'],
                'worker_cv': cvs[-1],
                'fev_count': res['fev_count'],
                'runtime_s': elapsed,
            })

        arr = np.array(robust_means)
        all_rows.append({
            'instance': name,
            'n_runs': N_RUNS,
            'best_robust': float(arr.min()),
            'worst_robust': float(arr.max()),
            'mean_robust': float(arr.mean()),
            'std_robust': float(arr.std()),
            'median_robust': float(np.median(arr)),
            'mean_nominal': float(np.mean(nominal_makespans)),
            'mean_deterioration_ratio': float(np.mean(deterioration_ratios)),
            'mean_worker_cv': float(np.mean(cvs)),
            'mean_fev': float(np.mean(fevs)),
            'mean_runtime_s': float(np.mean(runtimes)),
        })

        print(f"[{idx+1:>2}/{len(instances)}] {name:<40} "
              f"robust={arr.mean():.0f} std={arr.std():.1f} "
              f"R={np.mean(deterioration_ratios):.3f} "
              f"({np.mean(runtimes):.1f}s/run)")

    df = pd.DataFrame(all_rows)
    out_path = f"{RESULTS_DIR}/scenario2_uncertainty.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")

    raw_df = pd.DataFrame(all_raw)
    raw_path = f"{RESULTS_DIR}/scenario2_uncertainty_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved {len(all_raw)} raw runs to {raw_path}")

    if all_rows:
        print(f"\nSummary:")
        print(f"  Mean robust makespan:    {df['mean_robust'].mean():.1f}")
        print(f"  Mean deterioration (R):  {df['mean_deterioration_ratio'].mean():.4f}")
        print(f"  Mean worker CV:          {df['mean_worker_cv'].mean()*100:.1f}%")
        print(f"  Total FEV (all runs):    {df['mean_fev'].sum()*N_RUNS:.0f}")


if __name__ == "__main__":
    main()