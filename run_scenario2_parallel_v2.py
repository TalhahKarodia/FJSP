"""
Scenario 2: FJSSP-W under uncertainty (Parallelized).

Runs uncertainty-aware GA on all competition instances x 30 runs each.
Reports robust makespan and deterioration ratio.
"""

import sys
import io
import contextlib
import time
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
import pandas as pd

# Fix relative import paths
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / 'src'))

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w_v2 import FJSSPW_GA_V2
from ga.fjssp_w.uncertainty_sim import (
    create_seeded_uncertainty_vector,
    evaluate_robustness,
)

# --- Configuration ---
N_RUNS = 30
POP_SIZE = 30
GENERATIONS = 50
S_EVOLVE = 10          # scenarios per fitness eval during GA
S_FINAL = 50           # competition-mandated 50 for final evaluation
BALANCE_WEIGHT = 0.0

MAX_WORKERS = os.cpu_count() - 2 or 4

INSTANCES_DIR = "data-sample/FJSSP-W-Competition/instances/fjssp-w"
RESULTS_DIR = SCRIPT_DIR / "results"


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


def run_single_seed(args):
    """Worker task executing a single uncertainty run for a given seed."""
    path_str, seed, pop_size, generations, s_evolve, s_final, balance_weight = args
    name = Path(path_str).stem

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        instance = parse_competition_instance(path_str)

        # Fixed evaluation uncertainty vector for this instance
        eval_uv = create_seeded_uncertainty_vector(
            instance['n_workers'], seed=9999
        )

        ga = FJSSPW_GA_V2(
            instance,
            population_size=pop_size,
            generations=generations,
            random_seed=seed,
            fitness_mode='uncertainty',
            n_scenarios=s_evolve,
            uncertainty_seed=seed + 1000,
            scenario_seed=seed + 2000,
            balance_weight=balance_weight,
        )
        res = ga.evolve()

        # Final robustness evaluation
        metrics = evaluate_robustness(
            res['best_solution'], instance, ga.decoder,
            eval_uv, n_scenarios=s_final, seed=1000 + seed
        )
        cv = worker_cv(res['best_solution'], ga.decoder, instance['durations'])

    elapsed = time.time() - t0

    return {
        'instance': name,
        'run': seed,
        'robust_mean': float(metrics['mean_makespan']),
        'robust_std': float(metrics['std_makespan']),
        'nominal': float(metrics['nominal_makespan']),
        'deterioration_ratio': float(metrics['deterioration_ratio']),
        'worker_cv': float(cv),
        'fev_count': int(res['fev_count']),
        'runtime_s': elapsed,
    }


def find_instances(target_dir_str):
    """Resolves target directory relative to script and recursively finds .fjs files."""
    target_path = SCRIPT_DIR / target_dir_str
    if not target_path.exists():
        target_path = Path(target_dir_str)

    if not target_path.exists():
        return [], target_path.resolve()

    files = sorted([str(p) for p in target_path.rglob("*.fjs")])
    return files, target_path.resolve()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    instances, resolved_path = find_instances(INSTANCES_DIR)

    if not instances:
        print(f"ERROR: No .fjs files found in directory: {resolved_path}")
        return

    # Build work tasks for all (instance, seed) combinations
    jobs = []
    for path in instances:
        for seed in range(N_RUNS):
            jobs.append((path, seed, POP_SIZE, GENERATIONS, S_EVOLVE, S_FINAL, BALANCE_WEIGHT))

    total_jobs = len(jobs)
    print(f"Running Scenario 2 (uncertainty) on {len(instances)} instances x {N_RUNS} runs ({total_jobs} tasks)")
    print(f"Evolution S={S_EVOLVE}, final evaluation S={S_FINAL}")
    print(f"Parallel Execution: {MAX_WORKERS} workers | POP={POP_SIZE}, GEN={GENERATIONS}")
    print("=" * 75)

    all_raw = []
    total_start = time.time()

    # Execute seed runs concurrently across process workers
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_seed, job): job for job in jobs}
        completed = 0

        for future in as_completed(futures):
            completed += 1
            try:
                res = future.result()
                all_raw.append(res)
                if completed % max(1, total_jobs // 10) == 0 or completed == total_jobs:
                    pct = (completed / total_jobs) * 100
                    print(f"Progress: {completed}/{total_jobs} tasks completed ({pct:.1f}%)")
            except Exception as e:
                job_info = futures[future]
                print(f"  [ERROR] Job failed on {Path(job_info[0]).stem} (seed {job_info[1]}): {e}")

    total_time = time.time() - total_start
    print(f"\nAll tasks completed in {total_time/60:.2f} minutes")

    if not all_raw:
        print("ERROR: No runs completed successfully.")
        return

    # Save raw outputs
    raw_df = pd.DataFrame(all_raw)
    raw_path = RESULTS_DIR / "scenario2_uncertainty_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved {len(all_raw)} raw runs to {raw_path}")

    # Aggregate outputs per instance
    all_rows = []
    grouped = raw_df.groupby('instance')
    for name, group in sorted(grouped):
        robust_arr = group['robust_mean'].values
        all_rows.append({
            'instance': name,
            'n_runs': len(group),
            'best_robust': float(robust_arr.min()),
            'worst_robust': float(robust_arr.max()),
            'mean_robust': float(robust_arr.mean()),
            'std_robust': float(robust_arr.std()),
            'median_robust': float(np.median(robust_arr)),
            'mean_nominal': float(group['nominal'].mean()),
            'mean_deterioration_ratio': float(group['deterioration_ratio'].mean()),
            'mean_worker_cv': float(group['worker_cv'].mean()),
            'mean_fev': float(group['fev_count'].mean()),
            'mean_runtime_s': float(group['runtime_s'].mean()),
        })

        print(
            f"[{name:<35}] robust={robust_arr.mean():.0f} std={robust_arr.std():.1f} "
            f"R={group['deterioration_ratio'].mean():.3f} ({group['runtime_s'].mean():.1f}s/run)"
        )

    df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / "scenario2_uncertainty.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(all_rows)} summary rows to {out_path}")

    print(f"\nSummary:")
    print(f"  Mean robust makespan:    {df['mean_robust'].mean():.1f}")
    print(f"  Mean deterioration (R):  {df['mean_deterioration_ratio'].mean():.4f}")
    print(f"  Mean worker CV:          {df['mean_worker_cv'].mean()*100:.1f}%")
    print(f"  Total FEV (all runs):    {df['mean_fev'].sum()*N_RUNS:.0f}")


if __name__ == "__main__":
    main()