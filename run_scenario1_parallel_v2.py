"""
Scenario 1: Deterministic FJSSP-W (Parallelized).

Runs deterministic GA on all competition instances x 30 runs each.
Saves makespan, worker balance, runtime, and FEV counts per run.
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

# --- Configuration ---
N_RUNS = 30
POP_SIZE = 30
GENERATIONS = 50
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
    """Worker task executing a single instance run for a given seed."""
    path_str, seed, pop_size, generations, balance_weight = args
    name = Path(path_str).stem

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        instance = parse_competition_instance(path_str)
        ga = FJSSPW_GA_V2(
            instance,
            population_size=pop_size,
            generations=generations,
            random_seed=seed,
            fitness_mode='deterministic',
            balance_weight=balance_weight,
        )
        res = ga.evolve()
        cv = worker_cv(res['best_solution'], ga.decoder, instance['durations'])
    elapsed = time.time() - t0

    return {
        'instance': name,
        'seed': seed,
        'makespan': float(res['best_makespan']),
        'cv': float(cv),
        'fev': int(res['fev_count']),
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
            jobs.append((path, seed, POP_SIZE, GENERATIONS, BALANCE_WEIGHT))

    total_jobs = len(jobs)
    print(f"Running Scenario 1 (deterministic) on {len(instances)} instances x {N_RUNS} runs ({total_jobs} tasks)")
    print(f"Parallel Execution: {MAX_WORKERS} workers | POP={POP_SIZE}, GEN={GENERATIONS}")
    print("=" * 75)

    instance_results = defaultdict(list)
    total_start = time.time()

    # Execute seed runs concurrently across process workers
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_seed, job): job for job in jobs}
        completed = 0

        for future in as_completed(futures):
            completed += 1
            try:
                res = future.result()
                instance_results[res['instance']].append(res)
                if completed % max(1, total_jobs // 10) == 0 or completed == total_jobs:
                    pct = (completed / total_jobs) * 100
                    print(f"Progress: {completed}/{total_jobs} tasks completed ({pct:.1f}%)")
            except Exception as e:
                job_info = futures[future]
                print(f"  [ERROR] Job failed on {Path(job_info[0]).stem} (seed {job_info[1]}): {e}")

    # Aggregate run outputs per instance
    all_rows = []
    for name in sorted(instance_results.keys()):
        runs = instance_results[name]
        if not runs:
            continue
        makespans = [r['makespan'] for r in runs]
        cvs = [r['cv'] for r in runs]
        fevs = [r['fev'] for r in runs]
        runtimes = [r['runtime_s'] for r in runs]

        arr = np.array(makespans)
        all_rows.append({
            'instance': name,
            'n_runs': len(runs),
            'best': int(arr.min()),
            'worst': int(arr.max()),
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'median': float(np.median(arr)),
            'mean_worker_cv': float(np.mean(cvs)),
            'mean_fev': float(np.mean(fevs)),
            'mean_runtime_s': float(np.mean(runtimes)),
        })

        print(f"[{name:<35}] best={arr.min():<8} mean={arr.mean():.0f} std={arr.std():.1f} CV={np.mean(cvs)*100:.1f}%")

    total_time = time.time() - total_start
    print(f"\nAll tasks completed in {total_time/60:.2f} minutes")

    if not all_rows:
        print("ERROR: No runs completed successfully.")
        return

    df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / "scenario1_deterministic.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")

    print(f"\nSummary:")
    print(f"  Mean best makespan:   {df['best'].mean():.1f}")
    print(f"  Mean std across runs: {df['std'].mean():.2f}")
    print(f"  Mean worker CV:       {df['mean_worker_cv'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()