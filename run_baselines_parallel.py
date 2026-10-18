"""
Run all baselines + proposed algorithm on all competition instances in parallel.
Produces results/baselines_raw.csv and results/baselines_summary.csv

Parallelized version using concurrent.futures.ProcessPoolExecutor.
"""

import sys
import io
import contextlib
import time
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# Fix relative pathing relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / 'src'))

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
from baselines.standard_ga import StandardGA
from baselines.hybrid_ga import HybridGA
from baselines.pso import DiscretePSO

# --- Execution Settings ---
N_RUNS = 30
POP = 30
GEN = 50
BALANCE_W = 0.0

# Number of parallel workers (defaults to total physical/logical CPU cores)
MAX_WORKERS = os.cpu_count() or 4

INSTANCES_DIR = "data-sample/FJSSP-W-Competition/instances/fjssp-w"
RESULTS_DIR = SCRIPT_DIR / "results"

ALGORITHMS = ['Proposed_FPC_GA', 'Standard_GA', 'Hybrid_GA', 'PSO']


def run_single_job(args):
    """Worker function executed across process pool workers."""
    path_str, algo_name, seed, pop, gen, balance_w = args
    name = Path(path_str).stem

    t0 = time.time()
    # Suppress output from individual algorithm iterations
    with contextlib.redirect_stdout(io.StringIO()):
        inst = parse_competition_instance(path_str)

        if algo_name == 'Proposed_FPC_GA':
            ga = FJSSPW_GA(
                inst, population_size=pop, generations=gen, random_seed=seed,
                fitness_mode='deterministic', balance_weight=balance_w,
            )
            r = ga.evolve()
        elif algo_name == 'Standard_GA':
            ga = StandardGA(inst, population_size=pop, generations=gen, random_seed=seed)
            r = ga.evolve()
        elif algo_name == 'Hybrid_GA':
            ga = HybridGA(inst, population_size=pop, generations=gen, random_seed=seed)
            r = ga.evolve()
        elif algo_name == 'PSO':
            pso = DiscretePSO(inst, swarm_size=pop, iterations=gen, random_seed=seed)
            r = pso.evolve()
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        mk = float(r['best_makespan'])
        fev = int(r['fev_count'])

    elapsed = time.time() - t0
    return {
        'instance': name,
        'algorithm': algo_name,
        'run': seed,
        'makespan': mk,
        'fev_count': fev,
        'runtime_s': elapsed,
    }


def find_instances(target_dir_str):
    """Resolves target directory and recursively finds all .fjs files."""
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
        print("Please verify the INSTANCES_DIR setting.")
        return

    # Build work items for all (instance, algorithm, seed) combinations
    jobs = []
    for path in instances:
        for algo_name in ALGORITHMS:
            for seed in range(N_RUNS):
                jobs.append((path, algo_name, seed, POP, GEN, BALANCE_W))

    total_jobs = len(jobs)
    print(f"Loaded {len(instances)} instances from: {resolved_path}")
    print(f"Running {len(ALGORITHMS)} algorithms x {N_RUNS} runs ({total_jobs} total tasks)")
    print(f"Parallel Execution: {MAX_WORKERS} workers | POP={POP}, GEN={GEN}")
    print("=" * 75)

    all_raw = []
    total_start = time.time()

    # Process jobs concurrently
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_job, job): job for job in jobs}
        completed = 0

        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
                all_raw.append(result)
                if completed % max(1, (total_jobs // 10)) == 0 or completed == total_jobs:
                    pct = (completed / total_jobs) * 100
                    print(f"Progress: {completed}/{total_jobs} tasks completed ({pct:.1f}%)")
            except Exception as e:
                job_info = futures[future]
                print(f"  [ERROR] Job failed {job_info[1]} on {Path(job_info[0]).stem} (seed {job_info[2]}): {e}")

    total_time = time.time() - total_start
    print(f"\nAll tasks completed in {total_time / 60:.2f} minutes")

    if not all_raw:
        print("ERROR: No runs completed successfully. Summary files were not generated.")
        return

    # Save outputs
    df = pd.DataFrame(all_raw)
    raw_path = RESULTS_DIR / "baselines_raw.csv"
    df.to_csv(raw_path, index=False)

    agg = df.groupby(['instance', 'algorithm']).agg(
        best=('makespan', 'min'),
        worst=('makespan', 'max'),
        mean=('makespan', 'mean'),
        std=('makespan', 'std'),
        mean_fev=('fev_count', 'mean'),
        mean_runtime=('runtime_s', 'mean'),
    ).reset_index()

    summary_path = RESULTS_DIR / "baselines_summary.csv"
    agg.to_csv(summary_path, index=False)

    print(f"Saved {len(all_raw)} raw runs to {raw_path}")
    print(f"Saved {len(agg)} summary rows to {summary_path}")


if __name__ == "__main__":
    main()