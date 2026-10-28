"""
Unified Competition Orchestrator for Scenario 1 (Deterministic FJSSP-W).

- Runs Proposed FPC-GA, Standard GA, Hybrid GA, and Discrete PSO in parallel.
- Ensures 100% fair baseline evaluation via shared initial population seeding.
- Computes Official Competition Metrics: Makespan, Worker Load CV (Tie-Breaker), 
  FEV Counts, Runtime, and Rank-Sum Scoring.

Outputs:
  - results/baselines_raw.csv
  - results/baselines_summary.csv
"""

import sys
import io
import contextlib
import time
import os
import copy
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / 'src'))

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.decoder_w_v2 import FJSSPWDecoder_V2
from ga.fjssp_w.genetic_algorithm_w_v2 import FJSSPW_GA_V2
from baselines.standard_ga_v2 import StandardGA_V2
from baselines.hybrid_ga_v2 import HybridGA_V2
from baselines.pso_v2 import DiscretePSO_V2

# --- Execution Settings ---
N_RUNS = 30
POP = 30
GEN = 50
BALANCE_W = 0.0
GREEDY_SEED_RATIO = 0.20

MAX_WORKERS = os.cpu_count() - 2 or 4

INSTANCES_DIR = "data-sample/FJSSP-W-Competition/instances/fjssp-w"
RESULTS_DIR = SCRIPT_DIR / "results"

ALGORITHMS = ['Proposed_FPC_GA', 'Standard_GA', 'Hybrid_GA', 'PSO']


def compute_worker_cv(solution: dict, n_workers: int, durations: list) -> float:
    """
    Computes Coefficient of Variation (CV) of worker workloads (Official Tie-Breaker).
    Handles both GA dicts ('machine_assignment') and PSO dicts ('machines').
    """
    loads = defaultdict(float)
    machines = solution.get('machine_assignment', solution.get('machines', []))
    workers = solution.get('worker_assignment', solution.get('workers', []))

    for op_idx in range(len(machines)):
        m = machines[op_idx]
        w = workers[op_idx]
        loads[w] += durations[op_idx][m][w]

    vals = [loads.get(w_idx, 0.0) for w_idx in range(n_workers)]
    mean_val = sum(vals) / len(vals) if vals else 0.0
    if mean_val == 0.0:
        return 0.0

    var = sum((v - mean_val) ** 2 for v in vals) / len(vals)
    return float((var ** 0.5) / mean_val)


def convert_chrom_to_pso_particle(chrom: dict, n_ops: int) -> dict:
    """Maps a GA chromosome into a discrete PSO particle vector."""
    seq = chrom['operation_sequence']
    priority = [0.0] * n_ops
    for rank, op in enumerate(seq):
        priority[op] = rank / max(1, n_ops - 1)
    return {
        'priority': priority,
        'machines': list(chrom['machine_assignment']),
        'workers': list(chrom['worker_assignment']),
    }


def create_shared_initial_population(decoder: FJSSPWDecoder_V2, pop_size: int, seed: int) -> list:
    """Generates an identical population (20% greedy, 80% random) for a given run seed."""
    rng = random.Random(seed)
    n_greedy = int(GREEDY_SEED_RATIO * pop_size)
    n_random = pop_size - n_greedy

    population = []
    for _ in range(n_greedy):
        population.append(decoder.create_greedy_chromosome(rng))
    for _ in range(n_random):
        population.append(decoder.create_random_chromosome(rng))

    return population


def run_single_job(args):
    """Worker task executing a single instance run for a given seed and algorithm."""
    path_str, algo_name, seed, pop, gen, balance_w = args
    name = Path(path_str).stem

    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        inst = parse_competition_instance(path_str)
        decoder = FJSSPWDecoder_V2(inst)

        # Generate shared initial population for perfect baseline fairness
        shared_pop = create_shared_initial_population(decoder, pop, seed)

        if algo_name == 'Proposed_FPC_GA':
            ga = FJSSPW_GA_V2(
                inst, population_size=pop, generations=gen, random_seed=seed,
                fitness_mode='deterministic', balance_weight=balance_w,
                initial_population=copy.deepcopy(shared_pop)
            )
            r = ga.evolve()

        elif algo_name == 'Standard_GA':
            ga = StandardGA_V2(
                inst, population_size=pop, generations=gen, random_seed=seed,
                initial_population=copy.deepcopy(shared_pop)
            )
            r = ga.evolve()

        elif algo_name == 'Hybrid_GA':
            ga = HybridGA_V2(
                inst, population_size=pop, generations=gen, random_seed=seed,
                initial_population=copy.deepcopy(shared_pop)
            )
            r = ga.evolve()

        elif algo_name == 'PSO':
            pso_pop = [
                convert_chrom_to_pso_particle(c, decoder.n_operations) 
                for c in copy.deepcopy(shared_pop)
            ]
            pso = DiscretePSO_V2(
                inst, swarm_size=pop, iterations=gen, random_seed=seed,
                initial_population=pso_pop
            )
            r = pso.evolve()

        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        mk = float(r['best_makespan'])
        fev = int(r['fev_count'])
        cv = compute_worker_cv(r['best_solution'], decoder.n_workers, inst['durations'])

    elapsed = time.time() - t0
    return {
        'instance': name,
        'algorithm': algo_name,
        'seed': seed,
        'makespan': mk,
        'worker_cv': cv,
        'fev_count': fev,
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

    # Build work items across (instance, algorithm, seed) combinations
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

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_job, job): job for job in jobs}
        completed = 0

        for future in as_completed(futures):
            completed += 1
            try:
                result = future.result()
                all_raw.append(result)
                if completed % max(1, total_jobs // 10) == 0 or completed == total_jobs:
                    pct = (completed / total_jobs) * 100
                    print(f"Progress: {completed}/{total_jobs} tasks completed ({pct:.1f}%)")
            except Exception as e:
                job_info = futures[future]
                print(f"  [ERROR] Job failed {job_info[1]} on {Path(job_info[0]).stem} (seed {job_info[2]}): {e}")

    total_time = time.time() - total_start
    print(f"\nAll tasks completed in {total_time / 60:.2f} minutes")

    if not all_raw:
        print("ERROR: No runs completed successfully.")
        return

    # 1. Save Raw Dataset
    df = pd.DataFrame(all_raw)
    raw_path = RESULTS_DIR / "baselines_raw.csv"
    df.to_csv(raw_path, index=False)

    # 2. Compute Competition Ranking per (instance, seed) run using Primary + Secondary criteria
    # Primary: Makespan (asc), Secondary: Worker CV (asc), Tertiary: FEV Count (asc)
    df = df.sort_values(
        by=['instance', 'seed', 'makespan', 'worker_cv', 'fev_count'],
        ascending=[True, True, True, True, True]
    )
    df['competition_rank'] = df.groupby(['instance', 'seed']).cumcount() + 1

    # 3. Aggregate Summary Table
    agg = df.groupby(['instance', 'algorithm']).agg(
        rank_sum=('competition_rank', 'sum'),
        mean_rank=('competition_rank', 'mean'),
        best=('makespan', 'min'),
        worst=('makespan', 'max'),
        mean=('makespan', 'mean'),
        std=('makespan', 'std'),
        mean_worker_cv=('worker_cv', 'mean'),
        mean_fev=('fev_count', 'mean'),
        mean_runtime_s=('runtime_s', 'mean'),
    ).reset_index()

    summary_path = RESULTS_DIR / "baselines_summary.csv"
    agg.to_csv(summary_path, index=False)

    print(f"Saved {len(all_raw)} raw runs to {raw_path}")
    print(f"Saved {len(agg)} summary rows (with competition rank-sums) to {summary_path}")


if __name__ == "__main__":
    main()