"""
Unified Competition Orchestrator for Scenario 1 (Deterministic FJSSP-W).

- Runs Proposed FPC-GA, Standard GA, Hybrid GA, and Discrete PSO in parallel.
- Shared initial population per (instance, seed) for fair baselines.
- Official metrics: Makespan, Worker Load CV (tie-breaker), FEV, Runtime, Rank-Sum.

Optimised: one task per (instance, seed) runs all four algorithms, so the
instance is parsed once per worker process (cached), the decoder tables are
built once, and the shared population is generated once instead of 4x.
runtime_s now measures the algorithm run only (no parsing / population setup).

Outputs:
  - results/baselines_raw.csv
  - results/baselines_summary.csv
"""

import sys
import io
import contextlib
import time
import os
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
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

MAX_WORKERS = max(1, (os.cpu_count() or 4))

INSTANCES_DIR = "data-sample/FJSSP-W-Competition/instances/fjssp-w"
RESULTS_DIR = SCRIPT_DIR / "results"

ALGORITHMS = ['Proposed_FPC_GA', 'Standard_GA', 'Hybrid_GA', 'PSO']

# Per-process cache: path -> (instance_data, decoder)
_INSTANCE_CACHE = {}


def compute_worker_cv(solution: dict, n_workers: int, durations: list) -> float:
    """CV of worker workloads (official tie-breaker). Handles GA and PSO dicts."""
    machines = solution.get('machine_assignment', solution.get('machines', []))
    workers = solution.get('worker_assignment', solution.get('workers', []))

    loads = [0.0] * n_workers
    for op_idx in range(len(machines)):
        m = machines[op_idx]
        w = workers[op_idx]
        loads[w] += durations[op_idx][m][w]

    mean_val = sum(loads) / n_workers if n_workers else 0.0
    if mean_val == 0.0:
        return 0.0

    var = sum((v - mean_val) ** 2 for v in loads) / n_workers
    return float((var ** 0.5) / mean_val)


def convert_chrom_to_pso_particle(chrom: dict, n_ops: int) -> dict:
    seq = chrom['operation_sequence']
    priority = [0.0] * n_ops
    denom = max(1, n_ops - 1)
    for rank, op in enumerate(seq):
        priority[op] = rank / denom
    return {
        'priority': priority,
        'machines': list(chrom['machine_assignment']),
        'workers': list(chrom['worker_assignment']),
    }


def create_shared_initial_population(decoder: FJSSPWDecoder_V2, pop_size: int, seed: int) -> list:
    """Identical population (20% greedy, 80% random) for a given run seed."""
    rng = random.Random(seed)
    n_greedy = int(GREEDY_SEED_RATIO * pop_size)
    n_random = pop_size - n_greedy

    population = []
    for _ in range(n_greedy):
        population.append(decoder.create_greedy_chromosome(rng))
    for _ in range(n_random):
        population.append(decoder.create_random_chromosome(rng))
    return population


def _get_instance(path_str):
    entry = _INSTANCE_CACHE.get(path_str)
    if entry is None:
        inst = parse_competition_instance(path_str)
        entry = (inst, FJSSPWDecoder_V2(inst))
        _INSTANCE_CACHE[path_str] = entry
    return entry


def _clone_pop(pop):
    return [{
        'machine_assignment': c['machine_assignment'][:],
        'worker_assignment': c['worker_assignment'][:],
        'operation_sequence': c['operation_sequence'][:],
    } for c in pop]


def _run_algorithm(algo_name, inst, decoder, shared_pop, seed, pop, gen, balance_w):
    if algo_name == 'Proposed_FPC_GA':
        return FJSSPW_GA_V2(
            inst, population_size=pop, generations=gen, random_seed=seed,
            fitness_mode='deterministic', balance_weight=balance_w,
            initial_population=_clone_pop(shared_pop), decoder=decoder,
        ).evolve()
    if algo_name == 'Standard_GA':
        return StandardGA_V2(
            inst, population_size=pop, generations=gen, random_seed=seed,
            initial_population=_clone_pop(shared_pop), decoder=decoder,
        ).evolve()
    if algo_name == 'Hybrid_GA':
        return HybridGA_V2(
            inst, population_size=pop, generations=gen, random_seed=seed,
            initial_population=_clone_pop(shared_pop), decoder=decoder,
        ).evolve()
    if algo_name == 'PSO':
        pso_pop = [convert_chrom_to_pso_particle(c, decoder.n_operations) for c in shared_pop]
        return DiscretePSO_V2(
            inst, swarm_size=pop, iterations=gen, random_seed=seed,
            initial_population=pso_pop, decoder=decoder,
        ).evolve()
    raise ValueError(f"Unknown algorithm: {algo_name}")


def run_instance_seed(args):
    """One task = all algorithms for one (instance, seed). Returns (rows, errors)."""
    path_str, seed, pop, gen, balance_w = args
    name = Path(path_str).stem
    rows, errors = [], []

    with contextlib.redirect_stdout(io.StringIO()):
        inst, decoder = _get_instance(path_str)
        shared_pop = create_shared_initial_population(decoder, pop, seed)

        for algo_name in ALGORITHMS:
            t0 = time.time()
            try:
                r = _run_algorithm(algo_name, inst, decoder, shared_pop, seed, pop, gen, balance_w)
                mk = float(r['best_makespan'])
                fev = int(r['fev_count'])
                cv = compute_worker_cv(r['best_solution'], decoder.n_workers, inst['durations'])
            except Exception as e:
                errors.append((algo_name, name, seed, repr(e)))
                continue
            rows.append({
                'instance': name,
                'algorithm': algo_name,
                'seed': seed,
                'makespan': mk,
                'worker_cv': cv,
                'fev_count': fev,
                'runtime_s': time.time() - t0,
            })
    return rows, errors


def find_instances(target_dir_str):
    target_path = SCRIPT_DIR / target_dir_str
    if not target_path.exists():
        target_path = Path(target_dir_str)

    if not target_path.exists():
        return [], target_path.resolve()

    files = sorted(str(p) for p in target_path.rglob("*.fjs"))
    return files, target_path.resolve()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    instances, resolved_path = find_instances(INSTANCES_DIR)

    if not instances:
        print(f"ERROR: No .fjs files found in directory: {resolved_path}")
        return

    # Seed-major ordering within an instance keeps a worker's parse cache warm
    jobs = [(path, seed, POP, GEN, BALANCE_W) for path in instances for seed in range(N_RUNS)]
    total_jobs = len(jobs)
    print(f"Loaded {len(instances)} instances from: {resolved_path}")
    print(f"Running {len(ALGORITHMS)} algorithms x {N_RUNS} runs "
          f"({total_jobs * len(ALGORITHMS)} runs in {total_jobs} tasks)")
    print(f"Parallel Execution: {MAX_WORKERS} workers | POP={POP}, GEN={GEN}")
    print("=" * 75)

    all_raw = []
    total_start = time.time()
    step = max(1, total_jobs // 10)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_instance_seed, job) for job in jobs]
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                rows, errors = future.result()
                all_raw.extend(rows)
                for algo, inst_name, seed, msg in errors:
                    print(f"  [ERROR] {algo} on {inst_name} (seed {seed}): {msg}")
            except Exception as e:
                print(f"  [ERROR] task failed: {e}")
            if completed % step == 0 or completed == total_jobs:
                print(f"Progress: {completed}/{total_jobs} tasks ({completed / total_jobs * 100:.1f}%)")

    print(f"\nAll tasks completed in {(time.time() - total_start) / 60:.2f} minutes")

    if not all_raw:
        print("ERROR: No runs completed successfully.")
        return

    df = pd.DataFrame(all_raw)
    raw_path = RESULTS_DIR / "baselines_raw.csv"
    df.to_csv(raw_path, index=False)

    # Rank per (instance, seed): makespan, then worker CV, then FEV (all ascending)
    df = df.sort_values(by=['instance', 'seed', 'makespan', 'worker_cv', 'fev_count'])
    df['competition_rank'] = df.groupby(['instance', 'seed']).cumcount() + 1

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