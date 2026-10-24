"""
Generate competition submission files (Scenario 1 and 2) in parallel.

Format:
- For each instance, for each of 10 runs (minimum):
  - start_times: list
  - machine_assignments: list
  - worker_assignments: list
  - fev_count: int
  - (Scenario 2) uncertainty_parameters
"""

import sys
import os
import json
import io
import contextlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Fix module imports relative to script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1] if (SCRIPT_DIR.parents[1] / "src").exists() else SCRIPT_DIR.parents[0]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w_v2 import FJSSPW_GA_V2
from ga.fjssp_w.uncertainty_sim import (
    create_seeded_uncertainty_vector,
    evaluate_robustness,
)

N_RUNS = 10   # competition minimum
POP = 30
GEN = 50
BALANCE_W = 0.0

MAX_WORKERS = os.cpu_count() or 4

# Path relative to project root
INSTANCES_REL_DIR = "data-sample/FJSSP-W-Competition/instances/fjssp-w"
SUBMISSION_DIR = PROJECT_ROOT / "results" / "submission"


def get_instance_files():
    """Locates instance files regardless of current working directory."""
    target_path = PROJECT_ROOT / INSTANCES_REL_DIR
    if not target_path.exists():
        target_path = Path(INSTANCES_REL_DIR)

    if not target_path.exists():
        return [], target_path.resolve()

    files = sorted([str(p) for p in target_path.rglob("*.fjs")])
    return files, target_path.resolve()


def run_s1_single_seed(args):
    """Worker task for Scenario 1."""
    path_str, seed, pop, gen, balance_w = args
    name = Path(path_str).stem

    with contextlib.redirect_stdout(io.StringIO()):
        instance = parse_competition_instance(path_str)
        ga = FJSSPW_GA_V2(
            instance, population_size=pop, generations=gen,
            random_seed=seed, fitness_mode='deterministic',
            balance_weight=balance_w,
        )
        res = ga.evolve()
        decoded = ga.decoder.decode(res['best_solution'])

    run_data = {
        'run': seed,
        'instance': name,
        'start_times': [float(x) for x in decoded['start_times']],
        'machine_assignments': [int(x) for x in decoded['machine_assignment']],
        'worker_assignments': [int(x) for x in decoded['worker_assignment']],
        'makespan': float(res['best_makespan']),
        'fev_count': int(res['fev_count']),
    }
    return name, run_data


def run_s2_single_seed(args):
    """Worker task for Scenario 2."""
    path_str, seed, pop, gen, balance_w = args
    name = Path(path_str).stem

    with contextlib.redirect_stdout(io.StringIO()):
        instance = parse_competition_instance(path_str)
        eval_uv = create_seeded_uncertainty_vector(instance['n_workers'], seed=9999)
        ga = FJSSPW_GA_V2(
            instance, population_size=pop, generations=gen,
            random_seed=seed, fitness_mode='uncertainty',
            n_scenarios=10, uncertainty_seed=seed + 1000,
            scenario_seed=seed + 2000, balance_weight=balance_w,
        )
        res = ga.evolve()

        metrics = evaluate_robustness(
            res['best_solution'], instance, ga.decoder,
            eval_uv, n_scenarios=50, seed=1000 + seed,
        )
        decoded = ga.decoder.decode(res['best_solution'])

    run_data = {
        'run': seed,
        'instance': name,
        'start_times': [float(x) for x in decoded['start_times']],
        'machine_assignments': [int(x) for x in decoded['machine_assignment']],
        'worker_assignments': [int(x) for x in decoded['worker_assignment']],
        'nominal_makespan': float(metrics['nominal_makespan']),
        'robust_makespan': float(metrics['mean_makespan']),
        'robust_std': float(metrics['std_makespan']),
        'deterioration_ratio': float(metrics['deterioration_ratio']),
        'uncertainty_parameters': eval_uv,
        'fev_count': int(res['fev_count']),
    }
    return name, run_data


def submit_scenario1():
    out_dir = SUBMISSION_DIR / "scenario1"
    os.makedirs(out_dir, exist_ok=True)
    
    instances, resolved_path = get_instance_files()
    if not instances:
        print(f"[Scenario 1 ERROR] No .fjs files found in: {resolved_path}")
        return

    jobs = [(path, seed, POP, GEN, BALANCE_W) for path in instances for seed in range(N_RUNS)]
    total_jobs = len(jobs)
    print(f"Generating Scenario 1 submissions across {len(instances)} instance(s) ({total_jobs} tasks, {MAX_WORKERS} workers)...")

    instance_runs = defaultdict(list)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_s1_single_seed, job): job for job in jobs}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            name, run_data = future.result()
            instance_runs[name].append(run_data)
            if completed % max(1, total_jobs // 10) == 0 or completed == total_jobs:
                print(f"  Scenario 1 progress: {completed}/{total_jobs} completed ({(completed/total_jobs)*100:.1f}%)")

    for name, runs in instance_runs.items():
        sorted_runs = sorted(runs, key=lambda x: x['run'])
        out_file = out_dir / f"{name}.json"
        with open(out_file, 'w') as f:
            json.dump(sorted_runs, f, indent=2)

    print(f"Scenario 1 submissions written to {out_dir}/")


def submit_scenario2():
    out_dir = SUBMISSION_DIR / "scenario2"
    os.makedirs(out_dir, exist_ok=True)
    
    instances, resolved_path = get_instance_files()
    if not instances:
        print(f"[Scenario 2 ERROR] No .fjs files found in: {resolved_path}")
        return

    jobs = [(path, seed, POP, GEN, BALANCE_W) for path in instances for seed in range(N_RUNS)]
    total_jobs = len(jobs)
    print(f"Generating Scenario 2 submissions across {len(instances)} instance(s) ({total_jobs} tasks, {MAX_WORKERS} workers)...")

    instance_runs = defaultdict(list)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_s2_single_seed, job): job for job in jobs}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            name, run_data = future.result()
            instance_runs[name].append(run_data)
            if completed % max(1, total_jobs // 10) == 0 or completed == total_jobs:
                print(f"  Scenario 2 progress: {completed}/{total_jobs} completed ({(completed/total_jobs)*100:.1f}%)")

    for name, runs in instance_runs.items():
        sorted_runs = sorted(runs, key=lambda x: x['run'])
        out_file = out_dir / f"{name}.json"
        with open(out_file, 'w') as f:
            json.dump(sorted_runs, f, indent=2)

    print(f"Scenario 2 submissions written to {out_dir}/")


if __name__ == "__main__":
    which = sys.argv[1].lower() if len(sys.argv) > 1 else 'both'
    if which in ('1', 'both'):
        submit_scenario1()
    if which in ('2', 'both'):
        submit_scenario2()