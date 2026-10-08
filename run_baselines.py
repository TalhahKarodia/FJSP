"""
Run all baselines + proposed algorithm on all competition instances.
Produces results/baselines_raw.csv and results/baselines_summary.csv

Reduced-scope version for faster completion:
  - POP = 20, GEN = 30 (faster GA)
  - N_RUNS = 30 (competition minimum exceeded; proposal promised this)
  - Robust_GA removed: runtime exceeds experiment budget (S=2 still
    requires ~86 min/instance). Documented in report Section 4.5.
"""

import sys, io, contextlib, time, glob, os
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
from baselines.standard_ga import StandardGA
from baselines.hybrid_ga import HybridGA
from baselines.pso import DiscretePSO
# from baselines.robust_ga import RobustGA  # removed for runtime reasons
import numpy as np
import pandas as pd

# --- Reduced settings for quick run ---
N_RUNS = 30
POP = 30
GEN = 50
BALANCE_W = 0.0

INSTANCES_DIR = "../FJSSP-W-Competition/instances/fjssp-w"
RESULTS_DIR = "results"


def run_proposed(inst, seed):
    ga = FJSSPW_GA(
        inst, population_size=POP, generations=GEN, random_seed=seed,
        fitness_mode='deterministic', balance_weight=BALANCE_W,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        r = ga.evolve()
    return r['best_makespan'], r['fev_count']


def run_standard(inst, seed):
    ga = StandardGA(inst, population_size=POP, generations=GEN, random_seed=seed)
    r = ga.evolve()
    return r['best_makespan'], r['fev_count']


def run_hga(inst, seed):
    ga = HybridGA(inst, population_size=POP, generations=GEN, random_seed=seed)
    r = ga.evolve()
    return r['best_makespan'], r['fev_count']


def run_pso(inst, seed):
    pso = DiscretePSO(inst, swarm_size=POP, iterations=GEN, random_seed=seed)
    r = pso.evolve()
    return r['best_makespan'], r['fev_count']


# Robust_GA removed — see module docstring

ALGORITHMS = {
    'Proposed_FPC_GA': run_proposed,
    'Standard_GA': run_standard,
    'Hybrid_GA': run_hga,
    'PSO': run_pso,
    # 'Robust_GA': run_robust,  # excluded
}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    instances = sorted(glob.glob(f"{INSTANCES_DIR}/*.fjs"))
    print(f"Running {len(ALGORITHMS)} algorithms on {len(instances)} instances x {N_RUNS} runs")
    print(f"Settings: POP={POP}, GEN={GEN}")
    print("=" * 75)

    all_raw = []
    total_start = time.time()

    for idx, path in enumerate(instances):
        name = os.path.basename(path).replace(".fjs", "")
        try:
            inst = parse_competition_instance(path)
        except Exception as e:
            print(f"  [{name}] PARSE ERROR: {e}")
            continue

        inst_start = time.time()
        for algo_name, runner in ALGORITHMS.items():
            try:
                algo_start = time.time()
                for seed in range(N_RUNS):
                    t0 = time.time()
                    mk, fev = runner(inst, seed)
                    elapsed = time.time() - t0
                    all_raw.append({
                        'instance': name,
                        'algorithm': algo_name,
                        'run': seed,
                        'makespan': float(mk),
                        'fev_count': int(fev),
                        'runtime_s': elapsed,
                    })
                algo_time = time.time() - algo_start
                print(f"  [{name}] {algo_name:<16} done ({algo_time:.1f}s)")
            except Exception as e:
                print(f"  [{name}] {algo_name} ERROR: {e}")

        inst_time = time.time() - inst_start
        print(f"[{idx+1:>2}/{len(instances)}] {name} done in {inst_time:.1f}s")
        print()

    total_time = time.time() - total_start
    print(f"\nAll done in {total_time/60:.1f} minutes")

    df = pd.DataFrame(all_raw)
    df.to_csv(f"{RESULTS_DIR}/baselines_raw.csv", index=False)

    agg = df.groupby(['instance', 'algorithm']).agg(
        best=('makespan', 'min'),
        worst=('makespan', 'max'),
        mean=('makespan', 'mean'),
        std=('makespan', 'std'),
        mean_fev=('fev_count', 'mean'),
        mean_runtime=('runtime_s', 'mean'),
    ).reset_index()
    agg.to_csv(f"{RESULTS_DIR}/baselines_summary.csv", index=False)

    print(f"Saved {len(all_raw)} raw runs to {RESULTS_DIR}/baselines_raw.csv")
    print(f"Saved {len(agg)} summary rows to {RESULTS_DIR}/baselines_summary.csv")


if __name__ == "__main__":
    main()