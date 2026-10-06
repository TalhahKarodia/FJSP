"""
Run all baselines + proposed algorithm on all competition instances.
Produces results/baselines_comparison.csv
"""

import sys, io, contextlib, time, glob, os
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
from baselines.standard_ga import StandardGA
from baselines.hybrid_ga import HybridGA
from baselines.pso import DiscretePSO
from baselines.robust_ga import RobustGA
from analysis.metrics import worker_balance_cv
import numpy as np
import pandas as pd


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
    return r['best_makespan'], r['fev_count'], r['best_solution']


def run_standard(inst, seed):
    ga = StandardGA(inst, population_size=POP, generations=GEN, random_seed=seed)
    r = ga.evolve()
    return r['best_makespan'], r['fev_count'], r['best_solution']


def run_hga(inst, seed):
    ga = HybridGA(inst, population_size=POP, generations=GEN, random_seed=seed)
    r = ga.evolve()
    return r['best_makespan'], r['fev_count'], r['best_solution']


def run_pso(inst, seed):
    pso = DiscretePSO(inst, swarm_size=POP, iterations=GEN, random_seed=seed)
    r = pso.evolve()
    return r['best_makespan'], r['fev_count'], r['best_solution']


def run_robust(inst, seed):
    ga = RobustGA(inst, population_size=POP, generations=GEN, random_seed=seed,
                  n_scenarios=5)
    r = ga.evolve()
    return r['best_makespan'], r['fev_count'], r['best_solution']


ALGORITHMS = {
    'Proposed_FPC_GA': run_proposed,
    'Standard_GA': run_standard,
    'Hybrid_GA': run_hga,
    'PSO': run_pso,
    'Robust_GA': run_robust,
}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    instances = sorted(glob.glob(f"{INSTANCES_DIR}/*.fjs"))
    print(f"Running {len(ALGORITHMS)} algorithms on {len(instances)} instances x {N_RUNS} runs")

    all_raw = []

    for idx, path in enumerate(instances):
        name = os.path.basename(path).replace(".fjs", "")
        try:
            inst = parse_competition_instance(path)
        except Exception as e:
            print(f"  [{name}] PARSE ERROR - {e}")
            continue

        for algo_name, runner in ALGORITHMS.items():
            try:
                for seed in range(N_RUNS):
                    t0 = time.time()
                    mk, fev, chrom = runner(inst, seed)
                    elapsed = time.time() - t0
                    all_raw.append({
                        'instance': name,
                        'algorithm': algo_name,
                        'run': seed,
                        'makespan': float(mk),
                        'fev_count': fev,
                        'runtime_s': elapsed,
                    })
            except Exception as e:
                print(f"  [{name}] {algo_name} ERROR - {e}")

        print(f"[{idx+1:>2}/{len(instances)}] {name} done")

    df = pd.DataFrame(all_raw)
    raw_path = f"{RESULTS_DIR}/baselines_raw.csv"
    df.to_csv(raw_path, index=False)

    # Aggregate
    agg = df.groupby(['instance', 'algorithm']).agg(
        best=('makespan', 'min'),
        worst=('makespan', 'max'),
        mean=('makespan', 'mean'),
        std=('makespan', 'std'),
        mean_fev=('fev_count', 'mean'),
        mean_runtime=('runtime_s', 'mean'),
    ).reset_index()
    agg.to_csv(f"{RESULTS_DIR}/baselines_summary.csv", index=False)

    print(f"\nSaved {len(all_raw)} runs to {raw_path}")
    print(f"Saved {len(agg)} summary rows to {RESULTS_DIR}/baselines_summary.csv")


if __name__ == "__main__":
    main()