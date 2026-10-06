"""
Generate competition submission files (Scenario 1 and 2).

Format:
- For each instance, for each of 10 runs (minimum):
  - start_times: list
  - machine_assignments: list
  - worker_assignments: list
  - fev_count: int
  - (Scenario 2) uncertainty_parameters
"""

import sys, os, json, io, contextlib, glob
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
from ga.fjssp_w.uncertainty_sim import (
    create_seeded_uncertainty_vector,
    evaluate_robustness,
)

N_RUNS = 10   # competition minimum
POP = 30
GEN = 50
BALANCE_W = 0.0

INSTANCES_DIR = "../FJSSP-W-Competition/instances/fjssp-w"
SUBMISSION_DIR = "results/submission"


def submit_scenario1():
    os.makedirs(f"{SUBMISSION_DIR}/scenario1", exist_ok=True)
    instances = sorted(glob.glob(f"{INSTANCES_DIR}/*.fjs"))

    for path in instances:
        name = os.path.basename(path).replace(".fjs", "")
        instance = parse_competition_instance(path)
        runs = []

        for seed in range(N_RUNS):
            ga = FJSSPW_GA(
                instance, population_size=POP, generations=GEN,
                random_seed=seed, fitness_mode='deterministic',
                balance_weight=BALANCE_W,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                res = ga.evolve()

            decoded = ga.decoder.decode(res['best_solution'])
            runs.append({
                'run': seed,
                'instance': name,
                'start_times': [float(x) for x in decoded['start_times']],
                'machine_assignments': [int(x) for x in decoded['machine_assignment']],
                'worker_assignments': [int(x) for x in decoded['worker_assignment']],
                'makespan': float(res['best_makespan']),
                'fev_count': int(res['fev_count']),
            })

        with open(f"{SUBMISSION_DIR}/scenario1/{name}.json", 'w') as f:
            json.dump(runs, f)

    print(f"Scenario 1 submissions written to {SUBMISSION_DIR}/scenario1/")


def submit_scenario2():
    os.makedirs(f"{SUBMISSION_DIR}/scenario2", exist_ok=True)
    instances = sorted(glob.glob(f"{INSTANCES_DIR}/*.fjs"))

    for path in instances:
        name = os.path.basename(path).replace(".fjs", "")
        instance = parse_competition_instance(path)
        # Fixed uncertainty vector per instance (per competition rules)
        eval_uv = create_seeded_uncertainty_vector(instance['n_workers'], seed=9999)
        runs = []

        for seed in range(N_RUNS):
            ga = FJSSPW_GA(
                instance, population_size=POP, generations=GEN,
                random_seed=seed, fitness_mode='uncertainty',
                n_scenarios=10, uncertainty_seed=seed + 1000,
                scenario_seed=seed + 2000, balance_weight=BALANCE_W,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                res = ga.evolve()

            metrics = evaluate_robustness(
                res['best_solution'], instance, ga.decoder,
                eval_uv, n_scenarios=50, seed=1000 + seed,
            )
            decoded = ga.decoder.decode(res['best_solution'])

            runs.append({
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
            })

        with open(f"{SUBMISSION_DIR}/scenario2/{name}.json", 'w') as f:
            json.dump(runs, f)

    print(f"Scenario 2 submissions written to {SUBMISSION_DIR}/scenario2/")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('1', 'both'):
        submit_scenario1()
    if which in ('2', 'both'):
        submit_scenario2()