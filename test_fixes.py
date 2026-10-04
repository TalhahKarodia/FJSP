"""Verify fixes: seeds must produce different results."""
import sys
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
import numpy as np
import glob

paths = glob.glob("src/FJSSP-W-Competition/instances/fjssp-w/*Fattahi_20*")
if not paths:
    paths = glob.glob("src/FJSSP-W-Competition/instances/fjssp-w/*Fattahi*")
if not paths:
    print("No Fattahi instance found.")
    sys.exit(1)
print(f"Using instance: {paths[0]}\n")

instance = parse_competition_instance(paths[0])
print(f"Jobs={instance['n_jobs']}, Machines={instance['n_machines']}, "
      f"Workers={instance['n_workers']}, Ops={instance['n_operations']}\n")

print("Running 5 seeds x 20 generations each...")
results = []
for seed in [0, 1, 2, 3, 4]:
    ga = FJSSPW_GA(instance, population_size=30, generations=20, random_seed=seed)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        result = ga.evolve()
    results.append(result['best_makespan'])
    print(f"  Seed {seed}: best makespan = {result['best_makespan']}")

print(f"\nResults: {results}")
print(f"Min: {min(results)}, Max: {max(results)}")
print(f"Std dev: {np.std(results):.3f}")

if len(set(results)) > 1:
    print("\n[PASS] Seeds produce different results")
else:
    print("\n[FAIL] All seeds still produce the same result")