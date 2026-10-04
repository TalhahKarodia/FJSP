"""Run bigger GA (100 gen, pop=50) with 3 seeds."""
import sys
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
import glob, io, contextlib, numpy as np

path = glob.glob("src/FJSSP-W-Competition/instances/fjssp-w/*Fattahi_20*")[0]
instance = parse_competition_instance(path)

results = []
for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    ga = FJSSPW_GA(
        instance, population_size=50, generations=100, random_seed=seed
    )
    with contextlib.redirect_stdout(io.StringIO()):
        result = ga.evolve()
    results.append(result['best_makespan'])
    print(f"  Seed {seed}: makespan = {result['best_makespan']}")

print(f"\nResults: {results}")
print(f"Mean = {np.mean(results):.1f}, Std = {np.std(results):.1f}, "
      f"Min = {min(results)}, Max = {max(results)}")
print(f"Coefficient of variation = {np.std(results) / np.mean(results) * 100:.2f}%")