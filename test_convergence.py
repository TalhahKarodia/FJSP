"""Show GA convergence over generations."""
import sys
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
import glob

path = glob.glob("src/FJSSP-W-Competition/instances/fjssp-w/*Fattahi_20*")[0]
instance = parse_competition_instance(path)

ga = FJSSPW_GA(instance, population_size=30, generations=20, random_seed=1)
result = ga.evolve()

h = result['history']
print("\nGeneration |    Best |     Avg |   Worst")
print("-" * 45)
for i in range(len(h['best_makespan'])):
    print(f"  {i+1:>8} | {h['best_makespan'][i]:>7} | "
          f"{h['avg_makespan'][i]:>7.1f} | {h['worst_makespan'][i]:>7}")