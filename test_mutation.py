"""Test if higher mutation escapes the 5238 local optimum."""
import sys
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
import glob, io, contextlib

path = glob.glob("src/FJSSP-W-Competition/instances/fjssp-w/*Fattahi_20*")[0]
instance = parse_competition_instance(path)

for mut_rate in [0.1, 0.2, 0.3, 0.5]:
    results = []
    for seed in [0, 1, 2]:
        ga = FJSSPW_GA(
            instance,
            population_size=50,
            generations=100,
            mutation_rate=mut_rate,
            random_seed=seed,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = ga.evolve()
        results.append(result['best_makespan'])
    print(f"Mutation={mut_rate}: results = {results}")