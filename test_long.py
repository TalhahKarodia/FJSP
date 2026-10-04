"""Try 500 generations with pop=100 to see if we escape 5238."""
import sys
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
import glob, io, contextlib, time

path = glob.glob("src/FJSSP-W-Competition/instances/fjssp-w/*Fattahi_20*")[0]
instance = parse_competition_instance(path)

for seed in [0, 1, 2]:
    t0 = time.time()
    ga = FJSSPW_GA(
        instance,
        population_size=100,
        generations=500,
        mutation_rate=0.2,
        random_seed=seed,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        result = ga.evolve()
    elapsed = time.time() - t0
    print(f"Seed {seed}: makespan = {result['best_makespan']}  ({elapsed:.1f}s)")