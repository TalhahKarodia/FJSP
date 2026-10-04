"""Test if competition's Graph class works with our decoder output."""
import sys
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
from util.graph import Graph
from util.uncertainty import create_uncertainty_vector
import glob, io, contextlib

path = glob.glob("src/FJSSP-W-Competition/instances/fjssp-w/*Fattahi_20*")[0]
instance = parse_competition_instance(path)

# Get a solution
ga = FJSSPW_GA(instance, population_size=30, generations=20, random_seed=0)
with contextlib.redirect_stdout(io.StringIO()):
    result = ga.evolve()

chromosome = result['best_solution']
decoded = ga.decoder.decode(chromosome)

s = decoded['start_times']
e = decoded['end_times']
m = decoded['machine_assignment']
w = decoded['worker_assignment']
d = instance['durations']
js = instance['job_sequence']

print(f"Decoded makespan: {decoded['makespan']}")
print(f"Lengths: s={len(s)}, e={len(e)}, m={len(m)}, w={len(w)}, js={len(js) if isinstance(js, list) else 'N/A'}")

# Try building the Graph
try:
    g = Graph(s, e, m, w, js)
    print(f"Graph built successfully")
    print(f"Graph roots: {len(g.roots)}")
    print(f"Graph nodes: {len(g.all_nodes)}")
    print(f"max(g.e): {max(g.e)}")
except Exception as ex:
    import traceback
    traceback.print_exc()

# Try uncertainty vector
try:
    n_workers = instance['n_workers']
    uv = create_uncertainty_vector(n_workers)
    print(f"\nUncertainty vector for {n_workers} workers:")
    for i, params in enumerate(uv[:3]):
        print(f"  Worker {i}: alpha={params[0]:.3f}, beta={params[1]:.3f}, offset={params[2]}")
except Exception as ex:
    traceback.print_exc()
