"""Quick smoke test: 1 instance, 2 runs, 3 generations."""
import sys, io, contextlib, glob, os, time
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
from baselines.standard_ga import StandardGA
from baselines.hybrid_ga import HybridGA
from baselines.pso import DiscretePSO
from baselines.robust_ga import RobustGA

paths = sorted(glob.glob("../FJSSP-W-Competition/instances/fjssp-w/*.fjs"))[:1]
if not paths:
    print("No instances found")
    sys.exit(1)

inst = parse_competition_instance(paths[0])
print(f"Instance: {os.path.basename(paths[0])}")
print(f"Jobs={inst['n_jobs']}, Machines={inst['n_machines']}, "
      f"Workers={inst['n_workers']}, Ops={inst['n_operations']}\n")

print("=== Proposed FPC-GA ===")
t0 = time.time()
ga = FJSSPW_GA(inst, population_size=10, generations=3, random_seed=0,
               fitness_mode='deterministic', balance_weight=0.1)
with contextlib.redirect_stdout(io.StringIO()):
    r = ga.evolve()
print(f"  makespan={r['best_makespan']:.0f}, fev={r['fev_count']}, {time.time()-t0:.2f}s")

print("\n=== Standard GA ===")
t0 = time.time()
sga = StandardGA(inst, population_size=10, generations=3, random_seed=0)
r = sga.evolve()
print(f"  makespan={r['best_makespan']:.0f}, fev={r['fev_count']}, {time.time()-t0:.2f}s")

print("\n=== Hybrid GA ===")
t0 = time.time()
hga = HybridGA(inst, population_size=10, generations=3, random_seed=0)
r = hga.evolve()
print(f"  makespan={r['best_makespan']:.0f}, fev={r['fev_count']}, {time.time()-t0:.2f}s")

print("\n=== PSO ===")
t0 = time.time()
pso = DiscretePSO(inst, swarm_size=10, iterations=3, random_seed=0)
r = pso.evolve()
print(f"  makespan={r['best_makespan']:.0f}, fev={r['fev_count']}, {time.time()-t0:.2f}s")

print("\n=== Robust GA ===")
t0 = time.time()
rga = RobustGA(inst, population_size=10, generations=3, random_seed=0, n_scenarios=3)
r = rga.evolve()
print(f"  makespan={r['best_makespan']:.0f}, fev={r['fev_count']}, {time.time()-t0:.2f}s")

print("\n[SMOKE TEST PASSED]")
