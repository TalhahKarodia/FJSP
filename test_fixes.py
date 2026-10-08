import sys, io, contextlib, glob
sys.path.insert(0, 'src')
from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
import numpy as np

paths = glob.glob("../FJSSP-W-Competition/instances/fjssp-w/*Fattahi_20*")
if not paths:
    paths = glob.glob("../FJSSP-W-Competition/instances/fjssp-w/*Fattahi*")
print(f"Using: {paths[0]}\n")
inst = parse_competition_instance(paths[0])

print("=== Deterministic mode: 5 seeds x 20 gen ===")
results = []
for seed in [0, 1, 2, 3, 4]:
    ga = FJSSPW_GA(inst, population_size=30, generations=20, random_seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        r = ga.evolve()
    results.append(r['best_makespan'])
    print(f"  Seed {seed}: makespan = {r['best_makespan']}, FEV = {r['fev_count']}")

print(f"\nStd dev: {np.std(results):.3f}")
print(f"[PASS] varied" if len(set(results)) > 1 else "[NOTE] all same (instance attractor)")

print("\n=== Uncertainty mode: S=5 ===")
ga = FJSSPW_GA(inst, population_size=30, generations=20, random_seed=0,
               fitness_mode='uncertainty', n_scenarios=5,
               uncertainty_seed=42, scenario_seed=42)
with contextlib.redirect_stdout(io.StringIO()):
    r = ga.evolve()
print(f"  Robust best: {r['best_makespan']:.1f}")
print(f"  FEV count:   {r['fev_count']}")

print("\n=== Balance weight λ=0.1 ===")
ga = FJSSPW_GA(inst, population_size=30, generations=20, random_seed=0,
               balance_weight=0.1)
with contextlib.redirect_stdout(io.StringIO()):
    r = ga.evolve()
print(f"  Best: {r['best_makespan']:.1f}")

print("\n[ALL CHECKS COMPLETE]")
