"""Test uncertainty simulator: should give varied results across seeds/scenarios."""
import sys
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
from ga.fjssp_w.uncertainty_sim import (
    create_seeded_uncertainty_vector,
    simulate_one_scenario,
    evaluate_robustness,
)
import glob, io, contextlib, random
import numpy as np

# Load instance
path = glob.glob("src/FJSSP-W-Competition/instances/fjssp-w/*Fattahi_20*")[0]
instance = parse_competition_instance(path)
print(f"Instance: {path}")
print(f"Jobs={instance['n_jobs']}, Machines={instance['n_machines']}, "
      f"Workers={instance['n_workers']}, Ops={instance['n_operations']}\n")

# Get a good chromosome from the deterministic GA
ga = FJSSPW_GA(instance, population_size=30, generations=50, random_seed=0)
with contextlib.redirect_stdout(io.StringIO()):
    result = ga.evolve()
chromosome = result['best_solution']
decoder = ga.decoder

print(f"Nominal makespan (deterministic): {result['best_makespan']}\n")

# Create uncertainty vector
uv = create_seeded_uncertainty_vector(instance['n_workers'], seed=42)
print(f"Uncertainty vector (first 3 workers):")
for i in range(min(3, len(uv))):
    a, b, o = uv[i]
    print(f"  Worker {i}: alpha={a:.4f}, beta={b:.4f}, offset={o}")
print()

# Test 1: Single scenario, different seeds
print("=" * 60)
print("TEST 1: Different RNG seeds → different single-scenario makespans")
print("=" * 60)
for seed in [0, 1, 2, 3, 4]:
    rng = random.Random(seed)
    mk = simulate_one_scenario(chromosome, instance, decoder, uv, rng)
    print(f"  Seed {seed}: makespan = {mk:.1f}")

# Test 2: Full robustness evaluation
print("\n" + "=" * 60)
print("TEST 2: Robustness over 50 scenarios")
print("=" * 60)
metrics = evaluate_robustness(
    chromosome, instance, decoder, uv, n_scenarios=50, seed=100
)
print(f"  Nominal makespan:    {metrics['nominal_makespan']}")
print(f"  Mean robust:         {metrics['mean_makespan']:.1f}")
print(f"  Std robust:          {metrics['std_makespan']:.1f}")
print(f"  Min / Max robust:    {metrics['min_makespan']:.1f} / {metrics['max_makespan']:.1f}")
print(f"  Deterioration ratio: {metrics['deterioration_ratio']:.3f}")

# Test 3: Different chromosomes → different robustness
print("\n" + "=" * 60)
print("TEST 3: Different chromosomes → different robust makespans")
print("=" * 60)
for trial_seed in [0, 1, 2, 3, 4]:
    ga_trial = FJSSPW_GA(instance, population_size=30, generations=30, random_seed=trial_seed)
    with contextlib.redirect_stdout(io.StringIO()):
        res = ga_trial.evolve()
    chrom = res['best_solution']
    metrics = evaluate_robustness(
        chrom, instance, decoder, uv, n_scenarios=20, seed=100
    )
    print(f"  Trial {trial_seed}: nominal={res['best_makespan']:.0f}, "
          f"robust_mean={metrics['mean_makespan']:.0f}, "
          f"R={metrics['deterioration_ratio']:.3f}")

print("\n[PASS] Uncertainty simulator is working")