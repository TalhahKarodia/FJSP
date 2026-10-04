"""Test sensitivity to S, pop, gen in uncertainty mode."""
import sys
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
from ga.fjssp_w.uncertainty_sim import evaluate_robustness, create_seeded_uncertainty_vector
import glob, io, contextlib

path = glob.glob("src/FJSSP-W-Competition/instances/fjssp-w/*Fattahi_20*")[0]
instance = parse_competition_instance(path)
eval_uv = create_seeded_uncertainty_vector(instance['n_workers'], seed=9999)

configs = [
    # (label, pop, gen, S)
    ("pop30 gen30 S5",  30, 30, 5),
    ("pop30 gen30 S10", 30, 30, 10),
    ("pop30 gen30 S20", 30, 30, 20),
    ("pop50 gen50 S10", 50, 50, 10),
    ("pop30 gen100 S10", 30, 100, 10),
]

print(f"{'Config':<20} {'Nominal':<10} {'Robust':<10} {'R':<8} {'FEV':<8}")
print("-" * 60)

for label, pop, gen, S in configs:
    ga = FJSSPW_GA(
        instance, population_size=pop, generations=gen,
        random_seed=0, fitness_mode='uncertainty',
        n_scenarios=S, uncertainty_seed=42, scenario_seed=42
    )
    with contextlib.redirect_stdout(io.StringIO()):
        res = ga.evolve()
    
    nominal = ga.decoder.decode(res['best_solution'])['makespan']
    metrics = evaluate_robustness(
        res['best_solution'], instance, ga.decoder,
        eval_uv, n_scenarios=50, seed=1000
    )
    print(f"{label:<20} {nominal:<10} {metrics['mean_makespan']:<10.1f} "
          f"{metrics['deterioration_ratio']:<8.4f} {ga.fev_count:<8}")
