"""Verify uncertainty-aware GA works."""
import sys
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
from ga.fjssp_w.uncertainty_sim import evaluate_robustness, create_seeded_uncertainty_vector
import glob, io, contextlib

path = glob.glob("src/FJSSP-W-Competition/instances/fjssp-w/*Fattahi_20*")[0]
instance = parse_competition_instance(path)

# Fixed evaluation uncertainty vector (same for both models)
eval_uv = create_seeded_uncertainty_vector(instance['n_workers'], seed=9999)

print("=" * 60)
print("Deterministic mode (baseline)")
print("=" * 60)
ga_det = FJSSPW_GA(instance, population_size=30, generations=30,
                  random_seed=0, fitness_mode='deterministic')
with contextlib.redirect_stdout(io.StringIO()):
    res_det = ga_det.evolve()

det_nominal = res_det['best_makespan']
det_metrics = evaluate_robustness(
    res_det['best_solution'], instance, ga_det.decoder,
    eval_uv, n_scenarios=50, seed=1000
)
print(f"  Nominal makespan:      {det_nominal}")
print(f"  Robust mean:           {det_metrics['mean_makespan']:.1f}")
print(f"  Robust std:            {det_metrics['std_makespan']:.1f}")
print(f"  Deterioration ratio:   {det_metrics['deterioration_ratio']:.4f}")
print(f"  FEV count:             {ga_det.fev_count}")

print("\n" + "=" * 60)
print("Uncertainty mode (S=5 scenarios per eval)")
print("=" * 60)
ga_unc = FJSSPW_GA(instance, population_size=30, generations=30,
                   random_seed=0, fitness_mode='uncertainty',
                   n_scenarios=5, uncertainty_seed=42, scenario_seed=42)
with contextlib.redirect_stdout(io.StringIO()):
    res_unc = ga_unc.evolve()

# Note: res_unc['best_makespan'] IS the robust mean (fitness)
unc_nominal_decoded = ga_unc.decoder.decode(res_unc['best_solution'])
unc_nominal = unc_nominal_decoded['makespan']

unc_metrics = evaluate_robustness(
    res_unc['best_solution'], instance, ga_unc.decoder,
    eval_uv, n_scenarios=50, seed=1000
)
print(f"  Nominal makespan:      {unc_nominal}")
print(f"  Robust mean (from GA): {res_unc['best_makespan']:.1f}")
print(f"  Robust mean (eval):    {unc_metrics['mean_makespan']:.1f}")
print(f"  Robust std:            {unc_metrics['std_makespan']:.1f}")
print(f"  Deterioration ratio:   {unc_metrics['deterioration_ratio']:.4f}")
print(f"  FEV count:             {ga_unc.fev_count}")

print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)
print(f"{'Metric':<25} {'Deterministic':<15} {'Uncertainty':<15}")
print("-" * 55)
print(f"{'Nominal makespan':<25} {det_nominal:<15} {unc_nominal:<15}")
print(f"{'Robust mean':<25} {det_metrics['mean_makespan']:<15.1f} {unc_metrics['mean_makespan']:<15.1f}")
print(f"{'Deterioration ratio':<25} {det_metrics['deterioration_ratio']:<15.4f} {unc_metrics['deterioration_ratio']:<15.4f}")
print(f"{'FEV count':<25} {ga_det.fev_count:<15} {ga_unc.fev_count:<15}")

# Winner
if unc_metrics['mean_makespan'] < det_metrics['mean_makespan']:
    print("\n[PASS] Uncertainty-aware GA produces more robust schedules")
elif unc_metrics['mean_makespan'] == det_metrics['mean_makespan']:
    print("\n[NEUTRAL] Both algorithms produce identical robust schedules")
else:
    print("\n[NOTE] Uncertainty-aware GA has higher robust mean; check parameters")
