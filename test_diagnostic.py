"""Diagnose why proposed GA underperforms on Behnke42."""
import sys, io, contextlib, glob, os
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
from baselines.standard_ga import StandardGA

paths = sorted(glob.glob("../FJSSP-W-Competition/instances/fjssp-w/*.fjs"))
inst_path = [p for p in paths if 'BehnkeGeiger_42' in p][0]
print(f"Instance: {os.path.basename(inst_path)}")
inst = parse_competition_instance(inst_path)

print(f"Jobs={inst['n_jobs']}, Machines={inst['n_machines']}, "
      f"Workers={inst['n_workers']}, Ops={inst['n_operations']}")
print(f"job_sequence (first 15): {inst['job_sequence'][:15]}")
print(f"job_sequence length: {len(inst['job_sequence'])}")

# Check op_to_job in decoder
from ga.fjssp_w.decoder_w import FJSSPWDecoder
dec = FJSSPWDecoder(inst)
print(f"\ndecoder.op_to_job (first 10): {dict(list(dec.op_to_job.items())[:10])}")
print(f"decoder.job_op_indices (first 5): {dict(list(dec.job_op_indices.items())[:5])}")
print(f"decoder.n_workers: {dec.n_workers}")

# Check durations shape
d = inst['durations']
print(f"\ndurations shape: {len(d)} ops, each {len(d[0])} machines, each {len(d[0][0])} workers")

# Count nonzero duration entries for first op
op0 = d[0]
nonzero = sum(1 for m in range(len(op0)) for w in range(len(op0[m])) if op0[m][w] > 0)
print(f"Op 0: {nonzero} nonzero durations across 60 machines x 90 workers")

# Simple sanity: minimum makespan if all ops run in parallel
min_durs = []
for op_idx in range(len(d)):
    vals = [d[op_idx][m][w] for m in range(len(d[op_idx]))
            for w in range(len(d[op_idx][m])) if d[op_idx][m][w] > 0]
    if vals:
        min_durs.append(min(vals))
print(f"\nSum of per-op minimum durations: {sum(min_durs)}")
print(f"Max of per-op minimum durations: {max(min_durs)}")
print(f"Lower bound on makespan: {max(max(min_durs), sum(min_durs) // inst['n_machines'])}")

# Check with a quick proposed GA run
print("\n=== Proposed GA (10 gen) ===")
ga = FJSSPW_GA(inst, population_size=10, generations=10, random_seed=0,
               fitness_mode='deterministic', balance_weight=0.0)
with contextlib.redirect_stdout(io.StringIO()):
    r = ga.evolve()
print(f"  makespan: {r['best_makespan']}")

# Check balance weight effect
print("\n=== Proposed GA with balance_weight=0.0 vs 0.1 ===")
for bw in [0.0, 0.1, 0.5]:
    ga = FJSSPW_GA(inst, population_size=10, generations=10, random_seed=0,
                   fitness_mode='deterministic', balance_weight=bw)
    with contextlib.redirect_stdout(io.StringIO()):
        r = ga.evolve()
    decoded = ga.decoder.decode(r['best_solution'])
    print(f"  bw={bw}: fitness={r['best_makespan']:.1f}, actual makespan={decoded['makespan']}")
