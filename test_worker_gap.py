"""Compare makespan using assigned workers vs min workers."""
import sys, io, contextlib, glob, os
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA

paths = sorted(glob.glob("../FJSSP-W-Competition/instances/fjssp-w/*BehnkeGeiger_42*"))
inst = parse_competition_instance(paths[0])
print(f"Instance: {os.path.basename(paths[0])}")

# Run GA
ga = FJSSPW_GA(inst, population_size=30, generations=50, random_seed=0,
               fitness_mode='deterministic', balance_weight=0.0)
with contextlib.redirect_stdout(io.StringIO()):
    r = ga.evolve()

chrom = r['best_solution']
decoded = ga.decoder.decode(chrom)
print(f"\nProposed FPC-GA makespan:    {decoded['makespan']}")

# Compute what the makespan would be if we used the MIN worker per op
d = inst['durations']
total_assigned = 0
total_min = 0
for op_idx in range(inst['n_operations']):
    m = chrom['machine_assignment'][op_idx]
    w = chrom['worker_assignment'][op_idx]
    assigned_dur = d[op_idx][m][w]
    workers = inst['encoding'].get_workers_for_operation_on_machine(op_idx, m)
    min_dur = min(d[op_idx][m][ww] for ww in workers) if workers else assigned_dur
    total_assigned += assigned_dur
    total_min += min_dur

print(f"\nSum of assigned durations:  {total_assigned}")
print(f"Sum of min durations:       {total_min}")
print(f"Ratio (assigned/min):       {total_assigned/total_min:.2f}x")

# Also: how many ops have an assigned worker that is NOT the fastest?
not_fastest = 0
for op_idx in range(inst['n_operations']):
    m = chrom['machine_assignment'][op_idx]
    w = chrom['worker_assignment'][op_idx]
    assigned_dur = d[op_idx][m][w]
    workers = inst['encoding'].get_workers_for_operation_on_machine(op_idx, m)
    if workers:
        min_dur = min(d[op_idx][m][ww] for ww in workers)
        if assigned_dur > min_dur:
            not_fastest += 1
print(f"\nOps NOT using fastest worker: {not_fastest}/{inst['n_operations']} "
      f"({not_fastest/inst['n_operations']*100:.1f}%)")
