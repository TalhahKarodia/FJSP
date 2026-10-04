"""Run best solution through competition evaluation tools."""
import sys
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA
import glob, io, contextlib

path = glob.glob("src/FJSSP-W-Competition/instances/fjssp-w/*Fattahi_20*")[0]
instance = parse_competition_instance(path)

# Run GA
ga = FJSSPW_GA(instance, population_size=50, generations=100, random_seed=0)
with contextlib.redirect_stdout(io.StringIO()):
    result = ga.evolve()

# best_solution is the CHROMOSOME dict, not the decoded schedule
chromosome = result['best_solution']

# Decode to get start/end times
decoded = ga.decoder.decode(chromosome)

s = decoded['start_times']
e = decoded['end_times']
m = decoded['machine_assignment']
w = decoded['worker_assignment']
d = instance['durations']

print(f"GA makespan:                  {decoded['makespan']}")

# Try competition's makespan function
try:
    from util.evaluation import makespan as comp_makespan
    comp_result = comp_makespan(s, m, w, d)
    print(f"Competition eval makespan:    {comp_result}")
except ImportError:
    print("Competition eval tools not importable; skipping")
except Exception as ex:
    print(f"Competition eval failed: {ex}")

# Worker balance
try:
    from util.evaluation import workload_balance
    balance = workload_balance(m, w, d)
    print(f"Worker balance (CV or value): {balance}")
except ImportError:
    print("workload_balance not available")
except Exception as ex:
    print(f"workload_balance failed: {ex}")

# Per-worker workload computed manually
from collections import defaultdict
worker_load = defaultdict(float)
for op_idx, machine in enumerate(m):
    worker = w[op_idx]
    dur = d[op_idx][machine][worker]
    worker_load[worker] += dur

print("\nPer-worker workload (manual):")
for w_id in sorted(worker_load.keys()):
    print(f"  Worker {w_id:>2}: {worker_load[w_id]:.0f}")

loads = list(worker_load.values())
mean_load = sum(loads) / len(loads)
variance = sum((l - mean_load) ** 2 for l in loads) / len(loads)
std_dev = variance ** 0.5
print(f"\nMean load:          {mean_load:.1f}")
print(f"Std dev:            {std_dev:.1f}")
print(f"Coef of variation:  {std_dev / mean_load * 100:.2f}%")

# Per-machine workload
machine_load = defaultdict(float)
for op_idx, machine in enumerate(m):
    worker = w[op_idx]
    dur = d[op_idx][machine][worker]
    machine_load[machine] += dur

print("\nPer-machine workload:")
for m_id in sorted(machine_load.keys()):
    print(f"  Machine {m_id:>2}: {machine_load[m_id]:.0f}")
