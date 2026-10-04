"""Compute lower bounds for the Fattahi20 instance."""
import sys
sys.path.insert(0, 'src')

from parser.fjssp_w_parser import parse_competition_instance
import glob

path = glob.glob("src/FJSSP-W-Competition/instances/fjssp-w/*Fattahi_20*")[0]
inst = parse_competition_instance(path)

durations = inst['durations']
n_ops = inst['n_operations']
n_jobs = inst['n_jobs']

print(f"Instance: {path}")
print(f"Jobs={n_jobs}, Ops={n_ops}\n")

# LB1: sum of minimum durations across all ops (naive lower bound)
total_min = 0
for op_idx, d_op in enumerate(durations):
    # d_op is a 2D array [machine][worker]
    vals = []
    for m in range(len(d_op)):
        row = d_op[m]
        for w in range(len(row)):
            if row[w] > 0:
                vals.append(row[w])
    if vals:
        total_min += min(vals)

print(f"LB1 (sum of per-op minimum durations): {total_min}")

# Also compute longest path through jobs by summing per-job minimums
# Need op->job mapping. Use the decoder's logic.
from ga.fjssp_w.decoder_w import FJSSPWDecoder
dec = FJSSPWDecoder(inst)

# Reconstruct job durations
job_min_sum = {}
job_op_count = {}
for op_idx in range(n_ops):
    # Find min duration for this op
    vals = []
    for m in range(len(durations[op_idx])):
        for w in range(len(durations[op_idx][m])):
            if durations[op_idx][m][w] > 0:
                vals.append(durations[op_idx][m][w])
    if not vals:
        continue
    min_dur = min(vals)
    # Which job?
    job = dec.op_to_job.get(op_idx)
    if job is None:
        continue
    job_min_sum[job] = job_min_sum.get(job, 0) + min_dur

# LB2: max over jobs of (sum of min durations of its ops)
lb2 = max(job_min_sum.values()) if job_min_sum else 0

print(f"LB2 (longest job path): {lb2}")

# LB3: max of (LB1 / n_machines, LB2) — better bound
lb3 = max(total_min / inst['n_machines'], lb2)
print(f"LB3 (max of LB1/m, LB2): {lb3:.1f}")

print(f"\nGA result:               5238")
print(f"Gap vs LB3:              {(5238 - lb3) / lb3 * 100:.1f}%")