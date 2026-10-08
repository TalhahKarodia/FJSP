"""
Decoder for the FJSP (deterministic, no workers yet).

A chromosome is:
  - sequence: a list of job_ids, length = num_operations. The k-th occurrence
    of job_id j in this list refers to operation j's k-th operation (0-based).
    This is the standard "operation-based encoding" (Section 4.2.2 of proposal):
    any permutation is automatically precedence-feasible after decoding.
  - assignment: dict (job_id, op_index) -> machine_id, one entry per operation.

The decoder walks the sequence in order, and for each operation:
  - looks up its assigned machine
  - starts it at max(job's previous op finish time, machine's free time)
  - records finish time
Returns per-operation start/end times and the makespan.
"""
from collections import defaultdict


def decode(instance, sequence, assignment):
    """
    instance: FJSPInstance
    sequence: list[int] of length num_operations, values are job ids
    assignment: dict[(job_id, op_index)] -> machine_id

    Returns: (makespan, schedule) where schedule is a list of dicts:
        {job, op_index, machine, start, end}
    """
    # Track how many operations of each job we've already placed (to know op_index)
    next_op_index = defaultdict(int)
    # Track when each job's last operation finished (precedence constraint)
    job_ready_time = defaultdict(int)
    # Track when each machine becomes free
    machine_free_time = defaultdict(int)

    schedule = []

    for job_id in sequence:
        op_index = next_op_index[job_id]
        next_op_index[job_id] += 1

        op = instance.jobs[job_id][op_index]
        machine = assignment[(job_id, op_index)]

        if machine not in op.eligible:
            raise ValueError(
                f"Infeasible assignment: job {job_id} op {op_index} "
                f"cannot run on machine {machine}"
            )
        proc_time = op.eligible[machine]

        start = max(job_ready_time[job_id], machine_free_time[machine])
        end = start + proc_time

        job_ready_time[job_id] = end
        machine_free_time[machine] = end

        schedule.append({
            "job": job_id, "op_index": op_index, "machine": machine,
            "start": start, "end": end,
        })

    makespan = max(entry["end"] for entry in schedule) if schedule else 0
    return makespan, schedule


def random_feasible_chromosome(instance, rng):
    """Builds one uniformly random feasible chromosome: a random valid
    operation sequence (precedence-respecting) plus a random eligible
    machine per operation. Useful for testing and GA population seeding."""
    # sequence: each job_id appears (num_ops_in_job) times, then shuffled
    # (shuffling a bag of job-ids is precedence-safe under operation-based encoding)
    sequence = []
    for job_id, ops in enumerate(instance.jobs):
        sequence.extend([job_id] * len(ops))
    rng.shuffle(sequence)

    assignment = {}
    for job_id, ops in enumerate(instance.jobs):
        for op in ops:
            machine = rng.choice(list(op.eligible.keys()))
            assignment[(job_id, op.op_index)] = machine

    return sequence, assignment


if __name__ == "__main__":
    import random
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPO_ROOT / "src" / "parser"))
    from fjsp_parser import parse_fjsp_file

    inst = parse_fjsp_file(REPO_ROOT / "data-sample" / "kacem" / "k1.txt")
    rng = random.Random(42)
    seq, asg = random_feasible_chromosome(inst, rng)
    makespan, sched = decode(inst, seq, asg)
    print(f"Random chromosome makespan: {makespan} (known optimum: 11)")
    for entry in sched:
        print(entry)