"""
Worker-Aware Decoder for FJSSP-W

Decodes a chromosome (machine assignment, worker assignment, operation
sequence) into a feasible schedule.

Fixes applied:
1. create_random_chromosome accepts an optional `rng` parameter so each
   GA instance uses its own RNG stream.
2. op_to_job and job_op_indices are built correctly from job_sequence,
   supporting multi-operation jobs (Behnke, Hurink, Barnes, etc.) rather
   than assuming one operation per job.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import random
from typing import List, Dict, Tuple


class FJSSPWDecoder:
    """
    Decoder for FJSSP-W that creates feasible schedules.

    Chromosome structure:
    1. machine_assignment: [m0, m1, ..., m_(n_ops-1)]
    2. worker_assignment:  [w0, w1, ..., w_(n_ops-1)]
    3. operation_sequence: permutation of [0..n_ops-1]
    """

    def __init__(self, instance_data):
        self.n_jobs = instance_data['n_jobs']
        self.n_machines = instance_data['n_machines']
        self.n_workers = instance_data['n_workers']
        self.n_operations = instance_data['n_operations']
        self.durations = instance_data['durations']
        self.job_sequence = instance_data['job_sequence']
        self.encoding = instance_data['encoding']

        # --- Build op -> job mapping correctly ---
        # job_sequence[i] = job_id of operation i.
        # This works for any instance (1 or more operations per job).
        self.op_to_job = {}
        self.job_op_indices = {}

        for op_idx, job_id in enumerate(self.job_sequence):
            self.op_to_job[op_idx] = job_id
            if job_id not in self.job_op_indices:
                self.job_op_indices[job_id] = []
            self.job_op_indices[job_id].append(op_idx)

        # Position of each operation within its job
        self.op_position_in_job = {}
        for job_id, ops in self.job_op_indices.items():
            for pos, op_idx in enumerate(ops):
                self.op_position_in_job[op_idx] = pos

    def decode(self, chromosome: Dict) -> Dict:
        machine_assignment = chromosome['machine_assignment']
        worker_assignment = chromosome['worker_assignment']
        operation_sequence = chromosome['operation_sequence']

        assert len(machine_assignment) == self.n_operations, \
            f"Machine assignment length {len(machine_assignment)} != {self.n_operations}"
        assert len(worker_assignment) == self.n_operations, \
            f"Worker assignment length {len(worker_assignment)} != {self.n_operations}"
        assert len(operation_sequence) == self.n_operations, \
            f"Operation sequence length {len(operation_sequence)} != {self.n_operations}"

        start_times = [0] * self.n_operations
        end_times = [0] * self.n_operations
        machine_end_times = [0] * self.n_machines
        worker_end_times = [0] * self.n_workers

        # Track last end per job_id using a dict (job ids may not be 0..n_jobs-1)
        job_last_end_time = {}

        for op_idx in operation_sequence:
            machine = machine_assignment[op_idx]
            worker = worker_assignment[op_idx]

            if not self.encoding.is_possible(op_idx, machine, worker):
                raise ValueError(
                    f"Operation {op_idx}: Machine {machine} + Worker {worker} is infeasible!"
                )

            proc_time = self.durations[op_idx][machine][worker]
            job_id = self.op_to_job.get(op_idx, 0)

            earliest_start = max(
                machine_end_times[machine],
                worker_end_times[worker],
                job_last_end_time.get(job_id, 0),
            )

            start_times[op_idx] = earliest_start
            end_times[op_idx] = earliest_start + proc_time

            machine_end_times[machine] = end_times[op_idx]
            worker_end_times[worker] = end_times[op_idx]
            job_last_end_time[job_id] = end_times[op_idx]

        makespan = max(end_times) if end_times else 0

        return {
            'start_times': start_times,
            'end_times': end_times,
            'machine_assignment': machine_assignment,
            'worker_assignment': worker_assignment,
            'makespan': makespan
        }

    def get_processing_time(self, op_idx: int, machine: int, worker: int) -> float:
        return self.durations[op_idx][machine][worker]

    def is_feasible_assignment(self, op_idx: int, machine: int, worker: int) -> bool:
        return self.encoding.is_possible(op_idx, machine, worker)

    def get_eligible_workers(self, op_idx: int, machine: int) -> List[int]:
        return self.encoding.get_workers_for_operation_on_machine(op_idx, machine)

    def get_eligible_machines(self, op_idx: int) -> List[int]:
        machines = []
        for m in range(self.n_machines):
            workers = self.get_eligible_workers(op_idx, m)
            if workers:
                machines.append(m)
        return machines

    def create_random_chromosome(self, rng: random.Random = None) -> Dict:
        """Create a random feasible chromosome.

        Args:
            rng: optional random.Random instance. If None, a fresh one is used.
        """
        if rng is None:
            rng = random.Random()

        n_ops = self.n_operations
        machine_assignment = []
        worker_assignment = []

        for op_idx in range(n_ops):
            feasible_pairs = []
            for m in range(self.n_machines):
                for w in self.get_eligible_workers(op_idx, m):
                    feasible_pairs.append((m, w))

            if not feasible_pairs:
                raise ValueError(f"No feasible pair for operation {op_idx}")

            m, w = rng.choice(feasible_pairs)
            machine_assignment.append(m)
            worker_assignment.append(w)

        operation_sequence = list(range(n_ops))
        rng.shuffle(operation_sequence)

        return {
            'machine_assignment': machine_assignment,
            'worker_assignment': worker_assignment,
            'operation_sequence': operation_sequence
        }