"""
Worker-Aware Decoder for FJSSP-W

Decodes a chromosome (machine assignment, worker assignment, operation
sequence) into a feasible schedule.

Fixes applied:
1. Implemented Topological List Decoding in `decode()` to strictly enforce intra-job 
   precedence (an operation is only scheduled after its in-job predecessors complete).
2. Maintains `operation_sequence` as a priority list, preserving JOX and OX crossover preferences.
3. Fully compatible with `FJSSPW_GA_V2`.
"""

import sys
import os
import random
from typing import List, Dict, Tuple


class FJSSPWDecoder_V2:
    """
    Decoder for FJSSP-W that creates feasible schedules using Topological List Decoding.

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

        # Build op -> job mapping correctly from job_sequence
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
        """
        Decodes a chromosome into a feasible schedule.

        Uses Topological List Decoding to enforce intra-job precedence constraints:
        an operation is eligible only after all previous operations in the same job
        have finished. Among eligible operations, the one appearing earliest in 
        `operation_sequence` is scheduled first.
        """
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
        job_last_end_time = {}

        # Priority lookup based on sequence order
        seq_priority = {op_idx: rank for rank, op_idx in enumerate(operation_sequence)}

        # Track position of next ready operation per job
        next_op_pos_in_job = {job_id: 0 for job_id in range(self.n_jobs)}

        scheduled_count = 0
        while scheduled_count < self.n_operations:
            # Find operations whose job predecessors have completed
            eligible_ops = []
            for job_id, pos in next_op_pos_in_job.items():
                if pos < len(self.job_op_indices.get(job_id, [])):
                    op_idx = self.job_op_indices[job_id][pos]
                    eligible_ops.append(op_idx)

            if not eligible_ops:
                break

            # Select ready operation with highest priority in operation_sequence
            op_idx = min(eligible_ops, key=lambda op: seq_priority[op])
            job_id = self.op_to_job[op_idx]

            machine = machine_assignment[op_idx]
            worker = worker_assignment[op_idx]

            if not self.encoding.is_possible(op_idx, machine, worker):
                raise ValueError(
                    f"Operation {op_idx}: Machine {machine} + Worker {worker} is infeasible!"
                )

            proc_time = self.durations[op_idx][machine][worker]

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

            next_op_pos_in_job[job_id] += 1
            scheduled_count += 1

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
        """Create a purely random feasible chromosome."""
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

    def create_greedy_chromosome(self, rng: random.Random = None,
                                  greedy_prob: float = 0.7) -> Dict:
        """
        Create a chromosome biased toward fast (machine, worker) pairs.

        For each operation:
          - with probability `greedy_prob`, pick the cheapest feasible pair
          - otherwise, pick randomly from the top 10% cheapest pairs
        This preserves diversity while seeding good building blocks.
        """
        if rng is None:
            rng = random.Random()

        n_ops = self.n_operations
        machine_assignment = []
        worker_assignment = []

        for op_idx in range(n_ops):
            feasible = []
            for m in range(self.n_machines):
                for w in self.get_eligible_workers(op_idx, m):
                    dur = self.durations[op_idx][m][w]
                    if dur > 0:
                        feasible.append((dur, m, w))

            if not feasible:
                raise ValueError(f"No feasible pair for operation {op_idx}")

            feasible.sort()  # ascending by duration

            if rng.random() < greedy_prob:
                _, m, w = feasible[0]
            else:
                top_k = max(1, len(feasible) // 10)
                _, m, w = rng.choice(feasible[:top_k])

            machine_assignment.append(m)
            worker_assignment.append(w)

        operation_sequence = list(range(n_ops))
        rng.shuffle(operation_sequence)

        return {
            'machine_assignment': machine_assignment,
            'worker_assignment': worker_assignment,
            'operation_sequence': operation_sequence
        }