"""
Worker-Aware Decoder for FJSSP-W

This module decodes a chromosome (machine assignment, worker assignment, 
operation sequence) into a feasible schedule.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import random
from typing import List, Dict, Tuple


class FJSSPWDecoder:
    """
    Decoder for FJSSP-W that creates feasible schedules.
    
    The chromosome consists of three parts:
    1. Machine assignment vector: [m0, m1, ..., m_n_ops-1]
    2. Worker assignment vector: [w0, w1, ..., w_n_ops-1]
    3. Operation sequence vector: [op0, op1, ..., op_n_ops-1]
    """
    
    def __init__(self, instance_data):
        """
        Initialize the decoder with instance data.
        
        Args:
            instance_data: dict from parse_competition_instance()
        """
        self.n_jobs = instance_data['n_jobs']
        self.n_machines = instance_data['n_machines']
        self.n_workers = instance_data['n_workers']
        self.n_operations = instance_data['n_operations']
        self.durations = instance_data['durations']
        self.job_sequence = instance_data['job_sequence']
        self.encoding = instance_data['encoding']
        
        # Build operation -> job mapping
        # For competition instances, each job typically has one operation
        # But we need to handle both cases
        self.op_to_job = {}
        self.job_op_indices = {}
        
        # Check if job_sequence is a flat list or list of lists
        if self.job_sequence and isinstance(self.job_sequence[0], int):
            # Flat list: each job has one operation
            for job_id in range(self.n_jobs):
                op_idx = job_id
                self.op_to_job[op_idx] = job_id
                self.job_op_indices[job_id] = [op_idx]
        else:
            # List of lists: each job has multiple operations
            for job_id, ops in enumerate(self.job_sequence):
                self.job_op_indices[job_id] = list(ops)
                for op_idx in ops:
                    self.op_to_job[op_idx] = job_id
        
        # Build operation position within job
        self.op_position_in_job = {}
        for job_id, ops in self.job_op_indices.items():
            for pos, op_idx in enumerate(ops):
                self.op_position_in_job[op_idx] = pos
    
    def decode(self, chromosome: Dict) -> Dict:
        """
        Decode a chromosome into a feasible schedule.
        
        Args:
            chromosome: dict with keys:
                - 'machine_assignment': list of machine indices
                - 'worker_assignment': list of worker indices  
                - 'operation_sequence': list of operation indices
        
        Returns:
            dict: {
                'start_times': list of start times for each operation,
                'end_times': list of end times for each operation,
                'machine_assignment': list of machine indices,
                'worker_assignment': list of worker indices,
                'makespan': float
            }
        """
        machine_assignment = chromosome['machine_assignment']
        worker_assignment = chromosome['worker_assignment']
        operation_sequence = chromosome['operation_sequence']
        
        # Validate chromosome length
        assert len(machine_assignment) == self.n_operations, \
            f"Machine assignment length {len(machine_assignment)} != {self.n_operations}"
        assert len(worker_assignment) == self.n_operations, \
            f"Worker assignment length {len(worker_assignment)} != {self.n_operations}"
        assert len(operation_sequence) == self.n_operations, \
            f"Operation sequence length {len(operation_sequence)} != {self.n_operations}"
        
        # Initialize data structures
        start_times = [0] * self.n_operations
        end_times = [0] * self.n_operations
        machine_end_times = [0] * self.n_machines
        worker_end_times = [0] * self.n_workers
        
        # For each job, track when the previous operation finished
        job_last_end_time = [0] * self.n_jobs
        
        # Decode each operation in sequence order
        for op_idx in operation_sequence:
            # Get assigned machine and worker
            machine = machine_assignment[op_idx]
            worker = worker_assignment[op_idx]
            
            # Validate feasibility
            if not self.encoding.is_possible(op_idx, machine, worker):
                raise ValueError(f"Operation {op_idx}: Machine {machine} + Worker {worker} is infeasible!")
            
            # Get processing time
            proc_time = self.durations[op_idx][machine][worker]
            
            # Find job ID for this operation
            job_id = self.op_to_job.get(op_idx, 0)
            
            # Calculate earliest start time (max of: machine availability, worker availability, job precedence)
            earliest_start = max(
                machine_end_times[machine],
                worker_end_times[worker],
                job_last_end_time[job_id]
            )
            
            # Schedule the operation
            start_times[op_idx] = earliest_start
            end_times[op_idx] = earliest_start + proc_time
            
            # Update resources
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
        """Get processing time for an operation on a machine-worker pair."""
        return self.durations[op_idx][machine][worker]
    
    def is_feasible_assignment(self, op_idx: int, machine: int, worker: int) -> bool:
        """Check if a machine-worker assignment is feasible."""
        return self.encoding.is_possible(op_idx, machine, worker)
    
    def get_eligible_workers(self, op_idx: int, machine: int) -> List[int]:
        """Get eligible workers for an operation on a specific machine."""
        return self.encoding.get_workers_for_operation_on_machine(op_idx, machine)
    
    def get_eligible_machines(self, op_idx: int) -> List[int]:
        """Get all machines that can process an operation."""
        machines = []
        for m in range(self.n_machines):
            workers = self.get_eligible_workers(op_idx, m)
            if workers:
                machines.append(m)
        return machines
    
    def create_random_chromosome(self) -> Dict:
        """Create a random feasible chromosome."""
        n_ops = self.n_operations
        
        # Create random machine assignments (feasible)
        machine_assignment = []
        worker_assignment = []
        
        for op_idx in range(n_ops):
            # Find a feasible machine-worker pair
            feasible_pairs = []
            for m in range(self.n_machines):
                for w in self.get_eligible_workers(op_idx, m):
                    feasible_pairs.append((m, w))
            
            if not feasible_pairs:
                raise ValueError(f"No feasible pair for operation {op_idx}")
            
            # Randomly choose a feasible pair
            m, w = random.choice(feasible_pairs)
            machine_assignment.append(m)
            worker_assignment.append(w)
        
        # Create random operation sequence
        operation_sequence = list(range(n_ops))
        random.shuffle(operation_sequence)
        
        return {
            'machine_assignment': machine_assignment,
            'worker_assignment': worker_assignment,
            'operation_sequence': operation_sequence
        }