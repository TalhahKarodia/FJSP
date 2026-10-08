"""
FJSSP-W Parser for Competition Instances

Parses .fjs files from the FJSSP-W-Competition.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.benchmark_parser import WorkerBenchmarkParser


def parse_competition_instance(filepath):
    """
    Parse a competition .fjs file and return FJSSP-W data.
    
    Returns:
        dict: {
            'n_jobs': int,
            'n_machines': int,
            'n_workers': int,
            'n_operations': int,
            'durations': 3D list [op][machine][worker] = time,
            'job_sequence': list of lists,
            'encoding': WorkerEncoding object
        }
    """
    parser = WorkerBenchmarkParser()
    encoding = parser.parse_benchmark(filepath)
    
    # Get basic info
    n_jobs = encoding.n_jobs()
    n_machines = encoding.n_machines()
    n_operations = encoding.n_operations()
    
    # Get durations
    durations = encoding.durations()
    
    # Infer number of workers from durations
    if n_operations > 0 and n_machines > 0:
        n_workers = len(durations[0][0]) if len(durations[0]) > 0 else 0
    else:
        n_workers = 0
    
    # Get job sequence
    job_sequence = encoding.job_sequence()
    
    return {
        'n_jobs': n_jobs,
        'n_machines': n_machines,
        'n_workers': n_workers,
        'n_operations': n_operations,
        'durations': durations,
        'job_sequence': job_sequence,
        'encoding': encoding
    }


def get_workers_for_operation(encoding, op_idx):
    """Get all workers that can process an operation."""
    return encoding.get_workers_for_operation(op_idx)


def get_workers_for_operation_on_machine(encoding, op_idx, machine_idx):
    """Get all workers that can process an operation on a specific machine."""
    return encoding.get_workers_for_operation_on_machine(op_idx, machine_idx)


def is_feasible(encoding, op_idx, machine_idx, worker_idx):
    """Check if a machine-worker pair is feasible for an operation."""
    return encoding.is_possible(op_idx, machine_idx, worker_idx)


if __name__ == "__main__":
    # Test the parser
    test_file = "../FJSSP-W-Competition/instances/fjssp-w/1_Brandimarte_7_workers.fjs"
    
    if os.path.exists(test_file):
        data = parse_competition_instance(test_file)
        print("=" * 60)
        print("PARSER TEST RESULTS")
        print("=" * 60)
        print(f"Instance: 1_Brandimarte_7_workers.fjs")
        print(f"Jobs: {data['n_jobs']}")
        print(f"Machines: {data['n_machines']}")
        print(f"Workers: {data['n_workers']}")
        print(f"Operations: {data['n_operations']}")
        print(f"Job Sequence (first 2 jobs): {data['job_sequence'][:2]}")
        print(f"Durations shape: {len(data['durations'])} ops")
        
        # Test feasibility check
        if data['n_operations'] > 0 and data['n_machines'] > 0 and data['n_workers'] > 0:
            encoding = data['encoding']
            op0 = 0
            print(f"\nOperation {op0} feasibility:")
            for m in range(min(3, data['n_machines'])):
                workers = encoding.get_workers_for_operation_on_machine(op0, m)
                if workers:
                    print(f"  Machine {m}: workers {workers[:5]}{'...' if len(workers) > 5 else ''}")
                else:
                    print(f"  Machine {m}: no feasible workers")
        
        print("\nParser working correctly!")
    else:
        print(f"Test file not found: {test_file}")