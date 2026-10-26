"""
Test the Worker-Aware Decoder
"""

import sys
import os

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.decoder_w import FJSSPWDecoder  # Changed from decoder to decoder_w


def main():
    print("=" * 60)
    print("TESTING WORKER-AWARE DECODER")
    print("=" * 60)
    
    # Load instance
    instance_path = "../FJSSP-W-Competition/instances/fjssp-w/1_Brandimarte_7_workers.fjs"
    instance_data = parse_competition_instance(instance_path)
    
    print(f"Instance: 1_Brandimarte_7_workers.fjs")
    print(f"Jobs: {instance_data['n_jobs']}")
    print(f"Machines: {instance_data['n_machines']}")
    print(f"Workers: {instance_data['n_workers']}")
    print(f"Operations: {instance_data['n_operations']}")
    print()
    
    # Create decoder
    decoder = FJSSPWDecoder(instance_data)
    
    # Create random chromosome
    print("Creating random feasible chromosome...")
    chromosome = decoder.create_random_chromosome()
    
    # Decode
    print("Decoding schedule...")
    schedule = decoder.decode(chromosome)
    
    # Display results
    print("\nSCHEDULE RESULTS:")
    print(f"Makespan: {schedule['makespan']}")
    print(f"First 5 operations:")
    for i in range(min(5, decoder.n_operations)):
        print(f"  Op {i}: start={schedule['start_times'][i]}, "
              f"end={schedule['end_times'][i]}, "
              f"machine={schedule['machine_assignment'][i]}, "
              f"worker={schedule['worker_assignment'][i]}")
    
    print("\nMachine utilization:")
    # Calculate machine completion times
    machine_end = [0] * decoder.n_machines
    for i in range(decoder.n_operations):
        m = schedule['machine_assignment'][i]
        if schedule['end_times'][i] > machine_end[m]:
            machine_end[m] = schedule['end_times'][i]
    
    for m in range(decoder.n_machines):
        util = (machine_end[m] / schedule['makespan']) * 100 if schedule['makespan'] > 0 else 0
        print(f"  Machine {m}: {util:.1f}% utilized")
    
    print("\nWorker utilization (first 10 workers):")
    worker_end = [0] * decoder.n_workers
    for i in range(decoder.n_operations):
        w = schedule['worker_assignment'][i]
        if schedule['end_times'][i] > worker_end[w]:
            worker_end[w] = schedule['end_times'][i]
    
    for w in range(min(10, decoder.n_workers)):
        util = (worker_end[w] / schedule['makespan']) * 100 if schedule['makespan'] > 0 else 0
        print(f"  Worker {w}: {util:.1f}% utilized")
    
    print("\nDecoder test complete!")


if __name__ == "__main__":
    main()