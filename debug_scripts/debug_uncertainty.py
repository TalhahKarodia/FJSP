"""
Debug script to test uncertainty simulation directly
"""
import sys
sys.path.append('src')

from util.benchmark_parser import WorkerBenchmarkParser
from util.uncertainty import create_uncertainty_vector
from util.graph import Graph
from parser.fjssp_w_parser import parse_competition_instance

print("=" * 60)
print("DEBUGGING UNCERTAINTY SIMULATION DIRECTLY")
print("=" * 60)

# Load instance
instance_path = "../FJSSP-W-Competition/instances/fjssp-w/1_Brandimarte_7_workers.fjs"
instance_data = parse_competition_instance(instance_path)

durations = instance_data['durations']
job_sequence = instance_data['job_sequence']
n_workers = instance_data['n_workers']
n_operations = instance_data['n_operations']

print(f"Instance: 1_Brandimarte_7_workers.fjs")
print(f"Workers: {n_workers}")
print(f"Operations: {n_operations}")
print()

# Create a simple schedule with known processing times
# Use operation 0 as an example
print("Checking operation 0 processing times:")
for machine in range(min(3, instance_data['n_machines'])):
    for worker in range(min(3, n_workers)):
        time = durations[0][machine][worker]
        if time > 0:
            print(f"  Machine {machine}, Worker {worker}: {time}")
print()

# Create uncertainty parameters
print("Creating uncertainty parameters...")
uncertainty_params = create_uncertainty_vector(n_workers)
print(f"Uncertainty params for first 3 workers:")
for i in range(min(3, n_workers)):
    print(f"  Worker {i}: {uncertainty_params[i]}")

print()
print("Running simulation on a simple schedule...")

# Create a schedule with only operation 0
start_times = [0] * n_operations
end_times = [0] * n_operations
machine_assignment = [0] * n_operations
worker_assignment = [0] * n_operations

# Set operation 0 to run on machine 0, worker 0
start_times[0] = 0
end_times[0] = durations[0][0][0]
machine_assignment[0] = 0
worker_assignment[0] = 0

print(f"Original processing time for op 0: {end_times[0]}")

# Create graph and simulate
g = Graph(start_times, end_times, machine_assignment, worker_assignment, job_sequence)
g.simulate(durations, uncertainty_params, processing_times=True)

print(f"After simulation, op 0 end time: {g.e[0]}")
print(f"Original: {end_times[0]}, Simulated: {g.e[0]}")

if g.e[0] == end_times[0]:
    print("WARNING: No change in processing time!")
    print("This means uncertainty simulation is not applying variations.")
else:
    print("SUCCESS: Processing time changed!")

print()
print("Running multiple simulations to test variation...")
results = []
for i in range(10):
    g = Graph(start_times, end_times, machine_assignment, worker_assignment, job_sequence)
    g.simulate(durations, uncertainty_params, processing_times=True)
    results.append(g.e[0])
    print(f"  Sim {i+1}: {g.e[0]}")

print()
print(f"Results: {results}")
if len(set(results)) == 1:
    print("ERROR: All simulations gave the same result!")
    print("The uncertainty module is not working as expected.")
else:
    print("SUCCESS: Simulations vary!")
