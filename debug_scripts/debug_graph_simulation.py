"""
Debug the Graph simulation to understand why processing times aren't changing
"""
import sys
sys.path.append('src')

from util.benchmark_parser import WorkerBenchmarkParser
from util.uncertainty import create_uncertainty_vector
from util.graph import Graph
from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.decoder_w import FJSSPWDecoder

print("=" * 60)
print("DEBUGGING GRAPH SIMULATION")
print("=" * 60)

# Load instance
instance_path = "../FJSSP-W-Competition/instances/fjssp-w/1_Brandimarte_7_workers.fjs"
instance_data = parse_competition_instance(instance_path)

durations = instance_data['durations']
job_sequence = instance_data['job_sequence']
n_workers = instance_data['n_workers']

# Create a real schedule
decoder = FJSSPWDecoder(instance_data)
chromosome = decoder.create_random_chromosome()
schedule = decoder.decode(chromosome)

start_times = schedule['start_times']
end_times = schedule['end_times']
machine_assignment = schedule['machine_assignment']
worker_assignment = schedule['worker_assignment']

print(f"Original makespan: {max(end_times)}")
print(f"Number of operations: {len(start_times)}")

# Create uncertainty parameters
uncertainty_params = create_uncertainty_vector(n_workers)
print(f"Uncertainty params for first 3 workers:")
for i in range(min(3, n_workers)):
    print(f"  Worker {i}: alpha={uncertainty_params[i][0]:.3f}, beta={uncertainty_params[i][1]:.3f}, offset={uncertainty_params[i][2]:.3f}")

print("\nRunning simulation and checking processing times...")

# Create graph
g = Graph(start_times, end_times, machine_assignment, worker_assignment, job_sequence)

# Check original processing times
original_processing_times = []
for i in range(min(10, len(start_times))):
    op = i
    m = machine_assignment[i]
    w = worker_assignment[i]
    orig_time = durations[op][m][w]
    original_processing_times.append(orig_time)
    print(f"Operation {i}: machine={m}, worker={w}, original time={orig_time}")

print("\nRunning simulation...")
g.simulate(durations, uncertainty_params, processing_times=True)

# Check simulated processing times
print("\nAfter simulation:")
for i in range(min(10, len(start_times))):
    op = i
    orig_time = original_processing_times[i]
    sim_time = g.e[i] - g.s[i]  # end - start
    print(f"Operation {i}: original={orig_time}, simulated={sim_time}, changed={sim_time != orig_time}")

print(f"\nSimulated makespan: {max(g.e)}")
print(f"Original makespan: {max(end_times)}")

# Check if any operation changed
any_changed = any(g.e[i] - g.s[i] != original_processing_times[i] for i in range(len(start_times)))
print(f"\nAny operation processing time changed? {any_changed}")

# Check if graph.e is different from original end_times
end_diff = any(g.e[i] != end_times[i] for i in range(len(end_times)))
print(f"Any end time changed? {end_diff}")

if not any_changed and not end_diff:
    print("\nPROBLEM: The simulation is not modifying any processing times!")
    print("This means the Graph.simulate() method is not applying uncertainty correctly.")
