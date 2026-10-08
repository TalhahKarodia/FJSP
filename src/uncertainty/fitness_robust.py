"""
Robust Fitness Evaluation for FJSSP-W under Uncertainty (Scenario 2)

This module evaluates schedules under stochastic processing times using
manual uncertainty application (since Graph.simulate() doesn't work properly).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import statistics
import random
import numpy as np
from typing import Dict, List, Tuple

from util.uncertainty import create_uncertainty_vector


class RobustFitnessEvaluator:
    """
    Evaluates FJSSP-W schedules under processing time uncertainty.
    
    For each schedule, runs multiple simulations with stochastic processing
    times and returns the average makespan (robust makespan).
    """
    
    def __init__(self, instance_data: Dict, n_simulations: int = 50):
        """
        Initialize the robust fitness evaluator.
        
        Args:
            instance_data: dict from parse_competition_instance()
            n_simulations: Number of uncertainty simulations per evaluation
        """
        self.instance_data = instance_data
        self.n_simulations = n_simulations
        
        # Get instance parameters
        self.n_jobs = instance_data['n_jobs']
        self.n_machines = instance_data['n_machines']
        self.n_workers = instance_data['n_workers']
        self.n_operations = instance_data['n_operations']
        self.durations = instance_data['durations']
        self.encoding = instance_data['encoding']
        self.job_sequence = instance_data['job_sequence']
        
        # Create uncertainty parameters once (same for all simulations)
        self.uncertainty_params = create_uncertainty_vector(self.n_workers)
        
        print(f"RobustFitnessEvaluator initialized:")
        print(f"  - Workers: {self.n_workers}")
        print(f"  - Simulations per evaluation: {n_simulations}")
    
    def _apply_uncertainty_to_durations(self, durations, uncertainty_params):
        """
        Apply uncertainty to processing times manually.
        
        For each worker, sample from a beta distribution and multiply
        the processing time by (sampled_value + offset).
        """
        # Create a copy of durations with uncertainty applied
        uncertain_durations = []
        
        for op_idx in range(self.n_operations):
            op_durations = []
            for machine_idx in range(self.n_machines):
                machine_workers = []
                for worker_idx in range(self.n_workers):
                    original_time = durations[op_idx][machine_idx][worker_idx]
                    if original_time > 0:
                        # Get uncertainty parameters for this worker
                        alpha, beta, offset = uncertainty_params[worker_idx]
                        
                        # Sample from beta distribution
                        sampled_value = np.random.beta(alpha, beta)
                        
                        # Apply offset and multiply by original time
                        new_time = original_time * (sampled_value + offset)
                        
                        # Ensure time is at least original time (no speedups)
                        # This matches the competition's behavior
                        new_time = max(original_time, new_time)
                        
                        machine_workers.append(new_time)
                    else:
                        machine_workers.append(-1)  # Infeasible
                op_durations.append(machine_workers)
            uncertain_durations.append(op_durations)
        
        return uncertain_durations
    
    def evaluate_robust(self, chromosome: Dict) -> Dict:
        """
        Evaluate a chromosome under uncertainty.
        
        Args:
            chromosome: dict with machine_assignment, worker_assignment, operation_sequence
        
        Returns:
            dict: {
                'deterministic_makespan': float,
                'robust_makespan': float,
                'robust_std': float,
                'simulation_results': list,
                'deterioration_ratio': float
            }
        """
        from ga.fjssp_w.decoder_w import FJSSPWDecoder
        
        # First, get the deterministic schedule
        decoder = FJSSPWDecoder(self.instance_data)
        schedule = decoder.decode(chromosome)
        
        deterministic_makespan = schedule['makespan']
        start_times = schedule['start_times']
        machine_assignment = schedule['machine_assignment']
        worker_assignment = schedule['worker_assignment']
        job_sequence = self.instance_data['job_sequence']
        
        # Run simulations manually with uncertainty applied to durations
        simulation_results = []
        
        for _ in range(self.n_simulations):
            # Create uncertain durations
            uncertain_durations = self._apply_uncertainty_to_durations(
                self.durations, 
                self.uncertainty_params
            )
            
            # Create a new instance data with uncertain durations
            uncertain_instance_data = self.instance_data.copy()
            uncertain_instance_data['durations'] = uncertain_durations
            
            # Create decoder with uncertain durations
            uncertain_decoder = FJSSPWDecoder(uncertain_instance_data)
            
            # Decode the chromosome with uncertain durations
            uncertain_schedule = uncertain_decoder.decode(chromosome)
            
            simulation_results.append(uncertain_schedule['makespan'])
        
        # Calculate statistics
        robust_makespan = statistics.mean(simulation_results)
        robust_std = statistics.stdev(simulation_results) if len(simulation_results) > 1 else 0
        deterioration_ratio = robust_makespan / deterministic_makespan if deterministic_makespan > 0 else 1.0
        
        return {
            'deterministic_makespan': deterministic_makespan,
            'robust_makespan': robust_makespan,
            'robust_std': robust_std,
            'simulation_results': simulation_results,
            'deterioration_ratio': deterioration_ratio
        }
    
    def evaluate_fitness(self, chromosome: Dict) -> float:
        """Fitness function for GA (minimize robust makespan)."""
        result = self.evaluate_robust(chromosome)
        return result['robust_makespan']
    
    def evaluate_robust_detailed(self, chromosome: Dict) -> Dict:
        """Evaluate with detailed breakdown."""
        result = self.evaluate_robust(chromosome)
        sorted_results = sorted(result['simulation_results'])
        
        return {
            'deterministic_makespan': result['deterministic_makespan'],
            'robust_makespan': result['robust_makespan'],
            'robust_std': result['robust_std'],
            'deterioration_ratio': result['deterioration_ratio'],
            'min_simulation': min(result['simulation_results']),
            'max_simulation': max(result['simulation_results']),
            'percentile_25': sorted_results[int(len(sorted_results) * 0.25)] if sorted_results else 0,
            'percentile_50': sorted_results[int(len(sorted_results) * 0.50)] if sorted_results else 0,
            'percentile_75': sorted_results[int(len(sorted_results) * 0.75)] if sorted_results else 0,
            'simulation_results': result['simulation_results']
        }


def test_robust_evaluator():
    """Test the robust fitness evaluator."""
    from parser.fjssp_w_parser import parse_competition_instance
    from ga.fjssp_w.decoder_w import FJSSPWDecoder
    
    print("=" * 60)
    print("TESTING ROBUST FITNESS EVALUATOR (Scenario 2)")
    print("=" * 60)
    
    instance_path = "../FJSSP-W-Competition/instances/fjssp-w/1_Brandimarte_7_workers.fjs"
    instance_data = parse_competition_instance(instance_path)
    
    print(f"Instance: 1_Brandimarte_7_workers.fjs")
    print(f"Jobs: {instance_data['n_jobs']}")
    print(f"Machines: {instance_data['n_machines']}")
    print(f"Workers: {instance_data['n_workers']}")
    print(f"Operations: {instance_data['n_operations']}")
    print()
    
    evaluator = RobustFitnessEvaluator(instance_data, n_simulations=10)
    
    decoder = FJSSPWDecoder(instance_data)
    chromosome = decoder.create_random_chromosome()
    
    print("Evaluating chromosome under uncertainty...")
    result = evaluator.evaluate_robust(chromosome)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Deterministic Makespan: {result['deterministic_makespan']:.1f}")
    print(f"Robust Makespan:        {result['robust_makespan']:.1f}")
    print(f"Robust Std Dev:         {result['robust_std']:.1f}")
    print(f"Deterioration Ratio:    {result['deterioration_ratio']:.3f}")
    print(f"Simulations:            {len(result['simulation_results'])}")
    
    print("\nFirst 5 simulation results:")
    for i, sim in enumerate(result['simulation_results'][:5]):
        print(f"  Sim {i+1}: {sim:.1f}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_robust_evaluator()
