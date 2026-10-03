"""
Genetic Algorithm for FJSSP-W with Feasibility-Preserving Crossover (FPC)

This module implements a Genetic Algorithm for the Flexible Job Shop Scheduling
Problem with Worker Flexibility (FJSSP-W) featuring a feasibility-preserving
crossover operator that maintains valid machine-worker assignments during
recombination.
"""

import sys, concurrent.futures, copy
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import random
import copy
from typing import List, Dict, Tuple, Optional
import time

from ga.fjssp_w.decoder_w import FJSSPWDecoder


class FJSSPW_GA:
    """
    Genetic Algorithm for FJSSP-W with Feasibility-Preserving Crossover.
    
    Chromosome structure:
    - machine_assignment: list of machine indices (length = n_operations)
    - worker_assignment: list of worker indices (length = n_operations)
    - operation_sequence: list of operation indices (length = n_operations)
    """
    
    def __init__(
        self,
        instance_data: Dict,
        population_size: int = 100,
        generations: int = 200,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        tournament_size: int = 3,
        elite_size: int = 2,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the GA.
        
        Args:
            instance_data: dict from parse_competition_instance()
            population_size: Number of individuals in population
            generations: Maximum number of generations
            crossover_rate: Probability of applying crossover
            mutation_rate: Probability of mutation per individual
            tournament_size: Size of tournament for selection
            elite_size: Number of best individuals to preserve
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        self.decoder = FJSSPWDecoder(instance_data)
        self.instance_data = instance_data
        
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        
        self.population = []
        self.fitness_values = []
        self.best_solution = None
        self.best_fitness = float('inf')
        self.best_makespan = float('inf')
        
        # Track progress
        self.history = {
            'best_makespan': [],
            'avg_makespan': [],
            'worst_makespan': []
        }
    
    def initialize_population(self):
        """Create initial random population."""
        self.population = []
        self.fitness_values = []
        
        for _ in range(self.population_size):
            chromosome = self.decoder.create_random_chromosome()
            self.population.append(chromosome)
            fitness = self.evaluate(chromosome)
            self.fitness_values.append(fitness)
        
        # Update best solution
        for i, fitness in enumerate(self.fitness_values):
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = copy.deepcopy(self.population[i])
                self.best_makespan = fitness
    
    def evaluate(self, chromosome: Dict) -> float:
        """
        Evaluate a chromosome (minimization problem).
        
        Returns:
            float: Makespan (lower is better)
        """
        schedule = self.decoder.decode(chromosome)
        return schedule['makespan']
    
    def selection(self) -> List[Dict]:
        """
        Tournament selection.
        
        Returns:
            List of selected parents
        """
        selected = []
        
        for _ in range(self.population_size):
            # Tournament
            tournament_indices = random.sample(range(self.population_size), self.tournament_size)
            best_idx = min(tournament_indices, key=lambda i: self.fitness_values[i])
            selected.append(copy.deepcopy(self.population[best_idx]))
        
        return selected
    
    def feasibility_preserving_crossover(
        self,
        parent1: Dict,
        parent2: Dict
    ) -> Tuple[Dict, Dict]:
        """
        Feasibility-Preserving Crossover (FPC) operator.
        
        Step 1: Apply order crossover (OX) to operation sequence
        Step 2: Inherit machine-worker assignments from parents
        Step 3: Repair infeasible assignments
        """
        n_ops = self.decoder.n_operations
        
        # Create children
        child1 = {
            'machine_assignment': [None] * n_ops,
            'worker_assignment': [None] * n_ops,
            'operation_sequence': [None] * n_ops
        }
        child2 = {
            'machine_assignment': [None] * n_ops,
            'worker_assignment': [None] * n_ops,
            'operation_sequence': [None] * n_ops
        }
        
        # Step 1: Order crossover (OX) on operation sequence
        pt1 = random.randint(0, n_ops - 2)
        pt2 = random.randint(pt1 + 1, n_ops - 1)
        
        # Copy segment from parents
        for i in range(pt1, pt2 + 1):
            child1['operation_sequence'][i] = parent1['operation_sequence'][i]
            child2['operation_sequence'][i] = parent2['operation_sequence'][i]
        
        # Helper to find next empty position
        def find_empty_pos(seq, start):
            for i in range(start, n_ops):
                if seq[i] is None:
                    return i
            for i in range(0, start):
                if seq[i] is None:
                    return i
            return None
        
        # Fill child1 with parent2's remaining operations
        for i in range(n_ops):
            op = parent2['operation_sequence'][i]
            if op not in child1['operation_sequence']:
                pos = find_empty_pos(child1['operation_sequence'], pt1)
                if pos is not None:
                    child1['operation_sequence'][pos] = op
        
        # Fill child2 with parent1's remaining operations
        for i in range(n_ops):
            op = parent1['operation_sequence'][i]
            if op not in child2['operation_sequence']:
                pos = find_empty_pos(child2['operation_sequence'], pt1)
                if pos is not None:
                    child2['operation_sequence'][pos] = op
        
        # Step 2: Inherit machine-worker assignments
        for i in range(n_ops):
            op_idx = child1['operation_sequence'][i]
            if op_idx is not None:
                if op_idx in parent1['operation_sequence']:
                    pos_p1 = parent1['operation_sequence'].index(op_idx)
                    child1['machine_assignment'][i] = parent1['machine_assignment'][pos_p1]
                    child1['worker_assignment'][i] = parent1['worker_assignment'][pos_p1]
                else:
                    pos_p2 = parent2['operation_sequence'].index(op_idx)
                    child1['machine_assignment'][i] = parent2['machine_assignment'][pos_p2]
                    child1['worker_assignment'][i] = parent2['worker_assignment'][pos_p2]
            
            op_idx2 = child2['operation_sequence'][i]
            if op_idx2 is not None:
                if op_idx2 in parent2['operation_sequence']:
                    pos_p2 = parent2['operation_sequence'].index(op_idx2)
                    child2['machine_assignment'][i] = parent2['machine_assignment'][pos_p2]
                    child2['worker_assignment'][i] = parent2['worker_assignment'][pos_p2]
                else:
                    pos_p1 = parent1['operation_sequence'].index(op_idx2)
                    child2['machine_assignment'][i] = parent1['machine_assignment'][pos_p1]
                    child2['worker_assignment'][i] = parent1['worker_assignment'][pos_p1]
        
        # Step 3: Repair any infeasible assignments
        child1 = self.repair_chromosome_optimised(child1)
        child2 = self.repair_chromosome_optimised(child2)
        
        return child1, child2
    
    def repair_chromosome(self, chromosome: Dict) -> Dict:
        """Repair infeasible machine-worker assignments."""
        n_ops = self.decoder.n_operations
        
        for op_idx in range(n_ops):
            machine = chromosome['machine_assignment'][op_idx]
            worker = chromosome['worker_assignment'][op_idx]
            
            # Skip if None
            if machine is None or worker is None:
                feasible_pairs = []
                for m in range(self.decoder.n_machines):
                    for w in self.decoder.get_eligible_workers(op_idx, m):
                        feasible_pairs.append((m, w))
                if feasible_pairs:
                    m, w = random.choice(feasible_pairs)
                    chromosome['machine_assignment'][op_idx] = m
                    chromosome['worker_assignment'][op_idx] = w
                continue
            
            # Check feasibility
            if not self.decoder.is_feasible_assignment(op_idx, machine, worker):
                feasible_pairs = []
                for m in range(self.decoder.n_machines):
                    for w in self.decoder.get_eligible_workers(op_idx, m):
                        feasible_pairs.append((m, w))
                
                if feasible_pairs:
                    best_pair = min(
                        feasible_pairs,
                        key=lambda p: self.decoder.get_processing_time(op_idx, p[0], p[1])
                    )
                    chromosome['machine_assignment'][op_idx] = best_pair[0]
                    chromosome['worker_assignment'][op_idx] = best_pair[1]
        
        return chromosome
    
    def repair_chromosome_optimised(self, chromosome) -> Dict:
        """repair infeasible machine-worker assignments"""
        n_ops = self.decoder.n_operations

        if not hasattr(self, '_feasible_pairs_cache'): # initialise caches if they don't exist
            self._feasible_pairs_cache = {}
            self._best_pair_cache = {}
        for op_idx in range(n_ops):
            machine = chromosome['machine_assignment'][op_idx]
            worker = chromosome['worker_assignment'][op_idx]

            if op_idx not in self._feasible_pairs_cache:
                pairs = []
                for m in range(self.decoder.n_machines):
                    for w in self.decoder.get_eligible_workers(op_idx, m):
                        pairs.append((m, w))
                self._feasible_pairs_cache[op_idx] = pairs

                if pairs:
                    best = min(pairs, key= lambda p: self.decoder.get_processing_time(op_idx, p[0], p[1]))
                    self._best_pair_cache[op_idx] = best
            feasible_pairs = self._feasible_pairs_cache[op_idx]

            if machine is None or worker is None:
                if feasible_pairs:
                    m, w = random.choice(feasible_pairs)
                    chromosome['machine_assignment'][op_idx] = m
                    chromosome['worker_assignment'][op_idx] = w
                continue

            # check feasibility and perform repair 
            if feasible_pairs:
                best_pair = self._best_pair_cache[op_idx]
                chromosome['machine_assignment'][op_idx] = best_pair[0]
                chromosome['worker_assignment'][op_idx] = best_pair[1]

        return chromosome
             
    def mutation(self, chromosome: Dict) -> Dict:
        """Apply mutation to a chromosome."""
        child = copy.deepcopy(chromosome)
        n_ops = self.decoder.n_operations
        
        # Mutation 1: Swap two operations in sequence
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(n_ops), 2)
            child['operation_sequence'][i], child['operation_sequence'][j] = \
                child['operation_sequence'][j], child['operation_sequence'][i]
        
        # Mutation 2: Reassign a machine-worker pair
        if random.random() < self.mutation_rate:
            op_idx = random.randint(0, n_ops - 1)
            feasible_pairs = []
            for m in range(self.decoder.n_machines):
                for w in self.decoder.get_eligible_workers(op_idx, m):
                    feasible_pairs.append((m, w))
            
            if feasible_pairs:
                current_pair = (child['machine_assignment'][op_idx], 
                              child['worker_assignment'][op_idx])
                other_pairs = [p for p in feasible_pairs if p != current_pair]
                if other_pairs:
                    new_m, new_w = random.choice(other_pairs)
                    child['machine_assignment'][op_idx] = new_m
                    child['worker_assignment'][op_idx] = new_w
        
        return child

    def evolve(self) -> Dict:
        """Run the genetic algorithm."""
        # Initialize
        self.initialize_population()
        
        best_overall = copy.deepcopy(self.best_solution)
        best_overall_makespan = self.best_makespan
        
        print(f"Initial best makespan: {self.best_makespan}")
        
        # Evolution loop
        for generation in range(self.generations):
            # Selection
            selected = self.selection()
            
            # Create new population
            new_population = []
            new_fitness = []
            
            # Elitism
            sorted_indices = sorted(
                range(len(self.fitness_values)),
                key=lambda i: self.fitness_values[i]
            )
            elites = sorted_indices[:self.elite_size]
            for idx in elites:
                new_population.append(copy.deepcopy(self.population[idx]))
                new_fitness.append(self.fitness_values[idx])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                p1 = random.choice(selected)
                p2 = random.choice(selected)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self.feasibility_preserving_crossover(p1, p2)
                else:
                    child1 = copy.deepcopy(p1)
                    child2 = copy.deepcopy(p2)
                
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                fitness1 = self.evaluate(child1)
                fitness2 = self.evaluate(child2)
                
                new_population.append(child1)
                new_fitness.append(fitness1)
                
                if len(new_population) < self.population_size:
                    new_population.append(child2)
                    new_fitness.append(fitness2)
            
            # Replace population
            self.population = new_population
            self.fitness_values = new_fitness
            
            # Update best
            best_idx = min(range(len(self.fitness_values)), key=lambda i: self.fitness_values[i])
            if self.fitness_values[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness_values[best_idx]
                self.best_solution = copy.deepcopy(self.population[best_idx])
                self.best_makespan = self.best_fitness
            
            # Track history
            self.history['best_makespan'].append(self.best_makespan)
            self.history['avg_makespan'].append(sum(self.fitness_values) / len(self.fitness_values))
            self.history['worst_makespan'].append(max(self.fitness_values))
            
            if (generation + 1) % 20 == 0:
                avg_fitness = sum(self.fitness_values) / len(self.fitness_values)
                print(f"Gen {generation+1}: Best={self.best_makespan:.1f}, Avg={avg_fitness:.1f}")
        
        print(f"Final best makespan: {self.best_makespan}")
        
        return {
            'best_solution': self.best_solution,
            'best_makespan': self.best_makespan,
            'history': self.history
        }
    
    def get_best_schedule(self) -> Dict:
        """Get the decoded schedule of the best solution."""
        if self.best_solution is None:
            return None
        return self.decoder.decode(self.best_solution)

def main():
    """Test the GA on a small instance."""
    from parser.fjssp_w_parser import parse_competition_instance
    
    print("=" * 60)
    print("TESTING FJSSP-W GENETIC ALGORITHM")
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
    
    # Create GA
    ga = FJSSPW_GA(
        instance_data,
        population_size=50,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        tournament_size=3,
        elite_size=2,
        random_seed=42
    )
    
    # Run GA
    result = ga.evolve()
    
    print(f"\nBest makespan found: {result['best_makespan']}")
    
    # Decode best schedule
    schedule = ga.get_best_schedule()
    if schedule:
        print("\nBest schedule details:")
        print(f"Start times: {schedule['start_times'][:10]}...")
        print(f"End times: {schedule['end_times'][:10]}...")
        print(f"Machine assignments: {schedule['machine_assignment'][:10]}...")
        print(f"Worker assignments: {schedule['worker_assignment'][:10]}...")
    
    print("\nGA test complete!")


if __name__ == "__main__":
    main()