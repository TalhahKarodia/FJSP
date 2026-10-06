"""
Genetic Algorithm for FJSSP-W.

Features:
- Feasibility-preserving crossover (basic order crossover + repair)
- Two fitness modes: 'deterministic' (Scenario 1) and 'uncertainty' (Scenario 2)
- Optional worker balance penalty
- FEV counting per competition rules
- Per-instance RNG for reproducible independent runs
"""

import sys, copy
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from ga.fjssp_w.decoder_w import FJSSPWDecoder
from ga.fjssp_w.uncertainty_sim import (
    create_seeded_uncertainty_vector,
    simulate_one_scenario,
)


class FJSSPW_GA:
    """Genetic Algorithm for FJSSP-W."""

    def __init__(
        self,
        instance_data: Dict,
        population_size: int = 100,
        generations: int = 200,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        tournament_size: int = 3,
        elite_size: int = 2,
        random_seed: Optional[int] = None,
        fitness_mode: str = 'deterministic',
        n_scenarios: int = 5,
        uncertainty_seed: Optional[int] = None,
        scenario_seed: Optional[int] = None,
        balance_weight: float = 0.0,
    ):
        # Per-instance RNG
        self.rng = random.Random(random_seed)
        self.seed = random_seed

        # Fitness config
        self.fitness_mode = fitness_mode
        self.n_scenarios = n_scenarios
        self.fev_count = 0
        self.balance_weight = balance_weight

        # Uncertainty
        self.uncertainty_vector = None
        self.scenario_rng = None
        if fitness_mode == 'uncertainty':
            unc_seed = (
                uncertainty_seed if uncertainty_seed is not None
                else (random_seed if random_seed is not None else 42)
            )
            self.uncertainty_vector = create_seeded_uncertainty_vector(
                instance_data['n_workers'], seed=unc_seed
            )
            self.scenario_rng = random.Random(
                scenario_seed if scenario_seed is not None
                else (random_seed if random_seed is not None else 0)
            )

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

        self.history = {
            'best_makespan': [],
            'avg_makespan': [],
            'worst_makespan': [],
        }

    def initialize_population(self):
        self.population = []
        self.fitness_values = []

        for _ in range(self.population_size):
            chromosome = self.decoder.create_random_chromosome(self.rng)
            self.population.append(chromosome)
            fitness = self.evaluate(chromosome)
            self.fitness_values.append(fitness)

        for i, fitness in enumerate(self.fitness_values):
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = copy.deepcopy(self.population[i])
                self.best_makespan = fitness

    def evaluate(self, chromosome: Dict) -> float:
        """Evaluate chromosome. FEV count: 1 (deterministic) or n_scenarios."""
        if self.fitness_mode == 'deterministic':
            self.fev_count += 1
            schedule = self.decoder.decode(chromosome)
            base = schedule['makespan']
        else:
            self.fev_count += self.n_scenarios
            total = 0.0
            for _ in range(self.n_scenarios):
                mk = simulate_one_scenario(
                    chromosome, self.instance_data, self.decoder,
                    self.uncertainty_vector, self.scenario_rng,
                )
                total += mk
            base = total / self.n_scenarios

        if self.balance_weight > 0:
            penalty = self._worker_balance_cv(chromosome)
            return base * (1.0 + self.balance_weight * penalty)
        return base

    def _worker_balance_cv(self, chromosome: Dict) -> float:
        d = self.instance_data['durations']
        m_assign = chromosome['machine_assignment']
        w_assign = chromosome['worker_assignment']

        loads = defaultdict(float)
        for op_idx in range(len(m_assign)):
            m = m_assign[op_idx]
            w = w_assign[op_idx]
            dur = d[op_idx][m][w]
            if dur > 0:
                loads[w] += dur

        n_workers = self.decoder.n_workers
        values = [loads.get(w, 0.0) for w in range(n_workers)]
        mean_load = sum(values) / len(values)
        if mean_load == 0:
            return 0.0
        variance = sum((v - mean_load) ** 2 for v in values) / len(values)
        return (variance ** 0.5) / mean_load

    def selection(self) -> List[Dict]:
        selected = []
        for _ in range(self.population_size):
            tournament_indices = self.rng.sample(
                range(self.population_size), self.tournament_size
            )
            best_idx = min(tournament_indices, key=lambda i: self.fitness_values[i])
            selected.append(copy.deepcopy(self.population[best_idx]))
        return selected

    def feasibility_preserving_crossover(
        self, parent1: Dict, parent2: Dict
    ) -> Tuple[Dict, Dict]:
        n_ops = self.decoder.n_operations

        child1 = {
            'machine_assignment': [None] * n_ops,
            'worker_assignment': [None] * n_ops,
            'operation_sequence': [None] * n_ops,
        }
        child2 = {
            'machine_assignment': [None] * n_ops,
            'worker_assignment': [None] * n_ops,
            'operation_sequence': [None] * n_ops,
        }

        pt1 = self.rng.randint(0, n_ops - 2)
        pt2 = self.rng.randint(pt1 + 1, n_ops - 1)

        for i in range(pt1, pt2 + 1):
            child1['operation_sequence'][i] = parent1['operation_sequence'][i]
            child2['operation_sequence'][i] = parent2['operation_sequence'][i]

        def find_empty_pos(seq, start):
            for i in range(start, n_ops):
                if seq[i] is None:
                    return i
            for i in range(0, start):
                if seq[i] is None:
                    return i
            return None

        for i in range(n_ops):
            op = parent2['operation_sequence'][i]
            if op not in child1['operation_sequence']:
                pos = find_empty_pos(child1['operation_sequence'], pt1)
                if pos is not None:
                    child1['operation_sequence'][pos] = op

        for i in range(n_ops):
            op = parent1['operation_sequence'][i]
            if op not in child2['operation_sequence']:
                pos = find_empty_pos(child2['operation_sequence'], pt1)
                if pos is not None:
                    child2['operation_sequence'][pos] = op

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

        child1 = self.repair_chromosome_optimised(child1)
        child2 = self.repair_chromosome_optimised(child2)

        return child1, child2

    def repair_chromosome_optimised(self, chromosome: Dict) -> Dict:
        """Repair only infeasible assignments; leave feasible ones untouched."""
        n_ops = self.decoder.n_operations

        if not hasattr(self, '_feasible_pairs_cache'):
            self._feasible_pairs_cache = {}

        for op_idx in range(n_ops):
            machine = chromosome['machine_assignment'][op_idx]
            worker = chromosome['worker_assignment'][op_idx]

            if op_idx not in self._feasible_pairs_cache:
                pairs = []
                for m in range(self.decoder.n_machines):
                    for w in self.decoder.get_eligible_workers(op_idx, m):
                        pairs.append((m, w))
                self._feasible_pairs_cache[op_idx] = pairs

            feasible_pairs = self._feasible_pairs_cache[op_idx]

            if machine is None or worker is None:
                if feasible_pairs:
                    m, w = self.rng.choice(feasible_pairs)
                    chromosome['machine_assignment'][op_idx] = m
                    chromosome['worker_assignment'][op_idx] = w
                continue

            if (machine, worker) not in feasible_pairs:
                if feasible_pairs:
                    best = min(
                        feasible_pairs,
                        key=lambda p: self.decoder.get_processing_time(op_idx, p[0], p[1])
                    )
                    chromosome['machine_assignment'][op_idx] = best[0]
                    chromosome['worker_assignment'][op_idx] = best[1]

        return chromosome

    def mutation(self, chromosome: Dict) -> Dict:
        child = copy.deepcopy(chromosome)
        n_ops = self.decoder.n_operations

        if self.rng.random() < self.mutation_rate:
            i, j = self.rng.sample(range(n_ops), 2)
            child['operation_sequence'][i], child['operation_sequence'][j] = \
                child['operation_sequence'][j], child['operation_sequence'][i]

        if self.rng.random() < self.mutation_rate:
            op_idx = self.rng.randint(0, n_ops - 1)
            feasible_pairs = []
            for m in range(self.decoder.n_machines):
                for w in self.decoder.get_eligible_workers(op_idx, m):
                    feasible_pairs.append((m, w))

            if feasible_pairs:
                current_pair = (child['machine_assignment'][op_idx],
                                child['worker_assignment'][op_idx])
                other_pairs = [p for p in feasible_pairs if p != current_pair]
                if other_pairs:
                    new_m, new_w = self.rng.choice(other_pairs)
                    child['machine_assignment'][op_idx] = new_m
                    child['worker_assignment'][op_idx] = new_w

        return child

    def evolve(self) -> Dict:
        self.initialize_population()

        for generation in range(self.generations):
            selected = self.selection()

            new_population = []
            new_fitness = []

            sorted_indices = sorted(
                range(len(self.fitness_values)),
                key=lambda i: self.fitness_values[i]
            )
            for idx in sorted_indices[:self.elite_size]:
                new_population.append(copy.deepcopy(self.population[idx]))
                new_fitness.append(self.fitness_values[idx])

            while len(new_population) < self.population_size:
                p1 = self.rng.choice(selected)
                p2 = self.rng.choice(selected)

                if self.rng.random() < self.crossover_rate:
                    child1, child2 = self.feasibility_preserving_crossover(p1, p2)
                else:
                    child1 = copy.deepcopy(p1)
                    child2 = copy.deepcopy(p2)

                child1 = self.mutation(child1)
                child2 = self.mutation(child2)

                f1 = self.evaluate(child1)
                f2 = self.evaluate(child2)

                new_population.append(child1)
                new_fitness.append(f1)

                if len(new_population) < self.population_size:
                    new_population.append(child2)
                    new_fitness.append(f2)

            self.population = new_population
            self.fitness_values = new_fitness

            best_idx = min(range(len(self.fitness_values)),
                           key=lambda i: self.fitness_values[i])
            if self.fitness_values[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness_values[best_idx]
                self.best_solution = copy.deepcopy(self.population[best_idx])
                self.best_makespan = self.best_fitness

            self.history['best_makespan'].append(min(self.fitness_values))
            self.history['avg_makespan'].append(
                sum(self.fitness_values) / len(self.fitness_values)
            )
            self.history['worst_makespan'].append(max(self.fitness_values))

        return {
            'best_solution': self.best_solution,
            'best_makespan': self.best_makespan,
            'history': self.history,
            'fev_count': self.fev_count,
        }

    def get_best_schedule(self) -> Dict:
        if self.best_solution is None:
            return None
        return self.decoder.decode(self.best_solution)