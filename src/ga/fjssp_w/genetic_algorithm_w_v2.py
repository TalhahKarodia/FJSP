"""
Optimized Genetic Algorithm for FJSSP-W.

Features & Optimizations:
- Fast chromosome shallow cloning (replacing deepcopy)
- Hybrid Job-Slice Crossover (Combines two-point slice OX with SOTA job precedence preservation & origin-paired assignments)
- Precomputed feasible (machine, worker) pairs, sets, and fastest pairs per operation
- Fast repair mechanism using cached best pairs
- Evaluation memoization cache for fitness calculations
- Worker load variation penalty calculated with flat arrays
"""

import sys
import os
import random
from typing import List, Dict, Tuple, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from ga.fjssp_w.decoder_w import FJSSPWDecoder
from ga.fjssp_w.uncertainty_sim import (
    create_seeded_uncertainty_vector,
    simulate_one_scenario,
)


def clone_chromosome(chrom: Dict) -> Dict:
    """Fast shallow clone of chromosome lists (much faster than deepcopy)."""
    return {
        'machine_assignment': chrom['machine_assignment'][:],
        'worker_assignment': chrom['worker_assignment'][:],
        'operation_sequence': chrom['operation_sequence'][:],
    }


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
        greedy_fraction: float = 0.5,
    ):
        self.rng = random.Random(random_seed)
        self.seed = random_seed

        self.fitness_mode = fitness_mode
        self.n_scenarios = n_scenarios
        self.fev_count = 0
        self.balance_weight = balance_weight
        self.greedy_fraction = greedy_fraction

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

        # Memoization cache for chromosome evaluation
        self.eval_cache = {}

        # Precompute feasible pairs, sets, and best processing time pairs
        self.feasible_pairs = {}
        self.feasible_pairs_set = {}
        self.best_feasible_pair = {}

        for op_idx in range(self.decoder.n_operations):
            pairs = []
            for m in range(self.decoder.n_machines):
                for w in self.decoder.get_eligible_workers(op_idx, m):
                    pairs.append((m, w))
            self.feasible_pairs[op_idx] = pairs
            self.feasible_pairs_set[op_idx] = set(pairs)
            if pairs:
                self.best_feasible_pair[op_idx] = min(
                    pairs, key=lambda p: self.decoder.get_processing_time(op_idx, p[0], p[1])
                )

    def initialize_population(self):
        """Seeded initialisation: greedy_fraction greedy + rest random."""
        self.population = []
        self.fitness_values = []

        n = self.population_size
        n_greedy = max(1, int(n * self.greedy_fraction))

        for i in range(n):
            if i < n_greedy:
                chromosome = self.decoder.create_greedy_chromosome(self.rng)
            else:
                chromosome = self.decoder.create_random_chromosome(self.rng)
            self.population.append(chromosome)
            fitness = self.evaluate(chromosome)
            self.fitness_values.append(fitness)

        for i, fitness in enumerate(self.fitness_values):
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = clone_chromosome(self.population[i])
                self.best_makespan = fitness

    def _hash_chromosome(self, chromosome: Dict) -> Tuple:
        """Hashable key for chromosome memoization."""
        return (
            tuple(chromosome['operation_sequence']),
            tuple(chromosome['machine_assignment']),
            tuple(chromosome['worker_assignment']),
        )

    def evaluate(self, chromosome: Dict) -> float:
        """Evaluate chromosome with memoization cache. FEV count: 1 (deterministic) or n_scenarios."""
        key = self._hash_chromosome(chromosome)
        if key in self.eval_cache:
            return self.eval_cache[key]

        if self.fitness_mode == 'deterministic':
            self.fev_count += 1
            schedule = self.decoder.decode(chromosome)
            base = schedule['makespan']
        else:
            self.fev_count += self.n_scenarios
            total = sum(
                simulate_one_scenario(
                    chromosome, self.instance_data, self.decoder,
                    self.uncertainty_vector, self.scenario_rng,
                )
                for _ in range(self.n_scenarios)
            )
            base = total / self.n_scenarios

        if self.balance_weight > 0:
            penalty = self._worker_balance_cv(chromosome)
            fit = base * (1.0 + self.balance_weight * penalty)
        else:
            fit = base

        self.eval_cache[key] = fit
        return fit

    def _worker_balance_cv(self, chromosome: Dict) -> float:
        """Calculate worker load variation coefficient using indexed array."""
        d = self.instance_data['durations']
        m_assign = chromosome['machine_assignment']
        w_assign = chromosome['worker_assignment']
        n_workers = self.decoder.n_workers

        loads = [0.0] * n_workers
        for op_idx in range(len(m_assign)):
            m, w = m_assign[op_idx], w_assign[op_idx]
            dur = d[op_idx][m][w]
            if dur > 0:
                loads[w] += dur

        mean_load = sum(loads) / n_workers
        if mean_load == 0:
            return 0.0

        variance = sum((v - mean_load) ** 2 for v in loads) / n_workers
        return (variance ** 0.5) / mean_load

    def selection(self) -> List[Dict]:
        """Tournament selection returning shallow clones."""
        selected = []
        for _ in range(self.population_size):
            tournament_indices = self.rng.sample(
                range(self.population_size), self.tournament_size
            )
            best_idx = min(tournament_indices, key=lambda i: self.fitness_values[i])
            selected.append(clone_chromosome(self.population[best_idx]))
        return selected

    def hybrid_job_slice_crossover(
        self, parent1: Dict, parent2: Dict
    ) -> Tuple[Dict, Dict]:
        """
        Hybrid Crossover for FJSSP-W:
        - Combines Two-Point Slice OX with Job-Aware Grouping.
        - Preserves job precedence inside slice windows.
        - Inherits paired (Machine, Worker) assignments consistent with sequence origin.
        """
        n_ops = self.decoder.n_operations
        p1_seq, p2_seq = parent1['operation_sequence'], parent2['operation_sequence']

        pt1 = self.rng.randint(0, n_ops - 2)
        pt2 = self.rng.randint(pt1 + 1, n_ops - 1)

        slice_ops_p1 = set(p1_seq[pt1:pt2 + 1])
        sliced_jobs = {self.decoder.get_job_id(op) for op in slice_ops_p1}

        def create_hybrid_child(primary_parent, secondary_parent, active_jobs):
            pri_seq = primary_parent['operation_sequence']
            sec_seq = secondary_parent['operation_sequence']

            c_seq = [None] * n_ops
            c_m = [None] * n_ops
            c_w = [None] * n_ops

            copied_ops = set()

            # A. Copy operations of selected jobs from Primary Parent at exact positions
            for i, op in enumerate(pri_seq):
                if self.decoder.get_job_id(op) in active_jobs:
                    c_seq[i] = op
                    c_m[i] = primary_parent['machine_assignment'][i]
                    c_w[i] = primary_parent['worker_assignment'][i]
                    copied_ops.add(op)

            # B. Fill empty slots using Secondary Parent's order & assignments
            sec_idx = 0
            for i in range(n_ops):
                if c_seq[i] is None:
                    while sec_idx < n_ops:
                        op_sec = sec_seq[sec_idx]
                        sec_pos = sec_idx
                        sec_idx += 1

                        if op_sec not in copied_ops:
                            c_seq[i] = op_sec
                            c_m[i] = secondary_parent['machine_assignment'][sec_pos]
                            c_w[i] = secondary_parent['worker_assignment'][sec_pos]
                            copied_ops.add(op_sec)
                            break

            # C. Uniform Assignment Blending (10% chance to explore assignment pairs)
            for i in range(n_ops):
                if self.rng.random() < 0.10:
                    alt_parent = secondary_parent if c_seq[i] in slice_ops_p1 else primary_parent
                    alt_m = alt_parent['machine_assignment'][i]
                    alt_w = alt_parent['worker_assignment'][i]
                    if alt_m is not None and alt_w is not None:
                        c_m[i], c_w[i] = alt_m, alt_w

            return {'machine_assignment': c_m, 'worker_assignment': c_w, 'operation_sequence': c_seq}

        child1 = create_hybrid_child(parent1, parent2, sliced_jobs)
        child2 = create_hybrid_child(parent2, parent1, sliced_jobs)

        return self.repair_chromosome_optimised(child1), self.repair_chromosome_optimised(child2)

    def repair_chromosome_optimised(self, chromosome: Dict) -> Dict:
        """Repair infeasible assignments using precomputed best pairs."""
        m_assign = chromosome['machine_assignment']
        w_assign = chromosome['worker_assignment']

        for op_idx in range(self.decoder.n_operations):
            m, w = m_assign[op_idx], w_assign[op_idx]
            feasible_set = self.feasible_pairs_set[op_idx]

            if m is None or w is None or (m, w) not in feasible_set:
                if op_idx in self.best_feasible_pair:
                    best_m, best_w = self.best_feasible_pair[op_idx]
                    m_assign[op_idx] = best_m
                    w_assign[op_idx] = best_w

        return chromosome

    def mutation(self, chromosome: Dict) -> Dict:
        """Optimized mutation utilizing shallow cloning and precomputed pairs."""
        child = clone_chromosome(chromosome)
        n_ops = self.decoder.n_operations

        if self.rng.random() < self.mutation_rate:
            i, j = self.rng.sample(range(n_ops), 2)
            seq = child['operation_sequence']
            seq[i], seq[j] = seq[j], seq[i]

        if self.rng.random() < self.mutation_rate:
            op_idx = self.rng.randint(0, n_ops - 1)
            feasible_pairs = self.feasible_pairs[op_idx]
            if feasible_pairs:
                cur_pair = (child['machine_assignment'][op_idx], child['worker_assignment'][op_idx])
                other_pairs = [p for p in feasible_pairs if p != cur_pair]
                if other_pairs:
                    new_m, new_w = self.rng.choice(other_pairs)
                    child['machine_assignment'][op_idx] = new_m
                    child['worker_assignment'][op_idx] = new_w

        return child

    def evolve(self) -> Dict:
        """Main evolutionary optimization loop."""
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
                new_population.append(clone_chromosome(self.population[idx]))
                new_fitness.append(self.fitness_values[idx])

            while len(new_population) < self.population_size:
                p1 = self.rng.choice(selected)
                p2 = self.rng.choice(selected)

                if self.rng.random() < self.crossover_rate:
                    child1, child2 = self.hybrid_job_slice_crossover(p1, p2)
                else:
                    child1 = clone_chromosome(p1)
                    child2 = clone_chromosome(p2)

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
                self.best_solution = clone_chromosome(self.population[best_idx])
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
        """Decode and return the best schedule found."""
        if self.best_solution is None:
            return None
        return self.decoder.decode(self.best_solution)