import sys
import os
import random
from heapq import nsmallest
from typing import List, Dict, Tuple, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from ga.fjssp_w.decoder_w_v2 import FJSSPWDecoder_V2
from ga.fjssp_w.uncertainty_sim import (
    create_seeded_uncertainty_vector,
    simulate_one_scenario,
)


def clone_chromosome(chrom: Dict) -> Dict:
    """Fast shallow clone of chromosome lists."""
    return {
        'machine_assignment': chrom['machine_assignment'][:],
        'worker_assignment': chrom['worker_assignment'][:],
        'operation_sequence': chrom['operation_sequence'][:],
    }


class FJSSPW_GA_V2:
    """Optimized Genetic Algorithm for FJSSP-W."""

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
        initial_population: Optional[List[Dict]] = None,
        max_fev: int = 5_000_000,
        decoder: Optional[FJSSPWDecoder_V2] = None,
        gene_mutation_rate: float = 0.04,
        fast_pair_fraction: Optional[float] = 0.10,
    ):
        # gene_mutation_rate: per-operation chance of re-picking its (machine,
        #   worker) pair in mutation(). 0 disables it.
        # fast_pair_fraction: re-picked pairs come from this fraction of the
        #   operation's fastest feasible pairs (None = any feasible pair).
        # Setting gene_mutation_rate=0 and fast_pair_fraction=None restores the
        # previous mutation behaviour exactly.
        self.rng = random.Random(random_seed)
        self.seed = random_seed

        self.fitness_mode = fitness_mode
        self.n_scenarios = n_scenarios
        self.fev_count = 0
        self.max_fev = max_fev
        self.balance_weight = balance_weight
        self.greedy_fraction = greedy_fraction
        self.initial_population = initial_population

        self.uncertainty_vector = None
        if fitness_mode == 'uncertainty':
            unc_seed = (
                uncertainty_seed if uncertainty_seed is not None
                else (random_seed if random_seed is not None else 42)
            )
            self.uncertainty_vector = create_seeded_uncertainty_vector(
                instance_data['n_workers'], seed=unc_seed
            )

        # A prebuilt decoder can be shared to avoid rebuilding its tables
        self.decoder = decoder if decoder is not None else FJSSPWDecoder_V2(instance_data)
        self.instance_data = instance_data

        self.op_to_job = self.decoder.op_to_job
        self.unique_jobs = list(set(self.op_to_job.values())) if self.op_to_job else []

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

        # Memoization: fitness, and (deterministic mode) raw makespan
        self.eval_cache = {}
        self.mk_cache = {}

        # Precomputed feasible pairs (shared lists from the decoder, same order)
        self.feasible_pairs = {}
        self.feasible_pairs_set = {}
        self.best_feasible_pair = {}
        dur = self.decoder.durations
        for op_idx in range(self.decoder.n_operations):
            pairs = self.decoder.feasible_pairs[op_idx]
            self.feasible_pairs[op_idx] = pairs
            self.feasible_pairs_set[op_idx] = set(pairs)
            if pairs:
                d = dur[op_idx]
                self.best_feasible_pair[op_idx] = min(pairs, key=lambda p: d[p[0]][p[1]])

        # Fastest-pairs pools used by mutation (durations > 0, ascending)
        self.gene_mutation_rate = gene_mutation_rate
        self.fast_pair_fraction = fast_pair_fraction
        self.mutation_pairs = {}
        for op_idx in range(self.decoder.n_operations):
            if fast_pair_fraction:
                lst = self.decoder._sorted_feasible(op_idx)
                k = max(3, int(len(lst) * fast_pair_fraction))
                pool = [(m, w) for _, m, w in lst[:k]]
                self.mutation_pairs[op_idx] = pool or self.feasible_pairs[op_idx]
            else:
                self.mutation_pairs[op_idx] = self.feasible_pairs[op_idx]

    # ------------------------------------------------------------------
    def initialize_population(self):
        self.population = []
        self.fitness_values = []
        self.eval_cache.clear()
        self.mk_cache.clear()
        self.fev_count = 0
        self.best_fitness = float('inf')
        self.best_makespan = float('inf')
        self.best_solution = None

        if self.initial_population is not None:
            for chrom in self.initial_population:
                c = clone_chromosome(chrom)
                self.population.append(c)
                self.fitness_values.append(self.evaluate(c))
        else:
            n = self.population_size
            n_greedy = max(1, int(n * self.greedy_fraction))
            for i in range(n):
                if i < n_greedy:
                    chromosome = self.decoder.create_greedy_chromosome(self.rng)
                else:
                    chromosome = self.decoder.create_random_chromosome(self.rng)
                self.population.append(chromosome)
                self.fitness_values.append(self.evaluate(chromosome))

        for i, fitness in enumerate(self.fitness_values):
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = clone_chromosome(self.population[i])
                self.best_makespan = self._makespan(self.population[i])

    def _hash_chromosome(self, chromosome: Dict) -> Tuple:
        return (
            tuple(chromosome['operation_sequence']),
            tuple(chromosome['machine_assignment']),
            tuple(chromosome['worker_assignment']),
        )

    def _makespan(self, chromosome: Dict) -> float:
        """Deterministic makespan, from cache when available (no re-decode)."""
        mk = self.mk_cache.get(self._hash_chromosome(chromosome))
        if mk is None:
            mk = self.decoder.decode_makespan(chromosome)
        return mk

    def evaluate(self, chromosome: Dict) -> float:
        key = self._hash_chromosome(chromosome)
        cached = self.eval_cache.get(key)
        if cached is not None:
            return cached

        if self.fev_count >= self.max_fev:
            return float('inf')

        if self.fitness_mode == 'deterministic':
            self.fev_count += 1
            base = self.decoder.decode_makespan(chromosome)
            self.mk_cache[key] = base
        else:
            self.fev_count += self.n_scenarios
            chrom_seed = hash(key) & 0xffffffff
            scen_rng = random.Random(chrom_seed)
            total = sum(
                simulate_one_scenario(
                    chromosome, self.instance_data, self.decoder,
                    self.uncertainty_vector, scen_rng,
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
        d = self.decoder.durations
        m_assign = chromosome['machine_assignment']
        w_assign = chromosome['worker_assignment']
        n_workers = self.decoder.n_workers

        loads = [0.0] * n_workers
        for op_idx in range(len(m_assign)):
            m, w = m_assign[op_idx], w_assign[op_idx]
            if m is None or w is None:
                continue
            dur = d[op_idx][m][w]
            if dur > 0:
                loads[w] += dur

        mean_load = sum(loads) / n_workers
        if mean_load == 0:
            return 0.0

        variance = sum((v - mean_load) ** 2 for v in loads) / n_workers
        return (variance ** 0.5) / mean_load

    # ------------------------------------------------------------------
    def selection(self) -> List[Dict]:
        """Tournament selection. Returns references: parents are never mutated
        in place (crossover builds new lists, mutation clones first)."""
        pop = self.population
        fit = self.fitness_values
        n = self.population_size
        k = self.tournament_size
        sample = self.rng.sample
        selected = []
        for _ in range(n):
            idxs = sample(range(n), k)
            best_idx = min(idxs, key=fit.__getitem__)
            selected.append(pop[best_idx])
        return selected

    def repair_chromosome_optimised(self, chromosome: Dict) -> Dict:
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

    def feasibility_preserving_crossover(
        self, parent1: Dict, parent2: Dict
    ) -> Tuple[Dict, Dict]:
        """
            Enhanced Job-Aware POX Crossover for FJSSP-W:
            - Two-Point Slice Windowing: Identifies co-scheduled job clusters in Parent 1.
            - POX Sequence Building: Active-job operations keep their Parent 1 positions; remaining slots are filled in Parent 2's relative order.
            - Origin-Paired Inheritance: Keeps (Machine, Worker) choices bound together per operation ID.
            - Feasibility-Guided Blending: Safely explores alternative parent assignments using precomputed sets.
        """
        n_ops = self.decoder.n_operations
        p1_seq = parent1['operation_sequence']

        # 1. Two-Point Slice Windowing (Extracts active job clusters from P1)
        pt1 = self.rng.randint(0, n_ops - 2)
        pt2 = self.rng.randint(pt1 + 1, n_ops - 1)
        
        slice_ops = p1_seq[pt1:pt2 + 1]
        sliced_jobs = {self._get_job_id(op) for op in slice_ops if op is not None}

        # Fallback safeguard if slice contains all or no jobs
        if 'n_jobs' in self.instance_data:
            total_jobs = self.instance_data['n_jobs']
        else:
            total_jobs = len({self._get_job_id(op) for op in range(n_ops)})

        if not sliced_jobs or len(sliced_jobs) >= total_jobs:
            all_jobs_list = list(range(total_jobs))
            k = max(1, total_jobs // 2)
            sliced_jobs = set(self.rng.sample(all_jobs_list, k))

        def create_job_aware_child(primary_parent: Dict, secondary_parent: Dict, active_jobs: set) -> Dict:
            pri_seq = primary_parent['operation_sequence']
            sec_seq = secondary_parent['operation_sequence']

            pri_m = primary_parent['machine_assignment']
            pri_w = primary_parent['worker_assignment']
            sec_m = secondary_parent['machine_assignment']
            sec_w = secondary_parent['worker_assignment']

            c_seq = [None] * n_ops
            c_m = [None] * n_ops
            c_w = [None] * n_ops

            # 2. Sequence Construction via POX
            # Step A: Copy active job operations at exact sequence positions from Primary Parent
            active_ops_set = set()
            for i, op in enumerate(pri_seq):
                if self._get_job_id(op) in active_jobs:
                    c_seq[i] = op
                    active_ops_set.add(op)

            # Step B: Fill remaining sequence slots using Secondary Parent's relative order
            sec_idx = 0
            for i in range(n_ops):
                if c_seq[i] is None:
                    while sec_idx < n_ops and sec_seq[sec_idx] in active_ops_set:
                        sec_idx += 1
                    if sec_idx < n_ops:
                        c_seq[i] = sec_seq[sec_idx]
                        sec_idx += 1

            # 3. Origin-Paired Assignment Inheritance & Blending (Indexed by Operation ID `op`)
            for op in range(n_ops):
                job_id = self._get_job_id(op)

                # Inherit intact (Machine, Worker) pair from the job's originating parent
                if job_id in active_jobs:
                    c_m[op], c_w[op] = pri_m[op], pri_w[op]
                    alt_m, alt_w = sec_m[op], sec_w[op]
                else:
                    c_m[op], c_w[op] = sec_m[op], sec_w[op]
                    alt_m, alt_w = pri_m[op], pri_w[op]

                # 4. Feasibility-Guided Assignment Blending (10% exploration chance)
                if self.rng.random() < 0.10:
                    # Only swap if the alternative parent's pair is verified feasible
                    if (alt_m, alt_w) in self.feasible_pairs_set[op]:
                        c_m[op], c_w[op] = alt_m, alt_w

            child = {
                'operation_sequence': c_seq,
                'machine_assignment': c_m,
                'worker_assignment': c_w,
            }
            return self.repair_chromosome_optimised(child)

        child1 = create_job_aware_child(parent1, parent2, sliced_jobs)
        child2 = create_job_aware_child(parent2, parent1, sliced_jobs)

        return child1, child2
    
    def _get_job_id(self, op: int) -> int:
        """Helper to map an operation index to its corresponding job ID."""
        return self.op_to_job[op]
    
    def mutation(self, chromosome: Dict) -> Dict:
        child = clone_chromosome(chromosome)
        n_ops = self.decoder.n_operations

        if self.rng.random() < self.mutation_rate:
            i, j = self.rng.sample(range(n_ops), 2)
            seq = child['operation_sequence']
            seq[i], seq[j] = seq[j], seq[i]

        ma = child['machine_assignment']
        wa = child['worker_assignment']
        rnd = self.rng.random
        choice = self.rng.choice
        pools = self.mutation_pairs

        if rnd() < self.mutation_rate:
            op_idx = self.rng.randint(0, n_ops - 1)
            pool = pools[op_idx]
            if pool:
                ma[op_idx], wa[op_idx] = choice(pool)

        g = self.gene_mutation_rate
        if g > 0:
            for op_idx in range(n_ops):
                if rnd() < g:
                    ma[op_idx], wa[op_idx] = choice(pools[op_idx])

        return child

    # ------------------------------------------------------------------
    def evolve(self) -> Dict:
        self.initialize_population()
        fit_get = None

        for generation in range(self.generations):
            if self.fev_count >= self.max_fev:
                break

            selected = self.selection()
            new_population = []
            new_fitness = []

            fv = self.fitness_values
            for idx in nsmallest(self.elite_size, range(len(fv)), key=fv.__getitem__):
                new_population.append(clone_chromosome(self.population[idx]))
                new_fitness.append(fv[idx])

            while len(new_population) < self.population_size and self.fev_count < self.max_fev:
                if len(selected) >= 2:
                    p1, p2 = self.rng.sample(selected, 2)
                else:
                    p1, p2 = selected[0], selected[0]

                if self.rng.random() < self.crossover_rate:
                    child1, child2 = self.feasibility_preserving_crossover(p1, p2)
                else:
                    child1, child2 = p1, p2  # mutation() clones

                child1 = self.mutation(child1)
                child2 = self.mutation(child2)

                f1 = self.evaluate(child1)
                f2 = self.evaluate(child2)

                new_population.append(child1)
                new_fitness.append(f1)

                if len(new_population) < self.population_size and self.fev_count < self.max_fev:
                    new_population.append(child2)
                    new_fitness.append(f2)

            self.population = new_population
            self.fitness_values = new_fitness

            best_idx = min(range(len(new_fitness)), key=new_fitness.__getitem__)
            curr_makespan = self._makespan(self.population[best_idx])

            if new_fitness[best_idx] < self.best_fitness:
                self.best_fitness = new_fitness[best_idx]
                self.best_solution = clone_chromosome(self.population[best_idx])
                self.best_makespan = curr_makespan

            self.history['best_makespan'].append(curr_makespan)

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