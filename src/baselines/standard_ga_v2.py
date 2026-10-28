"""
Standard GA baseline for FJSSP-W: machine+worker+sequence GA, no uncertainty.
Topological precedence decoding, shared initial population seeding.

Optimised: no deepcopy, O(n) order crossover (was O(n^2)-O(n^3)), precomputed
feasible pairs, shared decoder, single breeding routine reused by HybridGA.
"""

import sys, os, random
from heapq import nsmallest
from typing import Dict, Tuple, List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from ga.fjssp_w.decoder_w_v2 import FJSSPWDecoder_V2


def _clone(c: Dict) -> Dict:
    return {
        'machine_assignment': c['machine_assignment'][:],
        'worker_assignment': c['worker_assignment'][:],
        'operation_sequence': c['operation_sequence'][:],
    }


class StandardGA_V2:
    def __init__(
        self,
        instance_data,
        population_size=30,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        tournament_size=3,
        elite_size=2,
        random_seed=None,
        initial_population=None,
        max_fev=5_000_000,
        decoder: Optional[FJSSPWDecoder_V2] = None,
    ):
        self.rng = random.Random(random_seed)
        self.instance = instance_data
        self.n_ops = instance_data['n_operations']
        self.n_machines = instance_data['n_machines']
        self.n_workers = instance_data['n_workers']
        self.d = instance_data['durations']
        self.encoding = instance_data['encoding']
        self.js = instance_data['job_sequence']
        self.op_to_job = {op: job for op, job in enumerate(self.js)}

        self.pop_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.initial_population = initial_population
        self.max_fev = max_fev

        self.decoder = decoder if decoder is not None else FJSSPWDecoder_V2(instance_data)
        self._pairs = self.decoder.feasible_pairs

        self.population = []
        self.fitness = []
        self.best_makespan = float('inf')
        self.best_chromosome = None
        self.fev_count = 0
        self.cache_hits = 0
        self.eval_cache = {}

    # ------------------------------------------------------------------
    def _hash_chromosome(self, chrom: Dict) -> Tuple:
        return (
            tuple(chrom.get('operation_sequence', [])),
            tuple(chrom.get('machine_assignment', [])),
            tuple(chrom.get('worker_assignment', [])),
        )

    def _eligible_workers(self, op_idx: int, m: int) -> List[int]:
        return self.decoder.get_eligible_workers(op_idx, m)

    def _random_chromosome(self) -> Dict:
        return self.decoder.create_random_chromosome(self.rng)

    def _evaluate(self, chrom: Dict, key: Optional[Tuple] = None) -> float:
        """Makespan via topological decoder with memoization.
        `key` may be supplied by callers that already have it."""
        if key is None:
            key = self._hash_chromosome(chrom)
        cached = self.eval_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        if self.fev_count >= self.max_fev:
            return float('inf')

        self.fev_count += 1
        fit = self.decoder.decode(chrom)['makespan']
        self.eval_cache[key] = fit
        return fit

    # ------------------------------------------------------------------
    def _selection(self) -> List[Dict]:
        # References are safe: crossover builds new lists, mutation clones.
        pop, fit = self.population, self.fitness
        n, k = self.pop_size, self.tournament_size
        sample = self.rng.sample
        selected = []
        for _ in range(n):
            idxs = sample(range(n), k)
            selected.append(pop[min(idxs, key=fit.__getitem__)])
        return selected

    def _crossover(self, p1: Dict, p2: Dict) -> Tuple[Dict, Dict]:
        n = self.n_ops
        rnd = self.rng.random

        # Step 1: OX segment
        pt1 = self.rng.randint(0, n - 2)
        pt2 = self.rng.randint(pt1 + 1, n - 1)
        s1, s2 = p1['operation_sequence'], p2['operation_sequence']

        def ox_child(seg_parent_seq, donor_seq):
            cs = [None] * n
            seg = seg_parent_seq[pt1:pt2 + 1]
            cs[pt1:pt2 + 1] = seg
            used = set(seg)
            pos = 0
            for op in donor_seq:          # Step 2: fill free slots in order
                if op in used:
                    continue
                if pos == pt1:
                    pos = pt2 + 1
                cs[pos] = op
                pos += 1
            return cs

        c1s = ox_child(s1, s2)
        c2s = ox_child(s2, s1)

        # Step 3: inherit (machine, worker) per operation id
        p1m, p1w = p1['machine_assignment'], p1['worker_assignment']
        p2m, p2w = p2['machine_assignment'], p2['worker_assignment']
        c1m = [None] * n; c1w = [None] * n
        c2m = [None] * n; c2w = [None] * n
        for op in range(n):
            if rnd() < 0.5:
                c1m[op] = p1m[op]; c1w[op] = p1w[op]
                c2m[op] = p2m[op]; c2w[op] = p2w[op]
            else:
                c1m[op] = p2m[op]; c1w[op] = p2w[op]
                c2m[op] = p1m[op]; c2w[op] = p1w[op]

        return (
            {'machine_assignment': c1m, 'worker_assignment': c1w, 'operation_sequence': c1s},
            {'machine_assignment': c2m, 'worker_assignment': c2w, 'operation_sequence': c2s},
        )

    def _mutation(self, chrom: Dict) -> Dict:
        c = _clone(chrom)
        if self.rng.random() < self.mutation_rate:
            i, j = self.rng.sample(range(self.n_ops), 2)
            seq = c['operation_sequence']
            seq[i], seq[j] = seq[j], seq[i]
        if self.rng.random() < self.mutation_rate:
            op_idx = self.rng.randint(0, self.n_ops - 1)
            pairs = self._pairs[op_idx]
            if pairs:
                m, w = self.rng.choice(pairs)
                c['machine_assignment'][op_idx] = m
                c['worker_assignment'][op_idx] = w
        return c

    # ------------------------------------------------------------------
    def _init_run(self):
        self.eval_cache.clear()
        self.fev_count = 0
        self.cache_hits = 0
        self.best_makespan = float('inf')
        self.best_chromosome = None

        if self.initial_population is not None:
            self.population = [_clone(c) for c in self.initial_population]
        else:
            self.population = [self._random_chromosome() for _ in range(self.pop_size)]
        self.fitness = [self._evaluate(c) for c in self.population]

    def _breed(self):
        """One generation: selection, elitism, crossover, mutation, evaluation."""
        selected = self._selection()
        new_pop = []
        new_fit = []

        fit = self.fitness
        for idx in nsmallest(self.elite_size, range(len(fit)), key=fit.__getitem__):
            new_pop.append(_clone(self.population[idx]))
            new_fit.append(fit[idx])

        choice = self.rng.choice
        while len(new_pop) < self.pop_size and self.fev_count < self.max_fev:
            p1 = choice(selected)
            p2 = choice(selected)
            if self.rng.random() < self.crossover_rate:
                c1, c2 = self._crossover(p1, p2)
            else:
                c1, c2 = p1, p2          # _mutation clones
            c1 = self._mutation(c1)
            c2 = self._mutation(c2)

            new_pop.append(c1)
            new_fit.append(self._evaluate(c1))
            if len(new_pop) < self.pop_size and self.fev_count < self.max_fev:
                new_pop.append(c2)
                new_fit.append(self._evaluate(c2))

        self.population = new_pop
        self.fitness = new_fit

    def _update_best(self):
        fit = self.fitness
        bi = min(range(len(fit)), key=fit.__getitem__)
        if fit[bi] < self.best_makespan:
            self.best_makespan = fit[bi]
            self.best_chromosome = _clone(self.population[bi])

    def _result(self) -> Dict:
        return {
            'best_makespan': self.best_makespan,
            'best_solution': self.best_chromosome,
            'fev_count': self.fev_count,
            'cache_hits': self.cache_hits,
            'eval_cache_size': len(self.eval_cache),
        }

    def evolve(self) -> Dict:
        self._init_run()
        for _ in range(self.generations):
            if self.fev_count >= self.max_fev:
                break
            self._breed()
            self._update_best()
        return self._result()