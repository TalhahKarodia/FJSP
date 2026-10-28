"""
Discrete PSO baseline for FJSSP-W. Machine+worker+sequence, no uncertainty.
Compares to Kacem et al. [31] adapted to the FJSSP-W setting.

Updated to enforce topological precedence decoding via FJSSPWDecoder_V2,
support shared initial population seeding, and respect max_fev limits.
"""

import sys, os, copy, random
from typing import Dict, List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from ga.fjssp_w.decoder_w_v2 import FJSSPWDecoder_V2


class DiscretePSO_V2:
    def __init__(
        self,
        instance_data,
        swarm_size=30,
        iterations=50,
        w=0.7,
        c1=1.5,
        c2=1.5,
        random_seed=None,
        initial_population=None,
        max_fev=5_000_000,
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

        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w, self.c1, self.c2 = w, c1, c2
        self.initial_population = initial_population
        self.max_fev = max_fev

        # External decoder enforcing topological precedence
        self.decoder = FJSSPWDecoder_V2(instance_data)

        self.fev_count = 0
        self.cache_hits = 0
        self.eval_cache = {}

    def _eligible_pairs(self, op_idx: int) -> List[Tuple[int, int]]:
        pairs = []
        for m in range(self.n_machines):
            for w in self.encoding.get_workers_for_operation_on_machine(op_idx, m):
                pairs.append((m, w))
        return pairs

    def _random_particle(self) -> Dict:
        priority = [self.rng.random() for _ in range(self.n_ops)]
        machines, workers = [], []
        for op in range(self.n_ops):
            m, w = self.rng.choice(self._eligible_pairs(op))
            machines.append(m)
            workers.append(w)
        return {'priority': priority, 'machines': machines, 'workers': workers}

    def _hash_particle(self, particle: Dict) -> Tuple:
        """
        Creates a discrete, hashable key from particle priorities and assignments.
        Mapping priorities to topological sequence rank allows continuous priority
        vectors that resolve to the same sequence to hit the cache.
        """
        seq = tuple(sorted(range(self.n_ops), key=lambda i: particle['priority'][i]))
        machines = tuple(particle['machines'])
        workers = tuple(particle['workers'])
        return (seq, machines, workers)

    def _decode(self, particle: Dict) -> float:
        """Decodes particle to makespan with memoization cache and topological precedence."""
        key = self._hash_particle(particle)
        if key in self.eval_cache:
            self.cache_hits += 1
            return self.eval_cache[key]

        if self.fev_count >= self.max_fev:
            return float('inf')

        self.fev_count += 1
        seq = sorted(range(self.n_ops), key=lambda i: particle['priority'][i])
        chrom = {
            'operation_sequence': seq,
            'machine_assignment': particle['machines'],
            'worker_assignment': particle['workers'],
        }
        res = self.decoder.decode(chrom)
        makespan = res['makespan']
        self.eval_cache[key] = makespan
        return makespan

    def evolve(self) -> Dict:
        self.eval_cache.clear()
        self.cache_hits = 0
        self.fev_count = 0

        # Load shared initial population if provided by orchestrator; otherwise generate random
        if self.initial_population is not None:
            swarm = copy.deepcopy(self.initial_population)
        else:
            swarm = [self._random_particle() for _ in range(self.swarm_size)]

        fits = [self._decode(p) for p in swarm]

        pbest = [copy.deepcopy(p) for p in swarm]
        pbest_fit = list(fits)

        gi = min(range(len(fits)), key=lambda i: fits[i])
        gbest = copy.deepcopy(swarm[gi])
        gbest_fit = fits[gi]

        velocities = [{'priority': [0.0] * self.n_ops} for _ in range(len(swarm))]

        for _ in range(self.iterations):
            if self.fev_count >= self.max_fev:
                break

            for i in range(len(swarm)):
                v = velocities[i]['priority']
                p = swarm[i]
                r1, r2 = self.rng.random(), self.rng.random()

                for j in range(self.n_ops):
                    v[j] = (self.w * v[j]
                            + self.c1 * r1 * (pbest[i]['priority'][j] - p['priority'][j])
                            + self.c2 * r2 * (gbest['priority'][j] - p['priority'][j]))
                    p['priority'][j] = max(0.0, min(1.0, p['priority'][j] + v[j]))
                    if self.rng.random() < 0.15:
                        m, w = self.rng.choice(self._eligible_pairs(j))
                        p['machines'][j] = m
                        p['workers'][j] = w

                f = self._decode(p)
                fits[i] = f
                if f < pbest_fit[i]:
                    pbest[i] = copy.deepcopy(p)
                    pbest_fit[i] = f
                if f < gbest_fit:
                    gbest = copy.deepcopy(p)
                    gbest_fit = f

        return {
            'best_makespan': gbest_fit,
            'best_solution': gbest,
            'fev_count': self.fev_count,
            'cache_hits': self.cache_hits,
            'eval_cache_size': len(self.eval_cache),
        }