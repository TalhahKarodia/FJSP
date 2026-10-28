"""
Discrete PSO baseline for FJSSP-W. Machine+worker+sequence, no uncertainty.
Compares to Kacem et al. [31] adapted to the FJSSP-W setting.

Topological precedence decoding via FJSSPWDecoder_V2, shared initial
population seeding, max_fev respected.

Optimised: eligible pairs precomputed (were rebuilt on every mutation), the
priority sort is done once per decode, no deepcopy, localised inner loop.
"""

import sys, os, random
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from ga.fjssp_w.decoder_w_v2 import FJSSPWDecoder_V2


def _copy_particle(p: Dict) -> Dict:
    return {
        'priority': p['priority'][:],
        'machines': p['machines'][:],
        'workers': p['workers'][:],
    }


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

        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w, self.c1, self.c2 = w, c1, c2
        self.initial_population = initial_population
        self.max_fev = max_fev

        self.decoder = decoder if decoder is not None else FJSSPWDecoder_V2(instance_data)
        self._pairs = self.decoder.feasible_pairs

        self.fev_count = 0
        self.cache_hits = 0
        self.eval_cache = {}

    def _eligible_pairs(self, op_idx: int) -> List[Tuple[int, int]]:
        return self._pairs[op_idx]

    def _random_particle(self) -> Dict:
        priority = [self.rng.random() for _ in range(self.n_ops)]
        machines, workers = [], []
        for op in range(self.n_ops):
            m, w = self.rng.choice(self._pairs[op])
            machines.append(m)
            workers.append(w)
        return {'priority': priority, 'machines': machines, 'workers': workers}

    def _decode(self, particle: Dict) -> float:
        """Particle -> makespan, memoised on (sequence rank, machines, workers)."""
        pr = particle['priority']
        seq = sorted(range(self.n_ops), key=pr.__getitem__)
        key = (tuple(seq), tuple(particle['machines']), tuple(particle['workers']))
        cached = self.eval_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        if self.fev_count >= self.max_fev:
            return float('inf')

        self.fev_count += 1
        makespan = self.decoder.decode({
            'operation_sequence': seq,
            'machine_assignment': particle['machines'],
            'worker_assignment': particle['workers'],
        })['makespan']
        self.eval_cache[key] = makespan
        return makespan

    def evolve(self) -> Dict:
        self.eval_cache.clear()
        self.cache_hits = 0
        self.fev_count = 0

        if self.initial_population is not None:
            swarm = [_copy_particle(p) for p in self.initial_population]
        else:
            swarm = [self._random_particle() for _ in range(self.swarm_size)]

        fits = [self._decode(p) for p in swarm]

        pbest = [_copy_particle(p) for p in swarm]
        pbest_fit = list(fits)

        gi = min(range(len(fits)), key=fits.__getitem__)
        gbest = _copy_particle(swarm[gi])
        gbest_fit = fits[gi]

        n = self.n_ops
        velocities = [[0.0] * n for _ in range(len(swarm))]
        rnd = self.rng.random
        choice = self.rng.choice
        pairs = self._pairs
        iw, c1, c2 = self.w, self.c1, self.c2

        for _ in range(self.iterations):
            if self.fev_count >= self.max_fev:
                break

            for i in range(len(swarm)):
                p = swarm[i]
                v = velocities[i]
                pp = p['priority']
                pm = p['machines']
                pw = p['workers']
                pb = pbest[i]['priority']
                gb = gbest['priority']
                r1, r2 = rnd(), rnd()

                for j in range(n):
                    x = pp[j]
                    vj = iw * v[j] + c1 * r1 * (pb[j] - x) + c2 * r2 * (gb[j] - x)
                    v[j] = vj
                    x += vj
                    pp[j] = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
                    if rnd() < 0.15:
                        m, wk = choice(pairs[j])
                        pm[j] = m
                        pw[j] = wk

                f = self._decode(p)
                fits[i] = f
                if f < pbest_fit[i]:
                    pbest[i] = _copy_particle(p)
                    pbest_fit[i] = f
                if f < gbest_fit:
                    gbest = _copy_particle(p)
                    gbest_fit = f

        return {
            'best_makespan': gbest_fit,
            'best_solution': gbest,
            'fev_count': self.fev_count,
            'cache_hits': self.cache_hits,
            'eval_cache_size': len(self.eval_cache),
        }