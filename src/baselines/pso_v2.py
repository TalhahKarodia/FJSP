"""
Discrete PSO baseline for FJSSP-W. Machine+worker+sequence, no uncertainty.
Compares to Kacem et al. [31] adapted to the FJSSP-W setting.

Includes evaluation memoization caching using the discrete operation sequence rank
and machine/worker assignments to prevent redundant decodes during swarm iterations.
"""

import sys, os, copy, random
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class DiscretePSO:
    def __init__(self, instance_data, swarm_size=30, iterations=50,
                 w=0.7, c1=1.5, c2=1.5, random_seed=None):
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
        self.fev_count = 0
        self.cache_hits = 0
        self.eval_cache = {}

    def _eligible_pairs(self, op_idx):
        pairs = []
        for m in range(self.n_machines):
            for w in self.encoding.get_workers_for_operation_on_machine(op_idx, m):
                pairs.append((m, w))
        return pairs

    def _random_particle(self):
        priority = [self.rng.random() for _ in range(self.n_ops)]
        machines = []
        workers = []
        for op in range(self.n_ops):
            m, w = self.rng.choice(self._eligible_pairs(op))
            machines.append(m)
            workers.append(w)
        return {'priority': priority, 'machines': machines, 'workers': workers}

    def _hash_particle(self, particle: dict) -> tuple:
        """
        Creates a discrete, hashable key from particle priorities and assignments.
        Mapping priorities to a topological sequence index allows distinct continuous 
        priority vectors that resolve to the same scheduling sequence to hit the cache.
        """
        seq = tuple(sorted(range(self.n_ops), key=lambda i: particle['priority'][i]))
        machines = tuple(particle['machines'])
        workers = tuple(particle['workers'])
        return (seq, machines, workers)

    def _decode(self, particle):
        """Decode particle to makespan with memoization cache."""
        key = self._hash_particle(particle)
        if key in self.eval_cache:
            self.cache_hits += 1
            return self.eval_cache[key]

        seq = sorted(range(self.n_ops), key=lambda i: particle['priority'][i])
        m_end = [0] * self.n_machines
        w_end = [0] * self.n_workers
        job_last = {}
        end = [0] * self.n_ops

        for op_idx in seq:
            m = particle['machines'][op_idx]
            w = particle['workers'][op_idx]
            job = self.op_to_job[op_idx]
            dur = self.d[op_idx][m][w]
            start = max(m_end[m], w_end[w], job_last.get(job, 0))
            e = start + dur
            end[op_idx] = e
            m_end[m] = e
            w_end[w] = e
            job_last[job] = e

        self.fev_count += 1
        makespan = max(end) if end else 0
        self.eval_cache[key] = makespan
        return makespan

    def evolve(self):
        self.eval_cache.clear()
        self.cache_hits = 0
        self.fev_count = 0

        swarm = [self._random_particle() for _ in range(self.swarm_size)]
        fits = [self._decode(p) for p in swarm]

        pbest = [copy.deepcopy(p) for p in swarm]
        pbest_fit = list(fits)

        gi = min(range(self.swarm_size), key=lambda i: fits[i])
        gbest = copy.deepcopy(swarm[gi])
        gbest_fit = fits[gi]

        velocities = [{'priority': [0.0] * self.n_ops} for _ in range(self.swarm_size)]

        for _ in range(self.iterations):
            for i in range(self.swarm_size):
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