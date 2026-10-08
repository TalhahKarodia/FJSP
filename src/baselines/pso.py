"""
Discrete PSO baseline for FJSSP-W. Machine+worker+sequence, no uncertainty.
Compares to Kacem et al. [31] adapted to the FJSSP-W setting.
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

    def _decode(self, particle):
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
        return max(end) if end else 0

    def evolve(self):
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
        }