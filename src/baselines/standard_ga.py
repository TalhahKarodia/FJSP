"""
Standard GA baseline for FJSSP-W: machine+worker+sequence GA, no uncertainty.
Uses assigned worker durations and enforces worker availability.

Compares to Driss et al. [10] adapted to the FJSSP-W setting.
"""

import sys, os, copy, random
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class StandardGA:
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

        self.population = []
        self.fitness = []
        self.best_makespan = float('inf')
        self.best_chromosome = None
        self.fev_count = 0

    def _eligible_workers(self, op_idx, m):
        return self.encoding.get_workers_for_operation_on_machine(op_idx, m)

    def _random_chromosome(self):
        machines = []
        workers = []
        for op in range(self.n_ops):
            pairs = []
            for m in range(self.n_machines):
                for w in self._eligible_workers(op, m):
                    pairs.append((m, w))
            m, w = self.rng.choice(pairs)
            machines.append(m)
            workers.append(w)
        seq = list(range(self.n_ops))
        self.rng.shuffle(seq)
        return {
            'machine_assignment': machines,
            'worker_assignment': workers,
            'operation_sequence': seq,
        }

    def _decode(self, chrom):
        s = [0] * self.n_ops
        e = [0] * self.n_ops
        m_end = [0] * self.n_machines
        w_end = [0] * self.n_workers
        job_last = {}

        for op_idx in chrom['operation_sequence']:
            m = chrom['machine_assignment'][op_idx]
            w = chrom['worker_assignment'][op_idx]
            if not self.encoding.is_possible(op_idx, m, w):
                raise ValueError(f"Infeasible: op {op_idx} m {m} w {w}")
            job = self.op_to_job.get(op_idx, 0)
            dur = self.d[op_idx][m][w]
            start = max(m_end[m], w_end[w], job_last.get(job, 0))
            end = start + dur
            s[op_idx] = start
            e[op_idx] = end
            m_end[m] = end
            w_end[w] = end
            job_last[job] = end
        return max(e) if e else 0

    def _evaluate(self, chrom):
        self.fev_count += 1
        return self._decode(chrom)

    def _selection(self):
        selected = []
        for _ in range(self.pop_size):
            idxs = self.rng.sample(range(self.pop_size), self.tournament_size)
            best = min(idxs, key=lambda i: self.fitness[i])
            selected.append(copy.deepcopy(self.population[best]))
        return selected

    def _crossover(self, p1, p2):
        n = self.n_ops
        c1 = {'machine_assignment': [None] * n,
              'worker_assignment': [None] * n,
              'operation_sequence': [None] * n}
        c2 = {'machine_assignment': [None] * n,
              'worker_assignment': [None] * n,
              'operation_sequence': [None] * n}

        # Step 1: order crossover on sequence
        pt1 = self.rng.randint(0, n - 2)
        pt2 = self.rng.randint(pt1 + 1, n - 1)
        for i in range(pt1, pt2 + 1):
            c1['operation_sequence'][i] = p1['operation_sequence'][i]
            c2['operation_sequence'][i] = p2['operation_sequence'][i]

        # Step 2: fill remaining positions with donor's ops
        for child, donor in [(c1, p2), (c2, p1)]:
            for i in range(n):
                op = donor['operation_sequence'][i]
                if op not in child['operation_sequence']:
                    for j in range(n):
                        if child['operation_sequence'][j] is None:
                            child['operation_sequence'][j] = op
                            break

        # Step 3: inherit assignments per-op (indexed by op, not position)
        for op in range(n):
            if self.rng.random() < 0.5:
                c1['machine_assignment'][op] = p1['machine_assignment'][op]
                c1['worker_assignment'][op] = p1['worker_assignment'][op]
                c2['machine_assignment'][op] = p2['machine_assignment'][op]
                c2['worker_assignment'][op] = p2['worker_assignment'][op]
            else:
                c1['machine_assignment'][op] = p2['machine_assignment'][op]
                c1['worker_assignment'][op] = p2['worker_assignment'][op]
                c2['machine_assignment'][op] = p1['machine_assignment'][op]
                c2['worker_assignment'][op] = p1['worker_assignment'][op]

        return c1, c2

    def _mutation(self, chrom):
        c = copy.deepcopy(chrom)
        if self.rng.random() < self.mutation_rate:
            i, j = self.rng.sample(range(self.n_ops), 2)
            c['operation_sequence'][i], c['operation_sequence'][j] = \
                c['operation_sequence'][j], c['operation_sequence'][i]
        if self.rng.random() < self.mutation_rate:
            op_idx = self.rng.randint(0, self.n_ops - 1)
            pairs = []
            for m in range(self.n_machines):
                for w in self._eligible_workers(op_idx, m):
                    pairs.append((m, w))
            if pairs:
                m, w = self.rng.choice(pairs)
                c['machine_assignment'][op_idx] = m
                c['worker_assignment'][op_idx] = w
        return c

    def evolve(self):
        self.population = [self._random_chromosome() for _ in range(self.pop_size)]
        self.fitness = [self._evaluate(c) for c in self.population]

        for _ in range(self.generations):
            selected = self._selection()
            new_pop = []
            new_fit = []
            elites = sorted(range(len(self.fitness)), key=lambda i: self.fitness[i])[:self.elite_size]
            for idx in elites:
                new_pop.append(copy.deepcopy(self.population[idx]))
                new_fit.append(self.fitness[idx])

            while len(new_pop) < self.pop_size:
                p1 = self.rng.choice(selected)
                p2 = self.rng.choice(selected)
                if self.rng.random() < self.crossover_rate:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                c1 = self._mutation(c1)
                c2 = self._mutation(c2)
                new_pop.append(c1); new_fit.append(self._evaluate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2); new_fit.append(self._evaluate(c2))

            self.population = new_pop
            self.fitness = new_fit

            bi = min(range(len(self.fitness)), key=lambda i: self.fitness[i])
            if self.fitness[bi] < self.best_makespan:
                self.best_makespan = self.fitness[bi]
                self.best_chromosome = copy.deepcopy(self.population[bi])

        return {
            'best_makespan': self.best_makespan,
            'best_solution': self.best_chromosome,
            'fev_count': self.fev_count,
        }