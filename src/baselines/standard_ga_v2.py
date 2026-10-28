import sys, os, copy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from baselines.standard_ga_v2 import StandardGA_V2


class HybridGA_V2(StandardGA_V2):
    """
    Fixed Memetic Hybrid GA: Integrates 2-opt local search per generation 
    on top k individuals. Fully compatible with StandardGA_V2 and the shared population.
    """
    def __init__(self, *args, local_search_k=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_search_k = local_search_k

    def _local_search(self, chrom: dict) -> dict:
        """2-opt sequence swap keeping Operation-ID machine/worker assignments intact."""
        best = copy.deepcopy(chrom)
        best_fit = self._evaluate(best)
        improved = True
        attempts = 0
        max_attempts = 20

        while improved and attempts < max_attempts and self.fev_count < self.max_fev:
            improved = False
            for i in range(self.n_ops - 1):
                for j in range(i + 1, min(i + 5, self.n_ops)):
                    cand = {
                        'operation_sequence': best['operation_sequence'][:],
                        'machine_assignment': best['machine_assignment'][:],
                        'worker_assignment': best['worker_assignment'][:]
                    }
                    cand['operation_sequence'][i], cand['operation_sequence'][j] = \
                        cand['operation_sequence'][j], cand['operation_sequence'][i]
                    
                    cand_fit = self._evaluate(cand)
                    attempts += 1

                    if cand_fit < best_fit:
                        best = cand
                        best_fit = cand_fit
                        improved = True
                        break
                if improved:
                    break
        return best

    def evolve(self) -> dict:
        self.eval_cache.clear()
        self.fev_count = 0
        self.cache_hits = 0
        self.best_makespan = float('inf')
        self.best_chromosome = None

        # 1. Load shared initial population if provided by orchestrator; otherwise generate random
        if self.initial_population is not None:
            self.population = copy.deepcopy(self.initial_population)
        else:
            self.population = [self._random_chromosome() for _ in range(self.pop_size)]

        self.fitness = [self._evaluate(c) for c in self.population]

        for _ in range(self.generations):
            if self.fev_count >= self.max_fev:
                break

            # 2. Standard GA Step (Selection, Crossover, Mutation, Elitism)
            selected = self._selection()
            new_pop = []
            new_fit = []
            
            elites = sorted(range(len(self.fitness)), key=lambda i: self.fitness[i])[:self.elite_size]
            for idx in elites:
                new_pop.append(copy.deepcopy(self.population[idx]))
                new_fit.append(self.fitness[idx])

            while len(new_pop) < self.pop_size and self.fev_count < self.max_fev:
                p1 = self.rng.choice(selected)
                p2 = self.rng.choice(selected)
                if self.rng.random() < self.crossover_rate:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                c1 = self._mutation(c1)
                c2 = self._mutation(c2)
                
                new_pop.append(c1)
                new_fit.append(self._evaluate(c1))
                if len(new_pop) < self.pop_size and self.fev_count < self.max_fev:
                    new_pop.append(c2)
                    new_fit.append(self._evaluate(c2))

            self.population = new_pop
            self.fitness = new_fit

            # 3. Interleave Per-Generation Local Search on top k individuals
            sorted_indices = sorted(range(len(self.population)), key=lambda idx: self.fitness[idx])
            for idx in sorted_indices[:self.local_search_k]:
                if self.fev_count >= self.max_fev:
                    break
                improved_ind = self._local_search(self.population[idx])
                improved_fit = self._evaluate(improved_ind)
                self.population[idx] = improved_ind
                self.fitness[idx] = improved_fit

            # 4. Update best overall solution
            bi = min(range(len(self.fitness)), key=lambda i: self.fitness[i])
            if self.fitness[bi] < self.best_makespan:
                self.best_makespan = self.fitness[bi]
                self.best_chromosome = copy.deepcopy(self.population[bi])

        return {
            'best_makespan': self.best_makespan,
            'best_solution': self.best_chromosome,
            'fev_count': self.fev_count,
            'cache_hits': self.cache_hits,
            'eval_cache_size': len(self.eval_cache),
        }