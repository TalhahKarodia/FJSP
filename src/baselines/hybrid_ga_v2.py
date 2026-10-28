"""
Hybrid GA baseline: standard GA + local search (2-opt on sequence).
No worker flexibility, no uncertainty. Compares to Amjad et al. [32].

Includes evaluation memoization caching to skip redundant decodes during 
GA evolution and 2-opt local search iterations.
"""

import sys, os, copy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from baselines.standard_ga import StandardGA


class HybridGA(StandardGA):
    """Adds 2-opt local search applied to top individuals each generation with memoization caching."""

    def __init__(self, *args, local_search_k=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_search_k = local_search_k  # how many individuals to improve
        self.eval_cache = {}
        self.cache_hits = 0

    def _hash_chromosome(self, chrom: dict) -> tuple:
        """Create hashable key for chromosome caching."""
        return (
            tuple(chrom.get('operation_sequence', [])),
            tuple(chrom.get('machine_assignment', [])),
            tuple(chrom.get('worker_assignment', [])),
        )

    def _evaluate(self, chrom: dict) -> float:
        """
        Evaluates chromosome with memoization cache.
        Skips expensive decoder runs and FEV increments on cache hits.
        """
        key = self._hash_chromosome(chrom)
        if key in self.eval_cache:
            self.cache_hits += 1
            return self.eval_cache[key]

        # Call parent evaluation on cache miss
        fit = super()._evaluate(chrom)
        self.eval_cache[key] = fit
        return fit

    def _local_search(self, chrom):
        """2-opt: try swapping pairs, keep if improves."""
        best = chrom
        best_fit = self._evaluate(best)
        improved = True
        attempts = 0
        max_attempts = 30  # cap for speed

        while improved and attempts < max_attempts:
            improved = False
            for i in range(self.n_ops - 1):
                for j in range(i + 1, min(i + 5, self.n_ops)):  # neighbourhood window
                    # Shallow clone for fast neighborhood iteration
                    cand = {
                        'operation_sequence': best['operation_sequence'][:],
                        'machine_assignment': best['machine_assignment'][:] if 'machine_assignment' in best else [],
                    }
                    if 'worker_assignment' in best:
                        cand['worker_assignment'] = best['worker_assignment'][:]

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

    def evolve(self):
        # Clear evaluation cache per run to prevent stale entries across runs
        self.eval_cache.clear()
        self.cache_hits = 0

        result = super().evolve()

        # Apply local search to best chromosome
        if self.best_chromosome is not None:
            improved = self._local_search(self.best_chromosome)
            improved_fit = self._evaluate(improved)
            if improved_fit < self.best_makespan:
                self.best_makespan = improved_fit
                self.best_chromosome = improved

        return {
            'best_makespan': self.best_makespan,
            'best_solution': self.best_chromosome,
            'fev_count': self.fev_count,
            'cache_hits': self.cache_hits,
            'eval_cache_size': len(self.eval_cache),
        }