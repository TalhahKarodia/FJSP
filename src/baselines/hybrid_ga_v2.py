"""
Hybrid GA baseline: standard GA + post-processing local search (2-opt on sequence).
Compares to Amjad et al. [32]. Fully compatible with shared population orchestrator.
"""

import sys, os, copy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from baselines.standard_ga_v2 import StandardGA_V2


class HybridGA_V2(StandardGA_V2):
    """Adds post-processing 2-opt local search applied to the best individual after GA evolution."""

    def __init__(self, *args, local_search_k=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_search_k = local_search_k

    def _local_search(self, chrom: dict) -> dict:
        """2-opt sequence swap keeping Operation-ID machine/worker assignments intact."""
        best = copy.deepcopy(chrom)
        best_fit = self._evaluate(best)
        improved = True
        attempts = 0
        max_attempts = 30  # cap for neighborhood search

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
        # Run full standard GA evolution with shared initial population
        result = super().evolve()

        # Post-processing local search on the best solution found
        if self.best_chromosome is not None and self.fev_count < self.max_fev:
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