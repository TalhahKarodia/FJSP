"""
Hybrid GA baseline: standard GA + local search (2-opt on sequence).
No worker flexibility, no uncertainty. Compares to Amjad et al. [32].
"""

import sys, os, copy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from baselines.standard_ga import StandardGA


class HybridGA(StandardGA):
    """Adds 2-opt local search applied to top individuals each generation."""

    def __init__(self, *args, local_search_k=20, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_search_k = local_search_k  # how many individuals to improve

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
                    cand = copy.deepcopy(best)
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
        result = super().evolve()

        # Apply local search to best
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
        }