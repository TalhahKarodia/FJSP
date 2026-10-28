"""
Memetic Hybrid GA baseline for FJSSP-W.
Adds a per-generation 2-opt local search on the top-k individuals to
StandardGA_V2 (topological decoding, shared initial population, FEV limits).
"""

import sys
import os
from heapq import nsmallest
from typing import Dict

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from baselines.standard_ga_v2 import StandardGA_V2, _clone


class HybridGA_V2(StandardGA_V2):
    def __init__(self, *args, local_search_k=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_search_k = local_search_k

    def _local_search(self, chrom: Dict) -> Dict:
        """2-opt sequence swap; machine/worker assignments are never changed,
        so their lists/hash parts are shared and computed once."""
        m_ref = chrom['machine_assignment']
        w_ref = chrom['worker_assignment']
        mt = tuple(m_ref)
        wt = tuple(w_ref)

        best_seq = chrom['operation_sequence']
        best = chrom
        best_fit = self._evaluate(best, (tuple(best_seq), mt, wt))
        improved = True
        attempts = 0
        max_attempts = 20
        n = self.n_ops

        while improved and attempts < max_attempts and self.fev_count < self.max_fev:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, min(i + 5, n)):
                    seq = best_seq[:]
                    seq[i], seq[j] = seq[j], seq[i]
                    cand = {
                        'operation_sequence': seq,
                        'machine_assignment': m_ref,
                        'worker_assignment': w_ref,
                    }
                    cand_fit = self._evaluate(cand, (tuple(seq), mt, wt))
                    attempts += 1

                    if cand_fit < best_fit:
                        best = cand
                        best_seq = seq
                        best_fit = cand_fit
                        improved = True
                        break
                if improved:
                    break
        return best

    def evolve(self) -> Dict:
        self._init_run()

        # Record initial best before the loop
        fit = self.fitness
        bi = min(range(len(fit)), key=fit.__getitem__)
        self.best_makespan = fit[bi]
        self.best_chromosome = _clone(self.population[bi])

        for _ in range(self.generations):
            if self.fev_count >= self.max_fev:
                break

            self._breed()

            fit = self.fitness
            for idx in nsmallest(self.local_search_k, range(len(fit)), key=fit.__getitem__):
                if self.fev_count >= self.max_fev:
                    break
                improved_ind = self._local_search(self.population[idx])
                self.population[idx] = improved_ind
                self.fitness[idx] = self._evaluate(improved_ind)

            self._update_best()

        return self._result()