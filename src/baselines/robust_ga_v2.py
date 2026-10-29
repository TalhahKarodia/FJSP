"""
Robust GA baseline: GA with machine breakdown uncertainty only.
No worker flexibility. Compares to Al-Hinai et al. [44].

Machine breakdowns are simulated as multiplicative perturbations
of machine processing times (machines effectively slower).

Includes evaluation memoization caching with deterministic scenario seeding
to eliminate duplicate scenario evaluations and avoid scenario RNG drift.
"""

import sys, os, copy, random
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from baselines.standard_ga import StandardGA


class RobustGA(StandardGA):
    """StandardGA with S-scenario robust fitness (machine perturbations) and caching."""

    def __init__(self, *args, n_scenarios=5, breakdown_scale=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_scenarios = n_scenarios
        self.breakdown_scale = breakdown_scale
        self.seed = kwargs.get('random_seed', 42)
        self.eval_cache = {}
        self.cache_hits = 0

    def _hash_chromosome(self, chrom: dict) -> tuple:
        """Create hashable key for chromosome caching."""
        return (
            tuple(chrom.get('operation_sequence', [])),
            tuple(chrom.get('machine_assignment', [])),
            tuple(chrom.get('worker_assignment', [])),
        )

    def _evaluate(self, chrom):
        """
        Robust evaluation: mean makespan over S machine-perturbed scenarios.
        Uses memoization cache to skip redundant scenario decodes.
        """
        key = self._hash_chromosome(chrom)
        if key in self.eval_cache:
            self.cache_hits += 1
            return self.eval_cache[key]

        # Deterministic scenario RNG derived from chromosome key
        chrom_seed = hash(key) & 0xffffffff
        scen_rng = random.Random(chrom_seed)

        total = 0.0
        for _ in range(self.n_scenarios):
            # Perturb durations: multiply by random factor in [1, 1+scale]
            perturbed_d = []
            for op_idx in range(self.n_ops):
                row = []
                for m in range(self.n_machines):
                    wrow = []
                    for w in range(len(self.d[op_idx][m])):
                        base = self.d[op_idx][m][w]
                        if base > 0:
                            mult = 1.0 + scen_rng.random() * self.breakdown_scale
                            wrow.append(base * mult)
                        else:
                            wrow.append(0)
                    row.append(wrow)
                perturbed_d.append(row)

            saved = self.d
            self.d = perturbed_d
            total += self._decode(chrom)
            self.d = saved

        self.fev_count += self.n_scenarios
        fit = total / self.n_scenarios
        self.eval_cache[key] = fit
        return fit

    def evolve(self):
        # Clear evaluation cache per run to prevent stale entries across runs
        self.eval_cache.clear()
        self.cache_hits = 0

        res = super().evolve()

        if isinstance(res, dict):
            res['cache_hits'] = self.cache_hits
            res['eval_cache_size'] = len(self.eval_cache)
        return res