"""
Robust GA baseline: GA with machine breakdown uncertainty only.
No worker flexibility. Compares to Al-Hinai et al. [44].

Machine breakdowns are simulated as multiplicative perturbations
of machine processing times (machines effectively slower).
"""

import sys, os, copy, random
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from baselines.standard_ga import StandardGA


class RobustGA(StandardGA):
    """StandardGA with S-scenario robust fitness (machine perturbations)."""

    def __init__(self, *args, n_scenarios=5, breakdown_scale=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_scenarios = n_scenarios
        self.breakdown_scale = breakdown_scale
        self.rng_scenarios = random.Random(kwargs.get('random_seed', 42))

    def _evaluate(self, chrom):
        """Robust evaluation: mean makespan over S machine-perturbed scenarios."""
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
                            mult = 1.0 + self.rng_scenarios.random() * self.breakdown_scale
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
        return total / self.n_scenarios