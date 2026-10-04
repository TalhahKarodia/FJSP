"""
Uncertainty Simulator for FJSSP-W.

Replicates the competition's uncertainty math (from util/graph.py's
real_duration method) without relying on the buggy Graph class.

The competition's formula (per docs):
    d_new = d_nominal * (offset + Beta(alpha, beta))

Where (alpha, beta, offset) are the uncertainty parameters for the
worker assigned to the operation.

Note: The competition's graph.py code uses `1.0 + offset + Beta(...)`,
but their documentation says offset=1.1 gives "at least 10% longer"
processing times. We follow the documentation: `offset + Beta(...)`.

This module perturbs only the (machine, worker) pairs actually used
by a given chromosome, so it's ~10x faster than perturbing the full
durations matrix.
"""

import random
import numpy as np
from typing import List, Dict


def create_seeded_uncertainty_vector(n_workers: int, seed: int = None,
                                     factor: float = 10.0,
                                     offset: float = 1.0) -> List[List[float]]:
    """
    Create uncertainty parameters for each worker.

    Args:
        n_workers: number of workers
        seed: RNG seed (per-run reproducibility)
        factor: ratio of alpha to beta in the Beta distribution (default 10.0)
        offset: minimum multiplier shift (default 1.0)

    Returns:
        List of [alpha, beta, offset] for each worker.
    """
    rng = random.Random(seed)
    uncertainty_parameters = []
    for _ in range(n_workers):
        alpha = rng.random()               # in (0, 1)
        beta = factor * alpha               # in (0, 10)
        uncertainty_parameters.append([alpha, beta, offset])
    return uncertainty_parameters


def simulate_one_scenario(
    chromosome: Dict,
    instance: Dict,
    decoder,
    uncertainty_vector: List[List[float]],
    rng: random.Random,
    use_documented_formula: bool = True,
) -> float:
    """
    Sample one uncertainty scenario and return the perturbed makespan.

    Args:
        chromosome: dict with 'machine_assignment', 'worker_assignment',
                    'operation_sequence'
        instance: dict from parse_competition_instance()
        decoder: FJSSPWDecoder instance (durations will be temporarily swapped)
        uncertainty_vector: list of [alpha, beta, offset] per worker
        rng: per-call random.Random instance (do NOT use global random)
        use_documented_formula: if True, use `offset + Beta`; if False,
                                use graph.py's `1.0 + offset + Beta`

    Returns:
        float: makespan under this perturbed scenario
    """
    d_nominal = instance['durations']
    m_assign = chromosome['machine_assignment']
    w_assign = chromosome['worker_assignment']

    # Copy durations; only perturb used pairs
    d_perturbed = np.array(d_nominal, dtype=float, copy=True)

    n_ops = len(m_assign)
    for op_idx in range(n_ops):
        m = m_assign[op_idx]
        w = w_assign[op_idx]
        d_val = d_nominal[op_idx][m][w]
        if d_val == 0:
            continue  # infeasible pair, skip (shouldn't happen)

        alpha, beta, offset = uncertainty_vector[w]
        delay = rng.betavariate(alpha, beta)

        if use_documented_formula:
            multiplier = offset + delay            # per docs
        else:
            multiplier = 1.0 + offset + delay      # per graph.py code

        d_perturbed[op_idx][m][w] = d_val * multiplier

    # Temporarily swap decoder durations
    original_durations = decoder.durations
    decoder.durations = d_perturbed
    try:
        decoded = decoder.decode(chromosome)
        makespan = decoded['makespan']
    finally:
        decoder.durations = original_durations

    return makespan


def evaluate_robustness(
    chromosome: Dict,
    instance: Dict,
    decoder,
    uncertainty_vector: List[List[float]],
    n_scenarios: int = 50,
    seed: int = None,
    use_documented_formula: bool = True,
) -> Dict:
    """
    Evaluate a chromosome across n_scenarios uncertainty realisations.

    Returns:
        dict with keys: mean_makespan, std_makespan, min_makespan,
                        max_makespan, all_makespans, nominal_makespan,
                        deterioration_ratio (mean / nominal)
    """
    rng = random.Random(seed)

    # Nominal makespan (no perturbation)
    nominal_decoded = decoder.decode(chromosome)
    nominal_makespan = nominal_decoded['makespan']

    makespans = []
    for _ in range(n_scenarios):
        mk = simulate_one_scenario(
            chromosome, instance, decoder, uncertainty_vector, rng,
            use_documented_formula=use_documented_formula
        )
        makespans.append(mk)

    mean_mk = float(np.mean(makespans))
    std_mk = float(np.std(makespans))

    return {
        'mean_makespan': mean_mk,
        'std_makespan': std_mk,
        'min_makespan': float(np.min(makespans)),
        'max_makespan': float(np.max(makespans)),
        'all_makespans': makespans,
        'nominal_makespan': nominal_makespan,
        'deterioration_ratio': mean_mk / nominal_makespan if nominal_makespan > 0 else 0,
    }