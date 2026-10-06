"""
Uncertainty simulator for FJSSP-W.

Replicates the competition's uncertainty math (from util/graph.py) without
relying on the buggy Graph class. Perturbs only the (machine, worker) pairs
actually used by a chromosome.

Formula (competition documentation):
    d_new = d_nominal * (offset + Beta(alpha, beta))

Where (alpha, beta, offset) are the uncertainty params for the worker
assigned to the operation.
"""

import random
import numpy as np
from typing import List, Dict


def create_seeded_uncertainty_vector(
    n_workers: int,
    seed: int = None,
    factor: float = 10.0,
    offset: float = 1.0,
) -> List[List[float]]:
    """
    Create [alpha, beta, offset] uncertainty params for each worker.

    Args:
        n_workers: number of workers
        seed: RNG seed for reproducibility (per-run)
        factor: ratio of beta to alpha in the Beta distribution
        offset: minimum multiplier shift
    """
    rng = random.Random(seed)
    params = []
    for _ in range(n_workers):
        alpha = rng.random()               # in (0, 1)
        beta = factor * alpha               # in (0, 10)
        params.append([alpha, beta, offset])
    return params


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
        decoder: FJSSPWDecoder instance (durations swapped temporarily)
        uncertainty_vector: list of [alpha, beta, offset] per worker
        rng: per-call random.Random instance
        use_documented_formula: if True use `offset + Beta` (matches docs),
                                else use graph.py's `1.0 + offset + Beta`

    Returns:
        float: makespan under this perturbed scenario
    """
    d_nominal = instance['durations']
    m_assign = chromosome['machine_assignment']
    w_assign = chromosome['worker_assignment']

    d_perturbed = np.array(d_nominal, dtype=float, copy=True)

    for op_idx in range(len(m_assign)):
        m = m_assign[op_idx]
        w = w_assign[op_idx]
        d_val = d_nominal[op_idx][m][w]
        if d_val == 0:
            continue

        alpha, beta, offset = uncertainty_vector[w]
        delay = rng.betavariate(alpha, beta)

        if use_documented_formula:
            multiplier = offset + delay
        else:
            multiplier = 1.0 + offset + delay

        d_perturbed[op_idx][m][w] = d_val * multiplier

    original = decoder.durations
    decoder.durations = d_perturbed
    try:
        decoded = decoder.decode(chromosome)
        makespan = decoded['makespan']
    finally:
        decoder.durations = original

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
    Evaluate chromosome across n_scenarios realisations.

    Returns dict with mean, std, min, max makespans, nominal makespan,
    and deterioration ratio (mean / nominal).
    """
    rng = random.Random(seed)

    nominal = decoder.decode(chromosome)['makespan']

    makespans = []
    for _ in range(n_scenarios):
        mk = simulate_one_scenario(
            chromosome, instance, decoder, uncertainty_vector, rng,
            use_documented_formula=use_documented_formula,
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
        'nominal_makespan': nominal,
        'deterioration_ratio': mean_mk / nominal if nominal > 0 else 0,
    }