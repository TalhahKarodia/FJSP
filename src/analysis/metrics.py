"""Evaluation metrics for FJSSP-W experiments."""

import numpy as np
from collections import defaultdict


def worker_loads(chromosome, instance):
    """Per-worker total processing time."""
    d = instance['durations']
    loads = defaultdict(float)
    for op_idx in range(len(chromosome['machine_assignment'])):
        m = chromosome['machine_assignment'][op_idx]
        w = chromosome['worker_assignment'][op_idx]
        dur = d[op_idx][m][w]
        if dur > 0:
            loads[w] += dur
    n_workers = instance['n_workers']
    return [loads.get(w, 0.0) for w in range(n_workers)]


def machine_loads(chromosome, instance):
    """Per-machine total processing time."""
    d = instance['durations']
    loads = defaultdict(float)
    for op_idx in range(len(chromosome['machine_assignment'])):
        m = chromosome['machine_assignment'][op_idx]
        w = chromosome['worker_assignment'][op_idx]
        dur = d[op_idx][m][w]
        if dur > 0:
            loads[m] += dur
    n_machines = instance['n_machines']
    return [loads.get(m, 0.0) for m in range(n_machines)]


def worker_balance_cv(chromosome, instance):
    """Coefficient of variation of worker loads."""
    loads = worker_loads(chromosome, instance)
    mean = np.mean(loads)
    if mean == 0:
        return 0.0
    return float(np.std(loads) / mean)


def machine_balance_cv(chromosome, instance):
    loads = machine_loads(chromosome, instance)
    mean = np.mean(loads)
    if mean == 0:
        return 0.0
    return float(np.std(loads) / mean)


def worker_utilisation(chromosome, instance, makespan):
    """Average fraction of makespan each worker is busy."""
    loads = worker_loads(chromosome, instance)
    if makespan <= 0:
        return 0.0
    return float(np.mean([l / makespan for l in loads]))


def machine_utilisation(chromosome, instance, makespan):
    loads = machine_loads(chromosome, instance)
    if makespan <= 0:
        return 0.0
    return float(np.mean([l / makespan for l in loads]))


def summarise_run(makespan, chromosome, instance):
    """Compute all metrics for one run."""
    return {
        'makespan': float(makespan),
        'worker_balance_cv': worker_balance_cv(chromosome, instance),
        'machine_balance_cv': machine_balance_cv(chromosome, instance),
        'worker_utilisation': worker_utilisation(chromosome, instance, makespan),
        'machine_utilisation': machine_utilisation(chromosome, instance, makespan),
    }