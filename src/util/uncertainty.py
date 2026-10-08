import random

def create_uncertainty_vector(n_required, factor : float = 10.0, offset : float = 1.0):
    uncertainty_parameters = []
    for _ in range(n_required):
        alpha = random.random()
        beta = factor * alpha
        offset = offset
        uncertainty_parameters.append([alpha, beta, offset])
    return uncertainty_parameters