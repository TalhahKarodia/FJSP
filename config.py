"""Locked-in GA configuration for FJSSP-W experiments."""

# GA parameters
POPULATION_SIZE = 30
GENERATIONS = 50
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
TOURNAMENT_SIZE = 3
ELITE_SIZE = 2

# Uncertainty parameters
N_SCENARIOS = 10          # per fitness evaluation during evolution
FINAL_EVAL_SCENARIOS = 50 # per competition rules for final evaluation

# Balance weight (tuned from test_balance.py)
BALANCE_WEIGHT = 0.1

# Number of independent runs per instance
N_RUNS = 30
