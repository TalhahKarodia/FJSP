# FJSSP-W Genetic Algorithm Framework

An optimization framework for solving the Flexible Job Shop Scheduling Problem with Workers (FJSSP-W). This solver handles operation precedence, flexible machine routing, and worker assignment constraints under both deterministic operational conditions and stochastic uncertainty scenarios.

---

## Table of Contents

- [Overview](#overview)
- [Repository Layout](#repository-layout)
- [Requirements & Installation](#requirements--installation)
- [Usage & Execution](#usage--execution)
  - [Master Orchestrator](#1-master-orchestrator-run_allpy)
  - [Post-Processing & Analysis](#2-submission-generation--analysis)
- [Algorithm & Pseudocode](#algorithm--pseudocode)
  - [Crossover Mechanics](#crossover-mechanics)
  - [Pseudocode](#pseudocode)

---

## Overview

* **Job-Aware Feasibility-Preserving Crossover:** Combines Order Crossover (OX) with job-level grouping and uniform assignment blending to ensure precedence validity and effective solution space exploration.
* **Dual-Scenario Evaluation:** Supports both deterministic scheduling (Scenario 1) and stochastic disruption/uncertainty scenarios (Scenario 2).
* **Automated Repair Mechanism:** Includes chromosome repair operators (`repair_chromosome_optimised`) to enforce constraint satisfaction post-crossover.
* **Experiment Pipeline:** Includes automated scripts for running baselines, scenario benchmarks, competition submission packaging, and report table generation.

---

## Repository Layout

```text
.
├── run_all_parallel.py                   # Master orchestrator script
├── run_scenario1_parallel_v2.py          # Scenario 1 runner (Deterministic parallel execution)
├── run_scenario2_parallel_v2.py          # Scenario 2 runner (Uncertainty parallel execution)
├── run_baselines_parallel.py             # Baseline algorithms parallel runner
├── build_competition_package.py          # Competition package generator
├── full_analysis.py                      # Full statistical and analysis execution
├── report_tables.py                      # Report summary table generator
├── requirements.txt                      # Project dependencies
├── results/                              # Output directory for benchmarks and submissions
│   ├── submission/                       # Formatted competition submission outputs
│   └── summary/                          # Summarized experiment logs and metrics
└── src/
    ├── analysis/                         # Statistical evaluation tools
    │   └── statistical_tests.py          # Hypothesis testing and statistical metrics
    ├── baselines/                        # Baseline benchmark algorithm implementations
    │   ├── hybrid_ga.py                  # Hybrid Genetic Algorithm
    │   ├── pso.py                        # Particle Swarm Optimization baseline
    │   └── standard_ga.py                # Standard Genetic Algorithm baseline
    ├── ga/                               # Genetic Algorithm core modules
    │   ├── fjssp_w/                      # Worker-extended FJSSP components
    │   │   ├── decoder_w.py              # FJSSP-W schedule decoder
    │   │   ├── genetic_algorithm_w_v2.py # Core GA solver (contains feasibility crossover)
    │   │   └── uncertainty_sim.py        # Uncertainty and disruption simulator
    │   ├── genetic_algorithm.py          # Standard FJSP GA implementation
    │   └── sanity_check.py               # Chromosome & schedule verification scripts
    └── parser/                           # Benchmark dataset parsers
        ├── fjsp_parser.py                # Standard FJSP data parser
        └── fjssp_w_parser.py             # FJSSP-W instance parser with worker data
```
# Requirements & Installation
Environment
Python 3.9+

# Dependencies
Install the required dependencies via pip:
```bash
pip install -r requirements.txt
```

# Usage & Execution
## 1. Master Orchestrator (run_all_parallel.py)
Run experiments, baselines, and submission workflows through the central CLI interface:

```bash
# Run complete pipeline (Scenario 1, Scenario 2, Baselines, Submission)
python run_all_parallel.py all

# Run Scenario 1 only (Deterministic)
python run_all_parallel.py s1

# Run Scenario 2 only (Uncertainty)
python run_all_parallel.py s2

# Run baseline comparisons only
python run_all_parallel.py baselines

# Generate competition submission files
python run_all_parallel.py submission
```
## 2. Submission Generation & Analysis
To run stand-alone submission generation, package files, and output analytical tables:
```bash
# Generate submission files for all scenarios
python src/experiments/generate_submissions_v2.py all

# Build competition package archive
python build_competition_package.py

# Execute complete experimental analysis
python full_analysis.py

# Generate report summary tables
python report_tables.py
```

# Algorithm & Pseudocode
## Crossover Mechanics
**Phase 1: Localized Temporal Slicing**

A contiguous spatial window [pt1, pt2] is randomly initialized on Parent 1's operation sequence. The subset of unique jobs executing within this chronological window is extracted into an active_jobs set.

**Phase 2: Spatial & Synergistic Anchoring**

Operations corresponding to the active_jobs set are inherited by the offspring at their exact sequence indices. This perfectly preserves both the chronological timing and the primary Machine-Worker assignment synergies of the localized bottleneck.

**Phase 3: Relative Topological Fill**

The remaining unallocated sequence indices in the offspring are populated sequentially using the operations from Parent 2, strictly preserving their relative order of appearance to maintain global job precedence.

**Phase 4: Uniform Synergistic Blending**

To inject localized genetic diversity, each gene position undergoes a uniform 10% probability check. If triggered, the gene's Machine-Worker allocation pair is swapped with the alternate parent's assignment via an O(1) lookup table, facilitating rapid resource exploration.

**Phase 5: Feasibility Restoration**

The resultant offspring is passed through an optimized deterministic repair heuristic (repair_chromosome_optimised) to resolve any residual precedence violations or dual-resource allocation conflicts, guaranteeing strict constraint adherence.



## Pseudocode
![pseudocode](images/pseudocode_1.png)
![images](images/pseudocode_2.png)
