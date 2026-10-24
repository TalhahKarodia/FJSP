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
**The feasibility_preserving_crossover operator operates via three primary phases:**

1. **Job Slicing:** A random window $[pt1, pt2]$ is selected from Parent 1's operation sequence. The set of unique Job IDs present in this window is stored in active_jobs.

2. **Primary Sequence & Assignment Transfer:** Operations matching active_jobs are placed in the exact same sequence positions in the child chromosome, preserving their primary parent machine and worker assignments.
3. **Secondary Fill:** Empty slots are filled sequentially using operations from the secondary parent in their relative order of appearance.
4. **Uniform Assignment Blending:** For each gene position, a 10% probability step permits swapping Machine-Worker assignment pairs with the alternate parent using $O(1)$ lookup tables.
5. **Repair Step:** Offspring are processed through repair_chromosome_optimised to fix precedence or allocation conflicts.


## Pseudocode
![images](pseudocode_1.png)
![images](pseudocode_2.png)
