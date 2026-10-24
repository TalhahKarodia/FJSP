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
$$
\begin{array}{ll}
\mathbf{Algorithm:} & \mathbf{FeasibilityPreservingCrossover}(P_1, P_2) \\
\hline
\mathbf{Input:} & \text{Parent chromosomes } P_1, P_2 \\
\mathbf{Output:} & \text{Feasible offspring chromosomes } C_1, C_2 \\
\hline
1: & n_{\text{ops}} \leftarrow \text{decoder.n\_operations} \\
2: & pt_1 \leftarrow \text{RandomInt}(0, n_{\text{ops}} - 2) \\
3: & pt_2 \leftarrow \text{RandomInt}(pt_1 + 1, n_{\text{ops}} - 1) \\
4: & S_{\text{ops}} \leftarrow P_1.\text{sequence}[pt_1 : pt_2 + 1] \\
5: & J_{\text{active}} \leftarrow \{ \text{GetJobId}(op) \mid op \in S_{\text{ops}} \land op \neq \text{null} \} \\
6: & C_1^{\text{raw}} \leftarrow \mathbf{CreateHybridChild}(P_1, P_2, J_{\text{active}}) \\
7: & C_2^{\text{raw}} \leftarrow \mathbf{CreateHybridChild}(P_2, P_1, J_{\text{active}}) \\
8: & C_1 \leftarrow \mathbf{RepairChromosomeOptimised}(C_1^{\text{raw}}) \\
9: & C_2 \leftarrow \mathbf{RepairChromosomeOptimised}(C_2^{\text{raw}}) \\
10: & \mathbf{return}\; C_1, C_2 \\
\hline
\\
\mathbf{Function:} & \mathbf{CreateHybridChild}(P_{\text{pri}}, P_{\text{sec}}, J_{\text{active}}) \\
\hline
1: & \text{Initialize arrays } c_{\text{seq}}, c_m, c_w \text{ of length } n_{\text{ops}} \text{ filled with } \text{null} \\
2: & \mathcal{O}_{\text{copied}} \leftarrow \emptyset \\
3: & \mathbf{for}\; i \leftarrow 0 \;\mathbf{to}\; n_{\text{ops}} - 1 \;\mathbf{do} \\
4: & \quad op \leftarrow P_{\text{pri}}.\text{sequence}[i] \\
5: & \quad \mathbf{if}\; \text{GetJobId}(op) \in J_{\text{active}} \;\mathbf{then} \\
6: & \qquad c_{\text{seq}}[i] \leftarrow op,\quad c_m[i] \leftarrow P_{\text{pri}}.m[i],\quad c_w[i] \leftarrow P_{\text{pri}}.w[i] \\
7: & \qquad \mathcal{O}_{\text{copied}} \leftarrow \mathcal{O}_{\text{copied}} \cup \{op\} \\
8: & \quad \mathbf{end\;if} \\
9: & \mathbf{end\;for} \\
10: & idx_{\text{sec}} \leftarrow 0 \\
11: & \mathbf{for}\; i \leftarrow 0 \;\mathbf{to}\; n_{\text{ops}} - 1 \;\mathbf{do} \\
12: & \quad \mathbf{if}\; c_{\text{seq}}[i] = \text{null} \;\mathbf{then} \\
13: & \qquad \mathbf{while}\; idx_{\text{sec}} < n_{\text{ops}} \;\mathbf{do} \\
14: & \qquad\quad op_{\text{sec}} \leftarrow P_{\text{sec}}.\text{sequence}[idx_{\text{sec}}], \quad pos_{\text{sec}} \leftarrow idx_{\text{sec}} \\
15: & \qquad\quad idx_{\text{sec}} \leftarrow idx_{\text{sec}} + 1 \\
16: & \qquad\quad \mathbf{if}\; op_{\text{sec}} \notin \mathcal{O}_{\text{copied}} \;\mathbf{then} \\
17: & \qquad\qquad c_{\text{seq}}[i] \leftarrow op_{\text{sec}},\quad c_m[i] \leftarrow P_{\text{sec}}.m[pos_{\text{sec}}],\quad c_w[i] \leftarrow P_{\text{sec}}.w[pos_{\text{sec}}] \\
18: & \qquad\qquad \mathcal{O}_{\text{copied}} \leftarrow \mathcal{O}_{\text{copied}} \cup \{op_{\text{sec}}\} \\
19: & \qquad\qquad \mathbf{break} \\
20: & \qquad\quad \mathbf{end\;if} \\
21: & \qquad \mathbf{end\;while} \\
22: & \quad \mathbf{end\;if} \\
23: & \mathbf{end\;for} \\
24: & \text{Construct maps } M_{\text{pri}}, M_{\text{sec}} : op \mapsto \text{index in } P_{\text{pri}}.\text{sequence}, P_{\text{sec}}.\text{sequence} \\
25: & \mathbf{for}\; i \leftarrow 0 \;\mathbf{to}\; n_{\text{ops}} - 1 \;\mathbf{do} \\
26: & \quad op \leftarrow c_{\text{seq}}[i] \\
27: & \quad \mathbf{if}\; \text{RandomFloat}(0, 1) < 0.10 \;\mathbf{then} \\
28: & \qquad \text{is\_primary} \leftarrow (\text{GetJobId}(op) \in J_{\text{active}}) \\
29: & \qquad (P_{\text{alt}}, M_{\text{alt}}) \leftarrow \mathbf{if}\; \text{is\_primary} \;\mathbf{then}\; (P_{\text{sec}}, M_{\text{sec}}) \;\mathbf{else}\; (P_{\text{pri}}, M_{\text{pri}}) \\
30: & \qquad idx_{\text{alt}} \leftarrow M_{\text{alt}}[op] \\
31: & \qquad m_{\text{alt}} \leftarrow P_{\text{alt}}.m[idx_{\text{alt}}], \quad w_{\text{alt}} \leftarrow P_{\text{alt}}.w[idx_{\text{alt}}] \\
32: & \qquad \mathbf{if}\; m_{\text{alt}} \neq \text{null} \land w_{\text{alt}} \neq \text{null} \;\mathbf{then} \\
33: & \qquad\qquad c_m[i] \leftarrow m_{\text{alt}},\quad c_w[i] \leftarrow w_{\text{alt}} \\
34: & \qquad \mathbf{end\;if} \\
35: & \quad \mathbf{end\;if} \\
36: & \mathbf{end\;for} \\
37: & \mathbf{return}\; \{ \text{machine\_assignment}: c_m, \text{worker\_assignment}: c_w, \text{operation\_sequence}: c_{\text{seq}} \} \\
\hline
\end{array}
$$
