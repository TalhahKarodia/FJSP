"""
Master orchestrator: runs all experiments + generates submission files.

Usage:
    python run_all.py s1        # Scenario 1 only
    python run_all.py s2        # Scenario 2 only
    python run_all.py baselines # Baselines only
    python run_all.py all       # Everything
"""

import sys, subprocess, time


def run(label, cmd):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - t0
    print(f"[{label}] finished in {elapsed/60:.1f} min (exit code {result.returncode})")
    return result.returncode


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'

    tasks = []
    if which in ('s1', 'all'):
        tasks.append(("Scenario 1 (deterministic)", "python run_scenario1.py"))
    if which in ('s2', 'all'):
        tasks.append(("Scenario 2 (uncertainty)", "python run_scenario2.py"))
    if which in ('baselines', 'all'):
        tasks.append(("Baselines comparison", "python run_baselines.py"))
    if which in ('submission', 'all'):
        tasks.append(("Competition submission", "python src/experiments/generate_submission.py both"))

    for label, cmd in tasks:
        rc = run(label, cmd)
        if rc != 0:
            print(f"ERROR in {label}. Stopping.")
            sys.exit(rc)

    print(f"\n{'='*70}\nALL DONE\n{'='*70}")


if __name__ == "__main__":
    main()