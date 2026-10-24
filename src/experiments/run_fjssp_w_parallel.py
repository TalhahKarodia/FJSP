"""
Experiment Runner for FJSSP-W

Key Improvements:
- Pre-parses instance files ONCE per instance (eliminates 30 redundant file reads per instance)
- Reuses a persistent ProcessPoolExecutor across all instances (eliminates process pool creation/destruction overhead)
- Eliminates builtins.print monkey-patching
- Lightweight inter-process communication (IPC) payload to minimize pickle/unpickle serialization overhead
"""

import sys
import os
import csv
import time
import glob
import statistics
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# For statistical tests
try:
    from scipy.stats import mannwhitneyu, friedmanchisquare
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not installed. Statistical tests will be skipped.")
    print("Install with: pip install scipy")

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA


def run_single_ga_run(args: Tuple) -> Dict:
    """Worker task for running a single GA execution on pre-parsed instance data."""
    (
        run_idx,
        instance_data,
        instance_name,
        pop_size,
        gens,
        seed,
        crossover_rate,
        mutation_rate,
        tournament_size,
        elite_size,
        keep_history,
    ) = args

    start_time = time.time()

    # Instantiate GA with pre-parsed data
    ga = FJSSPW_GA(
        instance_data,
        population_size=pop_size,
        generations=gens,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        elite_size=elite_size,
        random_seed=seed,
    )

    result = ga.evolve()
    runtime = time.time() - start_time

    output = {
        'run': run_idx + 1,
        'instance': instance_name,
        'best_makespan': float(result['best_makespan']),
        'runtime': runtime,
    }

    if keep_history:
        output['schedule'] = ga.get_best_schedule()
        output['history'] = result['history']

    return output


def perform_statistical_tests(results: Dict) -> Dict:
    """Perform statistical tests (Friedman, Mann-Whitney U, Descriptive) on experiment results."""
    if not SCIPY_AVAILABLE:
        return {'error': 'scipy not available'}

    stats_results = {}
    instance_names = list(results.keys())
    makespan_lists = [results[name] for name in instance_names]

    # 1. Friedman test
    try:
        friedman_stat, friedman_p = friedmanchisquare(*makespan_lists)
        stats_results['friedman'] = {
            'statistic': friedman_stat,
            'p_value': friedman_p,
            'significant': friedman_p < 0.05,
        }
    except Exception as e:
        stats_results['friedman'] = {'error': str(e)}

    # 2. Pairwise Mann-Whitney U tests
    pairwise_results = []
    n_instances = len(instance_names)

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            try:
                stat, p = mannwhitneyu(
                    results[instance_names[i]],
                    results[instance_names[j]],
                    alternative='two-sided',
                )
                pairwise_results.append({
                    'instance1': instance_names[i],
                    'instance2': instance_names[j],
                    'statistic': stat,
                    'p_value': p,
                    'significant_raw': p < 0.05,
                    'significant_bonferroni': p < (0.05 / (n_instances * (n_instances - 1) / 2)),
                })
            except Exception as e:
                pairwise_results.append({
                    'instance1': instance_names[i],
                    'instance2': instance_names[j],
                    'error': str(e),
                })

    stats_results['pairwise'] = pairwise_results

    # 3. Descriptive statistics
    descriptive = {}
    for name in instance_names:
        data = results[name]
        descriptive[name] = {
            'mean': statistics.mean(data),
            'median': statistics.median(data),
            'std': statistics.stdev(data) if len(data) > 1 else 0.0,
            'min': min(data),
            'max': max(data),
            'range': max(data) - min(data),
            'n': len(data),
        }
    stats_results['descriptive'] = descriptive

    return stats_results


def run_all_instances(
    instance_dir: str,
    output_file: str = None,
    n_runs_per_instance: int = 30,
    population_size: int = 100,
    generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    tournament_size: int = 3,
    elite_size: int = 2,
    test_all: bool = True,
    keep_detailed_history: bool = False,
) -> Tuple[List[Dict], Dict]:
    """Run experiments across all instances using a persistent process pool."""
    instance_files = sorted(glob.glob(os.path.join(instance_dir, "*.fjs")))

    if not instance_files:
        print(f"No .fjs files found in {instance_dir}")
        return [], {}

    print(f"\nFound {len(instance_files)} instances")

    if not test_all:
        instance_files = instance_files[:5]
        print(f"Running on {len(instance_files)} instances (first 5 for testing)")
    else:
        print(f"Running on ALL {len(instance_files)} instances")

    all_results = []
    all_makespans = {}

    # Use a single persistent ProcessPoolExecutor across all instances
    max_workers = os.cpu_count()
    print(f"Initializing process pool with {max_workers} worker processes...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, instance_path in enumerate(instance_files):
            instance_name = os.path.basename(instance_path)

            print(f"\n{'#'*60}")
            print(f"Instance {i+1}/{len(instance_files)}: {instance_name}")
            print(f"{'#'*60}")

            # Parse instance file ONCE per instance
            instance_data = parse_competition_instance(instance_path)

            print(f"Starting {n_runs_per_instance} independent parallel runs...")

            # Prepare execution tasks for this instance
            run_args = [
                (
                    run_idx,
                    instance_data,
                    instance_name,
                    population_size,
                    generations,
                    42 + run_idx,
                    crossover_rate,
                    mutation_rate,
                    tournament_size,
                    elite_size,
                    keep_detailed_history,
                )
                for run_idx in range(n_runs_per_instance)
            ]

            # Submit and gather results with process pool
            chunksize = max(1, n_runs_per_instance // (max_workers * 2))
            completed_runs = list(executor.map(run_single_ga_run, run_args, chunksize=chunksize))

            instance_results = []
            instance_makespans = []

            for result in completed_runs:
                instance_results.append(result)
                instance_makespans.append(result['best_makespan'])
                print(f"Run {result['run']:2d} | Best Makespan: {result['best_makespan']:.1f} | Time: {result['runtime']:.2f}s")

            all_makespans[instance_name] = instance_makespans

            avg_makespan = statistics.mean(instance_makespans)
            best_makespan = min(instance_makespans)
            worst_makespan = max(instance_makespans)
            std_makespan = statistics.stdev(instance_makespans) if len(instance_makespans) > 1 else 0.0
            avg_runtime = statistics.mean([r['runtime'] for r in completed_runs])

            print(f"\n{instance_name} Results:")
            print(f"  Best:         {best_makespan:.1f}")
            print(f"  Average:      {avg_makespan:.1f}")
            print(f"  Worst:        {worst_makespan:.1f}")
            print(f"  Std Dev:      {std_makespan:.1f}")
            print(f"  Avg Run Time: {avg_runtime:.2f}s")

            all_results.append({
                'instance': instance_name,
                'best': best_makespan,
                'average': avg_makespan,
                'worst': worst_makespan,
                'std': std_makespan,
                'avg_runtime': avg_runtime,
                'runs': n_runs_per_instance,
                'population_size': population_size,
                'generations': generations,
                'detailed_results': instance_results,
            })

    # Statistical Analysis
    print("\n" + "="*60)
    print("PERFORMING STATISTICAL TESTS")
    print("="*60)

    statistical_results = {}
    if SCIPY_AVAILABLE and len(all_makespans) > 1:
        statistical_results = perform_statistical_tests(all_makespans)

        if 'friedman' in statistical_results:
            fr = statistical_results['friedman']
            if 'error' not in fr:
                print("\nFriedman Test (overall significance):")
                print(f"  Statistic:   {fr['statistic']:.4f}")
                print(f"  P-value:     {fr['p_value']:.4f}")
                print(f"  Significant: {fr['significant']} (p < 0.05)")

        if 'pairwise' in statistical_results:
            significant_pairs = [
                p for p in statistical_results['pairwise']
                if p.get('significant_bonferroni', False)
            ]
            print("\nSignificant pairwise differences (Bonferroni corrected):")
            if significant_pairs:
                for p in significant_pairs:
                    print(f"  {p['instance1']} vs {p['instance2']}: p={p['p_value']:.4f}")
            else:
                print("  No significant pairwise differences found")

    # CSV Exports
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'instance', 'best_makespan', 'avg_makespan', 'worst_makespan',
                'std_makespan', 'avg_runtime_sec', 'runs', 'population_size', 'generations'
            ])
            for r in all_results:
                writer.writerow([
                    r['instance'],
                    r['best'],
                    r['average'],
                    r['worst'],
                    r['std'],
                    f"{r['avg_runtime']:.3f}",
                    r['runs'],
                    r['population_size'],
                    r['generations'],
                ])
        print(f"\nResults saved to: {output_file}")

        if SCIPY_AVAILABLE and statistical_results:
            stats_file = output_file.replace('.csv', '_statistics.csv')
            with open(stats_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['instance', 'mean', 'median', 'std', 'min', 'max', 'n'])
                for name, desc in statistical_results.get('descriptive', {}).items():
                    writer.writerow([
                        name, desc['mean'], desc['median'], desc['std'],
                        desc['min'], desc['max'], desc['n']
                    ])
            print(f"Statistical results saved to: {stats_file}")

    # Final Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Total instances: {len(all_results)}")
    print(f"Runs per instance: {n_runs_per_instance}")
    print(f"Population size: {population_size}")
    print(f"Generations: {generations}")
    print("\nResults:")
    print("-" * 90)
    print(f"{'Instance':35} {'Best':>10} {'Avg':>10} {'Worst':>10} {'Std':>10} {'Avg Time':>10}")
    print("-" * 90)
    for r in all_results:
        print(
            f"{r['instance'][:35]:35} {r['best']:>10.1f} {r['average']:>10.1f} "
            f"{r['worst']:>10.1f} {r['std']:>10.1f} {r['avg_runtime']:>9.2f}s"
        )

    return all_results, statistical_results


def main():
    """Main entry point for experiments."""
    competition_root = "../FJSSP-W-Competition"
    instance_dir = os.path.join(competition_root, "instances/fjssp-w")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/wfu/experiment_results_{timestamp}.csv"

    print("="*60)
    print("FJSSP-W EXPERIMENT RUNNER (OPTIMIZED)")
    print("="*60)
    print(f"Instance directory: {instance_dir}")
    print(f"Output file: {output_file}\n")

    if not os.path.exists(instance_dir):
        print(f"Error: Instance directory not found: {instance_dir}")
        print(f"Current directory: {os.getcwd()}")
        return

    run_all_instances(
        instance_dir=instance_dir,
        output_file=output_file,
        n_runs_per_instance=30,
        population_size=100,
        generations=200,
        test_all=True,
        keep_detailed_history=False,
    )

    print("\nExperiments complete!")


if __name__ == "__main__":
    main()