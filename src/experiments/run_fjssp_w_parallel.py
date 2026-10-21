"""
Experiment Runner for FJSSP-W

Includes statistical analysis (Mann-Whitney U test, Friedman test)
with continuous 100% worker saturation, live progress, and incremental CSV checkpointing.
"""

import sys
import os
import concurrent.futures
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import csv
import time
import random
from datetime import datetime
from typing import List, Dict, Tuple
import glob
import statistics

# For statistical tests
try:
    from scipy import stats
    from scipy.stats import mannwhitneyu, friedmanchisquare
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not installed. Statistical tests will be skipped.")
    print("Install with: pip install scipy")

from parser.fjssp_w_parser import parse_competition_instance
from ga.fjssp_w.genetic_algorithm_w import FJSSPW_GA


def run_experiment(
    instance_path: str,
    instance_name: str,
    population_size: int = 100,
    generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.1,
    tournament_size: int = 3,
    elite_size: int = 2,
    random_seed: int = 42,
    verbose: bool = False
) -> Dict:
    """Run a single experiment on an instance."""
    start_time = time.time()
    instance_data = parse_competition_instance(instance_path)
    
    ga = FJSSPW_GA(
        instance_data,
        population_size=population_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        elite_size=elite_size,
        random_seed=random_seed
    )
    
    # Run GA silently
    original_print = print
    if not verbose:
        import builtins
        builtins.print = lambda *args, **kwargs: None
    
    result = ga.evolve()
    
    if not verbose:
        import builtins
        builtins.print = original_print
    
    runtime = time.time() - start_time
    
    return {
        'instance': instance_name,
        'best_makespan': float(result['best_makespan']),
        'runtime': runtime,
        'generations': generations,
        'population_size': population_size
    }


def run_worker(args):
    """Worker task wrapping run_experiment for multiprocessing."""
    run_idx, instance_path, instance_name, pop_size, gens, seed, verbose = args

    result = run_experiment(
        instance_path, 
        instance_name,
        population_size=pop_size,
        generations=gens,
        random_seed=seed,
        verbose=verbose
    )
    result['run'] = run_idx + 1
    return result


def perform_statistical_tests(results: Dict) -> Dict:
    """Perform statistical tests on experiment results."""
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
            'significant': friedman_p < 0.05
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
                    alternative='two-sided'
                )
                pairwise_results.append({
                    'instance1': instance_names[i],
                    'instance2': instance_names[j],
                    'statistic': stat,
                    'p_value': p,
                    'significant_raw': p < 0.05,
                    'significant_bonferroni': p < (0.05 / (n_instances * (n_instances - 1) / 2))
                })
            except Exception as e:
                pairwise_results.append({
                    'instance1': instance_names[i],
                    'instance2': instance_names[j],
                    'error': str(e)
                })
    
    stats_results['pairwise'] = pairwise_results
    
    # 3. Descriptive statistics
    descriptive = {}
    for name in instance_names:
        data = results[name]
        descriptive[name] = {
            'mean': statistics.mean(data),
            'median': statistics.median(data),
            'std': statistics.stdev(data) if len(data) > 1 else 0,
            'min': min(data),
            'max': max(data),
            'range': max(data) - min(data),
            'n': len(data)
        }
    stats_results['descriptive'] = descriptive
    
    return stats_results


def run_all_instances(
    instance_dir: str,
    output_file: str = None,
    n_runs_per_instance: int = 30,
    population_size: int = 100,
    generations: int = 200,
    test_all: bool = True,
    max_workers: int = 12,
    verbose: bool = False
) -> Tuple[List[Dict], Dict]:
    """Run experiments on instances with global process pooling and live tracking."""
    instance_files = sorted(glob.glob(os.path.join(instance_dir, "*.fjs")))
    
    if not instance_files:
        print(f"No .fjs files found in {instance_dir}")
        return [], {}
    
    if not test_all:
        instance_files = instance_files[:5]
    
    print(f"\nFound {len(instance_files)} instances.")
    print(f"Total runs to execute: {len(instance_files) * n_runs_per_instance} across {max_workers} worker cores.")

    # Flatten all tasks across all instances
    tasks = []
    for instance_path in instance_files:
        instance_name = os.path.basename(instance_path)
        for run in range(n_runs_per_instance):
            tasks.append((run, instance_path, instance_name, population_size, generations, 42 + run, verbose))
    
    total_tasks = len(tasks)
    completed_counter = 0
    start_wall_time = time.time()
    
    # Grouped storage
    instance_data_store = {os.path.basename(p): {'makespans': [], 'detailed': []} for p in instance_files}

    # Open CSV early for live incremental writes
    csv_file = None
    csv_writer = None
    if output_file:
        raw_runs_file = output_file.replace('.csv', '_raw_runs.csv')
        csv_file = open(raw_runs_file, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['instance', 'run', 'best_makespan', 'runtime', 'population_size', 'generations'])

    print("\nStarting execution pool...")
    print("-" * 80)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(run_worker, task): task for task in tasks}
        
        for future in concurrent.futures.as_completed(future_to_task):
            res = future.result()
            completed_counter += 1
            inst_name = res['instance']
            
            # Store result
            instance_data_store[inst_name]['makespans'].append(res['best_makespan'])
            instance_data_store[inst_name]['detailed'].append(res)
            
            # Write to raw CSV immediately
            if csv_writer:
                csv_writer.writerow([
                    res['instance'], res['run'], res['best_makespan'], 
                    f"{res['runtime']:.2f}", res['population_size'], res['generations']
                ])
                csv_file.flush()

            # Calculate progress metrics
            elapsed = time.time() - start_wall_time
            avg_per_task = elapsed / completed_counter
            remaining_tasks = total_tasks - completed_counter
            eta_seconds = remaining_tasks * (avg_per_task / max_workers)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            
            print(
                f"[{completed_counter:03d}/{total_tasks:03d}] "
                f"Instance: {inst_name[:25]:<25} | Run {res['run']:02d} Makespan: {res['best_makespan']:<6.1f} | "
                f"Time/Run: {res['runtime']:5.1f}s | ETA: {eta_str}", 
                flush=True
            )

    if csv_file:
        csv_file.close()

    # Aggregate final summaries
    all_results = []
    all_makespans = {}
    
    for instance_name, data in instance_data_store.items():
        makespans = data['makespans']
        if not makespans:
            continue
        
        all_makespans[instance_name] = makespans
        avg_makespan = sum(makespans) / len(makespans)
        best_makespan = min(makespans)
        worst_makespan = max(makespans)
        std_makespan = statistics.stdev(makespans) if len(makespans) > 1 else 0
        
        all_results.append({
            'instance': instance_name,
            'best': best_makespan,
            'average': avg_makespan,
            'worst': worst_makespan,
            'std': std_makespan,
            'runs': len(makespans),
            'population_size': population_size,
            'generations': generations,
            'detailed_results': data['detailed']
        })

    # Perform statistical tests
    print("\n" + "="*60)
    print("PERFORMING STATISTICAL TESTS")
    print("="*60)
    
    statistical_results = {}
    if SCIPY_AVAILABLE and len(all_makespans) > 1:
        statistical_results = perform_statistical_tests(all_makespans)

    # Save summary CSV
    if output_file:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'instance', 'best_makespan', 'avg_makespan', 'worst_makespan', 
                'std_makespan', 'runs', 'population_size', 'generations'
            ])
            for r in all_results:
                writer.writerow([
                    r['instance'], r['best'], r['average'], r['worst'],
                    r['std'], r['runs'], r['population_size'], r['generations']
                ])
        print(f"\nSummary results saved to: {output_file}")

    return all_results, statistical_results


def main():
    """Main entry point."""
    competition_root = "../FJSSP-W-Competition"
    instance_dir = os.path.join(competition_root, "instances/fjssp-w")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/wfu/experiment_results_{timestamp}.csv"
    os.makedirs("results/wfu", exist_ok=True)
    
    if not os.path.exists(instance_dir):
        print(f"Error: Instance directory not found: {instance_dir}")
        return
    
    run_all_instances(
        instance_dir=instance_dir,
        output_file=output_file,
        n_runs_per_instance=30,
        population_size=100,
        generations=200,
        test_all=True,
        max_workers=12,
        verbose=False
    )
    print("\nExperiments complete!")


if __name__ == "__main__":
    main()