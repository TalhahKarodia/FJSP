"""
Experiment Runner for FJSSP-W

This script runs experiments on FJSSP-W instances using the GA with FPC.
Includes statistical analysis (Mann-Whitney U test, Friedman test) as per proposal.
"""

import sys
import os
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
    """
    Run a single experiment on an instance.
    
    Returns:
        dict: Results including makespan and runtime
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running experiment on: {instance_name}")
        print(f"{'='*60}")
    
    # Load instance
    start_time = time.time()
    instance_data = parse_competition_instance(instance_path)
    
    # Create GA
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
    
    # Run GA (silent mode)
    original_print = print
    if not verbose:
        import builtins
        builtins.print = lambda *args, **kwargs: None
    
    result = ga.evolve()
    
    # Restore print
    if not verbose:
        import builtins
        builtins.print = original_print
    
    # Get schedule
    schedule = ga.get_best_schedule()
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    return {
        'instance': instance_name,
        'best_makespan': float(result['best_makespan']),
        'runtime': runtime,
        'generations': generations,
        'population_size': population_size,
        'schedule': schedule,
        'history': result['history']
    }


def perform_statistical_tests(results: Dict) -> Dict:
    """
    Perform statistical tests on experiment results.
    
    Args:
        results: Dictionary with instance names as keys and lists of makespans as values
        
    Returns:
        dict: Statistical test results
    """
    if not SCIPY_AVAILABLE:
        return {'error': 'scipy not available'}
    
    stats_results = {}
    
    # Get all makespan lists
    instance_names = list(results.keys())
    makespan_lists = [results[name] for name in instance_names]
    
    # 1. Friedman test (non-parametric test for multiple groups)
    try:
        friedman_stat, friedman_p = friedmanchisquare(*makespan_lists)
        stats_results['friedman'] = {
            'statistic': friedman_stat,
            'p_value': friedman_p,
            'significant': friedman_p < 0.05
        }
    except Exception as e:
        stats_results['friedman'] = {'error': str(e)}
    
    # 2. Pairwise Mann-Whitney U tests (with Bonferroni correction)
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
    verbose: bool = False
) -> Tuple[List[Dict], Dict]:
    """
    Run experiments on all instances in a directory.
    
    Args:
        instance_dir: Directory containing .fjs files
        output_file: Path to output CSV file
        n_runs_per_instance: Number of runs per instance
        population_size: GA population size
        generations: GA generations
        test_all: If True, test all instances. If False, test first 5.
        verbose: If True, print detailed output
    
    Returns:
        Tuple: (all_results, statistical_results)
    """
    # Find all .fjs files
    instance_files = sorted(glob.glob(os.path.join(instance_dir, "*.fjs")))
    
    if not instance_files:
        print(f"No .fjs files found in {instance_dir}")
        return [], {}
    
    print(f"\nFound {len(instance_files)} instances")
    
    # Filter instances
    if not test_all:
        instance_files = instance_files[:5]
        print(f"Running on {len(instance_files)} instances (first 5 for testing)")
    else:
        print(f"Running on ALL {len(instance_files)} instances")
    
    # Prepare results
    all_results = []
    all_makespans = {}  # For statistical tests
    
    # Run each instance
    for i, instance_path in enumerate(instance_files):
        instance_name = os.path.basename(instance_path)
        
        print(f"\n{'#'*60}")
        print(f"Instance {i+1}/{len(instance_files)}: {instance_name}")
        print(f"{'#'*60}")
        
        # Run multiple times
        instance_results = []
        instance_makespans = []
        
        for run in range(n_runs_per_instance):
            if verbose:
                print(f"\nRun {run+1}/{n_runs_per_instance}")
            
            random_seed = 42 + run
            
            result = run_experiment(
                instance_path,
                instance_name,
                population_size=population_size,
                generations=generations,
                random_seed=random_seed,
                verbose=verbose
            )
            result['run'] = run + 1
            instance_results.append(result)
            instance_makespans.append(result['best_makespan'])
        
        # Store for statistical tests
        all_makespans[instance_name] = instance_makespans
        
        # Calculate statistics
        avg_makespan = sum(instance_makespans) / len(instance_makespans)
        best_makespan = min(instance_makespans)
        worst_makespan = max(instance_makespans)
        std_makespan = statistics.stdev(instance_makespans) if len(instance_makespans) > 1 else 0
        
        print(f"\n{instance_name} Results:")
        print(f"  Best:    {best_makespan:.1f}")
        print(f"  Average: {avg_makespan:.1f}")
        print(f"  Worst:   {worst_makespan:.1f}")
        print(f"  Std Dev: {std_makespan:.1f}")
        print(f"  Runs:    {n_runs_per_instance}")
        
        # Store summary
        all_results.append({
            'instance': instance_name,
            'best': best_makespan,
            'average': avg_makespan,
            'worst': worst_makespan,
            'std': std_makespan,
            'runs': n_runs_per_instance,
            'population_size': population_size,
            'generations': generations,
            'detailed_results': instance_results
        })
    
    # Perform statistical tests
    print("\n" + "="*60)
    print("PERFORMING STATISTICAL TESTS")
    print("="*60)
    
    statistical_results = {}
    if SCIPY_AVAILABLE and len(all_makespans) > 1:
        statistical_results = perform_statistical_tests(all_makespans)
        
        # Print Friedman test results
        if 'friedman' in statistical_results:
            fr = statistical_results['friedman']
            if 'error' not in fr:
                print(f"\nFriedman Test (overall significance):")
                print(f"  Statistic: {fr['statistic']:.4f}")
                print(f"  P-value:   {fr['p_value']:.4f}")
                print(f"  Significant: {fr['significant']} (p < 0.05)")
        
        # Print pairwise results summary
        if 'pairwise' in statistical_results:
            significant_pairs = [p for p in statistical_results['pairwise'] 
                               if p.get('significant_bonferroni', False)]
            print(f"\nSignificant pairwise differences (Bonferroni corrected):")
            if significant_pairs:
                for p in significant_pairs:
                    print(f"  {p['instance1']} vs {p['instance2']}: p={p['p_value']:.4f}")
            else:
                print("  No significant pairwise differences found")
    else:
        print("\nStatistical tests skipped (scipy not available or insufficient data)")
    
    # Save to CSV
    if output_file:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'instance', 'best_makespan', 'avg_makespan', 'worst_makespan', 
                'std_makespan', 'runs', 'population_size', 'generations'
            ])
            for r in all_results:
                writer.writerow([
                    r['instance'],
                    r['best'],
                    r['average'],
                    r['worst'],
                    r['std'],
                    r['runs'],
                    r['population_size'],
                    r['generations']
                ])
        print(f"\nResults saved to: {output_file}")
        
        # Also save statistical results if available
        if SCIPY_AVAILABLE and statistical_results:
            stats_file = output_file.replace('.csv', '_statistics.csv')
            with open(stats_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['instance', 'mean', 'median', 'std', 'min', 'max', 'n'])
                for name, desc in statistical_results.get('descriptive', {}).items():
                    writer.writerow([
                        name,
                        desc['mean'],
                        desc['median'],
                        desc['std'],
                        desc['min'],
                        desc['max'],
                        desc['n']
                    ])
            print(f"Statistical results saved to: {stats_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Total instances: {len(all_results)}")
    print(f"Runs per instance: {n_runs_per_instance}")
    print(f"Population size: {population_size}")
    print(f"Generations: {generations}")
    print("\nResults:")
    print("-"*80)
    print(f"{'Instance':35} {'Best':>10} {'Avg':>10} {'Worst':>10} {'Std':>10}")
    print("-"*80)
    for r in all_results:
        print(f"{r['instance'][:35]:35} {r['best']:>10.1f} {r['average']:>10.1f} {r['worst']:>10.1f} {r['std']:>10.1f}")
    
    return all_results, statistical_results


def main():
    """Main entry point for experiments."""
    # Define paths
    competition_root = "../FJSSP-W-Competition"
    instance_dir = os.path.join(competition_root, "instances/fjssp-w")
    
    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/wfu/experiment_results_{timestamp}.csv"
    
    # Create results directory if it doesn't exist
    os.makedirs("results/wfu", exist_ok=True)
    
    # Run experiments
    print("="*60)
    print("FJSSP-W EXPERIMENT RUNNER")
    print("="*60)
    print(f"Instance directory: {instance_dir}")
    print(f"Output file: {output_file}")
    print()
    
    # Check if instance directory exists
    if not os.path.exists(instance_dir):
        print(f"Error: Instance directory not found: {instance_dir}")
        print(f"Current directory: {os.getcwd()}")
        return
    
    # Run all instances with full settings
    results, stats = run_all_instances(
        instance_dir=instance_dir,
        output_file=output_file,
        n_runs_per_instance=30,    # As per proposal
        population_size=100,        # As per proposal
        generations=200,           # As per proposal
        test_all=True,             # Test ALL instances
        verbose=False              # Quiet mode for speed
    )
    
    print("\nExperiments complete!")


if __name__ == "__main__":
    main()