"""
Visualization utilities for FJSSP-W results.

Creates convergence plots, box plots, and performance profiles.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import csv
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
import glob


def load_results(csv_file: str) -> Dict:
    """
    Load experiment results from CSV file.
    
    Returns:
        dict: {
            'instances': list of instance names,
            'best': list of best makespans,
            'avg': list of average makespans,
            'worst': list of worst makespans,
            'std': list of standard deviations
        }
    """
    instances = []
    best = []
    avg = []
    worst = []
    std = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instances.append(row['instance'])
            best.append(float(row['best_makespan']))
            avg.append(float(row['avg_makespan']))
            worst.append(float(row['worst_makespan']))
            # Handle missing std column
            if 'std_makespan' in row and row['std_makespan']:
                std.append(float(row['std_makespan']))
            else:
                std.append(0.0)
    
    return {
        'instances': instances,
        'best': best,
        'avg': avg,
        'worst': worst,
        'std': std
    }


def plot_convergence(history: Dict, instance_name: str, save_path: str = None):
    """
    Plot convergence curve (makespan vs generations).
    
    Args:
        history: dict with 'best_makespan', 'avg_makespan', 'worst_makespan'
        instance_name: Name of the instance
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    generations = range(1, len(history['best_makespan']) + 1)
    
    plt.plot(generations, history['best_makespan'], 'g-', linewidth=2, label='Best')
    plt.plot(generations, history['avg_makespan'], 'b-', linewidth=1.5, label='Average')
    plt.plot(generations, history['worst_makespan'], 'r-', linewidth=1.5, label='Worst')
    
    plt.xlabel('Generation')
    plt.ylabel('Makespan')
    plt.title(f'Convergence Plot - {instance_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to: {save_path}")
    else:
        plt.show()


def plot_boxplot_from_summary(data: Dict, save_path: str = None):
    """
    Create box plot from summary data (best, avg, worst).
    
    Args:
        data: dict from load_results()
        save_path: Path to save the figure
    """
    instances = [i.replace('.fjs', '') for i in data['instances']]
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(instances))
    width = 0.25
    
    plt.bar(x - width, data['best'], width, label='Best', color='green', alpha=0.8)
    plt.bar(x, data['avg'], width, label='Average', color='blue', alpha=0.8)
    plt.bar(x + width, data['worst'], width, label='Worst', color='red', alpha=0.8)
    
    plt.xlabel('Instance')
    plt.ylabel('Makespan')
    plt.title('Makespan Results Summary')
    plt.xticks(x, instances, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary bar chart saved to: {save_path}")
    else:
        plt.show()


def plot_boxplot(results: Dict, save_path: str = None):
    """
    Create box plot of makespan distribution across instances.
    
    Args:
        results: dict with 'instances' and 'detailed_results'
        save_path: Path to save the figure
    """
    # Check if we have detailed data
    has_detailed = False
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, dict) and 'detailed_results' in value:
                has_detailed = True
                break
    
    if not has_detailed:
        print("No detailed data found for box plot - using summary data")
        return
    
    # Extract data for each instance
    data = []
    labels = []
    
    for instance_name, instance_data in results.items():
        if isinstance(instance_data, dict) and 'detailed_results' in instance_data:
            makespans = [r['best_makespan'] for r in instance_data['detailed_results']]
            data.append(makespans)
            labels.append(instance_name.replace('.fjs', ''))
    
    if not data:
        print("No detailed data found for box plot")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Create box plot
    bp = plt.boxplot(data, labels=labels, patch_artist=True)
    
    # Customize colors
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    plt.xlabel('Instance')
    plt.ylabel('Makespan')
    plt.title('Makespan Distribution Across Instances')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Box plot saved to: {save_path}")
    else:
        plt.show()


def plot_performance_profile_from_summary(data: Dict, save_path: str = None):
    """
    Create performance profile from summary data.
    
    Args:
        data: dict from load_results()
        save_path: Path to save the figure
    """
    best_makespans = data['best']
    
    if not best_makespans:
        print("No data found for performance profile")
        return
    
    # Calculate relative gaps
    best_overall = min(best_makespans)
    gaps = [(m - best_overall) / best_overall * 100 for m in best_makespans]
    
    # Sort gaps
    gaps.sort()
    
    # Calculate proportion of instances solved within each gap
    proportions = []
    gap_values = np.linspace(0, max(gaps) + 1, 100)
    
    for gap in gap_values:
        proportion = sum(1 for g in gaps if g <= gap) / len(gaps)
        proportions.append(proportion)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(gap_values, proportions, 'b-', linewidth=2)
    plt.xlabel('Relative Gap to Best (%)')
    plt.ylabel('Proportion of Instances Solved')
    plt.title('Performance Profile')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, min(max(gaps) + 5, 50))
    plt.ylim(0, 1.05)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance profile saved to: {save_path}")
    else:
        plt.show()


def plot_performance_profile(results: Dict, save_path: str = None):
    """
    Create performance profile (MiniZinc-style).
    
    Args:
        results: dict with instance results
        save_path: Path to save the figure
    """
    # Check if we have detailed data
    has_detailed = False
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, dict) and 'detailed_results' in value:
                has_detailed = True
                break
    
    if not has_detailed:
        print("No detailed data found for performance profile - using summary data")
        return
    
    # Extract best makespans
    best_makespans = []
    instance_names = []
    
    for instance_name, instance_data in results.items():
        if isinstance(instance_data, dict) and 'detailed_results' in instance_data:
            best_makespans.append(min(instance_data['detailed_results'], 
                                       key=lambda x: x['best_makespan'])['best_makespan'])
            instance_names.append(instance_name)
    
    if not best_makespans:
        print("No data found for performance profile")
        return
    
    # Calculate relative gaps
    best_overall = min(best_makespans)
    gaps = [(m - best_overall) / best_overall * 100 for m in best_makespans]
    
    # Sort gaps
    gaps.sort()
    
    # Calculate proportion of instances solved within each gap
    proportions = []
    gap_values = np.linspace(0, max(gaps) + 1, 100)
    
    for gap in gap_values:
        proportion = sum(1 for g in gaps if g <= gap) / len(gaps)
        proportions.append(proportion)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(gap_values, proportions, 'b-', linewidth=2)
    plt.xlabel('Relative Gap to Best (%)')
    plt.ylabel('Proportion of Instances Solved')
    plt.title('Performance Profile')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, min(max(gaps) + 5, 50))
    plt.ylim(0, 1.05)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance profile saved to: {save_path}")
    else:
        plt.show()


def plot_std_deviation(data: Dict, save_path: str = None):
    """
    Plot standard deviation of makespan across runs.
    
    Args:
        data: dict from load_results()
        save_path: Path to save the figure
    """
    instances = [i.replace('.fjs', '') for i in data['instances']]
    
    plt.figure(figsize=(12, 6))
    plt.bar(instances, data['std'], color='orange')
    plt.xlabel('Instance')
    plt.ylabel('Standard Deviation')
    plt.title('Makespan Variability Across Runs')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Standard deviation plot saved to: {save_path}")
    else:
        plt.show()


def plot_all_visualizations(results_csv: str, output_dir: str = "results/wfu/figures"):
    """
    Generate all visualizations from results CSV.
    
    Args:
        results_csv: Path to results CSV file
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    data = load_results(results_csv)
    
    # Plot 1: Summary bar chart
    bar_path = os.path.join(output_dir, 'results_summary.png')
    plot_boxplot_from_summary(data, bar_path)
    
    # Plot 2: Standard deviation
    std_path = os.path.join(output_dir, 'std_deviation.png')
    plot_std_deviation(data, std_path)
    
    # Plot 3: Performance profile
    profile_path = os.path.join(output_dir, 'performance_profile.png')
    plot_performance_profile_from_summary(data, profile_path)
    
    print(f"\nAll visualizations saved to: {output_dir}")


def main():
    """Generate visualizations from results."""
    # Find latest results file
    results_files = glob.glob("results/wfu/experiment_results_*.csv")
    
    if not results_files:
        print("No experiment results files found in results/wfu/")
        results_files = glob.glob("results/wfu/test_results.csv")
    
    if not results_files:
        print("No results files found!")
        print("Please run experiments first.")
        return
    
    latest_file = max(results_files, key=os.path.getctime)
    print(f"Using results file: {latest_file}")
    
    # Generate visualizations
    plot_all_visualizations(latest_file)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()