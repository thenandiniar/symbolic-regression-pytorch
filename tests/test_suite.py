"""
Comprehensive test suite for symbolic regression.

Tests the algorithm on multiple target functions.

Author: Anand
Date: March 2, 2025
GSoC 2026 - DeepChem Symbolic ML
"""

import sys
import os

# Get absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from expression import ExpressionTree
from genetic_algo import SymbolicRegressor


def test_linear():
    """Test: y = 2*x + 3"""
    print("\n" + "="*60)
    print("TEST 1: LINEAR FUNCTION (y = 2*x + 3)")
    print("="*60)
    
    x_train = torch.linspace(-5, 5, 100)
    y_train = 2 * x_train + 3
    
    regressor = SymbolicRegressor(
        population_size=100,
        max_depth=4,
        mutation_rate=0.3,
        complexity_weight=0.01,
    )
    
    best_tree, history = regressor.evolve(
        x_train, y_train,
        generations=50,
        verbose=True
    )
    
    return best_tree, history, x_train, y_train


def test_quadratic():
    """Test: y = x^2"""
    print("\n" + "="*60)
    print("TEST 2: QUADRATIC FUNCTION (y = x²)")
    print("="*60)
    
    x_train = torch.linspace(-3, 3, 100)
    y_train = x_train ** 2
    
    regressor = SymbolicRegressor(
        population_size=100,
        max_depth=4,
        mutation_rate=0.3,
        complexity_weight=0.01,
    )
    
    best_tree, history = regressor.evolve(
        x_train, y_train,
        generations=50,
        verbose=True
    )
    
    return best_tree, history, x_train, y_train


def test_sine():
    """Test: y = sin(x)"""
    print("\n" + "="*60)
    print("TEST 3: SINE FUNCTION (y = sin(x))")
    print("="*60)
    
    x_train = torch.linspace(-np.pi, np.pi, 100)
    y_train = torch.sin(x_train)
    
    regressor = SymbolicRegressor(
        population_size=100,
        max_depth=4,
        mutation_rate=0.3,
        complexity_weight=0.01,
    )
    
    best_tree, history = regressor.evolve(
        x_train, y_train,
        generations=50,
        verbose=True
    )
    
    return best_tree, history, x_train, y_train


def plot_results(best_tree, x_train, y_train, title, filename):
    """Plot true vs predicted function."""
    y_pred = best_tree.evaluate(x_train)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train.numpy(), y_train.numpy(), 
                alpha=0.5, label='True data', s=20)
    plt.plot(x_train.numpy(), y_pred.detach().numpy(), 
             'r-', linewidth=2, label=f'Discovered: {best_tree}')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create docs directory using absolute path
    docs_dir = os.path.join(project_root, 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    filepath = os.path.join(docs_dir, filename)
    plt.savefig(filepath, dpi=150)
    print(f"✓ Plot saved: {filepath}")
    plt.close()


def plot_evolution(history, title, filename):
    """Plot evolution of fitness over generations."""
    generations = range(len(history['best_fitness']))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Fitness over time
    ax1.plot(generations, history['best_fitness'], 'b-', linewidth=2, label='Best')
    ax1.plot(generations, history['mean_fitness'], 'r--', linewidth=2, label='Mean')
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Fitness (lower is better)', fontsize=12)
    ax1.set_title('Fitness Evolution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Best expression length over time
    complexities = [len(expr) for expr in history['best_expression']]
    ax2.plot(generations, complexities, 'g-', linewidth=2)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Expression Length (characters)', fontsize=12)
    ax2.set_title('Expression Complexity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create docs directory using absolute path
    docs_dir = os.path.join(project_root, 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    filepath = os.path.join(docs_dir, filename)
    plt.savefig(filepath, dpi=150)
    print(f"✓ Plot saved: {filepath}")
    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("SYMBOLIC REGRESSION - COMPREHENSIVE TEST SUITE")
    print("="*60)
    print(f"Project root: {project_root}")
    print(f"Docs will be saved in: {os.path.join(project_root, 'docs')}")
    print("="*60)
    
    # Run all tests
    tests = [
        ("Linear", test_linear, "linear"),
        ("Quadratic", test_quadratic, "quadratic"),
        ("Sine", test_sine, "sine"),
    ]
    
    results = []
    
    for name, test_func, code in tests:
        try:
            best_tree, history, x_train, y_train = test_func()
            
            # Plot results
            plot_results(
                best_tree, x_train, y_train,
                f"{name} Function - Symbolic Regression",
                f"result_{code}.png"
            )
            
            # Plot evolution
            plot_evolution(
                history,
                f"{name} - Evolution Progress",
                f"evolution_{code}.png"
            )
            
            results.append({
                'name': name,
                'expression': str(best_tree),
                'fitness': history['best_fitness'][-1],
                'complexity': best_tree.complexity(),
                'success': '✓'
            })
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': name,
                'expression': 'Failed',
                'fitness': float('inf'),
                'complexity': 0,
                'success': '✗'
            })
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY OF ALL TESTS")
    print("="*60)
    print(f"{'Test':<15} {'Status':<8} {'Best Expression':<30} {'Fitness':<12} {'Complexity':<12}")
    print("-"*60)
    for r in results:
        expr_short = r['expression'][:28] + '..' if len(r['expression']) > 30 else r['expression']
        print(f"{r['name']:<15} {r['success']:<8} {expr_short:<30} {r['fitness']:<12.6f} {r['complexity']:<12}")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print(f"Check plots in: {os.path.join(project_root, 'docs')}")
    print("="*60)