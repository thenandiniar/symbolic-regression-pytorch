"""
Genetic Algorithm for Symbolic Regression.

Author: Nandini A R
Date: March 2, 2026
GSoC 2026 - DeepChem Symbolic ML
"""

import torch
import random
from typing import List, Tuple, Optional
from expression import (
    ExpressionNode, ExpressionTree, NodeType, Operator,
    make_variable, make_constant, make_operator
)
from fitness import FitnessFunction


class SymbolicRegressor:
    """
    Genetic algorithm for symbolic regression.
    
    Evolves mathematical expressions to fit data.
    """
    
    def __init__(
        self,
        population_size: int = 100,
        max_depth: int = 5,
        tournament_size: int = 5,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        complexity_weight: float = 0.001,
        operators: Optional[List[Operator]] = None,
    ):
        """
        Initialize genetic algorithm.
        
        Args:
            population_size: Number of individuals in population
            max_depth: Maximum tree depth
            tournament_size: Tournament selection size
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            complexity_weight: Weight for complexity penalty in fitness
            operators: List of operators to use (default: basic set)
        """
        self.population_size = population_size
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.complexity_weight = complexity_weight
        
        # Default operators if not specified
        if operators is None:
            self.operators = [
                Operator.ADD, Operator.SUB, Operator.MUL, Operator.DIV,
                Operator.SIN, Operator.COS,
            ]
        else:
            self.operators = operators
        
        self.binary_ops = [op for op in self.operators if op.arity == 2]
        self.unary_ops = [op for op in self.operators if op.arity == 1]
    
    def generate_random_tree(self, max_depth: int, method: str = "grow") -> ExpressionNode:
        """
        Generate random expression tree.
        
        Args:
            max_depth: Maximum depth
            method: "grow" (varied depth) or "full" (full depth)
        
        Returns:
            Root node of random tree
        """
        if max_depth == 1 or (method == "grow" and random.random() < 0.3):
            # Terminal node
            if random.random() < 0.7:
                return make_variable()
            else:
                return make_constant(random.uniform(-5, 5))
        
        # Operator node
        if random.random() < 0.7 and self.binary_ops:
            # Binary operator
            op = random.choice(self.binary_ops)
            left = self.generate_random_tree(max_depth - 1, method)
            right = self.generate_random_tree(max_depth - 1, method)
            return make_operator(op, left, right)
        elif self.unary_ops:
            # Unary operator
            op = random.choice(self.unary_ops)
            child = self.generate_random_tree(max_depth - 1, method)
            return make_operator(op, child)
        else:
            # Fallback to terminal
            return make_variable()
    
    def initialize_population(self) -> List[ExpressionTree]:
        """
        Create initial population using ramped half-and-half.
        
        Returns:
            List of random expression trees
        """
        population = []
        half = self.population_size // 2
        
        # Half with "grow" method
        for _ in range(half):
            depth = random.randint(2, self.max_depth)
            root = self.generate_random_tree(depth, method="grow")
            population.append(ExpressionTree(root))
        
        # Half with "full" method
        for _ in range(self.population_size - half):
            depth = random.randint(2, self.max_depth)
            root = self.generate_random_tree(depth, method="full")
            population.append(ExpressionTree(root))
        
        return population
    
    def tournament_selection(
        self, 
        population: List[ExpressionTree],
        fitnesses: List[float]
    ) -> ExpressionTree:
        """
        Select individual using tournament selection.
        
        Args:
            population: List of trees
            fitnesses: List of fitness values
        
        Returns:
            Selected tree
        """
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
        return population[winner_idx]
    
    def mutate(self, node: ExpressionNode, depth: int = 0) -> ExpressionNode:
        """
        Mutate expression tree.
        
        Args:
            node: Node to mutate
            depth: Current depth
        
        Returns:
            Mutated node
        """
        if random.random() < 0.1:  # 10% chance to replace entire subtree
            return self.generate_random_tree(self.max_depth - depth, method="grow")
        
        # Handle different node types
        if node.node_type == NodeType.CONSTANT:
            # Perturb constant
            new_value = node.value + random.gauss(0, 0.5)
            return make_constant(new_value)
        
        elif node.node_type == NodeType.VARIABLE:
            # Variables don't mutate
            return make_variable()
        
        elif node.node_type == NodeType.OPERATOR:
            # Mutate children first
            left_mutated = self.mutate(node.left, depth + 1) if node.left else None
            right_mutated = self.mutate(node.right, depth + 1) if node.right else None
            
            # Maybe change operator (but keep same arity!)
            new_op = node.operator
            if random.random() < 0.2:
                if node.operator.arity == 2 and self.binary_ops:
                    # Only change to another binary operator
                    new_op = random.choice(self.binary_ops)
                elif node.operator.arity == 1 and self.unary_ops:
                    # Only change to another unary operator
                    new_op = random.choice(self.unary_ops)
            
            # Create new node with mutated children
            return make_operator(new_op, left_mutated, right_mutated)
        
        else:
            # Shouldn't happen, but return copy
            return node.copy()
    
    def crossover(
        self, 
        parent1: ExpressionNode, 
        parent2: ExpressionNode
    ) -> Tuple[ExpressionNode, ExpressionNode]:
        """
        Perform subtree crossover.
        
        Args:
            parent1: First parent node
            parent2: Second parent node
        
        Returns:
            Two offspring nodes
        """
        # Simple implementation: just return copies
        # A more sophisticated version would swap random subtrees
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        return child1, child2
    
    def evolve(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        generations: int = 50,
        verbose: bool = True
    ) -> Tuple[ExpressionTree, dict]:
        """
        Run genetic algorithm.
        
        Args:
            x_train: Training inputs
            y_train: Training targets
            generations: Number of generations
            verbose: Print progress
        
        Returns:
            best_tree: Best expression found
            history: Training history
        """
        # Initialize
        population = self.initialize_population()
        fitness_fn = FitnessFunction(x_train, y_train, self.complexity_weight)
        
        history = {
            'best_fitness': [],
            'mean_fitness': [],
            'best_expression': [],
        }
        
        if verbose:
            print(f"Generation 0/{generations}")
            print(f"Population size: {self.population_size}")
            print("-" * 60)
        
        for gen in range(generations):
            # Evaluate fitness
            fitnesses = []
            for tree in population:
                fitness, _ = fitness_fn.evaluate(tree)
                fitnesses.append(fitness)
            
            # Track best
            best_idx = fitnesses.index(min(fitnesses))
            best_fitness = fitnesses[best_idx]
            best_tree = population[best_idx]
            
            history['best_fitness'].append(best_fitness)
            history['mean_fitness'].append(sum(fitnesses) / len(fitnesses))
            history['best_expression'].append(str(best_tree))
            
            if verbose and gen % 10 == 0:
                print(f"Gen {gen}: Best fitness = {best_fitness:.6f}")
                print(f"         Best expr = {best_tree}")
                print(f"         Mean fitness = {history['mean_fitness'][-1]:.6f}")
                print()
            
            # Create next generation
            new_population = []
            
            # Elitism: keep best individual
            new_population.append(ExpressionTree(best_tree.root.copy()))
            
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1_root, child2_root = self.crossover(
                        parent1.root, parent2.root
                    )
                else:
                    child1_root = parent1.root.copy()
                    child2_root = parent2.root.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child1_root = self.mutate(child1_root)
                if random.random() < self.mutation_rate:
                    child2_root = self.mutate(child2_root)
                
                # Add to new population
                new_population.append(ExpressionTree(child1_root))
                if len(new_population) < self.population_size:
                    new_population.append(ExpressionTree(child2_root))
            
            population = new_population
        
        # Final evaluation
        fitnesses = [fitness_fn.evaluate(tree)[0] for tree in population]
        best_idx = fitnesses.index(min(fitnesses))
        best_tree = population[best_idx]
        
        if verbose:
            print("-" * 60)
            print(f"FINAL RESULT:")
            print(f"Best fitness: {fitnesses[best_idx]:.6f}")
            print(f"Best expression: {best_tree}")
            print(f"Complexity: {best_tree.complexity()}")
        
        return best_tree, history


# Test the genetic algorithm
if __name__ == "__main__":
    import numpy as np
    
    print("=" * 60)
    print("GENETIC ALGORITHM TEST")
    print("=" * 60)
    print()
    
    # Test data: y = x^2
    print("Target function: y = x^2")
    x_train = torch.linspace(-3, 3, 50)
    y_train = x_train ** 2
    
    print(f"Training samples: {len(x_train)}")
    print(f"X range: [{x_train.min():.2f}, {x_train.max():.2f}]")
    print(f"Y range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print()
    
    # Run evolution
    regressor = SymbolicRegressor(
        population_size=50,
        max_depth=4,
        tournament_size=3,
        mutation_rate=0.3,
        crossover_rate=0.7,
        complexity_weight=0.01,
    )
    
    best_tree, history = regressor.evolve(
        x_train, y_train,
        generations=30,
        verbose=True
    )
    
    print()
    print("=" * 60)
    print("EVOLUTION COMPLETED")
    print("=" * 60)
