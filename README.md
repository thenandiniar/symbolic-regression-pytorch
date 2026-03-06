# Symbolic Regression in PyTorch

**GSoC 2026 Prototype for DeepChem**

A pure PyTorch implementation of symbolic regression using genetic programming. Discovers mathematical equations from data automatically.

## 🎯 Project Goal

Replace Julia-based PySR with native PyTorch solution for DeepChem integration.

## ✨ Features

- ✅ Expression tree representation with PyTorch evaluation
- ✅ Genetic algorithm with tournament selection
- ✅ Structure-preserving mutation operators
- ✅ Automatic equation discovery from data
- ✅ Fitness evaluation (MSE + complexity penalty)
- ✅ Comprehensive test suite
- ✅ Visualization of evolution progress

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic test
python src/expression.py

# Run genetic algorithm test
python src/genetic_algo.py

# Run comprehensive test suite
python tests/test_suite.py