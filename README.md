# Symbolic Regression in PyTorch

**GSoC 2026 Prototype for DeepChem**

## Overview
Pure PyTorch implementation of symbolic regression using genetic programming.

## Goal
Replace Julia-based PySR with native PyTorch solution for DeepChem integration.

## Current Status
🚧 **Week 1 Prototype** - Core functionality implementation

### Implemented
- [x] Expression tree representation
- [x] Basic operators (add, mul, sin, cos, exp, log)
- [ ] Genetic algorithm core
- [ ] Fitness evaluation
- [ ] Constant optimization

### In Progress
- [ ] Expression evaluation in PyTorch
- [ ] Mutation operators
- [ ] Crossover operators

### Planned
- [ ] Neural-guided search (paper approach)
- [ ] Benchmarking vs PySR
- [ ] DeepChem TorchModel integration

## Quick Start
```bash
pip install -r requirements.txt
python src/expression.py