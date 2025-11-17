# TSP Algorithm Benchmark Suite

A comprehensive implementation and benchmarking suite for Traveling Salesman Problem (TSP) algorithms, including Nearest Neighbor, Genetic Algorithm, Brute Force, and Branch and Bound methods.

## 📋 Table of Contents

- [Overview](#overview)
- [Algorithms Implemented](#algorithms-implemented)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Benchmark Files](#benchmark-files)
- [Results](#results)
- [Performance Analysis](#performance-analysis)
- [Time Complexity](#time-complexity)

## 🎯 Overview

This project implements four different algorithms for solving the Traveling Salesman Problem (TSP) and tests them on multiple benchmark instances. The TSP is a classic optimization problem where the goal is to find the shortest possible route that visits each city exactly once and returns to the starting city.

## 🔧 Algorithms Implemented

### 1. Nearest Neighbor Algorithm
- **Type**: Greedy Heuristic
- **Time Complexity**: O(n²)
- **Space Complexity**: O(n)
- **Description**: Starts from a city and repeatedly visits the nearest unvisited city until all cities are visited.
- **Pros**: Very fast, simple to implement
- **Cons**: May not find optimal solution, can get stuck in local optima

### 2. Genetic Algorithm
- **Type**: Metaheuristic
- **Time Complexity**: O(g × p × n²) where g=generations, p=population size
- **Space Complexity**: O(p × n)
- **Description**: Population-based evolutionary algorithm that uses selection, crossover, and mutation to evolve solutions.
- **Pros**: Can find good solutions, handles large instances
- **Cons**: Requires parameter tuning, not guaranteed optimal

### 3. Brute Force Algorithm
- **Type**: Exact Algorithm
- **Time Complexity**: O(n!)
- **Space Complexity**: O(n)
- **Description**: Tries all possible permutations to find the optimal solution.
- **Pros**: Guaranteed to find optimal solution
- **Cons**: Only feasible for small instances (≤10 cities)

### 4. Branch and Bound Algorithm
- **Type**: Exact Algorithm
- **Time Complexity**: O(2^n) worst case, but often much better with pruning
- **Space Complexity**: O(n)
- **Description**: Uses tree search with pruning based on lower bounds to find optimal solution.
- **Pros**: Can find optimal solution, more efficient than brute force
- **Cons**: Still exponential, uses heuristic for large instances (>20 cities)

## ✨ Features

- ✅ Four different TSP solving algorithms
- ✅ Automatic benchmark testing on multiple TSP instances
- ✅ Performance metrics and statistical analysis
- ✅ PDF report generation for results
- ✅ Comparative analysis with visualizations
- ✅ Handles both small and large problem instances
- ✅ Automatic heuristic fallback for large instances

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Required Packages

```bash
pip install openpyxl reportlab
```

Or install all at once:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Basic Usage

Run a single algorithm on a benchmark file:

```bash
python3 tsp_algorithms.py att48.tsp.txt
```

### Run All Algorithms on All Benchmarks

```bash
python3 test_all_benchmarks.py
```

### Generate Results PDF

```bash
python3 generate_results_pdf.py
```

This creates `TSP_Algorithm_Results.pdf` with a comprehensive table of all results.

### Generate Comparative Analysis PDF

```bash
python3 generate_comparative_analysis.py
```

This creates `TSP_Comparative_Analysis.pdf` with statistical analysis and comparisons.

## 📁 Project Structure

```
.
├── tsp_algorithms.py              # Main algorithm implementations
├── test_all_benchmarks.py          # Test script for all benchmarks
├── generate_results_pdf.py         # PDF report generator
├── generate_comparative_analysis.py # Comparative analysis PDF generator
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore file
│
├── Benchmark Files/
│   ├── ali535.txt
│   ├── att48.tsp.txt
│   ├── att532.txt
│   ├── d198.txt
│   ├── d493.txt
│   ├── d657.txt
│   ├── d1291.txt
│   ├── d1655.txt
│   ├── fl1400.txt
│   ├── fl3795.txt
│   ├── fnl4461.txt
│   ├── gr431.txt
│   ├── pa561.txt
│   ├── pcb3038.txt
│   ├── pla7397.txt
│   ├── rat575.txt
│   └── rl11849.txt
│
└── Output Files/
    ├── TSP_Algorithm_Results.pdf
    ├── TSP_Comparative_Analysis.pdf
    └── benchmark_results.json
```

## 📊 Benchmark Files

The project includes 17 TSP benchmark instances with varying sizes:

| Benchmark | Cities | Optimal Distance |
|-----------|--------|------------------|
| ali535 | 100 | 202339 |
| att48 | 48 | 10628 |
| att532 | 100 | 27686 |
| d198 | 100 | 15780 |
| d493 | 100 | 35002 |
| d657 | 100 | 48912 |
| d1291 | 100 | 50801 |
| d1655 | 100 | 62128 |
| fl1400 | 100 | 20127 |
| fl3795 | 100 | 28772 |
| fnl4461 | 100 | 182566 |
| gr431 | 100 | 171414 |
| pa561 | 100 | 2763 |
| pcb3038 | 100 | 137694 |
| pla7397 | 100 | 23260728 |
| rat575 | 100 | 6773 |
| rl11849 | 100 | 923288 |

## 📈 Results

The algorithms are tested on all benchmark files and results are generated in two formats:

1. **TSP_Algorithm_Results.pdf**: Detailed table with all algorithm results including distances and execution times
2. **TSP_Comparative_Analysis.pdf**: Statistical analysis including:
   - Performance ratio analysis (distance/optimal)
   - Execution time statistics
   - Algorithm comparison by benchmark
   - Key findings and conclusions

## 🔍 Performance Analysis

### Algorithm Performance Summary

- **Nearest Neighbor**: Fastest execution, good for quick approximate solutions
- **Genetic Algorithm**: Better solution quality, moderate execution time
- **Brute Force**: Optimal solutions but only for small instances
- **Branch and Bound**: Optimal solutions for medium instances, uses heuristic for large ones

### Time Complexity Comparison

| Algorithm | Best Case | Average Case | Worst Case |
|-----------|-----------|--------------|------------|
| Nearest Neighbor | O(n²) | O(n²) | O(n²) |
| Genetic Algorithm | O(g×p×n²) | O(g×p×n²) | O(g×p×n²) |
| Brute Force | O(n!) | O(n!) | O(n!) |
| Branch and Bound | O(n²) | O(2^n) | O(2^n) |

## 🧪 Testing

To test the implementation on all benchmarks:

```bash
python3 test_all_benchmarks.py
```

Results are saved to `benchmark_results.json` in JSON format.

## 📝 Notes

- For instances with more than 10 cities, Brute Force uses Nearest Neighbor heuristic
- For instances with more than 20 cities, Branch and Bound uses Nearest Neighbor heuristic
- Genetic Algorithm parameters are automatically adjusted based on problem size
- Some benchmark results may show discrepancies with optimal values due to data format differences

## 🤝 Contributing

This is an academic project for algorithm implementation and benchmarking. Contributions and improvements are welcome!

## 📄 License

This project is provided for educational purposes.

## 👤 Author

TSP Algorithm Benchmark Suite - Academic Project

## 🔗 References

- TSPLIB: A library of sample instances for the TSP
- Traveling Salesman Problem: Classic optimization problem in computer science
- Algorithm Design: Various TSP solving techniques

---

**Last Updated**: 2025

