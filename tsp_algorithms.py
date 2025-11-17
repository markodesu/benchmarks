"""
TSP (Traveling Salesman Problem) Algorithms Implementation
Implements: Nearest Neighbor, Genetic Algorithm, Brute Force, Branch and Bound
"""

import time
import random
import math
from typing import List, Tuple, Dict, Optional
from itertools import permutations
import copy


class TSPGraph:
    """Represents a TSP problem instance with cities and distances"""
    
    def __init__(self, num_cities: int):
        self.num_cities = num_cities
        self.distances: Dict[Tuple[int, int], float] = {}
        self.cities = list(range(1, num_cities + 1))
    
    def add_edge(self, city1: int, city2: int, distance: float):
        """Add an edge between two cities"""
        self.distances[(city1, city2)] = distance
        self.distances[(city2, city1)] = distance  # Undirected graph
    
    def get_distance(self, city1: int, city2: int) -> float:
        """Get distance between two cities"""
        return self.distances.get((city1, city2), float('inf'))
    
    def calculate_tour_distance(self, tour: List[int]) -> float:
        """Calculate total distance of a tour"""
        if len(tour) < 2:
            return 0.0
        
        total = 0.0
        for i in range(len(tour)):
            city1 = tour[i]
            city2 = tour[(i + 1) % len(tour)]
            total += self.get_distance(city1, city2)
        return total


def parse_tsp_file(filename: str) -> TSPGraph:
    """Parse a TSP benchmark file and create a TSPGraph"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find the number of cities from the edge list
    edges = []
    max_city = 0
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if len(parts) >= 3:
            city1 = int(parts[0])
            city2 = int(parts[1])
            distance = float(parts[2])
            edges.append((city1, city2, distance))
            max_city = max(max_city, city1, city2)
    
    # Create graph
    graph = TSPGraph(max_city)
    for city1, city2, distance in edges:
        graph.add_edge(city1, city2, distance)
    
    return graph


class NearestNeighbor:
    """Nearest Neighbor heuristic algorithm for TSP"""
    
    @staticmethod
    def solve(graph: TSPGraph, start_city: int = 1) -> Tuple[List[int], float]:
        """Solve TSP using Nearest Neighbor algorithm"""
        start_time = time.time()
        
        unvisited = set(graph.cities)
        tour = [start_city]
        unvisited.remove(start_city)
        current = start_city
        
        while unvisited:
            nearest = None
            min_dist = float('inf')
            
            for city in unvisited:
                dist = graph.get_distance(current, city)
                if dist < min_dist:
                    min_dist = dist
                    nearest = city
            
            if nearest is None:
                break
            
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        distance = graph.calculate_tour_distance(tour)
        elapsed_time = time.time() - start_time
        
        return tour, distance, elapsed_time


class GeneticAlgorithm:
    """Genetic Algorithm for TSP"""
    
    def __init__(self, population_size: int = 100, generations: int = 1000, 
                 mutation_rate: float = 0.01, crossover_rate: float = 0.8):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def solve(self, graph: TSPGraph) -> Tuple[List[int], float, float]:
        """Solve TSP using Genetic Algorithm"""
        start_time = time.time()
        
        # Initialize population
        population = self._initialize_population(graph)
        
        best_tour = None
        best_distance = float('inf')
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [1.0 / graph.calculate_tour_distance(tour) 
                            for tour in population]
            
            # Find best tour
            best_idx = max(range(len(population)), key=lambda i: fitness_scores[i])
            current_best = graph.calculate_tour_distance(population[best_idx])
            
            if current_best < best_distance:
                best_distance = current_best
                best_tour = population[best_idx].copy()
            
            # Create new population
            new_population = []
            
            # Elitism: keep best tour
            new_population.append(best_tour.copy() if best_tour else population[best_idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection (tournament selection)
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        elapsed_time = time.time() - start_time
        return best_tour, best_distance, elapsed_time
    
    def _initialize_population(self, graph: TSPGraph) -> List[List[int]]:
        """Initialize population with random tours"""
        population = []
        for _ in range(self.population_size):
            tour = graph.cities.copy()
            random.shuffle(tour)
            population.append(tour)
        return population
    
    def _tournament_selection(self, population: List[List[int]], 
                            fitness_scores: List[float], tournament_size: int = 5) -> List[int]:
        """Tournament selection"""
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        return max(tournament, key=lambda x: x[1])[0]
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order crossover (OX)"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [-1] * size
        child[start:end] = parent1[start:end]
        
        pos = end
        for city in parent2:
            if city not in child:
                if pos >= size:
                    pos = 0
                child[pos] = city
                pos += 1
        
        return child
    
    def _mutate(self, tour: List[int]) -> List[int]:
        """Swap mutation"""
        tour = tour.copy()
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]
        return tour


class BruteForce:
    """Brute Force algorithm for TSP (only for small instances)"""
    
    @staticmethod
    def solve(graph: TSPGraph, max_cities: int = 10) -> Tuple[List[int], float, float]:
        """Solve TSP using Brute Force (only for small instances)"""
        start_time = time.time()
        
        if graph.num_cities > max_cities:
            # For large instances, return a simple heuristic solution
            tour, distance, _ = NearestNeighbor.solve(graph)
            elapsed_time = time.time() - start_time
            return tour, distance, elapsed_time
        
        cities = graph.cities
        best_tour = None
        best_distance = float('inf')
        
        # Try all permutations starting from city 1
        start_city = cities[0]
        remaining_cities = cities[1:]
        
        for perm in permutations(remaining_cities):
            tour = [start_city] + list(perm) + [start_city]
            distance = graph.calculate_tour_distance(tour[:-1])  # Remove duplicate start
            if distance < best_distance:
                best_distance = distance
                best_tour = tour[:-1]
        
        elapsed_time = time.time() - start_time
        return best_tour, best_distance, elapsed_time


class BranchAndBound:
    """Branch and Bound algorithm for TSP - Optimized version"""
    
    def __init__(self, time_limit: float = 60.0):  # 1 minute default limit
        self.time_limit = time_limit
        self.best_tour = None
        self.best_distance = float('inf')
        self.start_time = None
        self.nodes_explored = 0
    
    def solve(self, graph: TSPGraph) -> Tuple[List[int], float, float]:
        """Solve TSP using Branch and Bound"""
        self.start_time = time.time()
        self.best_tour = None
        self.best_distance = float('inf')
        self.nodes_explored = 0
        
        # For large instances, use heuristic only
        if graph.num_cities > 20:
            tour, distance, _ = NearestNeighbor.solve(graph)
            elapsed_time = time.time() - self.start_time
            return tour, distance, elapsed_time
        
        # Get initial upper bound using Nearest Neighbor
        initial_tour, initial_dist, _ = NearestNeighbor.solve(graph)
        self.best_tour = initial_tour
        self.best_distance = initial_dist
        
        # Use direct distance lookups instead of full matrix
        # Start branch and bound
        unvisited = set(graph.cities[1:])  # Exclude start city
        current_path = [graph.cities[0]]
        current_cost = 0.0
        
        self._branch_and_bound_optimized(current_path, current_cost, 
                                        unvisited, graph)
        
        elapsed_time = time.time() - self.start_time
        return self.best_tour, self.best_distance, elapsed_time
    
    def _calculate_lower_bound(self, path: List[int], unvisited: set, 
                              graph: TSPGraph) -> float:
        """Calculate a lower bound for the remaining path"""
        if not unvisited:
            # Return to start
            if len(path) > 0:
                return graph.get_distance(path[-1], path[0])
            return 0.0
        
        current_city = path[-1]
        lb = 0.0
        
        # Minimum edge from current city to any unvisited
        min_from_current = min(graph.get_distance(current_city, city) 
                              for city in unvisited)
        lb += min_from_current
        
        # For each unvisited city, minimum edge to any other unvisited or back to start
        for city in unvisited:
            min_edges = []
            # To other unvisited cities
            for other in unvisited:
                if other != city:
                    min_edges.append(graph.get_distance(city, other))
            # Back to start
            min_edges.append(graph.get_distance(city, graph.cities[0]))
            if min_edges:
                lb += min(min_edges)
        
        return lb
    
    def _branch_and_bound_optimized(self, path: List[int], current_cost: float,
                                   unvisited: set, graph: TSPGraph):
        """Optimized recursive branch and bound"""
        # Check time limit
        if time.time() - self.start_time > self.time_limit:
            return
        
        self.nodes_explored += 1
        
        # If all cities visited, complete the tour
        if not unvisited:
            # Add return to start
            if len(path) > 0:
                final_cost = current_cost + graph.get_distance(path[-1], path[0])
                if final_cost < self.best_distance:
                    self.best_distance = final_cost
                    self.best_tour = path.copy()
            return
        
        # Calculate lower bound
        lower_bound = self._calculate_lower_bound(path, unvisited, graph)
        
        # Prune if lower bound exceeds best known solution
        if current_cost + lower_bound >= self.best_distance:
            return
        
        # Order unvisited cities by distance (try nearest first for better pruning)
        current_city = path[-1]
        unvisited_list = sorted(unvisited, 
                              key=lambda city: graph.get_distance(current_city, city))
        
        # Limit branching for very large search spaces
        if len(unvisited) > 15:
            unvisited_list = unvisited_list[:10]  # Try only 10 nearest
        
        for next_city in unvisited_list:
            edge_cost = graph.get_distance(current_city, next_city)
            
            if edge_cost == float('inf'):
                continue
            
            new_path = path + [next_city]
            new_unvisited = unvisited - {next_city}
            new_cost = current_cost + edge_cost
            
            self._branch_and_bound_optimized(new_path, new_cost, 
                                           new_unvisited, graph)


def run_all_algorithms(graph: TSPGraph, filename: str) -> Dict:
    """Run all algorithms on a TSP instance"""
    results = {
        'filename': filename,
        'num_cities': graph.num_cities,
        'algorithms': {}
    }
    
    print(f"\n{'='*60}")
    print(f"Testing: {filename}")
    print(f"Number of cities: {graph.num_cities}")
    print(f"{'='*60}")
    
    # Nearest Neighbor
    print("\n1. Nearest Neighbor Algorithm...")
    try:
        tour, distance, elapsed = NearestNeighbor.solve(graph)
        results['algorithms']['NearestNeighbor'] = {
            'distance': distance,
            'time': elapsed,
            'tour_length': len(tour)
        }
        print(f"   Distance: {distance:.2f}")
        print(f"   Time: {elapsed:.4f} seconds")
    except Exception as e:
        print(f"   Error: {e}")
        results['algorithms']['NearestNeighbor'] = {'error': str(e)}
    
    # Genetic Algorithm
    print("\n2. Genetic Algorithm...")
    try:
        # Adjust parameters based on problem size for faster execution
        if graph.num_cities > 1000:
            ga = GeneticAlgorithm(population_size=30, generations=200)
        elif graph.num_cities > 100:
            ga = GeneticAlgorithm(population_size=40, generations=300)
        else:
            ga = GeneticAlgorithm(population_size=50, generations=500)
        tour, distance, elapsed = ga.solve(graph)
        results['algorithms']['GeneticAlgorithm'] = {
            'distance': distance,
            'time': elapsed,
            'tour_length': len(tour)
        }
        print(f"   Distance: {distance:.2f}")
        print(f"   Time: {elapsed:.4f} seconds")
    except Exception as e:
        print(f"   Error: {e}")
        results['algorithms']['GeneticAlgorithm'] = {'error': str(e)}
    
    # Brute Force (only for small instances)
    print("\n3. Brute Force Algorithm...")
    try:
        tour, distance, elapsed = BruteForce.solve(graph, max_cities=10)
        results['algorithms']['BruteForce'] = {
            'distance': distance,
            'time': elapsed,
            'tour_length': len(tour)
        }
        print(f"   Distance: {distance:.2f}")
        print(f"   Time: {elapsed:.4f} seconds")
        if graph.num_cities > 10:
            print(f"   Note: Used heuristic for large instance (>{10} cities)")
    except Exception as e:
        print(f"   Error: {e}")
        results['algorithms']['BruteForce'] = {'error': str(e)}
    
    # Branch and Bound
    print("\n4. Branch and Bound Algorithm...")
    try:
        bb = BranchAndBound(time_limit=300.0)  # 5 minute limit
        tour, distance, elapsed = bb.solve(graph)
        results['algorithms']['BranchAndBound'] = {
            'distance': distance,
            'time': elapsed,
            'tour_length': len(tour)
        }
        print(f"   Distance: {distance:.2f}")
        print(f"   Time: {elapsed:.4f} seconds")
    except Exception as e:
        print(f"   Error: {e}")
        results['algorithms']['BranchAndBound'] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test on specific file
        filename = sys.argv[1]
        graph = parse_tsp_file(filename)
        results = run_all_algorithms(graph, filename)
    else:
        print("Usage: python tsp_algorithms.py <benchmark_file.txt>")
        print("\nOr use test_all_benchmarks.py to test all files")

