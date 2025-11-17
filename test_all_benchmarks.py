"""
Test script to run all TSP algorithms on all benchmark files
"""

import os
import glob
from tsp_algorithms import parse_tsp_file, run_all_algorithms
import json


def find_benchmark_files(directory: str = ".") -> list:
    """Find all .txt benchmark files"""
    pattern = os.path.join(directory, "*.txt")
    files = glob.glob(pattern)
    # Filter out any non-benchmark files if needed
    return sorted([f for f in files if os.path.isfile(f)])


def main():
    """Run all algorithms on all benchmark files"""
    benchmark_files = find_benchmark_files()
    
    if not benchmark_files:
        print("No benchmark .txt files found!")
        return
    
    print(f"Found {len(benchmark_files)} benchmark files:")
    for f in benchmark_files:
        print(f"  - {os.path.basename(f)}")
    
    all_results = []
    
    for filename in benchmark_files:
        try:
            print(f"\n{'#'*80}")
            print(f"Processing: {os.path.basename(filename)}")
            print(f"{'#'*80}")
            
            graph = parse_tsp_file(filename)
            results = run_all_algorithms(graph, os.path.basename(filename))
            all_results.append(results)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results to JSON file
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_file}")
    print("\nResults Summary:")
    print(f"{'Filename':<20} {'Cities':<10} {'NN Distance':<15} {'GA Distance':<15} {'BF Distance':<15} {'BB Distance':<15}")
    print("-" * 100)
    
    for result in all_results:
        filename = result['filename']
        num_cities = result['num_cities']
        nn_dist = result['algorithms'].get('NearestNeighbor', {}).get('distance', 'N/A')
        ga_dist = result['algorithms'].get('GeneticAlgorithm', {}).get('distance', 'N/A')
        bf_dist = result['algorithms'].get('BruteForce', {}).get('distance', 'N/A')
        bb_dist = result['algorithms'].get('BranchAndBound', {}).get('distance', 'N/A')
        
        nn_str = f"{nn_dist:.2f}" if isinstance(nn_dist, (int, float)) else str(nn_dist)
        ga_str = f"{ga_dist:.2f}" if isinstance(ga_dist, (int, float)) else str(ga_dist)
        bf_str = f"{bf_dist:.2f}" if isinstance(bf_dist, (int, float)) else str(bf_dist)
        bb_str = f"{bb_dist:.2f}" if isinstance(bb_dist, (int, float)) else str(bb_dist)
        
        print(f"{filename:<20} {num_cities:<10} {nn_str:<15} {ga_str:<15} {bf_str:<15} {bb_str:<15}")


if __name__ == "__main__":
    main()

