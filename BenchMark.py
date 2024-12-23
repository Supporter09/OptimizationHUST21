import os
import csv
import time
from algorithm.CP_model1 import solveCP
from algorithm.LP_model2 import solveLP
from algorithm.Greedy3 import solveGreedy
from algorithm.Genetic_algorithm4 import solveGA
from algorithm.Local_search5 import solveLS
from algorithm.Particle_swarm_optimization6 import solvePSO
from algorithm.Simulated_annealing7 import solveSA
from algorithm.Tabu_search8 import solveTabu
from algorithm.Clarke_wright_savings9 import solve_vrp_clarke_wright
from algorithm.Ant_colony_optimization10 import solveAntColony

def read_test_case(file_path):
    """
    Reads the test case file and extracts N, K, and the distance matrix.

    Args:
        file_path (str): Path to the test case file.

    Returns:
        tuple: (N, K, distance_matrix) where
            - N (int): Number of locations.
            - K (int): Number of vehicles.
            - distance_matrix (list of list of int): N x N matrix of distances.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Extract N and K from the first line
    N, K = map(int, lines[0].strip().split())

    # Extract the distance matrix from the subsequent lines
    distance_matrix = []
    for line in lines[1:]:
        row = list(map(int, line.strip().split()))
        distance_matrix.append(row)

    return N, K, distance_matrix

# Function to benchmark an algorithm
def benchmark_algorithm(solver_class, test_case_path, solution_path):
    # Read the test case
    N, K, distance_matrix = read_test_case(test_case_path)

    # Start timer
    start_time = time.time()

    # Solve
    plans, max_route_distance = solver_class(N, K, distance_matrix)

    # End timer
    end_time = time.time()
    time_taken = end_time - start_time

    # Read the corresponding solution file
    with open(solution_path, "r") as f:
        lines = f.readlines()
        base_objective_value = int(lines[-1]) if lines[-1].strip().isdigit() else -1

    # Compute correctness
    if base_objective_value == 0:
        return 0  # Handle edge case where base objective is zero
    deviation = abs(max_route_distance - base_objective_value)
    correctness = 1 - (deviation / base_objective_value)

    return time_taken, max(correctness, 0), deviation # Ensure correctness doesn't go below 0

# Function to process all test cases and export results
def benchmark_and_export(test_case_dir, solution_dir, algorithms, output_csv):
    results = []

    # Iterate through all algorithms
    for algorithm_name, solver_class in algorithms.items():
        # Iterate through test sizes
        # Size: ["small", "large", "very-large", "enormous"]
        for size in ["small"]:
            for i in range(1, 11):  # Assuming 10 test cases per size
                test_case_file = os.path.join(test_case_dir, f"{size}{i}.txt")
                solution_file = os.path.join(solution_dir, f"{size}_answer{i}.txt")

                if not os.path.exists(test_case_file) or not os.path.exists(solution_file):
                    continue

                # Benchmark the algorithm
                time_taken, correctness, deviation = benchmark_algorithm(solver_class, test_case_file, solution_file)

                # Collect result
                results.append({
                    "Algorithm": algorithm_name,
                    "Test Case Size": size,
                    "Test Case ID": f"{size}{i}",
                    "Time Taken (s)": time_taken,
                    "Correctness": correctness,
                    "Deviation": deviation
                })

    # Export results to CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Algorithm", "Test Case Size", "Test Case ID", "Time Taken (s)", "Correctness", "Deviation"])
        writer.writeheader()
        writer.writerows(results)

# Main function to run the benchmarking
def main():
    test_case_dir = "test_cases"
    solution_dir = "solutions"
    output_csv = "benchmark_results.csv"

    # Define algorithms to benchmark
    algorithms = {
        "CP": solveCP,
        "LP": solveLP,
        # "Greedy": solveGreedy,
        # "Genetic": solveGA,
        # "LocalSearch": solveLS,
        # "PSO": solvePSO,
        # "Simulated Annealing": solveSA,
        # "TabuSearch": solveTabu,
        "CWS": solve_vrp_clarke_wright,
        # "AntColony": solveAntColony,
    }

    # Run benchmarking and export results
    benchmark_and_export(test_case_dir, solution_dir, algorithms, output_csv)

if __name__ == "__main__":
    main()
