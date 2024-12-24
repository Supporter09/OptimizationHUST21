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


def benchmark_algorithm(solver_class, test_case_path, solution_path):
    """
    Benchmark the given solver with the provided test case.

    Args:
        solver_class (function): The solving algorithm.
        test_case_path (str): Path to the test case file.
        solution_path (str): Path to the solution file.
        time_limit (int): Time limit in seconds (optional).

    Returns:
        tuple: Time taken, correctness, deviation, plans, number of vehicles.
    """
    # Read the test case
    N, K, distance_matrix = read_test_case(test_case_path)

    # Start timer
    start_time = time.time()

    # Solve the problem
    plans, max_route_distance = solver_class(N, K, distance_matrix)

    # Stop the timer
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

    return (
        time_taken,
        max(correctness, 0),
        deviation,
        plans,
        N,
    )  # Ensure correctness doesn't go below 0


# Function to process all test cases and export results
def benchmark_and_export(
    test_case_dir, solution_dir, algorithms, output_csv, output_dir
):
    """
    Benchmark all algorithms and export results to CSV. Additionally, save the best plans for each algorithm and size.

    Args:
        test_case_dir (str): Directory containing test cases.
        solution_dir (str): Directory containing solutions.
        algorithms (dict): Mapping of algorithm names to solver functions.
        output_csv (str): Path to export CSV results.
        output_dir (str): Directory to save best plans for each algorithm.
    """
    results = []
    best_plans = {
        algorithm: {} for algorithm in algorithms.keys()
    }  # Track best solutions

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all algorithms
    for algorithm_name, solver_class in algorithms.items():
        # Iterate through test sizes

        time_stats = {
            size: {"total_time": 0, "count": 0}
            for size in ["small", "large", "very-large", "enormous"]
        }
        correctness_stats = {
            size: {"correctness": 0, "count": 0}
            for size in ["small", "large", "very-large", "enormous"]
        }

        # Size: ["small", "large", "very-large", "enormous"]
        for size in ["small", "large", "very-large", "enormous"]:
            # Don't run LP in very large and enormous algo
            if size in ["very-large", "enormous"] and algorithm_name == "LP":
                continue

            for i in range(1, 11):  # Assuming 10 test cases per size
                print(f"Running {algorithm_name} size {size} test case {i}...")
                test_case_file = os.path.join(test_case_dir, f"{size}{i}.txt")
                solution_file = os.path.join(solution_dir, f"{size}_answer{i}.txt")

                if not os.path.exists(test_case_file) or not os.path.exists(
                    solution_file
                ):
                    continue

                # Benchmark the algorithm
                time_taken, correctness, deviation, plans, N = benchmark_algorithm(
                    solver_class, test_case_file, solution_file
                )

                # Fix plans format for CWS algo
                if algorithm_name == "CWS":
                    tmp_plans = []
                    for plan in plans:
                        tmp_plans.append([len(plan), plan])

                    plans = tmp_plans

                # Collect result
                results.append(
                    {
                        "Algorithm": algorithm_name,
                        "Test Case Size": size,
                        "Test Case ID": f"{size}{i}",
                        "Time Taken (s)": time_taken,
                        "Correctness": correctness,
                        "Deviation": deviation,
                    }
                )

                # Update cumulative time stats
                if time_taken != -1:  # Exclude timeouts
                    time_stats[size]["total_time"] += time_taken
                    time_stats[size]["count"] += 1

                # Update cumulative correctness
                if correctness != -1:  # Exclude timeouts
                    correctness_stats[size]["correctness"] += correctness
                    correctness_stats[size]["count"] += 1

                # Save best plan for this size and algorithm
                if (
                    size not in best_plans[algorithm_name]
                    or correctness > best_plans[algorithm_name][size]["correctness"]
                ):
                    best_plans[algorithm_name][size] = {
                        "num_of_vehicles": N,
                        "time_taken": time_taken,
                        "plans": plans,
                        "correctness": correctness,
                        "test_case_id": f"{size}{i}",
                    }

        # Calculate and save average time for each size
        for size in time_stats:
            total_time = time_stats[size]["total_time"]
            count = time_stats[size]["count"]
            avg_time = total_time / count if count > 0 else 0

            if size in best_plans[algorithm_name]:
                best_plans[algorithm_name][size]["avg_time_taken"] = avg_time

        # Calculate and save avg correctness for each size
        for size in correctness_stats:
            correctness = correctness_stats[size]["correctness"]
            count = correctness_stats[size]["count"]
            avg_correctness = correctness / count if count > 0 else 0

            if size in best_plans[algorithm_name]:
                best_plans[algorithm_name][size]["avg_correctness"] = avg_correctness

    # Save best plans to files
    for algorithm_name, size_data in best_plans.items():
        file_path = os.path.join(output_dir, f"{algorithm_name}_best_plans.txt")
        with open(file_path, "w") as f:
            for size, data in size_data.items():
                f.write(f"{size}\n")
                f.write(f"{data['test_case_id']}\n")
                f.write(f"{data['num_of_vehicles']}\n")
                f.write(f"{data['time_taken']}\n")
                f.write(f"{data['avg_time_taken']}\n")
                f.write(f"{data['correctness']}\n")
                f.write(f"{data['avg_correctness']}\n")
                for plan in data["plans"]:
                    f.write(f"{plan[0]}\n")  # Number of locations in the plan
                    if (isinstance(plan[1], str)):
                        f.write(plan[1].strip() + "\n")  # Route plan
                    else:
                        f.write(" ".join(map(str, plan[1])).strip()+ "\n")  # Route plan
                f.write("\n")

    # Export results to CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "Algorithm",
                "Test Case Size",
                "Test Case ID",
                "Time Taken (s)",
                "Correctness",
                "Deviation",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


# Main function to run the benchmarking
def main():
    test_case_dir = "test_cases"
    solution_dir = "solutions"
    output_csv = "benchmark_results.csv"
    output_dir = "best_plans"

    # Define algorithms to benchmark
    algorithms = {
        "CP": solveCP,
        "LP": solveLP,
        "Greedy": solveGreedy,
        "Genetic": solveGA,
        "LocalSearch": solveLS,
        "PSO": solvePSO,
        "Simulated Annealing": solveSA,
        "TabuSearch": solveTabu,
        "CWS": solve_vrp_clarke_wright,
        "AntColony": solveAntColony,
    }

    # Run benchmarking and export results
    benchmark_and_export(
        test_case_dir, solution_dir, algorithms, output_csv, output_dir
    )

if __name__ == "__main__":
    main()
