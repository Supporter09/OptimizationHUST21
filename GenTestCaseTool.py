import random
import os
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from algorithm.CP_model1 import solveCP

# Test case generation
def generate_test_case(n, k):
    """Generates a symmetric random distance matrix for VRP."""
    dis_matrix = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            distance = random.randint(1, 100)
            dis_matrix[i][j] = distance
            dis_matrix[j][i] = distance  # Ensure symmetry
    return n, k, dis_matrix

# Save test cases to .txt files
def save_test_case(n, k, distance_matrix, filename):
    with open(filename, "w") as f:
        f.write(f"{n} {k}\n")
        for row in distance_matrix:
            f.write(" ".join(map(str, row)) + "\n")

# Solve test cases using constraint programming
def solve_test_case(input_file, output_file):
    # Read test case from file
    with open(input_file, "r") as f:
        lines = f.readlines()
    n, k = map(int, lines[0].split())
    distance_matrix = [list(map(int, line.split())) for line in lines[1:]]
    plans, max_route_distance = solveCP(n, k, distance_matrix)

    # Write solution to output file
    with open(output_file, "w") as f:
        if max_route_distance > -1:
            # Number of vehicles
            f.write(f"{k}\n")

            # Print route of each vehicle
            for plan in plans:
                f.write(f"{plan[0]}\n")
                f.write("".join(map(str, plan[1])) + "\n")

            # Take longest_track as base to calculate correctness when using other model
            f.write(f"{max_route_distance}\n")
        else:
            f.write("No solution found\n")

# Main logic to generate test cases and solutions
def main():
    test_sizes = {
        "small": (10, 3),
        "large": (100, 10),
        # "very-large": (200, 10),
        # "enormous": (300, 20),
    }
    num_cases = 10

    # Create directories for inputs and outputs
    os.makedirs("test_cases", exist_ok=True)
    os.makedirs("solutions", exist_ok=True)

    for size, (n, k) in test_sizes.items():
        for i in range(1, num_cases + 1):
            # Generate test case
            n, k, distance_matrix = generate_test_case(n, k)
            test_case_file = f"test_cases/{size}{i}.txt"
            solution_file = f"solutions/{size}_answer{i}.txt"

            # Save test case
            save_test_case(n, k, distance_matrix, test_case_file)

            # Solve and save solution
            solve_test_case(test_case_file, solution_file)

if __name__ == "__main__":
    main()
