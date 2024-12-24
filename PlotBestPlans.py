import os
import matplotlib.pyplot as plt
import numpy as np
import itertools


def read_all_plans(output_dir):
    """
    Reads all saved plans from the best plans files for each algorithm and size.

    Args:
        output_dir (str): Directory where the best plans are saved.

    Returns:
        dict: A dictionary with structure:
            {
                "Algorithm1": {
                    "size1": {
                        "test_case_id": "test_case_id1",
                        "num_of_vehicles": num_of_vehicles,
                        "time_taken": best_time_taken,
                        "avg_time_taken": avg_time_taken,
                        "correctness": best_correctness,
                        "avg_correctness": avg_correctness,
                        "plans": [[...], [...], ...]
                    },
                    ...
                },
                ...
            }
    """
    all_plans = {}

    for filename in os.listdir(output_dir):
        if filename.endswith("_best_plans.txt"):
            algorithm_name = filename.replace("_best_plans.txt", "")
            file_path = os.path.join(output_dir, filename)
            all_plans[algorithm_name] = {}

            with open(file_path, "r") as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue

                current_size = line
                i += 1

                current_test_case_id = lines[i].strip()
                i += 1

                current_num_of_vehicles = int(lines[i].strip())
                i += 1

                current_time_taken = float(lines[i].strip())
                i += 1

                current_avg_time_taken = float(lines[i].strip())
                i += 1

                current_correctness = float(lines[i].strip())
                i += 1

                current_avg_correctness = float(lines[i].strip())
                i += 1

                current_plans = []
                while i < len(lines) and lines[i].strip():
                    num_locations = int(lines[i].strip())
                    i += 1
                    route_plan = list(map(int, lines[i].strip().split()))
                    current_plans.append(route_plan)
                    i += 1

                all_plans[algorithm_name][current_size] = {
                    "test_case_id": current_test_case_id,
                    "num_of_vehicles": current_num_of_vehicles,
                    "time_taken": current_time_taken,
                    "avg_time_taken": current_avg_time_taken,
                    "correctness": current_correctness,
                    "avg_correctness": current_avg_correctness,
                    "plans": current_plans,
                }

    return all_plans


def plotRoutes(N, K, routes, algo_name="", save_folder="route_plots"):
    color_palette = [
        "tab:red",
        "tab:blue",
        "tab:green",
        "tab:orange",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
    ]
    # Randomly assign coordinates to N nodes + depot (node 0)
    np.random.seed(34)
    coordinates = np.random.rand(N + 1, 2) * 100  # Scale coordinates to [0, 100]

    # Plot the points (nodes)
    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(coordinates):
        plt.scatter(
            x,
            y,
            s=200,
            c="black" if i == 0 else "tab:blue",
            label="Depot (0)" if i == 0 else "",
        )
        plt.text(x, y, str(i), fontsize=12, ha="center", va="center", color="white")

    # Plot the routes with arrows
    unique_colors = itertools.cycle(color_palette)  # Infinite iterator for colors
    route_colors = []  # Store colors to match legend

    for i, route in enumerate(routes):
        route_color = next(unique_colors)
        route_colors.append(route_color)

        for j in range(len(route) - 1):
            start, end = route[j], route[j + 1]
            x_start, y_start = coordinates[start]
            x_end, y_end = coordinates[end]

            # Plot a line for the route
            plt.plot([x_start, x_end], [y_start, y_end], color=route_color, linewidth=2)

            # Add an arrow to indicate direction
            plt.annotate(
                "",
                xy=(x_end, y_end),
                xycoords="data",
                xytext=(x_start, y_start),
                textcoords="data",
                arrowprops=dict(arrowstyle="->", color=route_color, lw=2),
                size=10,
            )

    # Add legend
    legend_labels = [f"Car {i+1}" for i in range(len(routes))]
    for i, label in enumerate(legend_labels):
        plt.scatter(
            [], [], color=route_colors[i], label=label
        )  # Empty scatter for legend entry

    plt.title(f"Vehicle Routing Problem Visualization {algo_name}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    # plt.show()

    # Save the plot to the folder
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{algo_name}_{N}_routes.png")
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory
    print(f"Saved plot to {save_path}")


def plot_comparison(results, output_dir):
    """
    Plots comparison between algorithms for different metrics and saves the plots.

    Args:
        results (dict): The results dictionary from `read_all_plans`.
        output_dir (str): Directory where the plots should be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = ["time_taken", "avg_time_taken", "correctness", "avg_correctness"]
    metric_titles = {
        "time_taken": "Best Time Taken (s)",
        "avg_time_taken": "Average Time Taken (s)",
        "correctness": "Best Correctness",
        "avg_correctness": "Average Correctness",
    }

    for metric in metrics:
        for size in next(iter(results.values())).keys():  # Iterate over sizes
            algorithms = [algo for algo in results.keys() if size in results[algo] and metric in results[algo][size]]
            values = [results[algo][size][metric] for algo in algorithms]

            # Plot
            plt.figure(figsize=(10, 6))
            plt.bar(algorithms, values, color="skyblue")
            plt.title(f"{metric_titles[metric]} for {size.capitalize()} Test Cases")
            plt.ylabel(metric_titles[metric])
            plt.xlabel("Algorithm")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(output_dir, f"{metric}_{size}.png")
            plt.savefig(plot_path)
            plt.close()

    print(f"Plots saved to {output_dir}")


output_dir = "best_plans"
all_plans = read_all_plans(output_dir)

for algorithm, sizes in all_plans.items():
    for size, data in sizes.items():
        print(f"Algorithm: {algorithm}, Size: {size}")
        print(f"Test Case ID: {data['test_case_id']}")
        print(f"Number of Vehicles: {data['num_of_vehicles']}")
        print("Plans:")
        for plan in data["plans"]:
            print(plan)

        plotRoutes(data['num_of_vehicles'], 0, data["plans"], algorithm)

# plot_output_dir = "comparison_plots"
# plot_comparison(all_plans, plot_output_dir)
