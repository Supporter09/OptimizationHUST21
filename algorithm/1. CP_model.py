"""Simple Vehicles Routing Problem (VRP).

   This is a sample using the routing library python wrapper to solve a VRP
   problem.
   A description of the problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.

   Distances are in meters.
"""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


# def create_data_model():
#     """Stores the data for the problem."""
#     data = {}
#     data["distance_matrix"] = [
#         # fmt: off
#       [0, 9, 9, 9, 7, 2, 9],
#         [9, 0, 3, 0, 2, 8, 1],
#         [9, 3, 0, 3, 4, 7, 4],
#         [9, 0, 3, 0, 2, 8, 1],
#         [7, 2, 4, 2, 0, 6, 2],
#         [2, 8, 7, 8, 6, 0, 8],
#         [9, 1, 4, 1, 2, 8, 0]
#         # fmt: on
#     ]
#     data["num_vehicles"] = 2
#     data["depot"] = 0
#     data["num_package"] = 6
#     return data

def read_input():
    # Đọc giá trị N và K
    N, K = map(int, input().split())
    
    # Đọc ma trận khoảng cách
    distance_matrix = []
    for _ in range(N + 1):
        distance_matrix.append(list(map(int, input().split())))
    '''N = 6
    K = 2
    distance_matrix = [
        [0, 9, 9, 9, 7, 2, 9],
        [9, 0, 3, 0, 2, 8, 1],
        [9, 3, 0, 3, 4, 7, 4],
        [9, 0, 3, 0, 2, 8, 1],
        [7, 2, 4, 2, 0, 6, 2],
        [2, 8, 7, 8, 6, 0, 8],
        [9, 1, 4, 1, 2, 8, 0]
    ]'''
    
    '''# Chuyển khoảng cách từ vị trí bất kì trở lại điểm ban đầu (0) thành 0:
    for i in range(len(distance_matrix)):
        distance_matrix[i][0] = 0'''
    for i in range(len(distance_matrix)):
        distance_matrix[i][0] = 0
    data = {}
    data["num_vehicles"] = K
    data["depot"] = 0
    data["distance_matrix"] = distance_matrix
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()}")
    print(data["num_vehicles"])
    max_route_distance = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        # plan_output = f"Route for vehicle {vehicle_id}:\n"
        track = 0
        plan_output = ""
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += f"{manager.IndexToNode(index)} "
            track += 1
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        # plan_output += f"{manager.IndexToNode(index)}\n"
        # plan_output += f"Distance of the route: {route_distance}m\n"
        print(track)
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print(f"Maximum of the route distances: {max_route_distance}m")



def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    # data = create_data_model()
    data = read_input()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        30000000000000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)
        # print(data)
    else:
        print("No solution found !")


if __name__ == "__main__":
    main()