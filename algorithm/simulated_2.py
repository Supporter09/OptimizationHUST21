# DYNAMIC SIMULATED ANNEALING (IMPROVED INTEGRATION WITH LOCAL SEARCH)
import random
import time
import math
import copy

time_limit = 8

start_time = time.time()

SEED = 17042005  # Seed for reproducibility

random.seed(SEED)


class Truck:
    def __init__(self, idx):
        self.idx = idx
        self.route = [0]
        self.cost = 0

    def copy(self):
        new_truck = Truck(self.idx)
        new_truck.route = self.route.copy()
        new_truck.cost = self.cost
        return new_truck


class Solver:

    def __init__(self, file=""):
        if file != "":
            self.read(file)
            self.reset()
        else:
            print("Run takeInput(N,K,distance_matrix) and reset() method")
        self.best_trucks = []
        self.best_cost = float('inf')

    def reset(self):
        self.trucks = [Truck(i) for i in range(self.K)]
        self.origin_reqs = [i for i in range(1, self.N + 1)]

    def takeInput(self, N, K, distance_matrix):
        self.N = N
        self.K = K
        self.distance_matrix = distance_matrix

    def read(self, file):
        with open(file, 'r') as f:
            self.N, self.K = map(int, f.readline().split())
            self.distance_matrix = [list(map(int, f.readline().split())) for _ in range(self.N + 1)]

    def calculate_total_cost(self):
        total = 0
        for truck in self.trucks:
            total += self.calculate_route_cost(truck.route)
        return total

    def calculate_route_cost(self, route):
        cost = 0
        for i in range(1, len(route)):
            cost += self.distance_matrix[route[i - 1]][route[i]]
        return cost

    def get_neighbor(self):
        neighbor = copy.deepcopy(self)

        # Select random trucks for operation
        truck1, truck2 = random.sample(neighbor.trucks, 2)

        if len(truck1.route) > 1 and len(truck2.route) > 1:
            # Swap random customers between trucks
            customer1 = random.choice(truck1.route[1:])
            customer2 = random.choice(truck2.route[1:])

            idx1 = truck1.route.index(customer1)
            idx2 = truck2.route.index(customer2)

            truck1.route[idx1], truck2.route[idx2] = truck2.route[idx2], truck1.route[idx1]

            # Update costs
            truck1.cost = self.calculate_route_cost(truck1.route)
            truck2.cost = self.calculate_route_cost(truck2.route)

        # Add more diverse operations for smaller problems
        elif len(truck1.route) > 2:
            # Relocate a random customer within the same truck
            customer = random.choice(truck1.route[1:])
            truck1.route.remove(customer)
            new_position = random.randint(1, len(truck1.route))
            truck1.route.insert(new_position, customer)
            truck1.cost = self.calculate_route_cost(truck1.route)

        return neighbor

    def simulated_annealing(self):
        current_cost = self.calculate_total_cost()
        best_cost = current_cost
        self.best_trucks = [truck.copy() for truck in self.trucks]

        # Simulated Annealing parameters (dynamic based on N)
        if self.N <= 100:
            T = 1500  # Higher initial temperature for smaller sizes
            alpha = 0.99  # Slower cooling rate
        elif self.N <= 200:
            T = 2000
            alpha = 0.985
        else:
            T = 2500
            alpha = 0.99

        T_min = 1e-3

        while T > T_min and (time.time() - start_time) < time_limit:
            neighbor = self.get_neighbor()
            neighbor_cost = neighbor.calculate_total_cost()

            delta = neighbor_cost - current_cost

            # Debugging information
            print(f"Current Cost: {current_cost}, Neighbor Cost: {neighbor_cost}, Delta: {delta}, T: {T}")

            # Accept new solution based on probability
            if delta < 0 or random.uniform(0, 1) < math.exp(-delta / T):
                self.trucks = neighbor.trucks
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_cost = current_cost
                    self.best_trucks = [truck.copy() for truck in self.trucks]

            T *= alpha  # Reduce temperature

        self.trucks = [truck.copy() for truck in self.best_trucks]

    def solve(self):
        self.reset()
        # Start with a local search-based solution
        self.local_search_initialization()
        self.simulated_annealing()
        self.best_routes = [truck.route for truck in self.best_trucks]

    def local_search_initialization(self):
        """A simple local search to initialize the solution."""
        self.greedy_initialization()
        improvement = True
        while improvement:
            improvement = False
            for truck in self.trucks:
                for i in range(1, len(truck.route) - 1):
                    for j in range(i + 1, len(truck.route)):
                        # Try swapping nodes
                        truck.route[i], truck.route[j] = truck.route[j], truck.route[i]
                        new_cost = self.calculate_route_cost(truck.route)
                        if new_cost < truck.cost:
                            truck.cost = new_cost
                            improvement = True
                        else:
                            # Revert swap
                            truck.route[i], truck.route[j] = truck.route[j], truck.route[i]

    def greedy_initialization(self):
        for req in self.origin_reqs:
            best_cost = float('inf')
            best_truck = None
            best_position = None

            for truck in self.trucks:
                for position in range(1, len(truck.route) + 1):
                    cost = self.insert_cost(truck, position, req)
                    if cost < best_cost:
                        best_cost = cost
                        best_truck = truck
                        best_position = position

            best_truck.route.insert(best_position, req)
            best_truck.cost = self.calculate_route_cost(best_truck.route)

    def insert_cost(self, truck, position, node):
        prev = truck.route[position - 1]
        if position == len(truck.route):
            return truck.cost + self.distance_matrix[prev][node]
        next_node = truck.route[position]
        return truck.cost - self.distance_matrix[prev][next_node] + self.distance_matrix[prev][node] + self.distance_matrix[node][next_node]

    def write(self, file=""):
        result = str(self.K) + "\n"
        for truck in self.best_trucks:
            result += str(len(truck.route)) + "\n"
            result += " ".join(map(str, truck.route)) + "\n"

        if file:
            with open(file, 'w') as f:
                f.write(result)
        else:
            print(result)

    def getResult(self):
        plans = []
        max_route_distance = 0
        for route in self.best_routes:
            plans.append([len(route), route])
            if len(route) >= 2:
                tmp_distance = 0
                for i in range(len(route) - 1):
                    tmp_distance += self.distance_matrix[route[i]][route[i + 1]]
                max_route_distance = max(max_route_distance, tmp_distance)
        return plans, max_route_distance

def solveSA2(N, K, distance_matrix):
    solver = Solver()
    solver.takeInput(N, K, distance_matrix)
    solver.solve()
    return solver.getResult()

def main():
    solver = Solver()
    solver.solve()
    solver.write()

if __name__ == "__main__":
    main()
