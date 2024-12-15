
import random
import time


time_limit = 5

SEED = 9042005 # seed for random number generator for reproducibility

random.seed(SEED)


class Truck:
    def __init__(self, idx):
        self.idx = idx
        self.route = [0]
        self.cost = 0

class Solver:
    def __init__(self, file = ""):
        self.read(file)
        self.reset()
        self.prev_truck = -1
        # print('Initialized Solver.')

    def reset(self):
        self.trucks = [Truck(i) for i in range(self.K)]
        self.combinations = [None for _ in range(self.K)]
        self.origin_reqs = [i for i in range(1, self.N+1)]
        self.attemp = 0
        # print('Solver reset.')

    def read(self, file):
        if file == "":
            readline_command = input
        else:
            f = open(file, 'r')
            readline_command = f.readline

        self.N, self.K = map(int, readline_command().split())
        self.distance_matrix = [list(map(int, readline_command().split())) for _ in range(self.N + 1)]

        if file != "":
            f.close()
        # print(f'Input read: N={self.N}, K={self.K}')

    def greedy(self):
        while self.reqs:
            # Get the best combination of request insertion
            combination = self.best_insert_combination()
            truck_idx = combination['truck_idx']
            route_idx = combination['idx']
            req = combination['req']
            cost = combination['cost']
            
            # Insert the request into the truck's route at the specified index
            self.trucks[truck_idx].route.insert(route_idx, req)
            # Remove the request from the list of pending requests
            self.reqs.remove(req)
            
            # Update combinations involving the inserted request or truck
            for i in range(self.K):
                if self.combinations[i]['req'] == req or self.combinations[i]['truck_idx'] == truck_idx:
                    self.combinations[i] = None
            
            # Update the cost of the truck after insertion
            self.trucks[truck_idx].cost = cost
        # print('Completed greedy construction.')

    def best_insert_combination(self):
        # Iterate over each truck
        for i in range(self.K):
            # Check if there's no existing combination for the current truck
            if self.combinations[i] == None:
                min_cost = float('inf')  # Initialize minimum cost
                results = []  # List to store potential insertions

                # Iterate over possible insertion positions in the truck's route
                for j in range(1, len(self.trucks[i].route)+1):
                    # Find the request that minimizes the insert cost at position j
                    req = min(self.reqs, key=lambda x: self.insert_cost(i, j, x))
                    current_cost = self.insert_cost(i, j, req)

                    if current_cost == min_cost:
                        # If cost equals current minimum, add to results
                        results.append({
                            'req': req,
                            'idx': j,
                        })
                    if current_cost < min_cost:
                        # Found a new minimum cost, update min_cost and reset results
                        min_cost = current_cost
                        results = [{
                            'req': req,
                            'idx': j,
                        }]

                # Randomly select one of the best insertions
                result = random.choice(results)
                route_idx = result['idx']
                req = result['req']
                min_cost = self.insert_cost(i, route_idx, req)

                # Create a combination dictionary for the insertion
                combination = {
                    'req': req,
                    'truck_idx': i,
                    'idx': route_idx,
                    'cost': min_cost
                }
                self.combinations[i] = combination  # Store the combination

        # Find the overall best combination based on cost
        best_combination = min(self.combinations, key=lambda x: x['cost'])
        # print(f'Best insert combination: {best_combination}')
        return best_combination

    def insert_cost(self, truck_idx, route_idx, node):
        # Verify that route_idx is greater than 0
        if route_idx <= 0:
            raise ValueError("route_idx cannot be <=0")
        # Ensure route_idx is within the route's length
        if len(self.trucks[truck_idx].route) < route_idx:
            raise ValueError("route_idx cannot be greater than the length of the route")
        # Get the previous node in the route
        prev = self.trucks[truck_idx].route[route_idx - 1]
        # Check if inserting at the end of the route
        if len(self.trucks[truck_idx].route) == route_idx:
            # Calculate cost by adding distance from prev to the new node
            cost = self.trucks[truck_idx].cost + self.distance_matrix[prev][node]
        else:
            # Get the current node at the insertion index
            current = self.trucks[truck_idx].route[route_idx]
            # Update cost by removing distance between prev and current
            # and adding distances from prev to new node and new node to current
            cost = self.trucks[truck_idx].cost - self.distance_matrix[prev][current] + \
                self.distance_matrix[prev][node] + self.distance_matrix[node][current]
        # print(f'Insert cost for truck {truck_idx} at position {route_idx} with node {node}: {cost}')
        return cost

    def route_cost(self, route):
        ret = 0  # Initialize total distance
        for i in range(1, len(route)):
            ret += self.distance_matrix[route[i-1]][route[i]]  # Add distance between consecutive points
        # print(f'Calculated route cost: {ret} for route {route}')
        return ret

    def aco(self):
        """
        Ant Colony Optimization (ACO) method for Vehicle Routing Problem.
        Improves the initial greedy solution through local search and route optimization.
        """
        # Increment attempt counter
        self.attemp += 1
        
        # Parameters for ACO
        num_ants = self.K  # Number of ants equals number of trucks
        alpha = 1.0  # Pheromone importance
        beta = 2.0  # Heuristic information importance
        evaporation_rate = 0.1  # Pheromone evaporation rate
        q0 = 0.9  # Probability of exploitation vs exploration
        
        # Initialize pheromone trails with a small initial value
        pheromones = [[1e-4 for _ in range(self.N + 1)] for _ in range(self.N + 1)]
        
        # Best solution tracking
        best_routes = [truck.route.copy() for truck in self.trucks]
        best_max_cost = max([truck.cost for truck in self.trucks])
        
        # Local search iterations
        for _ in range(10):  # Limit local search iterations
            # Ant path construction
            new_routes = [[] for _ in range(self.K)]
            
            for ant in range(num_ants):
                unvisited = set(range(1, self.N + 1))
                current_route = [0]  # Start from depot
                
                while unvisited:
                    # Last node in current route
                    current_node = current_route[-1]
                    
                    # Candidate selection using ACO probabilistic mechanism
                    candidates = []
                    for node in unvisited:
                        # Compute distance safely to avoid division by zero
                        distance = max(1, self.distance_matrix[current_node][node])
                        
                        # Exploitation: choose best path with probability q0
                        if random.random() < q0:
                            # Choose best candidate based on pheromone and distance
                            try:
                                score = (pheromones[current_node][node] ** alpha) / \
                                        (distance ** beta)
                            except ZeroDivisionError:
                                score = pheromones[current_node][node] ** alpha
                            candidates.append((node, score))
                        else:
                            # Exploration: probabilistic selection
                            candidates.append((node, random.random()))
                    
                    # Select next node
                    if candidates:
                        next_node = max(candidates, key=lambda x: x[1])[0]
                        current_route.append(next_node)
                        unvisited.remove(next_node)
                    else:
                        break
                
                # Ensure route ends at depot
                current_route.append(0)
                
                # Update truck route
                new_routes[ant] = current_route
            
            # Calculate route costs and update pheromones
            for i, route in enumerate(new_routes):
                route_cost = self.route_cost(route)
                
                # Prevent extremely small route costs
                route_cost = max(1, route_cost)
                
                # Update pheromones based on route quality
                for j in range(1, len(route)):
                    prev, curr = route[j-1], route[j]
                    pheromone_delta = 1.0 / route_cost
                    pheromones[prev][curr] += pheromone_delta
                    pheromones[curr][prev] += pheromone_delta
            
            # Pheromone evaporation
            for i in range(len(pheromones)):
                for j in range(len(pheromones[i])):
                    pheromones[i][j] *= (1 - evaporation_rate)
                    # Prevent pheromones from becoming too small
                    pheromones[i][j] = max(1e-4, pheromones[i][j])
            
            # Check if new solution is better
            current_max_cost = max(self.route_cost(route) for route in new_routes)
            if current_max_cost < best_max_cost:
                best_max_cost = current_max_cost
                best_routes = new_routes
                self.best_routes = best_routes
        
        # Update truck routes and costs
        for i, route in enumerate(best_routes):
            self.trucks[i].route = route
            self.trucks[i].cost = self.route_cost(route)
        
        print(f'ACO iteration completed. Best max cost: {best_max_cost}')

    def solve(self):
        # For small problem instances (N <= 200)
        if self.N <= 200:
            max_attemp = 4  # Maximum number of attempts before giving up
            max_cost = float('inf')  # Initialize best solution cost as infinity
            
            # Try 20 different initial solutions
            for _ in range(5):
                self.reset()  # Reset solver state
                n = min(10, self.N)  # Choose batch size, max 10
                
                # Split requests into batches
                for _ in range(n):
                    # Randomly sample subset of requests
                    self.reqs = random.sample(self.origin_reqs, self.N//n)
                    # print("randomly sample subset of requests:", self.reqs)
                    # Remove selected requests from original set
                    for j in self.reqs:
                        self.origin_reqs.remove(j)
                    self.greedy()  # Apply greedy algorithm to current batch
                
                # Process remaining requests
                self.reqs = self.origin_reqs
                self.greedy()

                # Apply aco until max attempts reached
                while True:
                    if self.attemp >= max_attemp:
                        break
                    self.aco()
                
                # Update best solution if current is better
                current_max_cost = max([x.cost for x in self.trucks])
                if current_max_cost < max_cost:
                    max_cost = current_max_cost
                    self.best_routes = [x.route for x in self.trucks]
                    
            print(f'Solving completed with best max cost: {max_cost}')

        # For large problem instances (N > 200)
        else:
            start_time = time.time()
            n = 10  # Fixed number of batches
            
            # Split requests into batches
            for _ in range(n):
                self.reqs = random.sample(self.origin_reqs, self.N//n)
                for j in self.reqs:
                    self.origin_reqs.remove(j)
                self.greedy()
            
            # Process remaining requests
            self.reqs = self.origin_reqs
            self.greedy()
            
            max_cost = float('inf')
            # Apply ACO until time limit reached
            while time.time() - start_time < time_limit:
                self.aco()
                current_max_cost = max([x.cost for x in self.trucks])
                max_cost = max(max_cost, current_max_cost)
                self.best_routes = [x.route for x in self.trucks]
            # Save best routes found
            self.best_routes = [x.route for x in self.trucks]
            print('Solving completed for large N, cost = ',max([x.cost for x in self.trucks]))


    def write(self, file = ""):
        ans = str(self.K) + "\n"
        for route in self.best_routes:
            ans += str(len(route)) + "\n"
            ans += " ".join(map(str, route)) + "\n"
        if file == "":
            print(ans)
        else:
            with open(file, 'w') as f:
                f.write(ans)
        # print('Solution written to ', file if file else 'stdout')

def main():
    inp_file = ""
    out_file = ""

    solver = Solver(inp_file)

    solver.solve()

    solver.write(out_file)


if __name__ == "__main__":
    main()