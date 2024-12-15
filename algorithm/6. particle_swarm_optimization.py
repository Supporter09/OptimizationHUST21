# PARTICLE SWARM OPTIMIZATION - Tối ưu hóa cho VRP
import random
import time
import math
import copy
import numpy as np

# Seed for reproducibility
SEED = 20230092
random.seed(SEED)
np.random.seed(SEED)

# PSO Hyperparameters
PSO_PARAMS = {
    'num_particles': 50,      # Number of particles in the swarm
    'max_iterations': 100,    # Maximum number of iterations
    'w': 0.7,                 # Inertia weight (Not used in current implementation)
    'c1': 1.5,                # Cognitive coefficient
    'c2': 1.5,                # Social coefficient
    'time_limit': 8           # Time limit for solving (seconds)
}

class Truck:
    def __init__(self, truck_id):
        self.id = truck_id
        self.route = [0]  # Start with depot
        self.cost = 0

class Particle:
    def __init__(self, num_trucks, num_requests, distance_matrix, initial_solution=None):
        """
        Initialize a particle representing a VRP solution.
        
        Args:
            num_trucks (int): Number of trucks in the solution.
            num_requests (int): Total number of requests.
            distance_matrix (list of list of int): Distance matrix.
            initial_solution (list of list of int): Optional initial solution to assign to particle.
        """
        self.position = [[] for _ in range(num_trucks)]  # Current solution configuration
        self.velocity = set()  # Movement/modification vector
        self.best_position = None  # Personal best solution
        self.best_fitness = float('inf')  # Fitness of personal best solution
        
        # Initialize routes starting with depot (0)
        for route in self.position:
            route.append(0)
        
        if initial_solution:
            # Assign initial solution to particle's position
            for truck_idx in range(num_trucks):
                # Avoid duplicating the depot
                if len(initial_solution[truck_idx]) > 0:
                    self.position[truck_idx] = initial_solution[truck_idx].copy()
        else:
            # Randomly distribute requests to trucks
            requests = list(range(1, num_requests + 1))
            random.shuffle(requests)
            
            for req in requests:
                truck_idx = random.randint(0, num_trucks - 1)
                self.position[truck_idx].append(req)

class PSO_VRP_Solver:
    def __init__(self, file=""):
        """
        Initialize PSO solver for Vehicle Routing Problem.
        
        Args:
            file (str): Input file path.
        """
        self.read(file)
        self.reset()
        # Precompute distance matrix for faster access
        self.precompute_distances()
        # print('Initialized PSO Solver.')

    def read(self, file):
        """
        Read problem input from file or stdin
        
        Args:
            file (str): Input file path
        """
        if file == "":
            readline_command = input
        else:
            f = open(file, 'r')
            readline_command = f.readline

        self.N, self.K = map(int, readline_command().split())
        self.distance_matrix = [list(map(int, readline_command().split())) for _ in range(self.N + 1)]

        if file != "":
            f.close()

    def precompute_distances(self):
        """
        Precompute and store distances for quick access.
        """
        self.distances = np.array(self.distance_matrix)

    def reset(self):
        """
        Reset solver state.
        """
        # Initialize trucks, requests, combinations
        self.trucks = [Truck(i) for i in range(self.K)]
        self.reqs = set(range(1, self.N + 1))  # Requests labeled from 1 to N
        self.combinations = [None for _ in range(self.K)]
        
        # Initialize Greedy solution
        self.greedy()
        
        # Initialize particles with the Greedy solution
        greedy_solution = [truck.route.copy() for truck in self.trucks]
        self.particles = [Particle(self.K, self.N, self.distance_matrix, initial_solution=greedy_solution) for _ in range(PSO_PARAMS['num_particles'])]
        
        self.global_best_solution = None
        self.global_best_fitness = float('inf')
        # print('PSO Solver reset.')

    def greedy(self):
        """
        Greedy algorithm to construct an initial solution.
        Assigns requests to trucks based on the best insertion cost.
        """
        # print('Starting greedy construction.')
        while self.reqs:
            combination = self.best_insert_combination()
            if combination is None:
                # print("No valid insertions found. Exiting greedy construction.")
                break  # Avoid infinite loop if no insertions possible
            
            truck_idx = combination['truck_idx']
            route_idx = combination['idx']
            req = combination['req']
            cost = combination['cost']
            
            # Insert the request into the truck's route at the specified index
            self.trucks[truck_idx].route.insert(route_idx, req)
            # Remove the request from the set of pending requests
            self.reqs.remove(req)
            
            # Update combinations involving the inserted request or truck
            for i in range(self.K):
                if self.combinations[i] is not None:
                    if self.combinations[i]['req'] == req or self.combinations[i]['truck_idx'] == truck_idx:
                        self.combinations[i] = None
            
            # Update the cost of the truck after insertion
            self.trucks[truck_idx].cost = cost
        
        # print('Completed greedy construction.')

    def best_insert_combination(self):
        """
        Find the best insertion combination across all trucks.
        
        Returns:
            dict: The best combination with 'req', 'truck_idx', 'idx', and 'cost'.
        """
        best_combination = None
        min_cost = float('inf')
        
        for i in range(self.K):
            if self.combinations[i] is None:
                # Iterate over possible insertion positions
                for j in range(1, len(self.trucks[i].route) + 1):
                    # Find the request that minimizes the insert cost at position j
                    if not self.reqs:
                        continue
                    req = min(self.reqs, key=lambda x: self.insert_cost(i, j, x))
                    current_cost = self.insert_cost(i, j, req)
                    
                    if current_cost < min_cost:
                        min_cost = current_cost
                        best_combination = {
                            'req': req,
                            'truck_idx': i,
                            'idx': j,
                            'cost': current_cost
                        }
        
        if best_combination:
            # print(f'Best insert combination: {best_combination}')
            return best_combination
        else:
            # print("No valid insert combination found.")
            return None

    def insert_cost(self, truck_idx, route_idx, node):
        """
        Calculate the cost of inserting a node into a truck's route at a specific position.
        
        Args:
            truck_idx (int): Index of the truck.
            route_idx (int): Position in the route to insert the node.
            node (int): The node to insert.
        
        Returns:
            float: The new cost after insertion.
        """
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
        """
        Calculate total cost for a single route.
        
        Args:
            route (list): Route to calculate cost for.
        
        Returns:
            float: Total route cost.
        """
        if len(route) <= 1:
            return 0
        indices = np.array(route)
        return np.sum(self.distances[indices[:-1], indices[1:]])

    def solution_fitness(self, solution):
        """
        Calculate fitness of a solution (max route cost).
        
        Args:
            solution (list): List of truck routes.
        
        Returns:
            float: Maximum route cost across all trucks.
        """
        # Using vectorized operations for efficiency
        route_costs = [self.route_cost(route) for route in solution]
        return max(route_costs)

    def velocity_update(self, particle, global_best):
        """
        Update particle velocity using PSO update rule.
        
        Args:
            particle (Particle): Current particle.
            global_best (list): Global best solution.
        """
        cognitive_component = set()
        social_component = set()
        
        # Cognitive component: difference between personal best and current position
        if particle.best_position:
            for i in range(self.K):
                cognitive_component.update(set(particle.best_position[i]) - set(particle.position[i]))
        
        # Social component: difference between global best and current position
        if global_best:
            for i in range(self.K):
                social_component.update(set(global_best[i]) - set(particle.position[i]))
        
        # Update velocity based on cognitive and social components
        new_velocity = set()
        # Apply cognitive component
        if cognitive_component:
            cognitive_sample_size = min(len(cognitive_component), max(1, int(PSO_PARAMS['c1'] * len(cognitive_component))))
            # Convert set to list before sampling
            cognitive_moves = set(random.sample(list(cognitive_component), cognitive_sample_size))
            new_velocity.update(cognitive_moves)
        # Apply social component
        if social_component:
            social_sample_size = min(len(social_component), max(1, int(PSO_PARAMS['c2'] * len(social_component))))
            # Convert set to list before sampling
            social_moves = set(random.sample(list(social_component), social_sample_size))
            new_velocity.update(social_moves)
        
        particle.velocity = new_velocity

    def update_particle_position(self, particle):
        """
        Update particle position based on velocity.
        
        Args:
            particle (Particle): Particle to update.
        """
        if not particle.velocity:
            return
        
        # Flatten all routes to a single list for easy manipulation
        all_requests = [req for route in particle.position for req in route if req != 0]
        all_requests_set = set(all_requests)
        velocity_set = set(particle.velocity)
        
        # Requests to remove are those in velocity_set
        requests_to_remove = velocity_set & all_requests_set
        
        # Remove requests from current routes
        for i in range(self.K):
            particle.position[i] = [req for req in particle.position[i] if req not in requests_to_remove]
        
        # Reinsert the removed requests into random positions
        for req in requests_to_remove:
            truck_idx = random.randint(0, self.K - 1)
            insert_pos = random.randint(1, len(particle.position[truck_idx]))
            particle.position[truck_idx].insert(insert_pos, req)
        
        # Optionally, you can shuffle to introduce more randomness
        # for route in particle.position:
        #     random.shuffle(route[1:])  # Shuffle except depot

    def solve(self):
        """
        Solve VRP using Particle Swarm Optimization.
        """
        start_time = time.time()
        iteration = 0
        
        while iteration < PSO_PARAMS['max_iterations']:
            current_time = time.time()
            if current_time - start_time > PSO_PARAMS['time_limit']:
                # print(f'Time limit reached at iteration {iteration}.')
                break
            
            # print(f'Iteration {iteration + 1}/{PSO_PARAMS["max_iterations"]}')
            
            for idx, particle in enumerate(self.particles):
                current_fitness = self.solution_fitness(particle.position)
                
                # Update personal best
                if current_fitness < particle.best_fitness:
                    particle.best_fitness = current_fitness
                    particle.best_position = copy.deepcopy(particle.position)
                    # print(f'Particle {idx} found new personal best: {current_fitness}')
                
                # Update global best
                if current_fitness < self.global_best_fitness:
                    self.global_best_fitness = current_fitness
                    self.global_best_solution = copy.deepcopy(particle.position)
                    # print(f'New global best fitness found: {self.global_best_fitness}')
            
            # Update velocities and positions
            for particle in self.particles:
                self.velocity_update(particle, self.global_best_solution)
                self.update_particle_position(particle)
            
            iteration += 1
        
        print(f'PSO solving completed in {iteration} iterations with best fitness: {self.global_best_fitness}')

    def write(self, file=""):
        """
        Write solution to file or stdout
        
        Args:
            file (str): Output file path
        """
        ans = f"{self.K}\n"
        for route in self.global_best_solution:
            ans += f"{len(route)}\n"
            ans += " ".join(map(str, route)) + "\n"
        
        if file == "":
            print(ans)
        else:
            with open(file, 'w') as f:
                f.write(ans)

def main():
    inp_file = ""
    out_file = ""

    solver = PSO_VRP_Solver(inp_file)
    solver.solve()
    solver.write(out_file)

if __name__ == "__main__":
    main()
