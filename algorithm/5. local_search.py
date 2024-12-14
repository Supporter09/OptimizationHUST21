# LOCAL SEARCH
import random
import time


time_limit = 8

start_time = time.time()

SEED = 20230081 # seed for random number generator for reproducibility

random.seed(SEED)


class Truck:
    def __init__(self, idx):
        self.idx = idx
        self.route = [0]
        self.cost = 0

class Solver:
    """
    A solver class for the Vehicle Routing Problem (VRP) using local search and greedy algorithms.
    This class implements a solution for the VRP where K trucks need to visit N nodes (requests)
    while minimizing the maximum route cost among all trucks. The solution combines greedy
    initialization with local search improvement.
    Attributes:
        N (int): Number of requests/nodes to visit
        K (int): Number of available trucks
        distance_matrix (list): Matrix containing distances between nodes
        trucks (list): List of Truck objects representing each truck's route
        combinations (list): Cache for storing best insertion combinations
        origin_reqs (list): List of original requests/nodes
        attemp (int): Counter for unsuccessful improvement attempts
        prev_truck (int): Index of previously modified truck
        best_routes (list): Storage for best solution found
    Methods:
        reset(): Reinitializes the solver's state
        read(file): Reads problem instance from file or standard input
        greedy(): Constructs initial solution using greedy approach
        best_insert_combination(): Finds best insertion position for requests
        insert_cost(truck_idx, route_idx, node): Calculates cost of inserting node
        route_cost(route): Calculates total cost of a route
        local_search(): Performs local search improvement
        solve(): Main solving method combining initialization and improvement
        write(file): Outputs solution to file or standard output
    The solver uses different strategies based on problem size:
    - For N <= 200: Multiple restarts with partial solutions and limited local search
    - For N > 200: Single construction with time-limited local search
    """

    def __init__(self, file = ""):
        self.read(file)
        self.reset()
        self.prev_truck = -1
    
    def reset(self):
        self.trucks = [Truck(i) for i in range(self.K)]

        self.combinations = [None for _ in range(self.K)]

        self.origin_reqs = [i for i in range(1, self.N+1)]
        self.attemp = 0

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

    def greedy(self):
        """
        Implements a greedy algorithm for vehicle routing by iteratively inserting requests into truck routes.
        The algorithm works as follows:
            1. While there are unassigned requests:
                - Finds best possible insertion combination for remaining requests
                - Inserts selected request into chosen truck's route at optimal position 
                - Removes assigned request from pool of unassigned requests
                - Updates combinations by removing entries for assigned request and modified truck
                - Updates cost for affected truck
        Parameters:
            None
        Returns:
            None
        Code explanation:
            while self.reqs: # Loop continues while there are unassigned requests
                combination = self.best_insert_combination() # Get best insertion combination based on cost
                # Extract details from best combination
                truck_idx = combination['truck_idx']  # Index of selected truck
                route_idx = combination['idx']        # Position to insert in route
                req = combination['req']              # Request to be inserted
                cost = combination['cost']            # New cost after insertion
                self.trucks[truck_idx].route.insert(route_idx, req) # Insert request into truck's route
                self.reqs.remove(req) # Remove assigned request from unassigned pool
                # Update combinations by removing entries for:
                # 1. The assigned request
                # 2. The truck that received a new request
                self.trucks[truck_idx].cost = cost # Update cost for modified truck
        """
        while self.reqs:
            combination = self.best_insert_combination()

            truck_idx = combination['truck_idx']
            route_idx = combination['idx']
            req = combination['req']
            cost = combination['cost']

            self.trucks[truck_idx].route.insert(route_idx, req)
            self.reqs.remove(req)
            
            for i in range(self.K):
                if self.combinations[i]['req'] == req or self.combinations[i]['truck_idx'] == truck_idx:
                    self.combinations[i] = None

            self.trucks[truck_idx].cost = cost
        

    def best_insert_combination(self):
        # Iterate through K trucks
        for i in range(self.K):
            
            # Check if there's no combination assigned for this truck
            if self.combinations[i] == None:
                min_cost = float('inf')  # Initialize minimum cost as infinity
                results = []  # List to store combinations with minimum cost
                
                # Try inserting at each possible position in the route
                for j in range(1, len(self.trucks[i].route)+1):
                    # Find request with minimum insertion cost at position j
                    req = min(self.reqs, key=lambda x: self.insert_cost(i, j, x))
                    # Calculate the cost for this request
                    current_cost = self.insert_cost(i, j, req)
                    
                    # If we found another combination with same minimum cost
                    if current_cost == min_cost:
                        results.append({
                            'req': req,
                            'idx': j,
                        })
                    
                    # If we found a new minimum cost
                    if current_cost < min_cost:
                        min_cost = current_cost
                        results = []  # Clear previous results
                        results.append({
                            'req': req,
                            'idx': j,
                        })
                
                # Randomly select one combination from those with minimum cost
                result = random.choice(results)
                
                # Extract values from chosen result
                route_idx = result['idx']
                req = result['req']
                
                # Calculate final insertion cost
                min_cost = self.insert_cost(i, route_idx, req)
                
                # Create combination dictionary
                combination = {
                    'req': req,
                    'truck_idx': i,
                    'idx': route_idx,
                    'cost': min_cost
                }
                # Store combination for this truck
                self.combinations[i] = combination
        
        # Return combination with overall minimum cost across all trucks
        return min(self.combinations, key=lambda x: x['cost'])

    def insert_cost(self, truck_idx, route_idx, node):
        if route_idx <= 0: # cannot happend but just in case
            raise ValueError("route_idx cannot be <=0")

        if (len(self.trucks[truck_idx].route) < route_idx): # just in case
            raise ValueError("route_idx cannot be greater than the length of the route")
        

        prev = self.trucks[truck_idx].route[route_idx-1]

        if (len(self.trucks[truck_idx].route) == route_idx): # if the node is inserted at the end of the route
            return self.trucks[truck_idx].cost + self.distance_matrix[prev][node]
        else: # if the node is inserted in the middle of the route
            current = self.trucks[truck_idx].route[route_idx]
            return self.trucks[truck_idx].cost - self.distance_matrix[prev][current] + self.distance_matrix[prev][node] + self.distance_matrix[node][current]

    def route_cost(self, route):
        ret = 0
        for i in range(1, len(route)):
            ret += self.distance_matrix[route[i-1]][route[i]]
        return ret

    def local_search(self):
        # Randomly select a truck index from 0 to K-1
        random_truck_idx = random.choice([i for i in range(self.K)])

        # Reset combinations array if we're looking at a different truck than last time
        if self.prev_truck != random_truck_idx:
            self.combinations = [None for _ in range(self.K)]
        
        # Update the previous truck index
        self.prev_truck = random_truck_idx

        # Get the selected truck object
        current_truck = self.trucks[random_truck_idx]

        # Add all nodes except depot (index 0) from current truck route to reqs list
        self.reqs.extend(current_truck.route[1:])

        # Try removing each node from the route
        for node_idx in range(1, len(current_truck.route)):
            # Create a copy of current route
            temp = current_truck.route[:]
            # Remove the node at current index
            temp.remove(current_truck.route[node_idx])

            # If removing the node doesn't improve cost, remove it from reqs
            if self.route_cost(temp) >= self.route_cost(current_truck.route):
                self.reqs.remove(current_truck.route[node_idx])
            
            # If no nodes left to consider, increment attempt counter and exit
            if not self.reqs:
                self.attemp += 1
                return
        
        # Find best possible insertion combination for remaining nodes
        combination = self.best_insert_combination()
        req = combination['req']           # Node to insert
        route_idx = combination['idx']     # Position to insert
        cost = combination['cost']         # Cost after insertion
        truck_idx = combination['truck_idx']  # Truck to insert into

        # Get current maximum and minimum costs across all trucks
        max_cost = max([x.cost for x in self.trucks])
        min_cost = min([x.cost for x in self.trucks])

        # If insertion improves cost or moves to different truck with same cost
        if cost < max_cost or (cost == max_cost and truck_idx != random_truck_idx):
            # Reset combinations that involve same request or truck
            for i in range(self.K):
                if self.combinations[i]['req'] == req or self.combinations[i]['truck_idx'] == truck_idx:
                    self.combinations[i] = None
            
            # Perform the move: insert node into new truck and remove from old truck
            self.trucks[truck_idx].route.insert(route_idx, req)
            self.trucks[random_truck_idx].route.remove(req)

            # Update costs for both trucks
            self.trucks[truck_idx].cost = cost
            self.trucks[random_truck_idx].cost = self.route_cost(self.trucks[random_truck_idx].route)

            # Calculate new maximum and minimum costs
            new_max_cost = max([x.cost for x in self.trucks])
            new_min_cost = min([x.cost for x in self.trucks])

            # If costs haven't changed, increment attempt counter
            if max_cost == new_max_cost and min_cost == new_min_cost:
                self.attemp += 1
            else:
                # Reset attempt counter if costs have changed
                self.attemp = 0

            # Clear the requests list
            self.reqs.clear()
        else:
            # If no improvement possible, clear requests and increment attempt counter
            self.reqs.clear()
            self.attemp += 1


    def solve(self):
        """Solves the capacitated vehicle routing problem using a hybrid approach of greedy and local search.
        The algorithm uses different strategies based on the number of nodes:
        - For N <= 200: Uses multiple random restarts with limited local search attempts
        - For N > 200: Uses a single run with time-limited local search
        The solve process involves:
        1. Breaking down the problem into smaller sub-problems
        2. Solving each sub-problem using greedy algorithm
        3. Applying local search to improve the solution 
        Returns:
            list: Best routes found for all trucks, where each route is a sequence of customer nodes
        Algorithm Steps:
            For N <= 200:
                1. Run 20 iterations of:
                    - Reset all data structures
                    - Split problem into n sub-problems (n = min(10, N))
                    - Solve each sub-problem with greedy algorithm
                    - Apply local search up to max_attemp times
                    - Track best solution found
            For N > 200:
                1. Split problem into 10 sub-problems
                2. Solve each sub-problem with greedy algorithm
                3. Apply local search until time limit reached
        Example:
            solver = VRPSolver(...)
            solver.solve()
            print(solver.best_routes)
        """

        # If Node < 200:
        # Branch for smaller problem sizes (N <= 200)
        if self.N <= 200:
            max_attemp = 5  # Maximum attempts for local search
            max_cost = float('inf')  # Track best solution cost
            
            # Try 20 different initial solutions
            for _ in range(20):
                self.reset()  # Reset all trucks and requests
                
                n = min(10, self.N)  # Number of chunks to split requests into
                
                # Split requests into chunks and solve each chunk greedily
                for _ in range(n):
                    # Randomly sample subset of requests
                    self.reqs = random.sample(self.origin_reqs, self.N//n)
                    # Remove selected requests from original pool
                    for j in self.reqs:
                        self.origin_reqs.remove(j)
                    self.greedy()  # Apply greedy algorithm to current chunk
                
                # Handle remaining requests
                self.reqs = self.origin_reqs
                self.greedy()

                # Apply local search until max attempts reached
                while True:
                    if self.attemp >= max_attemp:
                        break
                    self.local_search()

                # Update best solution if current is better
                current_max_cost = max([x.cost for x in self.trucks])
                if current_max_cost < max_cost:
                    max_cost = current_max_cost
                    self.best_routes = [x.route for x in self.trucks]

        # Branch for larger problem sizes (N > 200)
        else:
            n = 10  # Fixed number of chunks
            
            # Split and solve chunks greedily
            for _ in range(n):
                self.reqs = random.sample(self.origin_reqs, self.N//n)
                for j in self.reqs:
                    self.origin_reqs.remove(j)
                self.greedy()
            
            # Handle remaining requests
            self.reqs = self.origin_reqs
            self.greedy()
            
            # Apply local search until time limit reached
            while time.time() - start_time < time_limit:
                self.local_search()
            
            # Save final solution
            self.best_routes = [x.route for x in self.trucks]
            
            
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

def main():
    inp_file = ""
    out_file = ""

    solver = Solver(inp_file)

    solver.solve()

    solver.write(out_file)


if __name__ == "__main__":
    main()