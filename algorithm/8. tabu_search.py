import random
import time
import logging
import copy

random.seed(17042005)
start_time = time.time()
time_limit = 30



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


class TabuSolver(Solver):
    def __init__(self, file=""):
        super().__init__(file)
        # Tabu search specific parameters
        self.tabu_list = set()  # Change to set for faster lookup
        self.tabu_tenure = max(3, self.N // 20)  # Reduced tabu tenure
        self.max_iterations = 50  # Reduced iterations
        self.best_solution_cost = float('inf')
        
    def is_tabu(self, move):
        """
        Check if a move is in the tabu list
        Move is a hashable representation
        """
        return move in self.tabu_list
    
    def add_to_tabu_list(self, move):
        """
        Add a move to the tabu list and manage its size
        """
        self.tabu_list.add(move)
        # If tabu list is too large, remove oldest move
        if len(self.tabu_list) > self.tabu_tenure:
            # Convert to list, remove first item, convert back to set
            self.tabu_list = set(list(self.tabu_list)[1:])
    
    def generate_limited_neighborhood(self, max_moves=20):
        """
        Generate a limited set of moves to reduce computation time
        """
        moves = []
        # Randomly select trucks to reduce search space
        trucks_to_consider = random.sample(range(self.K), min(self.K, 3))
        
        for from_truck in trucks_to_consider:
            # Limit nodes to move per truck
            nodes_to_move = random.sample(
                self.trucks[from_truck].route[1:], 
                min(len(self.trucks[from_truck].route)-1, 5)
            )
            
            for node in nodes_to_move:
                # Limit destination trucks
                dest_trucks = random.sample(
                    [t for t in range(self.K) if t != from_truck], 
                    min(self.K-1, 3)
                )
                
                for to_truck in dest_trucks:
                    moves.append((from_truck, to_truck, node))
                    if len(moves) >= max_moves:
                        return moves
        
        return moves
    
    def fast_evaluate_move(self, from_truck, to_truck, node):
        """
        Faster move evaluation with approximation
        """
        # Quick cost estimation to reduce computation
        from_route = self.trucks[from_truck].route[:]
        from_route.remove(node)
        from_route_cost = self.route_cost(from_route)
        
        to_route = self.trucks[to_truck].route[:]
        
        # Approximate best insertion point
        best_index = len(to_route) // 2  # Middle of the route as default
        to_route.insert(best_index, node)
        to_route_cost = self.route_cost(to_route)
        
        max_route_cost_before = max(self.trucks[from_truck].cost, self.trucks[to_truck].cost)
        max_route_cost_after = max(from_route_cost, to_route_cost)
        
        return {
            'from_truck': from_truck,
            'to_truck': to_truck,
            'node': node,
            'insert_index': best_index,
            'cost_improvement': max_route_cost_before - max_route_cost_after
        }
    
    def tabu_search(self):
        """
        Optimized Tabu Search with reduced computational complexity
        """
        # Reset tabu list and iteration counter
        self.tabu_list = set()
        iterations_without_improvement = 0
        
        # Tracking best solution
        self.best_solution_cost = max(truck.cost for truck in self.trucks)
        self.best_routes = [truck.route[:] for truck in self.trucks]
        
        # Reduce computation time with limited iterations and move generation
        while (iterations_without_improvement < self.max_iterations and 
               time.time() - start_time < time_limit):
            
            # Generate a limited set of moves
            candidate_moves = self.generate_limited_neighborhood()
            
            # Find best move with early stopping
            best_move = None
            best_move_cost_improvement = float('-inf')
            
            for move in candidate_moves:
                from_truck, to_truck, node = move
                
                # Use faster move evaluation
                move_evaluation = self.fast_evaluate_move(from_truck, to_truck, node)
                
                # Aspiration criteria with early exit
                if (not self.is_tabu(move) or 
                    move_evaluation['cost_improvement'] > 0):
                    
                    if best_move is None or move_evaluation['cost_improvement'] > best_move_cost_improvement:
                        best_move = move_evaluation
                        best_move_cost_improvement = move_evaluation['cost_improvement']
                        
                        # Quick exit if a good move is found
                        if best_move_cost_improvement > 0:
                            break
            
            # Apply the best move if found
            if best_move and best_move_cost_improvement > 0:
                # Remove node from source truck
                self.trucks[best_move['from_truck']].route.remove(best_move['node'])
                
                # Insert node into destination truck
                self.trucks[best_move['to_truck']].route.insert(
                    best_move['insert_index'], 
                    best_move['node']
                )
                
                # Update route costs (approximate)
                self.trucks[best_move['from_truck']].cost = self.route_cost(
                    self.trucks[best_move['from_truck']].route
                )
                self.trucks[best_move['to_truck']].cost = self.route_cost(
                    self.trucks[best_move['to_truck']].route
                )
                
                # Add move to tabu list
                self.add_to_tabu_list((
                    best_move['from_truck'], 
                    best_move['to_truck'], 
                    best_move['node']
                ))
                
                # Update best solution
                current_solution_cost = max(truck.cost for truck in self.trucks)
                if current_solution_cost < self.best_solution_cost:
                    self.best_solution_cost = current_solution_cost
                    self.best_routes = [truck.route[:] for truck in self.trucks]
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
            else:
                # No improvement found
                iterations_without_improvement += 1
        
        print(f'Tabu Search completed. Best solution cost: {self.best_solution_cost}')

    def solve(self):
        """
        Override the original solve method with Tabu Search
        """
        # Initial construction phase (similar to original solver)
        if self.N <= 200:
            max_attemp = 4
            self.reset()
            
            # Batch processing of requests
            for _ in range(min(10, self.N)):
                self.reqs = random.sample(self.origin_reqs, self.N//10)
                for j in self.reqs:
                    self.origin_reqs.remove(j)
                self.greedy()
            
            # Process any remaining requests
            self.reqs = self.origin_reqs
            self.greedy()
        else:
            # For larger problems, do broader sampling
            n = 10
            for _ in range(n):
                self.reqs = random.sample(self.origin_reqs, self.N//n)
                for j in self.reqs:
                    self.origin_reqs.remove(j)
                self.greedy()
            
            # Process remaining requests
            self.reqs = self.origin_reqs
            self.greedy()
        
        # Apply Tabu Search
        self.tabu_search()

def main():
    inp_file = ""
    out_file = ""

    solver = TabuSolver(inp_file)
    solver.solve()
    solver.write(out_file)

if __name__ == "__main__":
    main()