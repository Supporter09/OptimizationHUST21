# SIMULATED ANNEALING
import random
import time
import math
import copy

time_limit = 8

start_time = time.time()

SEED = 17042005  # Seed for random number generator for reproducibility

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

    def calculate_total_cost(self):
        total = 0
        for truck in self.trucks:
            route_cost = 0
            for i in range(1, len(truck.route)):
                route_cost += self.distance_matrix[truck.route[i - 1]][truck.route[i]]
            total += route_cost
        return total

    def get_neighbor(self):
        # Tạo một bản sao sâu của giải pháp hiện tại
        neighbor = copy.deepcopy(self)

        # Chọn ngẫu nhiên hai xe khác nhau để hoán đổi khách hàng
        if neighbor.K < 2:
            return neighbor  # Không thể hoán đổi nếu chỉ có một xe

        truck1, truck2 = random.sample(neighbor.trucks, 2)
        if len(truck1.route) > 1 and len(truck2.route) > 1:
            # Chọn ngẫu nhiên một khách hàng từ mỗi xe (trừ kho)
            customer1 = random.choice(truck1.route[1:])
            customer2 = random.choice(truck2.route[1:])
            # Hoán đổi
            idx1 = truck1.route.index(customer1)
            idx2 = truck2.route.index(customer2)
            truck1.route[idx1], truck2.route[idx2] = truck2.route[idx2], truck1.route[idx1]
            # Cập nhật chi phí
            truck1.cost = self.calculate_route_cost(truck1.route)
            truck2.cost = self.calculate_route_cost(truck2.route)

        return neighbor

    def calculate_route_cost(self, route):
        cost = 0
        for i in range(1, len(route)):
            cost += self.distance_matrix[route[i - 1]][route[i]]
        return cost

    def simulated_annealing(self):
        current_cost = self.calculate_total_cost()
        best_cost = current_cost
        self.best_trucks = [truck.copy() for truck in self.trucks]
        self.best_cost = best_cost

        # Tham số của Simulated Annealing
        T = 2000  # Nhiệt độ ban đầu
        T_min = 1e-3
        alpha = 0.995  # Hệ số giảm nhiệt độ

        while T > T_min and (time.time() - start_time) < time_limit:
            # Tạo một giải pháp lân cận
            neighbor = self.get_neighbor()
            neighbor_cost = neighbor.calculate_total_cost()

            delta = neighbor_cost - current_cost

            if delta < 0 or random.uniform(0, 1) < math.exp(-delta / T):
                # Chấp nhận giải pháp lân cận
                self.trucks = neighbor.trucks
                current_cost = neighbor_cost

                # Cập nhật giải pháp tốt nhất
                if current_cost < best_cost:
                    best_cost = current_cost
                    self.best_trucks = [truck.copy() for truck in self.trucks]
                    self.best_cost = best_cost

            # Giảm nhiệt độ
            T *= alpha

        # Đặt giải pháp tốt nhất vào trucks
        self.trucks = [truck.copy() for truck in self.best_trucks]

    def solve(self):

        if self.N <= 200:
            max_attemp = 5
            max_cost = float('inf')

            for _ in range(10):
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

            while True:
                if self.attemp >= max_attemp:
                    break
                self.simulated_annealing()
                self.attemp += 1

                # Update best solution if current is better
            current_max_cost = max([x.cost for x in self.trucks])
            if current_max_cost < max_cost:
                max_cost = current_max_cost
                self.best_routes = [x.route for x in self.trucks]

        else:
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
            
            # Apply simulated annealing until time limit reached
            while time.time() - start_time < time_limit:
                self.simulated_annealing()
            
            # Save best routes found
            self.best_routes = [x.route for x in self.trucks]

    def write(self, file=""):
        ans = str(self.K) + "\n"
        for truck in self.best_trucks:
            ans += str(len(truck.route)) + "\n"
            ans += " ".join(map(str, truck.route)) + "\n"

        if file == "":
            print(ans)
        else:
            with open(file, 'w') as f:
                f.write(ans)


def main():
    inp_file = ""  # Để trống để sử dụng dữ liệu mẫu
    out_file = ""  # Để trống để in kết quả ra màn hình

    solver = Solver(inp_file)

    solver.solve()

    solver.write(out_file)


if __name__ == "__main__":
    main()
