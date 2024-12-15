import math
from typing import List, Tuple

class Customer:
    def __init__(self, id: int, is_depot: bool = False):
        self.id = id
        self.is_depot = is_depot

    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Customer) and self.id == other.id

def read_input():
	# Đọc giá trị N và K
	N, K = map(int, input().split())
	
	# Đọc ma trận khoảng cách
	distance_matrix = []
	for _ in range(N + 1):
		distance_matrix.append(list(map(int, input().split())))
	return N, K, distance_matrix

def compute_savings(N: int, distance_matrix: List[List[float]]) -> List[Tuple[float, int, int]]:
    """
    Computes the savings for all pairs of customers.

    Args:
        N (int): Number of customers.
        distance_matrix (List[List[float]]): Distance matrix.

    Returns:
        List[Tuple[float, int, int]]: List of savings with corresponding customer pairs.
    """
    savings = []
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            s = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
            savings.append((s, i, j))
    # Sort savings in descending order
    savings.sort(reverse=True, key=lambda x: x[0])
    return savings

def find_route(routes: List[List[int]], customer: int) -> int:
    """
    Finds the index of the route that contains the given customer.

    Args:
        routes (List[List[int]]): Current list of routes.
        customer (int): Customer ID to find.

    Returns:
        int: Index of the route containing the customer, -1 if not found.
    """
    for idx, route in enumerate(routes):
        if customer in route:
            return idx
    return -1

def solve_vrp_clarke_wright(N: int, K: int, distance_matrix: List[List[float]]) -> List[List[int]]:
    """
    Solves the Vehicle Routing Problem using the Clarke-Wright Savings heuristic with balanced distribution.

    Args:
        N (int): Number of customers.
        K (int): Number of vehicles.
        distance_matrix (List[List[float]]): Distance matrix.

    Returns:
        List[List[int]]: Routes for each vehicle, each route is a list of node indices.
    """
    # Tạo danh sách khách hàng (node 1 đến N)
    customers = [Customer(i, is_depot=(i == 0)) for i in range(N + 1)]
    
    # Khởi tạo các tuyến đường ban đầu: mỗi khách hàng một tuyến đường [Depot, customer, Depot]
    routes = [[0, i, 0] for i in range(1, N + 1)]
    
    # Tính toán savings và sắp xếp chúng
    savings = compute_savings(N, distance_matrix)
    
    # Tính số khách hàng tối đa mỗi xe có thể phục vụ
    max_customers_per_vehicle = math.ceil(N / K)
    
    # Iteratively merge routes based on savings
    for s, i, j in savings:
        # Tìm tuyến đường chứa khách hàng i và j
        route_i = find_route(routes, i)
        route_j = find_route(routes, j)
        
        # Không thể kết hợp nếu i và j thuộc cùng một tuyến đường hoặc một trong hai không có
        if route_i == -1 or route_j == -1 or route_i == route_j:
            continue
        
        # Kiểm tra xem i có ở cuối tuyến đường route_i không và j có ở đầu tuyến đường route_j không
        if routes[route_i][-2] == i and routes[route_j][1] == j:
            # Kiểm tra số lượng khách hàng sau khi kết hợp
            combined_customers = len(routes[route_i]) - 2 + len(routes[route_j]) - 2  # loại bỏ depot
            if combined_customers > max_customers_per_vehicle:
                continue  # Không kết hợp nếu vượt quá giới hạn
            
            # Kết hợp hai tuyến đường
            new_route = routes[route_i][:-1] + routes[route_j][1:]
            routes.append(new_route)
            # Xóa các tuyến đường đã kết hợp (xóa từ lớn đến nhỏ để tránh sai chỉ số)
            for index in sorted([route_i, route_j], reverse=True):
                routes.pop(index)
        elif routes[route_j][-2] == j and routes[route_i][1] == i:
            # Kiểm tra số lượng khách hàng sau khi kết hợp
            combined_customers = len(routes[route_j]) - 2 + len(routes[route_i]) - 2  # loại bỏ depot
            if combined_customers > max_customers_per_vehicle:
                continue  # Không kết hợp nếu vượt quá giới hạn
            
            # Kết hợp hai tuyến đường
            new_route = routes[route_j][:-1] + routes[route_i][1:]
            routes.append(new_route)
            # Xóa các tuyến đường đã kết hợp (xóa từ lớn đến nhỏ để tránh sai chỉ số)
            for index in sorted([route_i, route_j], reverse=True):
                routes.pop(index)
    
    # Sau khi kết hợp dựa trên savings, kiểm tra số tuyến đường
    # Nếu số tuyến đường vẫn nhiều hơn K, cần tiếp tục kết hợp
    while len(routes) > K:
        best_saving = -math.inf
        best_pair = (-1, -1)
        for idx1 in range(len(routes)):
            for idx2 in range(len(routes)):
                if idx1 == idx2:
                    continue
                # Giả sử kết hợp cuối của routes[idx1] với đầu của routes[idx2]
                i = routes[idx1][-2]
                j = routes[idx2][1]
                saving = distance_matrix[0][i] + distance_matrix[0][j] - distance_matrix[i][j]
                combined_customers = len(routes[idx1]) - 2 + len(routes[idx2]) - 2
                if combined_customers <= max_customers_per_vehicle and saving > best_saving:
                    best_saving = saving
                    best_pair = (idx1, idx2)
        if best_pair == (-1, -1):
            break  # Không thể kết hợp thêm
        # Kết hợp các tuyến đường
        idx1, idx2 = best_pair
        new_route = routes[idx1][:-1] + routes[idx2][1:]
        routes.append(new_route)
        # Xóa các tuyến đường đã kết hợp
        for index in sorted([idx1, idx2], reverse=True):
            routes.pop(index)
    
    # Nếu số tuyến đường sau khi kết hợp vẫn nhiều hơn K, chúng ta sẽ phải chấp nhận vượt quá số khách hàng tối đa
    # Đây là tình huống khó xử lý trong heuristic; bạn có thể cần xem xét điều chỉnh hoặc sử dụng thuật toán khác
    
    # Nếu số tuyến đường sau khi kết hợp nhỏ hơn K, thêm các tuyến đường chỉ quay lại depot
    while len(routes) < K:
        routes.append([0, 0])
    
    return routes

def main():
    """
    Main function to execute the VRP solver using Clarke-Wright Savings Algorithm with balanced distribution.
    """
    # Đọc dữ liệu đầu vào
    N, K, distance_matrix = read_input()
    
    # Kiểm tra điều kiện khả thi
    if K > N:
        print("Số xe K không thể lớn hơn số khách hàng N.")
        exit(1)
    
    # Giải quyết VRP
    routes = solve_vrp_clarke_wright(N, K, distance_matrix)
    
    # Kiểm tra xem tất cả khách hàng đã được phục vụ chưa
    served_customers = set()
    for route in routes:
        served_customers.update(route[1:-1])  # Loại bỏ depot đầu và cuối
    if len(served_customers) != N:
        print("Không thể phục vụ tất cả khách hàng với số xe đã cho.")
        exit(1)
    
    # In kết quả
    print(K)
    for route in routes:
        # Đếm số khách hàng trong tuyến đường (không bao gồm depot cuối)
        customer_count = len(route) - 2 if len(route) > 2 else 0
        print(customer_count)
        if customer_count > 0:
            # In các nút trong tuyến đường (không bao gồm depot cuối)
            route_str = " ".join(map(str, route[:-1]))
            print(route_str)
        else:
            # Nếu không có khách hàng nào, chỉ in depot
            print(route[0])

if __name__ == "__main__":
    main()
