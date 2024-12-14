from ortools.linear_solver import pywraplp

# Đọc giá trị N và K
N, K = map(int, input().split())

# Đọc ma trận khoảng cách
distance_matrix = []
for _ in range(N + 1):
    distance_matrix.append(list(map(int, input().split())))

# Chuyển khoảng cách từ vị trí bất kì trở lại điểm ban đầu (0) thành 0:
for i in range(len(distance_matrix)):
    distance_matrix[i][0] = 0

# Khởi tạo solver
solver = pywraplp.Solver.CreateSolver('CBC')

# Biến quyết định x[i][j][k]
x = {}
for i in range(N + 1):
    for j in range(N + 1):
        if i != j:
            for k in range(K):
                x[i, j, k] = solver.BoolVar(f'x[{i},{j},{k}]')

# Biến mục tiêu z: quãng đường dài nhất
z = solver.NumVar(0, solver.infinity(), 'z')

# Biến phụ u[i, k] để loại bỏ subtour
u = {}
for i in range(1, N + 1):
    for k in range(K):
        u[i, k] = solver.NumVar(0, N, f'u[{i},{k}]')

# Hàm mục tiêu: Tối thiểu hóa z
solver.Minimize(z)

# Ràng buộc 1: Mỗi điểm chỉ được ghé qua một lần
for i in range(1, N + 1):
    solver.Add(solver.Sum(x[i, j, k] for j in range(N + 1) for k in range(K) if i != j) == 1)

# Ràng buộc 2: Mỗi xe bắt đầu từ kho
for k in range(K):
    solver.Add(solver.Sum(x[0, j, k] for j in range(1, N + 1)) == 1)

# Ràng buộc 3: Liên tục chuyến đi (xe phải rời khỏi điểm nếu đi vào)
for k in range(K):
    for i in range(1, N + 1):
        solver.Add(solver.Sum(x[i, j, k] for j in range(N + 1) if i != j) ==
                   solver.Sum(x[j, i, k] for j in range(N + 1) if i != j))

# Ràng buộc 4: Tổng quãng đường của mỗi xe phải <= z
for k in range(K):
    solver.Add(solver.Sum(distance_matrix[i][j] * x[i, j, k]
                          for i in range(N + 1) for j in range(N + 1) if i != j) <= z)

# Ràng buộc 5: Loại bỏ chu trình con (Subtour Elimination)
for k in range(K):
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if i != j:
                solver.Add(u[i, k] - u[j, k] + (N * x[i, j, k]) <= N - 1)

# Hàm tính tổng cost
def calculate_cost():
    total_cost = 0
    for k in range(K):
        current_location = 0  # Bắt đầu từ kho
        while True:
            next_location = None
            for j in range(N + 1):
                if current_location != j and x[current_location, j, k].solution_value() > 0.5:
                    total_cost += distance_matrix[current_location][j]  # Cộng chi phí vào tổng
                    next_location = j
                    break
            if next_location is None or next_location == 0:
                break
            current_location = next_location
    return total_cost

# Giải bài toán
status = solver.Solve()

# In kết quả
if status == pywraplp.Solver.OPTIMAL:
    # Tính tổng chi phí (cost)
    total_cost = calculate_cost()
    print(f'Total cost: {total_cost}')  # In tổng chi phí

    #    # Uncomment the follow code to print out route
    # print(K)  # Line 1: Số lượng xe
    # for k in range(K):
    #     route = []
    #     current_location = 0  # Bắt đầu từ kho
    #     while True:
    #         next_location = None
    #         for j in range(N + 1):
    #             if current_location != j and x[current_location, j, k].solution_value() > 0.5:
    #                 route.append(j)
    #                 next_location = j
    #                 break
    #         if next_location is None or next_location == 0:
    #             break
    #         current_location = next_location
    #     print(len(route))  # Line 2 * k: Số điểm xe đi qua
    #     print('0 ' + ' '.join(map(str, route[:-1])))  # Line 2 * k + 1: Các điểm mà xe đi qua
else:
    print('The problem does not have an optimal solution.')
