# GREEDY
import sys

INF = int(1e9 + 7)
MAX_N = 1000 + 1
MAX_K = 100

class Truck:
    def __init__(self):
        self.route = [0]
        self.distance = 0

# Khởi tạo ma trận thời gian và các biến
distance_matrix = [[0] * MAX_N for _ in range(MAX_N)]
N = 0
K = 0
visited = [False] * MAX_N
trucks = [Truck() for _ in range(MAX_K)]

# Hàm nhập dữ liệu
def import_data():
    global N, K
    N, K = map(int, input().split())
    visited[0] = True
    for i in range(1, N + 1):
        visited[i] = False

    for i in range(N + 1):
        distance_matrix[i] = list(map(int, input().split()))

# Tính thời gian chạy cho một lộ trình
def calc_distance(route) -> int:
    _distance = 0
    for i in range(1, len(route)):
        _distance += distance_matrix[route[i - 1]][route[i]]
    return _distance

# Giải bài toán
def solve():
    for i in range(1, N + 1):
        best_pos = -1
        best_distance = INF
        best_node = -1

        for node in range(1, N + 1):
            if visited[node]:
                continue

            for pos in range(K):
                _distance = trucks[pos].distance + distance_matrix[trucks[pos].route[-1]][node]
                if _distance < best_distance:
                    best_pos = pos
                    best_distance = _distance
                    best_node = node

        trucks[best_pos].route.append(best_node)
        trucks[best_pos].distance = best_distance
        visited[best_node] = True

# In kết quả
def print_sol():
    print(K)
    for pos in range(K):
        print(len(trucks[pos].route))
        print(" ".join(map(str, trucks[pos].route)))

if __name__ == "__main__":
    import_data()

    # N = 6
    # K = 2
    # distance_matrix = [
    #     [0, 9, 9, 9, 7, 2, 9],
    #     [9, 0, 3, 0, 2, 8, 1],
    #     [9, 3, 0, 3, 4, 7, 4],
    #     [9, 0, 3, 0, 2, 8, 1],
    #     [7, 2, 4, 2, 0, 6, 2],
    #     [2, 8, 7, 8, 6, 0, 8],
    #     [9, 1, 4, 1, 2, 8, 0]
    # ]
    solve()
    print_sol()