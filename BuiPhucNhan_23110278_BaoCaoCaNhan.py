import pygame
import sys
from collections import deque
import heapq
import copy
import time
import random
import math

WIDTH, HEIGHT = 900, 800
GRID_SIZE = 3
TILE_SIZE = 100
GRID_WIDTH = GRID_SIZE * TILE_SIZE
GRID_HEIGHT = GRID_SIZE * TILE_SIZE
FONT_SIZE = 50
BUTTON_WIDTH = 75
BUTTON_HEIGHT = 40
BUTTON_SPACING = 15

BACKGROUND_COLOR = (245, 245, 250)
TILE_COLOR = (173, 216, 230)
TILE_BORDER = (70, 130, 180)
BUTTON_COLOR = (135, 206, 235)
BUTTON_HOVER = (100, 149, 237)
TEXT_COLOR = (34, 62, 74)
SOLVED_COLOR = (144, 238, 144)
MESSAGE_COLOR = (70, 130, 180)

STATE_AREA_WIDTH = 260
STATE_AREA_HEIGHT = HEIGHT - 100
STATE_AREA_X = WIDTH - STATE_AREA_WIDTH - 25
STATE_AREA_Y = 20
SCROLLBAR_WIDTH = 15
STATE_TILE_SIZE = 45
STATE_SPACING = 15 
TITLE_HEIGHT = 35 

INITIAL_STATE = [[1, 2, 3], [4, 5, 6], [0, 7, 8]]
GOAL_STATE = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

Q_LEARNING_EPISODES = 1000
Q_LEARNING_ALPHA = 0.1
Q_LEARNING_GAMMA = 0.9
Q_LEARNING_EPSILON = 0.1
Q_LEARNING_MAX_STEPS_PER_EPISODE = 200
Q_LEARNING_MAX_PATH_LEN = 50

def get_manhattan_distance(state):
# Tính tổng khoảng cách Manhattan từ vị trí hiện tại của các ô (trừ ô trống) đến vị trí mục tiêu trong lưới 8-puzzle. Dùng làm heuristic cho thuật toán tìm kiếm.
    distance = 0
    distance = 0
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if state[i][j] != 0:
                value = state[i][j]
                target_row = (value - 1) // GRID_SIZE
                target_col = (value - 1) % GRID_SIZE
                distance += abs(i - target_row) + abs(j - target_col)
    return distance

def print_state(state):
    for row in state:
        print(row)
    print()

def generic_bfs(start, goal, get_neighbors, is_goal, max_depth=float('inf')):
    start_time = time.perf_counter()
    queue = deque([(start, [])])      # Hàng đợi BFS, mỗi phần tử gồm (trạng thái, đường đi)
    visited = {start}                 # Tập hợp các trạng thái đã thăm để tránh lặp lại
    depth = 0                         # Độ sâu hiện tại của BFS
    while queue and depth < max_depth:
        level_size = len(queue)       # Số lượng trạng thái ở mức hiện tại (theo từng lớp)
        for _ in range(level_size):
            state, path = queue.popleft()  # Lấy trạng thái và đường đi hiện tại ra khỏi hàng đợi
            if is_goal(state, goal):       # Nếu đạt trạng thái đích thì trả về kết quả
                return path, time.perf_counter() - start_time
            for neighbor, move in get_neighbors(state):  # Duyệt các trạng thái kề
                if neighbor not in visited:              # Nếu chưa thăm thì thêm vào hàng đợi
                    queue.append((neighbor, path + [move]))
                    visited.add(neighbor)
        depth += 1
    return [], time.perf_counter() - start_time  # Không tìm thấy, trả về rỗng

def bfs_solve(start, goal):
    # Hàm sinh các trạng thái kề bằng cách di chuyển ô trống (0) lên/xuống/trái/phải
    def get_neighbors(state):
        r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if state[r][c] == 0)  # Tìm vị trí ô trống
        neighbors = []
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:  # Duyệt các hướng di chuyển
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:      # Kiểm tra hợp lệ trong lưới
                new_state = [list(row) for row in state]         # Sao chép trạng thái hiện tại
                new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]  # Đổi chỗ ô trống với ô kề
                neighbors.append((tuple(map(tuple, new_state)), (nr, nc)))  # Thêm trạng thái mới vào danh sách kề
        return neighbors

    # Kiểm tra trạng thái hiện tại có phải trạng thái đích không
    def is_goal(state, goal_tuple):
        return state == goal_tuple

    start_tuple = tuple(map(tuple, start))  # Chuyển trạng thái đầu vào thành tuple để dùng trong tập visited
    goal_tuple = tuple(map(tuple, goal))    # Chuyển trạng thái đích thành tuple

    # Gọi hàm BFS tổng quát với các hàm phụ trợ trên
    return generic_bfs(start_tuple, goal_tuple, get_neighbors, is_goal)

def dfs_solve(start, goal, depth_limit=31):
    start_time = time.perf_counter()
    stack = [(start, [], 0)]  # Stack lưu (trạng thái, đường đi, độ sâu hiện tại)
    visited_by_depth = [{} for _ in range(depth_limit + 1)]  # Đánh dấu trạng thái đã thăm theo từng độ sâu
    while stack:
        state, path, depth = stack.pop()
        if state == goal:
            return path, time.perf_counter() - start_time
        if depth < depth_limit:
            r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if state[r][c] == 0)  # Tìm vị trí ô trống
            for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:  # Duyệt các hướng di chuyển
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:  # Kiểm tra hợp lệ trong lưới
                    new_state = [row[:] for row in state]  # Sao chép trạng thái hiện tại
                    new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]  # Đổi chỗ ô trống với ô kề
                    new_tuple = tuple(map(tuple, new_state))  # Chuyển sang tuple để lưu vào visited
                    if new_tuple not in visited_by_depth[depth + 1]:  # Kiểm tra đã thăm ở độ sâu này chưa
                        stack.append((new_state, path + [(nr, nc)], depth + 1))
                        visited_by_depth[depth + 1][new_tuple] = True
    return [], time.perf_counter() - start_time

def ucs_solve(start, goal):
    start_time = time.perf_counter()
    queue = [(0, start, [])]  # Hàng đợi ưu tiên, mỗi phần tử gồm (cost, trạng thái, đường đi)
    visited = {tuple(map(tuple, start))}  # Tập hợp các trạng thái đã thăm (dạng tuple)
    while queue:
        cost, state, path = heapq.heappop(queue)  # Lấy trạng thái có cost nhỏ nhất ra khỏi hàng đợi
        if state == goal:
            return path, time.perf_counter() - start_time
        r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if state[r][c] == 0)  # Tìm vị trí ô trống
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:  # Duyệt các hướng di chuyển
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:  # Kiểm tra hợp lệ trong lưới
                new_state = [row[:] for row in state]  # Sao chép trạng thái hiện tại
                new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]  # Đổi chỗ ô trống với ô kề
                new_tuple = tuple(map(tuple, new_state))  # Chuyển sang tuple để lưu vào visited
                if new_tuple not in visited:  # Nếu trạng thái mới chưa được thăm
                    heapq.heappush(queue, (cost + 1, new_state, path + [(nr, nc)]))  # Thêm vào hàng đợi với cost tăng 1
                    visited.add(new_tuple)
    return [], time.perf_counter() - start_time  # Không tìm thấy, trả về rỗng

def greedy_solve(start, goal):
    start_time = time.perf_counter()
    queue = [(get_manhattan_distance(start), start, [])]  # Hàng đợi ưu tiên, mỗi phần tử gồm (giá trị heuristic, trạng thái, đường đi)
    visited = {tuple(map(tuple, start))}  # Tập hợp các trạng thái đã thăm (dạng tuple)
    while queue:
        _, state, path = heapq.heappop(queue)  # Lấy trạng thái có heuristic nhỏ nhất ra khỏi hàng đợi
        if state == goal:
            return path, time.perf_counter() - start_time
        r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if state[r][c] == 0)  # Tìm vị trí ô trống
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:  # Duyệt các hướng di chuyển
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:  # Kiểm tra hợp lệ trong lưới
                new_state = [row[:] for row in state]  # Sao chép trạng thái hiện tại
                new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]  # Đổi chỗ ô trống với ô kề
                new_tuple = tuple(map(tuple, new_state))  # Chuyển sang tuple để lưu vào visited
                if new_tuple not in visited:  # Nếu trạng thái mới chưa được thăm
                    heapq.heappush(queue, (get_manhattan_distance(new_state), new_state, path + [(nr, nc)]))  # Thêm vào hàng đợi với heuristic mới
                    visited.add(new_tuple)
    return [], time.perf_counter() - start_time  # Không tìm thấy, trả về rỗng

def iddfs_solve(start, goal):
    start_time = time.perf_counter()
    def dfs_limited(state, goal, depth, visited):
        if state == goal:
            return []
        if depth == 0:
            return None
        r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if state[r][c] == 0)  # Tìm vị trí ô trống
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:  # Duyệt các hướng di chuyển
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:  # Kiểm tra hợp lệ trong lưới
                new_state = [row[:] for row in state]  # Sao chép trạng thái hiện tại
                new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]  # Đổi chỗ ô trống với ô kề
                new_tuple = tuple(map(tuple, new_state))  # Chuyển sang tuple để lưu vào visited
                if new_tuple not in visited:  # Nếu trạng thái mới chưa được thăm
                    visited.add(new_tuple)
                    result = dfs_limited(new_state, goal, depth - 1, visited)  # Đệ quy với độ sâu giảm 1
                    visited.remove(new_tuple)
                    if result is not None:
                        return [(nr, nc)] + result
        return None
    for depth in range(32):  # Lặp tăng dần độ sâu giới hạn
        visited = {tuple(map(tuple, start))}  # Đánh dấu trạng thái bắt đầu đã thăm
        result = dfs_limited(start, goal, depth, visited)
        if result is not None:
            return result, time.perf_counter() - start_time
    return [], time.perf_counter() - start_time  # Không tìm thấy, trả về rỗng

def astar_solve(start, goal):
    start_time = time.perf_counter()
    queue = [(get_manhattan_distance(start), 0, start, [])]  # Hàng đợi ưu tiên, mỗi phần tử gồm (f, g, trạng thái, đường đi)
    visited = {tuple(map(tuple, start)): 0}  # Lưu trạng thái đã thăm kèm theo chi phí g nhỏ nhất
    while queue:
        f_score, g_score, state, path = heapq.heappop(queue)  # Lấy trạng thái có f nhỏ nhất ra khỏi hàng đợi
        if state == goal:
            return path, time.perf_counter() - start_time
        r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if state[r][c] == 0)  # Tìm vị trí ô trống
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:  # Duyệt các hướng di chuyển
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:  # Kiểm tra hợp lệ trong lưới
                new_state = [row[:] for row in state]  # Sao chép trạng thái hiện tại
                new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]  # Đổi chỗ ô trống với ô kề
                new_tuple = tuple(map(tuple, new_state))  # Chuyển sang tuple để lưu vào visited
                g_new = g_score + 1  # Tính chi phí mới (g)
                h_new = get_manhattan_distance(new_state)  # Tính heuristic mới (h)
                f_new = g_new + h_new  # Tổng chi phí f = g + h
                if new_tuple not in visited or g_new < visited[new_tuple]:  # Nếu chưa thăm hoặc tìm được đường đi tốt hơn
                    visited[new_tuple] = g_new
                    heapq.heappush(queue, (f_new, g_new, new_state, path + [(nr, nc)]))  # Thêm vào hàng đợi ưu tiên
    return [], time.perf_counter() - start_time  # Không tìm thấy, trả về rỗng

def idastar_solve(start, goal):
    start_time = time.perf_counter()
    def search(state, g_score, bound, path, visited):
        h_score = get_manhattan_distance(state)  # Tính heuristic (h) cho trạng thái hiện tại
        f_score = g_score + h_score  # Tổng chi phí f = g + h
        if f_score > bound:  # Nếu vượt quá ngưỡng hiện tại thì trả về
            return None, f_score
        if state == goal:
            return path, f_score
        min_bound = float('inf')  # Lưu ngưỡng nhỏ nhất cho lần lặp tiếp theo
        r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if state[r][c] == 0)  # Tìm vị trí ô trống
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:  # Duyệt các hướng di chuyển
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:  # Kiểm tra hợp lệ trong lưới
                new_state = [row[:] for row in state]  # Sao chép trạng thái hiện tại
                new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]  # Đổi chỗ ô trống với ô kề
                new_tuple = tuple(map(tuple, new_state))  # Chuyển sang tuple để lưu vào visited
                if new_tuple not in visited:  # Nếu trạng thái mới chưa được thăm
                    visited.add(new_tuple)
                    result, new_f = search(new_state, g_score + 1, bound, path + [(nr, nc)], visited)  # Đệ quy với g tăng 1
                    visited.remove(new_tuple)
                    if result is not None:
                        return result, new_f
                    min_bound = min(min_bound, new_f)  # Cập nhật ngưỡng nhỏ nhất cho lần lặp tiếp theo
        return None, min_bound
    bound = get_manhattan_distance(start)  # Khởi tạo ngưỡng ban đầu bằng heuristic của trạng thái đầu
    while True:
        visited = {tuple(map(tuple, start))}  # Đánh dấu trạng thái bắt đầu đã thăm
        result, new_bound = search(start, 0, bound, [], visited)
        if result is not None:
            return result, time.perf_counter() - start_time
        if new_bound == float('inf'):  # Nếu không còn trạng thái nào để mở rộng
            return [], time.perf_counter() - start_time
        bound = new_bound  # Cập nhật ngưỡng cho lần lặp tiếp theo

def simple_hill_climbing_solve(start, goal, max_iterations=1000):
    start_time = time.perf_counter()
    current_state = copy.deepcopy(start)
    path = []
    iterations = 0
    first_move = True
    while current_state != goal and iterations < max_iterations:
        r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if current_state[r][c] == 0)  # Tìm vị trí ô trống
        neighbors = []
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:  # Duyệt các hướng di chuyển
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:  # Kiểm tra hợp lệ trong lưới
                new_state = [row[:] for row in current_state]  # Sao chép trạng thái hiện tại
                new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]  # Đổi chỗ ô trống với ô kề
                neighbors.append((new_state, (nr, nc)))  # Thêm trạng thái mới vào danh sách neighbors
        current_h = get_manhattan_distance(current_state)  # Tính heuristic hiện tại
        if first_move:
            next_state, move = random.choice(neighbors)  # Bước đầu chọn ngẫu nhiên
            first_move = False
        else:
            better_neighbors = [(state, move) for state, move in neighbors if get_manhattan_distance(state) < current_h]  # Lọc neighbor tốt hơn
            if not better_neighbors:
                break  # Không còn neighbor tốt hơn, dừng lại (mắc kẹt local optimum)
            next_state, move = min(better_neighbors, key=lambda x: get_manhattan_distance(x[0]))  # Chọn neighbor tốt nhất
        current_state = next_state
        path.append(move)
        iterations += 1
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return (path, elapsed_time) if current_state == goal else ([], elapsed_time)

def steepest_ascent_hill_climbing_solve(start, goal, max_iterations=1000):
    start_time = time.perf_counter()
    current_state = copy.deepcopy(start)
    path = []
    iterations = 0
    while current_state != goal and iterations < max_iterations:
        r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if current_state[r][c] == 0)  # Tìm vị trí ô trống
        neighbors = []
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:  # Duyệt các hướng di chuyển
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:  # Kiểm tra hợp lệ trong lưới
                new_state = [row[:] for row in current_state]  # Sao chép trạng thái hiện tại
                new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]  # Đổi chỗ ô trống với ô kề
                neighbors.append((new_state, (nr, nc)))  # Thêm trạng thái mới vào danh sách neighbors

        current_h = get_manhattan_distance(current_state)  # Tính heuristic hiện tại
        better_neighbors = [(state, move) for state, move in neighbors if get_manhattan_distance(state) < current_h]  # Lọc neighbor tốt hơn
        if not better_neighbors:
            break  # Không còn neighbor tốt hơn, dừng lại (mắc kẹt local optimum)
        best_state, best_move = min(better_neighbors, key=lambda x: get_manhattan_distance(x[0]))  # Chọn neighbor tốt nhất
        current_state = best_state
        path.append(best_move)
        iterations += 1
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return (path, elapsed_time) if current_state == goal else ([], elapsed_time)

def stochastic_hill_climbing_solve(start, goal, max_iterations=1000):
    start_time = time.perf_counter()
    current_state = copy.deepcopy(start)
    path = []
    iterations = 0
    while current_state != goal and iterations < max_iterations:
        r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if current_state[r][c] == 0)  # Tìm vị trí ô trống
        neighbors = []
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:  # Duyệt các hướng di chuyển
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:  # Kiểm tra hợp lệ trong lưới
                new_state = [row[:] for row in current_state]  # Sao chép trạng thái hiện tại
                new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]  # Đổi chỗ ô trống với ô kề
                neighbors.append((new_state, (nr, nc)))  # Thêm trạng thái mới vào danh sách neighbors
        current_h = get_manhattan_distance(current_state)  # Tính heuristic hiện tại
        better_neighbors = [(state, move) for state, move in neighbors if get_manhattan_distance(state) <= current_h]  # Lọc neighbor tốt hơn hoặc bằng
        if not better_neighbors:
            break  # Không còn neighbor tốt hơn, dừng lại (mắc kẹt local optimum)
        next_state, move = random.choice(better_neighbors)  # Chọn ngẫu nhiên một neighbor tốt hơn hoặc bằng
        current_state = next_state
        path.append(move)
        iterations += 1
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return (path, elapsed_time) if current_state == goal else ([], elapsed_time)

def simulated_annealing_solve(start, goal, max_iterations=10000, initial_temp=1000, cooling_rate=0.99):
    start_time = time.perf_counter()
    current_state = copy.deepcopy(start)
    current_h = get_manhattan_distance(current_state)
    best_state = copy.deepcopy(current_state)
    best_h = current_h
    path = []
    temp = initial_temp
    iterations = 0
    while current_h > 0 and iterations < max_iterations and temp > 0.1:
        r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if current_state[r][c] == 0)  # Tìm vị trí ô trống
        neighbors = []
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:  # Duyệt các hướng di chuyển
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:  # Kiểm tra hợp lệ trong lưới
                new_state = [row[:] for row in current_state]  # Sao chép trạng thái hiện tại
                new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]  # Đổi chỗ ô trống với ô kề
                neighbors.append((new_state, (nr, nc), get_manhattan_distance(new_state)))  # Thêm neighbor kèm heuristic
        if not neighbors:
            break  # Không còn neighbor nào để mở rộng
        better_neighbors = [(state, move, h) for state, move, h in neighbors if h < current_h]  # Lọc neighbor tốt hơn
        if better_neighbors:
            next_state, move, next_h = random.choice(better_neighbors)  # Chọn ngẫu nhiên neighbor tốt hơn
        else:
            next_state, move, next_h = random.choice(neighbors)  # Chọn ngẫu nhiên neighbor bất kỳ
            delta_h = next_h - current_h
            if delta_h > 0 and random.random() >= math.exp(-delta_h / temp):  # Quyết định nhận neighbor xấu dựa trên xác suất
                iterations += 1
                temp *= cooling_rate
                continue
        current_state = next_state
        current_h = next_h
        path.append(move)
        if current_h < best_h:
            best_state = copy.deepcopy(current_state)  # Lưu trạng thái tốt nhất tìm được
            best_h = current_h
        temp *= cooling_rate  # Giảm nhiệt độ (làm lạnh)
        iterations += 1
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    return (path, elapsed_time) if best_state == goal else ([], elapsed_time)

def beam_search_solve(start, goal, beam_width=3):
    start_time = time.perf_counter()
    queue = [(get_manhattan_distance(start), start, [])]  # Hàng đợi ưu tiên, mỗi phần tử gồm (heuristic, trạng thái, đường đi)
    visited = set()
    while queue:
        level_states = []
        for _ in range(min(beam_width, len(queue))):
            if queue:
                f_score, state, path = heapq.heappop(queue)  # Lấy trạng thái có heuristic nhỏ nhất ở tầng hiện tại
                level_states.append((f_score, state, path))
        next_level_candidates = []
        for _, state, path in level_states:
            state_tuple = tuple(map(tuple, state))
            if state_tuple in visited:
                continue  # Bỏ qua nếu đã thăm
            visited.add(state_tuple)
            if state == goal:
                return path, time.perf_counter() - start_time
            r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if state[r][c] == 0)  # Tìm vị trí ô trống
            for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:  # Duyệt các hướng di chuyển
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:  # Kiểm tra hợp lệ trong lưới
                    new_state = [row[:] for row in state]  # Sao chép trạng thái hiện tại
                    new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]  # Đổi chỗ ô trống với ô kề
                    new_tuple = tuple(map(tuple, new_state))  # Chuyển sang tuple để lưu vào visited
                    if new_tuple not in visited:  # Nếu trạng thái mới chưa được thăm
                        h_score = get_manhattan_distance(new_state)  # Tính heuristic cho trạng thái mới
                        next_level_candidates.append((h_score, new_state, path + [(nr, nc)]))
        next_level_candidates.sort()  # Sắp xếp các ứng viên theo heuristic tăng dần
        queue = []
        for candidate in next_level_candidates[:beam_width]:  # Chỉ giữ lại beam_width ứng viên tốt nhất
            heapq.heappush(queue, candidate)
    return [], time.perf_counter() - start_time  # Không tìm thấy, trả về rỗng

def flatten_state(state):
    # Chuyển ma trận 2D (3x3) thành list 1D gồm 9 phần tử để dễ thao tác di truyền
    return [state[i][j] for i in range(GRID_SIZE) for j in range(GRID_SIZE)]

def unflatten_state(flat_state):
    # Chuyển list 1D về lại ma trận 2D (3x3)
    return [flat_state[i * GRID_SIZE:(i + 1) * GRID_SIZE] for i in range(GRID_SIZE)]

def is_valid_state(flat_state):
    # Kiểm tra trạng thái hợp lệ: phải chứa đủ các số từ 0 đến 8, không trùng, không thiếu
    return sorted(flat_state) == list(range(9))

def get_fitness(individual, goal_flat):
    # Đánh giá độ thích nghi (fitness) của một cá thể (trạng thái)
    # Fitness là tổng khoảng cách Manhattan (đảo dấu để tối ưu hóa)
    state = unflatten_state(individual)
    manhattan_dist = get_manhattan_distance(state)
    return -manhattan_dist

def generate_individual(goal_flat, start_flat):
    # Sinh cá thể mới bằng cách thực hiện một số bước trộn ngẫu nhiên từ trạng thái start
    individual = start_flat[:]
    for _ in range(random.randint(5, 15)):
        zero_idx = individual.index(0)
        r, c = zero_idx // GRID_SIZE, zero_idx % GRID_SIZE
        moves = []
        if r > 0: moves.append(zero_idx - GRID_SIZE)
        if r < GRID_SIZE - 1: moves.append(zero_idx + GRID_SIZE)
        if c > 0: moves.append(zero_idx - 1)
        if c < GRID_SIZE - 1: moves.append(zero_idx + 1)
        if moves:
            swap_idx = random.choice(moves)
            individual[zero_idx], individual[swap_idx] = individual[swap_idx], individual[zero_idx]
    return individual

def crossover(parent1, parent2):
    # Lai ghép hai cá thể cha mẹ để tạo cá thể con
    size = len(parent1)
    child = [-1] * size
    positions = random.sample(range(size), random.randint(2, size // 2))  # Chọn ngẫu nhiên vị trí lấy gen từ parent1
    for pos in positions:
        child[pos] = parent1[pos]
    remaining = [x for x in parent2 if x not in child]  # Lấy các giá trị còn lại từ parent2
    for i in range(size):
        if child[i] == -1:
            child[i] = remaining.pop(0)
    return child

def mutate(individual):
    # Đột biến cá thể bằng cách đổi chỗ ô trống với một ô kề
    zero_idx = individual.index(0)
    r, c = zero_idx // GRID_SIZE, zero_idx % GRID_SIZE
    moves = []
    if r > 0: moves.append(zero_idx - GRID_SIZE)
    if r < GRID_SIZE - 1: moves.append(zero_idx + GRID_SIZE)
    if c > 0: moves.append(zero_idx - 1)
    if c < GRID_SIZE - 1: moves.append(zero_idx + 1)
    if moves:
        swap_idx = random.choice(moves)
        individual[zero_idx], individual[swap_idx] = individual[swap_idx], individual[zero_idx]
    return individual

def find_path(start_state, goal_state):
    # Tìm đường đi từ start_state đến goal_state bằng BFS (trả về danh sách các bước di chuyển)
    queue = deque([(start_state, [])])
    visited = {tuple(map(tuple, start_state))}
    while queue:
        state, path = queue.popleft()
        if state == goal_state:
            return path
        r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if state[r][c] == 0)
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                new_state = [row[:] for row in state]
                new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
                new_tuple = tuple(map(tuple, new_state))
                if new_tuple not in visited:
                    queue.append((new_state, path + [(nr, nc)]))
                    visited.add(new_tuple)
    return []

def genetic_algorithm_solve(start, goal, population_size=100, generations=1000, mutation_rate=0.2):
    # Giải 8-puzzle bằng thuật toán di truyền
    start_time = time.perf_counter()
    start_flat = flatten_state(start)
    goal_flat = flatten_state(goal)
    # Khởi tạo quần thể ban đầu (population)
    population = [start_flat] + [generate_individual(goal_flat, start_flat) for _ in range(population_size - 1)]
    for generation in range(generations):
        # Đánh giá fitness từng cá thể
        fitness_scores = [(ind, get_fitness(ind, goal_flat)) for ind in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        best_individual = fitness_scores[0][0]
        best_state = unflatten_state(best_individual)
        if best_state == goal:
            path = find_path(start, best_state)
            return path, time.perf_counter() - start_time
        # Elitism: giữ lại top 20% cá thể tốt nhất
        elite_size = max(1, population_size // 5)
        new_population = [ind for ind, _ in fitness_scores[:elite_size]]
        # Tạo thế hệ mới bằng crossover và mutation
        while len(new_population) < population_size:
            parent1, parent2 = random.choices([ind for ind, _ in fitness_scores[:elite_size * 2]], k=2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)
        population = new_population
    # Nếu hết thế hệ mà chưa tìm thấy lời giải, trả về cá thể tốt nhất tìm được
    best_individual = max([(ind, get_fitness(ind, goal_flat)) for ind in population], key=lambda x: x[1])[0]
    best_state = unflatten_state(best_individual)
    path = find_path(start, best_state)
    return path, time.perf_counter() - start_time

def and_or_tree_search_solve(start, goal, max_depth=50):
    start_time = time.perf_counter()
    def or_search(state, path, depth, visited):
        if depth > max_depth:
            return None  # Vượt quá độ sâu cho phép, dừng tìm kiếm
        if state == goal:
            return path  # Đã đến trạng thái đích, trả về đường đi
        state_tuple = tuple(map(tuple, state))
        if state_tuple in visited:
            return None  # Trạng thái đã thăm, tránh lặp vô hạn
        visited.add(state_tuple)
        r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if state[r][c] == 0)  # Tìm vị trí ô trống
        moves = [(nr, nc) for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)] 
                 if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE]  # Các hướng di chuyển hợp lệ
        for move in moves:
            nr, nc = move
            new_state = [row[:] for row in state]  # Sao chép trạng thái hiện tại
            new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]  # Đổi chỗ ô trống với ô kề
            result = and_search(new_state, path + [(nr, nc)], depth + 1, visited.copy())  # Đệ quy AND với visited copy
            if result is not None:
                return result  # Nếu tìm được đường đi thì trả về
        return None  # Không tìm thấy đường đi ở nhánh này

    def and_search(state, path, depth, visited):
        result = or_search(state, path, depth, visited)  # Gọi lại or_search (AND-OR tree)
        return result

    visited = set()
    path = and_search(start, [], 0, visited)
    elapsed_time = time.perf_counter() - start_time
    return (path, elapsed_time) if path is not None else ([], elapsed_time)

def partially_observable_search_console(start, goal, observation_ratio=0.5, max_iterations=1000):
    start_time = time.perf_counter()
    print("\n======== PARTIALLY OBSERVABLE SEARCH ========")
    print(f"Observation ratio: {observation_ratio:.2f} ({int(9 * observation_ratio)}/9 tiles)")
    print("===========================================\n")
    
    # Chuyển trạng thái ban đầu thành list 1 chiều để dễ xử lý
    flat_state = flatten_state(start)
    
    # Xác định số lượng ô được quan sát (observable)
    num_observable = max(1, int(9 * observation_ratio))
    observable_indices = random.sample(range(9), num_observable)
    
    # Tạo mặt nạ quan sát: True nếu ô được quan sát, False nếu chưa biết
    observation_mask = [False] * 9
    for idx in observable_indices:
        observation_mask[idx] = True
    
    # In trạng thái ban đầu với các ô chưa biết được che dấu bằng '?'
    print("Initial state with partial observation:")
    masked_state = []
    for r in range(GRID_SIZE):
        row = []
        for c in range(GRID_SIZE):
            idx = r * GRID_SIZE + c
            if observation_mask[idx]:
                row.append(start[r][c])
                print(f" {start[r][c]} ", end="")
            else:
                row.append("?")
                print(" ? ", end="")
        masked_state.append(row)
        print()
    
    # Xác định các vị trí và giá trị chưa biết trong trạng thái ban đầu
    possible_states = []
    unknown_positions = [i for i, observed in enumerate(observation_mask) if not observed]  # Vị trí chưa quan sát được
    unknown_values = []
    for i in range(9):
        if i not in [flat_state[idx] for idx in observable_indices]:
            unknown_values.append(i)  # Giá trị chưa biết
    
    print(f"\nUnknown positions: {len(unknown_positions)}")
    print(f"Unknown values: {unknown_values}")
    
    # Sinh tất cả các trạng thái ban đầu có thể bằng hoán vị các giá trị chưa biết vào các vị trí chưa biết
    from itertools import permutations
    for perm in permutations(unknown_values):
        possible_state = flat_state.copy()
        for pos_idx, val_idx in enumerate(range(len(unknown_positions))):
            possible_state[unknown_positions[pos_idx]] = perm[val_idx]
        possible_states.append(unflatten_state(possible_state))  # Chuyển về ma trận 2D
    
    print(f"\nGenerated {len(possible_states)} possible initial states")
    
    solution_path = None
    solution_state = None
    visited_states = 0
    
    # Duyệt từng trạng thái ban đầu có thể, thử tìm đường đi đến goal bằng BFS
    for i, possible_start in enumerate(possible_states):
        print(f"\nExploring possible initial state {i+1}/{len(possible_states)}:")
        for row in possible_start:
            print(f"  {row}")
        
        queue = deque([(possible_start, [])])
        visited = {tuple(map(tuple, possible_start))}
        visited_states += 1
        steps_explored = 0
        
        while queue and steps_explored < max_iterations:
            state, path = queue.popleft()
            steps_explored += 1
            
            if state == goal:
                solution_path = path
                solution_state = possible_start
                print(f"\n✓ Solution found from possible state {i+1}!")
                print(f"  Path length: {len(path)}")
                print(f"  Steps explored: {steps_explored}")
                break
                
            # Tìm vị trí ô trống và sinh các trạng thái kề
            r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if state[r][c] == 0)
            for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    new_state = [row[:] for row in state]
                    new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
                    new_tuple = tuple(map(tuple, new_state))
                    if new_tuple not in visited:  # Nếu trạng thái mới chưa được thăm
                        queue.append((new_state, path + [(nr, nc)]))
                        visited.add(new_tuple)
                        visited_states += 1
        if solution_path:
            break
        print(f"  No solution found within {steps_explored} steps")
    
    elapsed_time = time.perf_counter() - start_time
    
    # Nếu tìm được lời giải, in ra các bước di chuyển
    if solution_path:
        print(f"\n✓ Solution found!")
        print(f"Explored {visited_states} states")
        print(f"Execution time: {elapsed_time:.6f} seconds")
        print("\n=== SOLUTION PATH ===")
        print("Starting state:")
        for row in solution_state:
            print(f"  {row}")
            
        current = copy.deepcopy(solution_state)
        
        print("\nSolution steps:")
        for step_num, (nr, nc) in enumerate(solution_path, 1):
            r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if current[r][c] == 0)
            current[r][c], current[nr][nc] = current[nr][nc], current[r][c]
            direction = ""
            if nr < r:
                direction = "UP"
            elif nr > r:
                direction = "DOWN"
            elif nc < c:
                direction = "LEFT"
            elif nc > c:
                direction = "RIGHT"
            print(f"\nStep {step_num}: Move {direction}")
            for row in current:
                print(f"  {row}")
        print("\n=== END OF SOLUTION ===")
        return solution_path, elapsed_time
    else:
        print(f"\n✗ No solution found")
        print(f"Explored {visited_states} states")
        print(f"Execution time: {elapsed_time:.6f} seconds")
        return [], elapsed_time

def belief_state_search_console(goal, start_states=None, max_depth=15):
    start_time = time.perf_counter()
    print("\n======== BELIEF STATE SEARCH ========")
    print("This algorithm searches through multiple possible start states")
    print("Results will be displayed in this console")
    print("======================================\n")
    
    # Nếu không truyền vào các trạng thái khởi đầu, tự sinh ngẫu nhiên 2 trạng thái ban đầu
    if start_states is None:
        initial_states = []
        for _ in range(2):
            state = list(range(9))
            random.shuffle(state)
            initial_states.append([state[i:i + GRID_SIZE] for i in range(0, len(state), GRID_SIZE)])
    else:
        initial_states = start_states
        
    print("Initial belief states:")
    for i, state in enumerate(initial_states):
        print(f"State {i + 1}:")
        for row in state:
            print(f"  {row}")
        print()
        
    goal_tuple = tuple(map(tuple, goal))
    # belief_state là một dict: key là trạng thái (tuple), value là đường đi đến đó
    belief_state = {tuple(map(tuple, state)): [] for state in initial_states}
    visited = set()
    
    # Lặp theo từng độ sâu (giống BFS nhiều trạng thái song song)
    for depth in range(max_depth + 1):
        print(f"\n==== DEPTH {depth} ====")
        print(f"Current belief state contains {len(belief_state)} possible states")
        
        # Kiểm tra có trạng thái nào đạt goal chưa
        for state_tuple in belief_state:
            if state_tuple == goal_tuple:
                path = belief_state[state_tuple]
                elapsed_time = time.perf_counter() - start_time
                print(f"\n🎉 Goal State Found at depth {depth}!")
                print(f"Path length: {len(path)}")
                print(f"Execution time: {elapsed_time:.6f} seconds")
                return path, elapsed_time
        
        if depth == max_depth:
            break
            
        new_belief_state = {}
        
        # Mở rộng từng trạng thái trong belief_state hiện tại
        for i, (state_tuple, path) in enumerate(belief_state.items()):
            if state_tuple in visited:
                continue
                
            visited.add(state_tuple)
            state = [list(row) for row in state_tuple]
            
            print(f"\nExpanding state {i+1}:")
            for row in state:
                print(f"  {row}")
                
            r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if state[r][c] == 0)
            
            print(f"Empty position at ({r}, {c})")
            print("Possible moves:")
            
            # Duyệt các hướng di chuyển hợp lệ của ô trống
            for dir_idx, (nr, nc) in enumerate([(r-1, c), (r+1, c), (r, c-1), (r, c+1)]):
                direction = ["UP", "DOWN", "LEFT", "RIGHT"][dir_idx]
                
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    new_state = [row[:] for row in state]
                    new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
                    new_state_tuple = tuple(map(tuple, new_state))
                    
                    print(f"  - Move {direction} ({nr}, {nc}):")
                    for row in new_state:
                        print(f"      {row}")
                        
                    if new_state_tuple not in visited:
                        new_path = path + [(nr, nc)]
                        new_belief_state[new_state_tuple] = new_path
                        print(f"      Added to belief state with path length {len(new_path)}")
                    else:
                        print(f"      Already visited")
                else:
                    print(f"  - Move {direction} out of bounds")
        
        belief_state = new_belief_state
        print(f"\nAfter expansion: belief state now contains {len(belief_state)} states")
        
        if not belief_state:
            print("\nNo more states to explore! Belief state is empty.")
            break

    elapsed_time = time.perf_counter() - start_time
    print("\n🚫 Goal State Not Found within max depth.")
    print(f"Explored {len(visited)} unique states")
    print(f"Execution time: {elapsed_time:.6f} seconds")
    return [], elapsed_time

def min_conflicts_solve(start, goal, max_steps=1000):
    start_time = time.perf_counter()
    if start == goal:
        # Nếu trạng thái ban đầu đã là trạng thái đích thì trả về luôn
        return [], time.perf_counter() - start_time
    current_state = copy.deepcopy(start)
    path = []
    
    for step in range(max_steps):
        if current_state == goal:
            # Nếu đã đạt trạng thái đích thì trả về đường đi và thời gian
            return path, time.perf_counter() - start_time
        conflicts = []
        # Xác định các ô đang ở sai vị trí so với goal (trừ ô trống)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if current_state[r][c] != 0 and current_state[r][c] != goal[r][c]:
                    conflicts.append((r, c))
        if not conflicts:
            # Nếu không còn xung đột nào thì trả về kết quả
            return path, time.perf_counter() - start_time
        # Tìm vị trí ô trống hiện tại
        r, c = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) 
                    if current_state[r][c] == 0)
        possible_moves = []
        # Xét các vị trí có thể di chuyển ô trống đến (lên/xuống/trái/phải)
        for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                possible_moves.append((nr, nc))
        if not possible_moves:
            break
        best_moves = []
        min_conflicts = float('inf')
        
        # Đánh giá từng nước đi, chọn nước đi làm giảm số xung đột nhiều nhất
        for nr, nc in possible_moves:
            new_state = [row[:] for row in current_state]
            new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
            new_conflicts = sum(1 for i in range(GRID_SIZE) for j in range(GRID_SIZE)
                              if new_state[i][j] != 0 and new_state[i][j] != goal[i][j])
            
            if new_conflicts < min_conflicts:
                min_conflicts = new_conflicts
                best_moves = [(nr, nc)]
            elif new_conflicts == min_conflicts:
                best_moves.append((nr, nc))
        # Chọn ngẫu nhiên một nước đi tốt nhất (ít xung đột nhất)
        move = random.choice(best_moves)
        nr, nc = move
        current_state[r][c], current_state[nr][nc] = current_state[nr][nc], current_state[r][c]
        path.append((nr, nc))
    # Nếu hết số bước mà chưa giải được thì trả về rỗng
    return [], time.perf_counter() - start_time

def backtracking_solve(goal=None, max_depth=15, screen=None, grid_offset_x=None, grid_offset_y=None):
    start_time = time.perf_counter()
    
    if goal is None:
        goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]  # Nếu không truyền goal thì dùng trạng thái đích mặc định

    empty_state = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # Khởi tạo ma trận rỗng
    used_values = set()  # Tập hợp các giá trị đã dùng
    all_states = []      # Lưu lại các trạng thái đã sinh ra (phục vụ hiển thị)
    
    cells_order = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
    random.shuffle(cells_order)  # Thứ tự duyệt các ô (ngẫu nhiên để đa dạng lời giải)
    
    goal_positions = {}
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            goal_positions[goal[r][c]] = (r, c)  # Lưu vị trí đích của từng giá trị

    def is_valid_assignment(state, r, c, value):
        # Kiểm tra giá trị đã được dùng chưa
        if value in used_values:
            return False
        return True

    def count_inversions(state):
        # Đếm số nghịch thế để kiểm tra trạng thái hợp lệ (giải được)
        flat = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if state[i][j] is not None and state[i][j] != 0:
                    flat.append(state[i][j])
        inversions = 0
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                if flat[i] > flat[j]:
                    inversions += 1
        return inversions

    def is_valid_state(state):
        # Kiểm tra trạng thái hiện tại có hợp lệ không (dựa vào số nghịch thế)
        filled_count = sum(1 for r in range(GRID_SIZE) for c in range(GRID_SIZE) if state[r][c] is not None)
        if filled_count == 8:
            empty_pos = next((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if state[r][c] is None)
            complete_state = [row[:] for row in state]
            complete_state[empty_pos[0]][empty_pos[1]] = 0
            inversions = count_inversions(complete_state)
            if inversions % 2 != 0:
                return False
        return True

    def is_goal_achievable(state):
        # Kiểm tra trạng thái hiện tại có thể đạt được goal không (dựa vào vị trí các số)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if state[r][c] is not None and state[r][c] != 0:
                    if goal_positions[state[r][c]] != (r, c):
                        return False
        return True

    def recursive_backtracking(state, cell_idx):
        nonlocal all_states

        # Nếu có truyền screen thì vẽ trạng thái hiện tại lên màn hình (phục vụ giao diện)
        if screen:
            display_state = [[0 if cell is None else cell for cell in row] for row in state]
            screen.fill(BACKGROUND_COLOR)
            draw_grid(screen, display_state, grid_offset_x, grid_offset_y)
            font = pygame.font.Font(None, 24)
            msg = f"Backtracking progress: {len(used_values)}/9 numbers placed"
            text = font.render(msg, True, MESSAGE_COLOR)
            screen.blit(text, (35, HEIGHT - 100))
            pygame.display.flip()
            pygame.time.delay(50)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        # Nếu đã điền hết các ô, kiểm tra có khớp goal không
        if cell_idx >= len(cells_order):
            current_display = [[0 if cell is None else cell for cell in row] for row in state]
            return current_display == goal

        r, c = cells_order[cell_idx]
        target_value = goal[r][c]
        values_to_try = [target_value]  # Ưu tiên điền đúng giá trị ở vị trí goal

        if target_value in used_values:
            # Nếu giá trị này đã dùng, thử các giá trị còn lại chưa dùng
            values_to_try = [v for v in range(9) if v not in used_values]
            random.shuffle(values_to_try)

        for value in values_to_try:
            if is_valid_assignment(state, r, c, value):
                state[r][c] = value
                used_values.add(value)
                current_display = [[0 if cell is None else cell for cell in row] for row in state]
                all_states.append(copy.deepcopy(current_display))
                # Đệ quy sang ô tiếp theo nếu trạng thái hợp lệ
                if is_valid_state(state) and recursive_backtracking(state, cell_idx + 1):
                    return True
                state[r][c] = None
                used_values.remove(value)

        return False

    result = recursive_backtracking(empty_state, 0)
    elapsed_time = time.perf_counter() - start_time
    return all_states, elapsed_time

def state_to_tuple(state):
    # Chuyển ma trận 2D thành tuple lồng nhau để dùng làm key trong dict (bất biến)
    return tuple(map(tuple, state))

def get_blank_pos_q(state_tuple): 
    # Tìm vị trí ô trống (giá trị 0) trong trạng thái tuple
    for r_idx, row in enumerate(state_tuple):
        for c_idx, val in enumerate(row):
            if val == 0:
                return r_idx, c_idx
    return -1, -1  # Không tìm thấy

def get_valid_actions_q(state_tuple):
    # Trả về danh sách các hành động hợp lệ (0: lên, 1: xuống, 2: trái, 3: phải) cho ô trống
    r, c = get_blank_pos_q(state_tuple)
    if r == -1: return []  # Không có ô trống
    valid_actions = []
    if r > 0: valid_actions.append(0)  # Lên
    if r < GRID_SIZE - 1: valid_actions.append(1)  # Xuống
    if c > 0: valid_actions.append(2)  # Trái
    if c < GRID_SIZE - 1: valid_actions.append(3)  # Phải
    return valid_actions

def apply_action_q(state_tuple, action_idx):
    # Thực hiện hành động (di chuyển ô trống) trên trạng thái tuple, trả về trạng thái mới và vị trí mới của ô trống
    state_list = [list(row) for row in state_tuple]
    r, c = get_blank_pos_q(state_tuple)
    if r == -1: return state_tuple, (-1,-1)  # Không hợp lệ

    dr, dc = 0, 0
    if action_idx == 0: dr = -1  # Lên
    elif action_idx == 1: dr = 1  # Xuống
    elif action_idx == 2: dc = -1  # Trái
    elif action_idx == 3: dc = 1   # Phải
    else: return state_tuple, (r,c)  # Không hợp lệ

    new_r_blank, new_c_blank = r + dr, c + dc

    # Đổi chỗ ô trống với ô kề theo hành động
    state_list[r][c], state_list[new_r_blank][new_c_blank] = state_list[new_r_blank][new_c_blank], state_list[r][c]
    return tuple(map(tuple, state_list)), (new_r_blank, new_c_blank)

def q_learning_solve(
    start, goal, 
    episodes=Q_LEARNING_EPISODES, 
    alpha=Q_LEARNING_ALPHA, 
    gamma=Q_LEARNING_GAMMA, 
    epsilon_start=Q_LEARNING_EPSILON, 
    max_steps_episode=Q_LEARNING_MAX_STEPS_PER_EPISODE, 
    max_path_len=Q_LEARNING_MAX_PATH_LEN
):
    start_time_total = time.perf_counter()
    q_table = {}  # Bảng Q: key là trạng thái, value là list Q-value cho 4 hành động
    
    start_tuple = state_to_tuple(start)
    goal_tuple = state_to_tuple(goal)

    # Q-learning training
    for episode in range(episodes):
        current_state_tuple = start_tuple 
        # Giảm epsilon dần theo số episode (exploration -> exploitation)
        epsilon = epsilon_start * math.exp(-episode / (episodes/5))

        for _step in range(max_steps_episode):
            valid_actions = get_valid_actions_q(current_state_tuple)
            if not valid_actions: 
                break

            current_q_values = q_table.get(current_state_tuple, [0.0] * 4)
            action_to_take = -1

            # Chọn hành động: epsilon-greedy (thăm dò hoặc khai thác)
            if random.random() < epsilon:
                action_to_take = random.choice(valid_actions) 
            else:
                best_q_val = -float('inf')
                shuffled_valid_actions = random.sample(valid_actions, len(valid_actions))
                for act_idx in shuffled_valid_actions:
                    if current_q_values[act_idx] > best_q_val:
                        best_q_val = current_q_values[act_idx]
                        action_to_take = act_idx
                if action_to_take == -1 : 
                    action_to_take = random.choice(valid_actions)

            next_state_tuple, _ = apply_action_q(current_state_tuple, action_to_take)
            
            # Đặt phần thưởng: -1 cho mỗi bước, +100 nếu đạt goal
            reward = -1 
            if next_state_tuple == goal_tuple:
                reward = 100
            
            old_q_value = current_q_values[action_to_take]
            
            # Q-learning update
            next_valid_actions = get_valid_actions_q(next_state_tuple)
            next_q_state_values = q_table.get(next_state_tuple, [0.0] * 4)
            max_next_q = 0.0
            if next_valid_actions:
                max_next_q = -float('inf')
                for next_act_idx in next_valid_actions:
                    if next_q_state_values[next_act_idx] > max_next_q:
                        max_next_q = next_q_state_values[next_act_idx]
                if max_next_q == -float('inf'): max_next_q = 0.0 

            # Cập nhật Q-value theo công thức Q-learning
            new_q_value = old_q_value + alpha * (reward + gamma * max_next_q - old_q_value)
        
            updated_q_values = list(current_q_values)
            updated_q_values[action_to_take] = new_q_value
            q_table[current_state_tuple] = updated_q_values
            
            current_state_tuple = next_state_tuple
            if current_state_tuple == goal_tuple:
                break
        
    # Sau khi train xong, truy vết đường đi tốt nhất từ start đến goal theo Q-table
    path = []
    current_state_tuple = start_tuple
    for _ in range(max_path_len):
        if current_state_tuple == goal_tuple:
            break
        
        valid_actions = get_valid_actions_q(current_state_tuple)
        if not valid_actions:
            path = []
            break

        current_q_values = q_table.get(current_state_tuple, [0.0] * 4)
        best_q_val = -float('inf')
        best_action = -1
        
        shuffled_valid_actions = random.sample(valid_actions, len(valid_actions)) 
        for act_idx in shuffled_valid_actions:
            if current_q_values[act_idx] > best_q_val:
                best_q_val = current_q_values[act_idx]
                best_action = act_idx
        
        if best_action == -1:
            if valid_actions: 
                best_action = random.choice(valid_actions) 
            else: 
                path = []
                break
        
        if best_action == -1: 
            path = []
            break

        next_state_tuple, new_blank_coords = apply_action_q(current_state_tuple, best_action)
        path.append(new_blank_coords) 
        current_state_tuple = next_state_tuple
    
    else:
        if current_state_tuple != goal_tuple: 
            path = []

    elapsed_time = time.perf_counter() - start_time_total
    return path, elapsed_time

def input_two_start_states_window(main_screen):
    input_width, input_height = 600, 500
    input_screen = pygame.display.set_mode((input_width, input_height))
    pygame.display.set_caption("Input Two Start States for Belief State Search")
    font = pygame.font.Font(None, 30)
    title_font = pygame.font.Font(None, 36)

    local_tile_size = 80
    local_grid_width = GRID_SIZE * local_tile_size
    local_grid_height = GRID_SIZE * local_tile_size

    state_1_x = (input_width // 2 - local_grid_width) // 2
    state_2_x = input_width // 2 + (input_width // 2 - local_grid_width) // 2
    grid_y = 120

    start_state_1 = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    start_state_2 = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    selected_cell = None
    message = "Click a cell in either grid and press 0-8 to set a number."

    ok_button_y = grid_y + local_grid_height + 50
    ok_button = pygame.Rect(input_width // 2 - 60, ok_button_y, 120, 50)

    highlight_color = (255, 200, 200)

    while True:
        input_screen.fill(BACKGROUND_COLOR)
        mouse_pos = pygame.mouse.get_pos()

        title = title_font.render("Belief State Search - Input Two Start States", True, TEXT_COLOR)
        title_rect = title.get_rect(center=(input_width // 2, 40))
        input_screen.blit(title, title_rect)

        label_1 = font.render("Start State 1", True, TEXT_COLOR)
        label_2 = font.render("Start State 2", True, TEXT_COLOR)
        label_1_rect = label_1.get_rect(center=(state_1_x + local_grid_width // 2, grid_y - 30))
        label_2_rect = label_2.get_rect(center=(state_2_x + local_grid_width // 2, grid_y - 30))
        input_screen.blit(label_1, label_1_rect)
        input_screen.blit(label_2, label_2_rect)

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                rect1 = pygame.Rect(state_1_x + c * local_tile_size, grid_y + r * local_tile_size, 
                                  local_tile_size, local_tile_size)
                if selected_cell == (r, c, 1):
                    pygame.draw.rect(input_screen, highlight_color, rect1)
                else:
                    val = start_state_1[r][c]
                    pygame.draw.rect(input_screen, TILE_COLOR if val != 0 else BACKGROUND_COLOR, rect1)
                pygame.draw.rect(input_screen, TILE_BORDER, rect1, 2)
                if start_state_1[r][c] != 0:
                    text = font.render(str(start_state_1[r][c]), True, TEXT_COLOR)
                    text_rect = text.get_rect(center=rect1.center)
                    input_screen.blit(text, text_rect)
                
                rect2 = pygame.Rect(state_2_x + c * local_tile_size, grid_y + r * local_tile_size, 
                                  local_tile_size, local_tile_size)
                if selected_cell == (r, c, 2):
                    pygame.draw.rect(input_screen, highlight_color, rect2)
                else:
                    val = start_state_2[r][c]
                    pygame.draw.rect(input_screen, TILE_COLOR if val != 0 else BACKGROUND_COLOR, rect2)
                pygame.draw.rect(input_screen, TILE_BORDER, rect2, 2)
                if start_state_2[r][c] != 0:
                    text = font.render(str(start_state_2[r][c]), True, TEXT_COLOR)
                    text_rect = text.get_rect(center=rect2.center)
                    input_screen.blit(text, text_rect)

        msg_font = pygame.font.Font(None, 24)
        text = msg_font.render(message, True, TEXT_COLOR)
        text_rect = text.get_rect(center=(input_width // 2, grid_y - 60))
        input_screen.blit(text, text_rect)

        button_color = BUTTON_HOVER if ok_button.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(input_screen, button_color, ok_button, border_radius=8)
        pygame.draw.rect(input_screen, TILE_BORDER, ok_button, 2, border_radius=8)
        ok_text = font.render("OK", True, TEXT_COLOR)
        text_rect = ok_text.get_rect(center=ok_button.center)
        input_screen.blit(ok_text, text_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = pygame.mouse.get_pos()
                if ok_button.collidepoint(pos):
                    if is_valid_state(flatten_state(start_state_1)) and is_valid_state(flatten_state(start_state_2)):
                        pygame.display.set_mode((WIDTH, HEIGHT))
                        pygame.display.set_caption("8-Puzzle Solver")
                        return [start_state_1, start_state_2]
                    else:
                        message = "Invalid state(s)! Each grid must use numbers 0-8 exactly once."
                elif state_1_x <= pos[0] < state_1_x + local_grid_width and grid_y <= pos[1] < grid_y + local_grid_height:
                    r = (pos[1] - grid_y) // local_tile_size
                    c = (pos[0] - state_1_x) // local_tile_size
                    selected_cell = (r, c, 1)
                elif state_2_x <= pos[0] < state_2_x + local_grid_width and grid_y <= pos[1] < grid_y + local_grid_height:
                    r = (pos[1] - grid_y) // local_tile_size
                    c = (pos[0] - state_2_x) // local_tile_size
                    selected_cell = (r, c, 2)
                else:
                    selected_cell = None
            if event.type == pygame.KEYDOWN and selected_cell:
                if event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, 
                               pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8]:
                    value = int(event.unicode)
                    r, c, state_id = selected_cell
                    if state_id == 1:
                        if is_valid_input(start_state_1, value):
                            start_state_1[r][c] = value
                            message = f"Set {value} at ({r}, {c}) in State 1."
                        else:
                            message = f"Invalid! {value} is already used or out of range in State 1."
                    elif state_id == 2:
                        if is_valid_input(start_state_2, value):
                            start_state_2[r][c] = value
                            message = f"Set {value} at ({r}, {c}) in State 2."
                        else:
                            message = f"Invalid! {value} is already used or out of range in State 2."

def input_observation_ratio_window(main_screen):
    input_width, input_height = 400, 300
    input_screen = pygame.display.set_mode((input_width, input_height))
    pygame.display.set_caption("Observation Ratio")
    font = pygame.font.Font(None, 30)
    title_font = pygame.font.Font(None, 36)
    
    ratios = [0.33, 0.44, 0.55, 0.66, 0.77]
    ratio_buttons = []
    
    for i, ratio in enumerate(ratios):
        button_rect = pygame.Rect(input_width//2 - 100, 100 + i*40, 200, 30)
        ratio_buttons.append((ratio, button_rect))
    
    while True:
        input_screen.fill(BACKGROUND_COLOR)
        mouse_pos = pygame.mouse.get_pos()
        
        title = title_font.render("Select Observation Ratio", True, TEXT_COLOR)
        title_rect = title.get_rect(center=(input_width//2, 40))
        input_screen.blit(title, title_rect)
        
        guide = font.render("(Number of visible tiles out of 9)", True, MESSAGE_COLOR)
        guide_rect = guide.get_rect(center=(input_width//2, 70))
        input_screen.blit(guide, guide_rect)
        
        for ratio, rect in ratio_buttons:
            visible_tiles = int(9 * ratio)
            button_color = BUTTON_HOVER if rect.collidepoint(mouse_pos) else BUTTON_COLOR
            pygame.draw.rect(input_screen, button_color, rect, border_radius=5)
            pygame.draw.rect(input_screen, TILE_BORDER, rect, 2, border_radius=5)
            
            text = font.render(f"{visible_tiles}/9 tiles ({int(ratio*100)}%)", True, TEXT_COLOR)
            text_rect = text.get_rect(center=rect.center)
            input_screen.blit(text, text_rect)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for ratio, rect in ratio_buttons:
                    if rect.collidepoint(event.pos):
                        pygame.display.set_mode((WIDTH, HEIGHT))
                        pygame.display.set_caption("8-Puzzle Solver")
                        return ratio

def is_valid_input(state, value):
    flat = [state[i][j] for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
    return 0 <= value <= 8 and value not in flat

def draw_grid(screen, state, offset_x, offset_y, selected_cell=None):
    font = pygame.font.Font(None, 45)
    
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            rect = pygame.Rect(offset_x + c * TILE_SIZE, offset_y + r * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            
            if selected_cell == (r, c):
                pygame.draw.rect(screen, BUTTON_HOVER, rect)
            else:
                if state[r][c] == 0:
                    pygame.draw.rect(screen, BACKGROUND_COLOR, rect)
                else:
                    pygame.draw.rect(screen, TILE_COLOR, rect)
            
            pygame.draw.rect(screen, TILE_BORDER, rect, 2)
            
            if state[r][c] != 0:
                text = font.render(str(state[r][c]), True, TEXT_COLOR)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

def draw_buttons(screen, buttons, mouse_pos):
    for button in buttons:
        if button['text'] == '':
            continue
            
        font_size = 20 if button['text'] == 'AND-OR' else 24
        font = pygame.font.Font(None, font_size)
        button_color = BUTTON_HOVER if button['rect'].collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(screen, button_color, button['rect'], border_radius=5)
        pygame.draw.rect(screen, TILE_BORDER, button['rect'], 2, border_radius=5)
        text = font.render(button['text'], True, TEXT_COLOR)
        text_rect = text.get_rect(center=button['rect'].center)
        screen.blit(text, text_rect)

def draw_message(screen, msg, x, y, color=MESSAGE_COLOR):
    font = pygame.font.Font(None, 28)
    text = font.render(msg, True, color)
    screen.blit(text, (x, y))

def draw_state_area(screen, states_history, scroll_offset=0):
    font = pygame.font.Font(None, 22)
    pygame.draw.rect(screen, BACKGROUND_COLOR, (STATE_AREA_X, STATE_AREA_Y, STATE_AREA_WIDTH, STATE_AREA_HEIGHT))
    pygame.draw.rect(screen, TILE_BORDER, (STATE_AREA_X, STATE_AREA_Y, STATE_AREA_WIDTH, STATE_AREA_HEIGHT), 2)
    
    if not states_history:
        return 0, 0
    title_text = font.render("Solution Steps:", True, TEXT_COLOR)
    title_rect = title_text.get_rect(topleft=(STATE_AREA_X + 10, STATE_AREA_Y + 10))
    screen.blit(title_text, title_rect)
    state_block_height = STATE_TILE_SIZE * GRID_SIZE + STATE_SPACING + 25
    total_content_height = len(states_history) * state_block_height
    content_area_y = STATE_AREA_Y + TITLE_HEIGHT
    content_area_height = STATE_AREA_HEIGHT - TITLE_HEIGHT
    max_scroll = max(0, total_content_height - content_area_height)
    scroll_offset = max(0, min(scroll_offset, max_scroll))
    visible_start_idx = max(0, int(scroll_offset // state_block_height))
    visible_end_idx = min(len(states_history), int((scroll_offset + content_area_height) // state_block_height) + 1)
    for i in range(visible_start_idx, visible_end_idx):
        state = states_history[i]
        state_y_top = content_area_y + (i * state_block_height) - scroll_offset
        state_y_bottom = state_y_top + state_block_height
        if state_y_bottom > content_area_y and state_y_top < content_area_y + content_area_height:
            step_text = font.render(f"Step {i}:", True, TEXT_COLOR)
            step_rect = step_text.get_rect(topleft=(STATE_AREA_X + 10, state_y_top))
            if content_area_y <= state_y_top < content_area_y + content_area_height:
                screen.blit(step_text, step_rect)
            grid_width = STATE_TILE_SIZE * GRID_SIZE
            grid_offset_x = STATE_AREA_X + (STATE_AREA_WIDTH - grid_width) // 2
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    tile_y = state_y_top + 25 + r * STATE_TILE_SIZE
                    tile_y_bottom = tile_y + STATE_TILE_SIZE
                    if tile_y >= content_area_y and tile_y_bottom <= content_area_y + content_area_height:
                        rect = pygame.Rect(
                            grid_offset_x + c * STATE_TILE_SIZE,
                            tile_y,
                            STATE_TILE_SIZE,
                            STATE_TILE_SIZE
                        )
                        pygame.draw.rect(screen, TILE_COLOR if state[r][c] != 0 else BACKGROUND_COLOR, rect)
                        pygame.draw.rect(screen, TILE_BORDER, rect, 1)
                        if state[r][c] != 0:
                            text = font.render(str(state[r][c]), True, TEXT_COLOR)
                            text_rect = text.get_rect(center=rect.center)
                            screen.blit(text, text_rect)
    scrollbar_height = content_area_height
    if total_content_height > content_area_height:
        scrollbar_height = max(30, int(content_area_height * content_area_height / total_content_height))
        scrollbar_pos = int((scroll_offset / max_scroll) * (content_area_height - scrollbar_height)) if max_scroll > 0 else 0
        scrollbar_rect = pygame.Rect(
            STATE_AREA_X + STATE_AREA_WIDTH - SCROLLBAR_WIDTH - 5,
            content_area_y + scrollbar_pos,
            SCROLLBAR_WIDTH,
            scrollbar_height
        )
        pygame.draw.rect(screen, BUTTON_COLOR, scrollbar_rect, border_radius=3)
        pygame.draw.rect(screen, TILE_BORDER, scrollbar_rect, 1, border_radius=3)
    return total_content_height, scrollbar_height

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("8-Puzzle Solver")
    
    GRID_OFFSET_X = (WIDTH - GRID_WIDTH - STATE_AREA_WIDTH) // 2
    GRID_OFFSET_Y = (HEIGHT - GRID_HEIGHT - 3 * (BUTTON_HEIGHT + BUTTON_SPACING) - 100) // 2
    
    reset_button_x = GRID_OFFSET_X + (GRID_WIDTH - 2 * BUTTON_WIDTH - BUTTON_SPACING) // 2
    restart_button_x = reset_button_x + BUTTON_WIDTH + BUTTON_SPACING
    button_y_top = GRID_OFFSET_Y - BUTTON_HEIGHT - BUTTON_SPACING
    
    buttons_count_row1 = 6
    buttons_count_row2 = 6
    buttons_count_row3 = 6 
    buttons_count_row4 = 1 

    total_buttons_width_row1 = buttons_count_row1 * BUTTON_WIDTH + (buttons_count_row1 - 1) * BUTTON_SPACING
    total_buttons_width_row2 = buttons_count_row2 * BUTTON_WIDTH + (buttons_count_row2 - 1) * BUTTON_SPACING
    total_buttons_width_row3 = buttons_count_row3 * BUTTON_WIDTH + (buttons_count_row3 - 1) * BUTTON_SPACING
    total_buttons_width_row4 = buttons_count_row4 * BUTTON_WIDTH + (buttons_count_row4 - 1) * BUTTON_SPACING 
    
    start_x_buttons_row1 = GRID_OFFSET_X + (GRID_WIDTH - total_buttons_width_row1) // 2
    start_x_buttons_row2 = GRID_OFFSET_X + (GRID_WIDTH - total_buttons_width_row2) // 2
    start_x_buttons_row3 = GRID_OFFSET_X + (GRID_WIDTH - total_buttons_width_row3) // 2
    start_x_buttons_row4 = GRID_OFFSET_X + (GRID_WIDTH - total_buttons_width_row4) // 2 
    
    button_y_row1 = GRID_OFFSET_Y + GRID_HEIGHT + BUTTON_SPACING
    button_y_row2 = button_y_row1 + BUTTON_HEIGHT + BUTTON_SPACING
    button_y_row3 = button_y_row2 + BUTTON_HEIGHT + BUTTON_SPACING
    button_y_row4 = button_y_row3 + BUTTON_HEIGHT + BUTTON_SPACING
    
    buttons = [
        {'text': 'Reset', 'rect': pygame.Rect(reset_button_x, button_y_top, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': lambda: 'reset'},
        {'text': 'Restart', 'rect': pygame.Rect(restart_button_x, button_y_top, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': lambda: 'restart'},
        
        {'text': 'BFS', 'rect': pygame.Rect(start_x_buttons_row1 + 0 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row1, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': bfs_solve},
        {'text': 'DFS', 'rect': pygame.Rect(start_x_buttons_row1 + 1 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row1, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': dfs_solve},
        {'text': 'UCS', 'rect': pygame.Rect(start_x_buttons_row1 + 2 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row1, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': ucs_solve},
        {'text': 'Greedy', 'rect': pygame.Rect(start_x_buttons_row1 + 3 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row1, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': greedy_solve},
        {'text': 'IDDFS', 'rect': pygame.Rect(start_x_buttons_row1 + 4 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row1, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': iddfs_solve},
        {'text': 'A*', 'rect': pygame.Rect(start_x_buttons_row1 + 5 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row1, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': astar_solve},
        
        {'text': 'IDA*', 'rect': pygame.Rect(start_x_buttons_row2 + 0 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row2, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': idastar_solve},
        {'text': 'SHC', 'rect': pygame.Rect(start_x_buttons_row2 + 1 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row2, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': simple_hill_climbing_solve},
        {'text': 'SAHC', 'rect': pygame.Rect(start_x_buttons_row2 + 2 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row2, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': steepest_ascent_hill_climbing_solve},
        {'text': 'STHC', 'rect': pygame.Rect(start_x_buttons_row2 + 3 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row2, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': stochastic_hill_climbing_solve},
        {'text': 'SA', 'rect': pygame.Rect(start_x_buttons_row2 + 4 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row2, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': simulated_annealing_solve},
        {'text': 'Beam', 'rect': pygame.Rect(start_x_buttons_row2 + 5 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row2, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': beam_search_solve},
        
        {'text': 'GA', 'rect': pygame.Rect(start_x_buttons_row3 + 0 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row3, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': genetic_algorithm_solve},
        {'text': 'AND-OR', 'rect': pygame.Rect(start_x_buttons_row3 + 1 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row3, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': and_or_tree_search_solve},
        {'text': 'Belief', 'rect': pygame.Rect(start_x_buttons_row3 + 2 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row3, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': belief_state_search_console},
        {'text': 'PO', 'rect': pygame.Rect(start_x_buttons_row3 + 3 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row3, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': partially_observable_search_console},
        {'text': 'MinConf', 'rect': pygame.Rect(start_x_buttons_row3 + 4 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row3, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': min_conflicts_solve},
        {'text': 'BT', 'rect': pygame.Rect(start_x_buttons_row3 + 5 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row3, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': backtracking_solve},
        
        {'text': 'Q-Learn', 'rect': pygame.Rect(start_x_buttons_row4 + 0 * (BUTTON_WIDTH + BUTTON_SPACING), button_y_row4, BUTTON_WIDTH, BUTTON_HEIGHT), 'func': q_learning_solve},
    ]
    
    current_state = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    initial_user_state = None 
    selected_cell = None
    last_message = "Click a cell and press 0-8 to set a number"
    last_time_message = ""
    states_history = []
    scroll_offset = 0
    dragging_scrollbar = False
    drag_start_y = 0
    drag_start_offset = 0
    
    screen.fill(BACKGROUND_COLOR)
    draw_grid(screen, current_state, GRID_OFFSET_X, GRID_OFFSET_Y, selected_cell)
    draw_buttons(screen, buttons, (0,0)) 
    draw_message(screen, last_message, 35, HEIGHT - 100)
    draw_state_area(screen, states_history, scroll_offset)
    pygame.display.flip()
    
    while True:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos 
                if (GRID_OFFSET_X <= pos[0] < GRID_OFFSET_X + GRID_WIDTH and
                        GRID_OFFSET_Y <= pos[1] < GRID_OFFSET_Y + GRID_HEIGHT):
                    r = (pos[1] - GRID_OFFSET_Y) // TILE_SIZE
                    c = (pos[0] - GRID_OFFSET_X) // TILE_SIZE
                    selected_cell = (r, c)
                    last_message = f"Selected cell ({r}, {c}). Press 0-8 to set a number."
                else:
                    total_hist_height, scrollbar_render_height = draw_state_area(screen, states_history, scroll_offset)
                    content_area_hist_height = STATE_AREA_HEIGHT - TITLE_HEIGHT
                    if total_hist_height > content_area_hist_height:
                        scroll_range_px = content_area_hist_height - scrollbar_render_height
                        current_scroll_px = 0
                        if (total_hist_height - content_area_hist_height) > 0:
                             current_scroll_px = int((scroll_offset / (total_hist_height - content_area_hist_height)) * scroll_range_px)

                        scrollbar_rect_check = pygame.Rect(
                            STATE_AREA_X + STATE_AREA_WIDTH - SCROLLBAR_WIDTH - 5,
                            STATE_AREA_Y + TITLE_HEIGHT + current_scroll_px,
                            SCROLLBAR_WIDTH,
                            scrollbar_render_height
                        )
                        if scrollbar_rect_check.collidepoint(pos):
                            dragging_scrollbar = True
                            drag_start_y = pos[1]
                            drag_start_offset = scroll_offset
                            selected_cell = None
                    button_clicked_on_this_event = False
                    for button in buttons:
                        if button['rect'].collidepoint(pos):
                            button_clicked_on_this_event = True
                            if button['text'] == 'Restart':
                                current_state = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
                                initial_user_state = None
                                selected_cell = None
                                last_message = "Puzzle restarted! Click a cell to start."
                                last_time_message = ""
                                states_history = []
                                scroll_offset = 0
                            elif button['text'] == 'Reset':
                                if initial_user_state:
                                    current_state = copy.deepcopy(initial_user_state)
                                    selected_cell = None
                                    last_message = "Reset to initial user state."
                                else:
                                    last_message = "No initial state to reset to. Use Restart."
                                last_time_message = ""
                                states_history = [] 
                                scroll_offset = 0
                            elif button['text'] == 'Belief':
                                last_message = "Belief State Search: Inputting states..."
                                last_time_message = "See console for detailed results."
                                screen.fill(BACKGROUND_COLOR)
                                draw_grid(screen, current_state, GRID_OFFSET_X, GRID_OFFSET_Y, selected_cell)
                                draw_buttons(screen, buttons, mouse_pos)
                                draw_state_area(screen, states_history, scroll_offset)
                                draw_message(screen, last_message, 35, HEIGHT - 100, MESSAGE_COLOR)
                                draw_message(screen, last_time_message, 35, HEIGHT - 60)
                                pygame.display.flip()

                                start_states = input_two_start_states_window(screen)
                                if start_states:
                                    solution, exec_time = belief_state_search_console(GOAL_STATE, start_states=start_states)
                                    if solution:
                                        last_message = "Belief Search completed (see console)."
                                    else:
                                        last_message = "Belief Search: No solution (see console)."
                                    last_time_message = f"Time: {exec_time:.6f} seconds"
                                else:
                                    last_message = "Belief Search cancelled."
                                    last_time_message = ""
                            elif button['text'] == 'PO':
                                last_message = "Partially Observable: Inputting ratio..."
                                last_time_message = "See console for detailed results."
                                screen.fill(BACKGROUND_COLOR)
                                draw_grid(screen, current_state, GRID_OFFSET_X, GRID_OFFSET_Y, selected_cell)
                                draw_buttons(screen, buttons, mouse_pos)
                                draw_state_area(screen, states_history, scroll_offset)
                                draw_message(screen, last_message, 35, HEIGHT - 100, MESSAGE_COLOR)
                                draw_message(screen, last_time_message, 35, HEIGHT - 60)
                                pygame.display.flip()

                                observation_ratio = input_observation_ratio_window(screen)
                                if observation_ratio is not None:
                                    solution, exec_time = partially_observable_search_console(current_state, GOAL_STATE, observation_ratio)
                                    if solution:
                                        last_message = "PO Search completed (see console)."
                                    else:
                                        last_message = "PO Search: No solution (see console)."
                                    last_time_message = f"Time: {exec_time:.6f} seconds"
                                else:
                                    last_message = "PO Search cancelled."
                                    last_time_message = ""
                            elif button['text'] == 'BT':
                                last_message = "Running Backtracking..."
                                last_time_message = ""
                                states_history = []
                                scroll_offset = 0
                                screen.fill(BACKGROUND_COLOR)
                                draw_grid(screen, current_state, GRID_OFFSET_X, GRID_OFFSET_Y, selected_cell)
                                draw_buttons(screen, buttons, mouse_pos)
                                draw_state_area(screen, states_history, scroll_offset)
                                draw_message(screen, last_message, 35, HEIGHT - 100, MESSAGE_COLOR)
                                pygame.display.flip()

                                solution_states, exec_time = backtracking_solve(
                                    goal=GOAL_STATE,
                                    screen=screen,
                                    grid_offset_x=GRID_OFFSET_X,
                                    grid_offset_y=GRID_OFFSET_Y
                                )
                                if solution_states:
                                    current_state = copy.deepcopy(solution_states[-1])
                                    states_history = solution_states
                                    last_message = "Backtracking completed!"
                                else:
                                    last_message = "Backtracking: No solution."
                                last_time_message = f"Time: {exec_time:.6f} seconds"

                            elif button['text'] not in ['Reset', 'Restart']:
                                flat_state_check = [current_state[i][j] for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
                                if sorted(flat_state_check) != list(range(9)):
                                    last_message = "Invalid state! Must use numbers 0-8 exactly once."
                                else:
                                    if initial_user_state is None:
                                        initial_user_state = copy.deepcopy(current_state)
                                    
                                    states_history = [copy.deepcopy(current_state)]
                                    scroll_offset = 0
                                    
                                    if button['text'] == 'Q-Learn':
                                        last_message = "Q-Learning: Training model..."
                                        last_time_message = "This may take a moment."
                                        screen.fill(BACKGROUND_COLOR)
                                        draw_grid(screen, current_state, GRID_OFFSET_X, GRID_OFFSET_Y, selected_cell)
                                        draw_buttons(screen, buttons, pos) 
                                        draw_state_area(screen, states_history, scroll_offset)
                                        draw_message(screen, last_message, 35, HEIGHT - 100, MESSAGE_COLOR)
                                        draw_message(screen, last_time_message, 35, HEIGHT - 60)
                                        pygame.display.flip()

                                    solution, exec_time = button['func'](current_state, GOAL_STATE)
                                    
                                    if not solution:
                                        last_message = f"{button['text']}: No solution found!"
                                    else:
                                        temp_anim_state = copy.deepcopy(states_history[0])
                                        
                                        for i, move_coord in enumerate(solution):
                                            r_old_blank, c_old_blank = -1, -1
                                            for r_idx in range(GRID_SIZE):
                                                for c_idx in range(GRID_SIZE):
                                                    if temp_anim_state[r_idx][c_idx] == 0:
                                                        r_old_blank, c_old_blank = r_idx, c_idx
                                                        break
                                                if r_old_blank != -1:
                                                    break
                                            
                                            if r_old_blank == -1:
                                                last_message = f"{button['text']}: Anim error (no blank)."
                                                solution = []
                                                break

                                            nr_new_blank, nc_new_blank = move_coord
                                            if abs(r_old_blank - nr_new_blank) + abs(c_old_blank - nc_new_blank) == 1:
                                                temp_anim_state[r_old_blank][c_old_blank], temp_anim_state[nr_new_blank][nc_new_blank] = \
                                                    temp_anim_state[nr_new_blank][nc_new_blank], temp_anim_state[r_old_blank][c_old_blank]
                                                
                                                states_history.append(copy.deepcopy(temp_anim_state))
                                            else:
                                                print(f"Warning: Invalid move in solution path for {button['text']}: from blank at ({r_old_blank},{c_old_blank}) to new blank at ({nr_new_blank},{nc_new_blank})")
                                                last_message = f"{button['text']}: Invalid step in solution!"
                                                solution = [] 
                                                break
                                            state_block_h = STATE_TILE_SIZE * GRID_SIZE + STATE_SPACING + 25
                                            total_hist_h_anim = len(states_history) * state_block_h
                                            content_area_h_anim = STATE_AREA_HEIGHT - TITLE_HEIGHT
                                            if total_hist_h_anim > content_area_h_anim:
                                                scroll_offset = max(0, total_hist_h_anim - content_area_h_anim)
                                            
                                            screen.fill(BACKGROUND_COLOR)
                                            draw_grid(screen, temp_anim_state, GRID_OFFSET_X, GRID_OFFSET_Y)
                                            draw_buttons(screen, buttons, pos) 
                                            draw_state_area(screen, states_history, scroll_offset)
                                            anim_msg = f"{button['text']}: Step {i+1}/{len(solution)}"
                                            draw_message(screen, anim_msg, 35, HEIGHT - 100, MESSAGE_COLOR)
                                            pygame.display.flip()
                                            pygame.time.delay(300)
                                        
                                        if solution:
                                            current_state = copy.deepcopy(states_history[-1])
                                            last_message = f"{button['text']}: Solution found!"
                                            
                                    last_time_message = f"Time: {exec_time:.6f} seconds"
                            selected_cell = None 
                            break 
                    
                    if not button_clicked_on_this_event and not dragging_scrollbar:
                        selected_cell = None

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging_scrollbar = False

            if event.type == pygame.MOUSEMOTION and dragging_scrollbar:
                total_hist_height_motion, scrollbar_render_height_motion = draw_state_area(screen, states_history, scroll_offset)
                content_area_hist_height_motion = STATE_AREA_HEIGHT - TITLE_HEIGHT
                if total_hist_height_motion > content_area_hist_height_motion:
                    delta_y = event.pos[1] - drag_start_y
                    scroll_range_content = total_hist_height_motion - content_area_hist_height_motion
                    scroll_range_bar_px = content_area_hist_height_motion - scrollbar_render_height_motion
                    if scroll_range_bar_px > 0:
                         scroll_offset = drag_start_offset + (delta_y / scroll_range_bar_px) * scroll_range_content
                    scroll_offset = max(0, min(scroll_offset, scroll_range_content))
            
            if event.type == pygame.MOUSEWHEEL:
                if (STATE_AREA_X <= mouse_pos[0] <= STATE_AREA_X + STATE_AREA_WIDTH and
                        STATE_AREA_Y <= mouse_pos[1] <= STATE_AREA_Y + STATE_AREA_HEIGHT):
                    
                    state_block_h_wheel = STATE_TILE_SIZE * GRID_SIZE + STATE_SPACING + 25
                    total_hist_h_wheel = len(states_history) * state_block_h_wheel
                    content_area_h_wheel = STATE_AREA_HEIGHT - TITLE_HEIGHT
                    
                    if total_hist_h_wheel > content_area_h_wheel:
                        scroll_offset -= event.y * 30
                        scroll_offset = max(0, min(scroll_offset, total_hist_h_wheel - content_area_h_wheel))

            if event.type == pygame.KEYDOWN and selected_cell:
                if event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                                 pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8]:
                    value = int(event.unicode)
                    r_sel, c_sel = selected_cell
                    temp_flat_state = []
                    for i_row in range(GRID_SIZE):
                        for i_col in range(GRID_SIZE):
                            if not (i_row == r_sel and i_col == c_sel):
                                temp_flat_state.append(current_state[i_row][i_col])
                    
                    if value not in temp_flat_state or value == 0:
                        current_state[r_sel][c_sel] = value
                        last_message = f"Set {value} at ({r_sel}, {c_sel})."
                        if initial_user_state is None and sorted([item for row in current_state for item in row if item is not None]) == list(range(9)):
                            initial_user_state = copy.deepcopy(current_state)
                    else:
                        last_message = f"Invalid! {value} is already used."

        
        screen.fill(BACKGROUND_COLOR)
        draw_grid(screen, current_state, GRID_OFFSET_X, GRID_OFFSET_Y, selected_cell)
        draw_buttons(screen, buttons, mouse_pos)
        draw_state_area(screen, states_history, scroll_offset)
        
        final_message_color = MESSAGE_COLOR
        if "Solution found!" in last_message or "completed!" in last_message :
            final_message_color = SOLVED_COLOR
        elif "Invalid" in last_message or "No solution" in last_message or "Error" in last_message or "cancelled" in last_message:
            final_message_color = (255,100,100)

        if last_message:
            draw_message(screen, last_message, 35, HEIGHT - 100, final_message_color)
        if last_time_message:
            draw_message(screen, last_time_message, 35, HEIGHT - 60)
            
        pygame.display.flip()

if __name__ == "__main__":
    main()
    pygame.quit()