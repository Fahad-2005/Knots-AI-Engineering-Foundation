import sys
import heapq

class Node:
    def __init__(self, state, parent=None, action=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)


class StackFrontier:
    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("Frontier is empty")
        return self.frontier.pop()


class QueueFrontier(StackFrontier):
    def remove(self):
        if self.empty():
            raise Exception("Frontier is empty")
        return self.frontier.pop(0)


class PriorityQueueFrontier:
    def __init__(self):
        self.frontier = []
        self.counter = 0

    def add(self, node):
        self.counter += 1
        priority = node.cost + node.heuristic
        heapq.heappush(self.frontier, (priority, self.counter, node))

    def contains_state(self, state):
        return any(item[2].state == state for item in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if self.empty():
            raise Exception("Frontier is empty")
        return heapq.heappop(self.frontier)[2]


class Maze:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            contents = f.read()

        lines = contents.splitlines()
        self.height = len(lines)
        self.width = max(len(line) for line in lines)
        self.walls = []

        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if lines[i][j] == "A":
                        self.start = (i, j)
                        row.append(False)
                    elif lines[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif lines[i][j] == "#":
                        row.append(True)
                    else:
                        row.append(False)
                except IndexError:
                    row.append(False)
            self.walls.append(row)

    def actions(self, state):
        row, col = state
        candidates = [
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]
        
        valid_moves = []
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                valid_moves.append((action, (r, c)))
        return valid_moves

    def manhattan_distance(self, state):
        return abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])

    def solve(self, algorithm="BFS"):
        self.num_explored = 0
        start_node = Node(state=self.start, cost=0, heuristic=self.manhattan_distance(self.start))

        if algorithm == "DFS":
            frontier = StackFrontier()
        elif algorithm == "BFS":
            frontier = QueueFrontier()
        elif algorithm in ["Greedy", "A*"]:
            frontier = PriorityQueueFrontier()

        frontier.add(start_node)
        self.explored = set()

        while True:
            if frontier.empty():
                return None, self.num_explored

            node = frontier.remove()
            self.num_explored += 1

            if node.state == self.goal:
                path = []
                while node.parent is not None:
                    path.append(node.state)
                    node = node.parent
                path.reverse()
                return path, self.num_explored

            self.explored.add(node.state)

            for action, state in self.actions(node.state):
                if state not in self.explored and not frontier.contains_state(state):
                    h = self.manhattan_distance(state)
                    g = 0 if algorithm == "Greedy" else node.cost + 1
                    child = Node(state=state, parent=node, action=action, cost=g, heuristic=h)
                    frontier.add(child)


def benchmark(maze_file):
    print(f"{'Algorithm':<12} | {'Path length':<12} | {'States explored':<15}")
    print("-" * 45)
    
    for algo in ["DFS", "BFS", "Greedy", "A*"]:
        maze = Maze(maze_file)
        path, explored = maze.solve(algorithm=algo)
        path_len = len(path) if path else 0
        print(f"{algo:<12} | {path_len:<12} | {explored:<15}")


if __name__ == "__main__":
    benchmark("mazes/maze1.txt")