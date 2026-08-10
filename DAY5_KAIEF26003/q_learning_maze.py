import random


class GridMazeEnv:
    def __init__(self):
        # 0 = open path, 1 = wall/obstacle
        self.grid = [
            [0, 0, 0, 0],
            [1, 1, 0, 1],
            [0, 0, 0, 0],
            [0, 1, 1, 0]
        ]
        self.start = (0, 0)
        self.goal = (3, 3)
        self.state = self.start
        # Action map: 0 = up, 1 = right, 2 = down, 3 = left
        self.actions = [0, 1, 2, 3]

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        r, c = self.state

        if action == 0:    # Up
            nr, nc = r - 1, c
        elif action == 1:  # Right
            nr, nc = r, c + 1
        elif action == 2:  # Down
            nr, nc = r + 1, c
        elif action == 3:  # Left
            nr, nc = r, c - 1
        else:
            nr, nc = r, c

        # Out-of-bounds or hit a wall
        if nr < 0 or nr >= 4 or nc < 0 or nc >= 4 or self.grid[nr][nc] == 1:
            return self.state, -5, False  # Penalty for invalid move

        self.state = (nr, nc)

        if self.state == self.goal:
            return self.state, 100, True  # Goal reward

        return self.state, -1, False  # Step cost to encourage shortest path


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration factor

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, possible_actions):
        # Epsilon-greedy action choice
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        
        # Pick action with maximum Q-value
        return max(possible_actions, key=lambda a: self.get_q(state, a))

    def learn(self, state, action, reward, next_state, possible_actions):
        # Bellman equation update rule
        max_next_q = max([self.get_q(next_state, a) for a in possible_actions], default=0.0)
        current_q = self.get_q(state, action)
        
        updated_q = current_q + self.alpha * (reward + (self.gamma * max_next_q) - current_q)
        self.q_table[(state, action)] = updated_q


def main():
    env = GridMazeEnv()
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2)

    episodes = 2000
    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 100:
            action = agent.choose_action(state, env.actions)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, env.actions)
            state = next_state
            steps += 1

    # Disable exploration to test learned policy
    agent.epsilon = 0.0
    state = env.reset()
    path = [state]
    done = False
    
    while not done and len(path) < 20:
        action = agent.choose_action(state, env.actions)
        next_state, _, done = env.step(action)
        state = next_state
        path.append(state)

    print("--- Q-Learning Maze Traversal ---")
    print("Learned optimal route from Start (0,0) to Goal (3,3):")
    print(" -> ".join(str(pos) for pos in path))


if __name__ == "__main__":
    main()