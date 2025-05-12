import numpy as np
from tqdm import trange

class GridQLearningAgent:
    def __init__(self,
                 env,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Discrete state dimensionss and action set
        self.n_rows, self.n_cols = env.n_rows, env.n_cols
        # up, right, down, left
        self.actions = [0, 1, 2, 3]
        # one value per row, column or action
        self.q_table = np.zeros((self.n_rows,
                                  self.n_cols,
                                  len(self.actions)))
    def select_action(self, state: tuple) -> int:
        r, c = state
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        # greedy choice
        return int(np.argmax(self.q_table[r, c, :]))

    def update(self,
               state: tuple,
               action: int,
               reward: float,
               next_state: tuple,
               done: bool):
        r, c = state
        nr, nc = next_state
        a = action
        # estimate of optimal future value
        best_next = 0.0 if done else np.max(self.q_table[nr, nc, :])
        target    = reward + self.gamma * best_next
        # TD error and Q-table update
        self.q_table[r, c, a] += self.alpha * (target - self.q_table[r, c, a])

    def train(self,
              episodes: int = 500,
              max_steps: int = 200) -> list:
        # Train the agent
        episode_rewards = []
        for ep in trange(episodes, desc="Training"):
            state = self.env.reset()
            total_reward = 0.0
            # Loop until done or reaching max steps
            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            episode_rewards.append(total_reward)
            # decay exploration
            self.epsilon *= self.epsilon_decay
        return episode_rewards
    
    # Q-learning evaluation
    def evaluate(self,
                 episodes: int = 100,
                 max_steps: int = 200) -> dict:
        total_returns   = []
        successes       = 0
        steps_to_success = []
        # temporarily turn off exploration
        old_epsilon, self.epsilon = self.epsilon, 0.0
        for _ in range(episodes):
            state = self.env.reset()
            cumulative_reward = 0.0

            for t in range(1, max_steps+1):
                # greedy choosing
                action = int(np.argmax(self.q_table[state[0], state[1], :]))
                state, reward, done = self.env.step(action)
                cumulative_reward += reward
                if done:
                    successes += 1
                    steps_to_success.append(t)
                    break
            total_returns.append(cumulative_reward)
        # restore exploration rate
        self.epsilon = old_epsilon
        avg_return   = np.mean(total_returns)
        success_rate = successes / episodes
        avg_steps    = (np.mean(steps_to_success)
                        if steps_to_success else float('nan'))
        return {
            'avg_return':   avg_return,
            'success_rate': success_rate,
            'avg_steps':    avg_steps
        }

