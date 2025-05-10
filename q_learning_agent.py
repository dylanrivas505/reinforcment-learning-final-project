import numpy as np
from tqdm import trange

# 1.3 Q Learning Process / Agent
class QLearningAgent:
    def __init__(self,
                 env,
                 num_x_bins: int,
                 num_v_bins: int,
                 action_list: list,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.999):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.actions = action_list

        # build the Q-table (x_bins, v_bins, n_actions)
        self.q_table = np.zeros((num_x_bins, num_v_bins, len(action_list)))

        # precompute bin edges
        self.x_bins = np.linspace(env.x_bounds[0],
                                  env.x_bounds[1],
                                  num_x_bins+1)[1:-1]
        self.v_bins = np.linspace(env.v_bounds[0],
                                  env.v_bounds[1],
                                  num_v_bins+1)[1:-1]

    def _discretize(self, state: np.ndarray) -> tuple:
        # Map the continuious state to indices
        x, v = state
        ix = np.digitize(x, self.x_bins)
        iv = np.digitize(v, self.v_bins)
        return ix, iv

    def select_action(self, state: np.ndarray) -> float:
        # Choose an action based off of epsilon
        ix, iv = self._discretize(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            a_idx = np.argmax(self.q_table[ix, iv, :])
            return self.actions[a_idx]

    def update(self,
               state: np.ndarray,
               action: float,
               reward: float,
               next_state: np.ndarray,
               done: bool):
        # Update Q-learning process
        ix, iv = self._discretize(state)
        a_idx = self.actions.index(action)
        nix, niv = self._discretize(next_state)
        # Make an estimate of optimal future value
        best_next = 0 if done else np.max(self.q_table[nix, niv, :])
        target = reward + self.gamma * best_next
        td_error = target - self.q_table[ix, iv, a_idx]
        self.q_table[ix, iv, a_idx] += self.alpha * td_error
    # train
    def train(self,
              episodes: int = 500,
              max_steps: int = 200,
              init_state: tuple = (5.0, 0.0)):
        # Run Q-Learning
        rewards = []
        # Keep track of progress over episodes
        for ep in trange(episodes, desc="Training"):
            state = self.env.reset(init_state)
            total_r = 0.0
            # Over time steps
            for _ in range(max_steps):
                u = self.select_action(state)
                next_state, r, done = self.env.step(u)
                self.update(state, u, r, next_state, done)
                state = next_state
                total_r += r
                if done:
                    break

            rewards.append(total_r)
            # decay epsilon
            self.epsilon *= self.epsilon_decay

        return rewards
    
    def evaluate(self,
                 episodes: int = 100,
                 max_steps: int = 200,
                 init_state: tuple = (5.0, 0.0)):
        total_returns = []
        successes = 0
        steps_to_success = []
        # Temporarily disable exploration
        old_epsilon = self.epsilon
        self.epsilon = 0.0

        for _ in range(episodes):
            state = self.env.reset(init_state)
            cum_r = 0.0

            for step in range(1, max_steps+1):
                # Greedy algorithm selection
                ix, iv = self._discretize(state)
                a_idx = np.argmax(self.q_table[ix, iv, :])
                u = self.actions[a_idx]

                next_state, r, done = self.env.step(u)
                cum_r += r
                state = next_state

                if done:
                    successes += 1
                    steps_to_success.append(step)
                    break

            total_returns.append(cum_r)
        # Restore exploration
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
        
