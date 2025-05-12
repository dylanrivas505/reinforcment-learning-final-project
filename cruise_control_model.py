import numpy as np
from scipy.integrate import odeint
from q_learning_agent import QLearningAgent


class CruiseControlModel:
    # Cruise Control Environment
    def __init__(self,
                 delta: float = 0.1,
                 x_bounds: tuple = (-5.0, 5.0),
                 v_bounds: tuple = (-5.0, 5.0),
                 # reward weights
                 w_x: float = 1.0,
                 w_v: float = 0.1,
                 w_u: float = 0.01,
                 # success thresholds
                 tol_x: float = 0.1,
                 tol_v: float = 0.1,
                 done_bonus: float = 250.0):
        # time step and bounds
        self.delta = delta
        self.x_bounds = x_bounds
        self.v_bounds = v_bounds
        self._t_grid = np.linspace(0, delta, 2)
        # reward parameters
        self.w_x, self.w_v, self.w_u = w_x, w_v, w_u
        self.tol_x, self.tol_v = tol_x, tol_v
        self.done_bonus = done_bonus
        self.discretizer = None
        self.actions = None
        self.state = None

    def model(self, s: np.ndarray, t: float, u: float) -> list:
        # ODE model right side
        x, v = s
        return [v, u]

    def reset(self, initial_state: tuple = (0.0, 0.0)) -> np.ndarray:
        # Returns new state for reset
        x0, v0 = initial_state
        x0 = np.clip(x0, *self.x_bounds)
        v0 = np.clip(v0, *self.v_bounds)
        self.state = np.array([x0, v0], dtype=float)
        return self.state

    def step(self, u: float):
        # Apply stepping of ODE for agent interaction
        # Use odeint to numerically compute reachable state of the system
        traj = odeint(self.model, self.state, self._t_grid, args=(u,))
        next_x, next_v = traj[-1]
        # Clip to bounds
        next_x = np.clip(next_x, *self.x_bounds)
        next_v = np.clip(next_v, *self.v_bounds)
        self.state = np.array([next_x, next_v])
        # compute cost function = x² + 0.1·v² + 0.01·u²
        cost = self.w_x * next_x**2 + self.w_v * next_v**2 + self.w_u * u**2
        reward = -cost
        # check if close to zero
        done = (abs(next_x) < self.tol_x) and (abs(next_v) < self.tol_v)
        if done:
            reward += self.done_bonus
        return self.state, reward, done

    def generate_trajectory(self, q_table, s0: tuple, max_steps: int = 200):
        # Generate trajectory from a given intitial state regarding to a given Q-table
        self.reset(s0)
        trajectory = []
        # Goes until reaching max steps
        for _ in range(max_steps):
            x, v = self.state
            u = self._select_action_from_q(q_table, x, v)
            next_s, r, done = self.step(u)
            trajectory.append((self.state.copy(), u, r))
            if done:
                break
        # Returns state, action, and reward trajectory
        return trajectory

    def _select_action_from_q(self, q_table, x: float, v: float) -> float:
        # Select action from q-table
        ix, iv = self.discretizer(np.array([x, v]))
        a_idx  = np.argmax(q_table[ix, iv, :])
        return self.actions[a_idx]