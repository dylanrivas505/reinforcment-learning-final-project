from PIL import Image
import numpy as np

class GridMapEnvironment:
    def __init__(self,
                 bmp_path: str,
                 threshold: int = 127,
                 start: tuple = (0,0),
                 goal: tuple = (0,0),
                 # reward parameters
                 step_cost: float = -0.5,
                 obstacle_cost: float = -5.0,
                 goal_reward: float = 500.0):
        # Load BMP into 2d array using Pillow library
        img = Image.open(bmp_path).convert('L') 
        data = np.array(img)
        # 1=obstacle, 0=free
        self.grid = (data <= threshold).astype(int)
        # Store dimensions in grid
        self.n_rows, self.n_cols = self.grid.shape
        # Store reward parameters
        self.step_cost     = step_cost
        self.obstacle_cost = obstacle_cost
        self.goal_reward   = goal_reward
        self.start = start
        self.goal  = goal
        self.reset()

    def reset(self) -> tuple:
        # reset state
        self.state = self.start
        return self.state
    
    def step(self, action: int):
        # use manhattan distance for more reward tracking
        old_dist = manhattan(self.state, self.goal)
        r, c = self.state
        # compute possible next cell
        if action == 0:    nr, nc = r-1, c
        elif action == 1:  nr, nc = r,   c+1
        elif action == 2:  nr, nc = r+1, c
        else:              nr, nc = r,   c-1
        # check validity
        if (0 <= nr < self.n_rows and 0 <= nc < self.n_cols 
            and self.grid[nr, nc] == 0):
            # valid move
            self.state = (nr, nc)
            # step cost
            reward = self.step_cost
            done   = False
            # check for the goal
            if self.state == self.goal:
                reward += self.goal_reward
                done = True
        else:
            # invalid move → penalize but stay in place
            reward = self.obstacle_cost
            done   = False
        new_dist = manhattan(self.state, self.goal)
        shaping = old_dist - new_dist
        reward += shaping
        return self.state, reward, done

def manhattan(a, b):
        # simple manhattan distance calculation
        return abs(a[0] - b[0]) + abs(a[1] - b[1])