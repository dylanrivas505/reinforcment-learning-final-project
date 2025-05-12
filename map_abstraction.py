from PIL import Image
import numpy as np

class GridMapEnv:
    def __init__(self,
                 bmp_path: str,
                 threshold: int = 127,
                 start: tuple = (0,0),
                 goal: tuple = (0,0)):
        # Load BMP into 2d array using Pillow library
        img = Image.open(bmp_path).convert('L') 
        data = np.array(img)
        # 1=obstacle, 0=free
        self.grid = (data > threshold).astype(int)\
        # Store dimensions in grid
        self.n_rows, self.n_cols = self.grid.shape
        # Set start and goal cells
        self.start = start
        self.goal  = goal
        # State back to start
        self.state = self.start
    def reset(self) -> tuple:
        # Reset agent
        self.state = self.start
        return self.state

    def step(self, action: int):
        # 0=up, 1=right, 2=down, 3=left
        r, c = self.state
        if action == 0:    nr, nc = r-1, c
        elif action == 1:  nr, nc = r,   c+1
        elif action == 2:  nr, nc = r+1, c
        else:              nr, nc = r,   c-1
        # Check bounds and obstacles
        if (0 <= nr < self.n_rows and 0 <= nc < self.n_cols 
            and self.grid[nr, nc] == 0):
            self.state = (nr, nc)
            # penalty for step
            reward = -1
            done   = False
            if self.state == self.goal:
                # bonus
                reward = +100
                done = True
        else:
            # penalty for wall or outside map
            reward = -5
            done   = False
        return self.state, reward, done
