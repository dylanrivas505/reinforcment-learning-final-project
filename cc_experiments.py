import itertools
import pandas as pd
import numpy as np

from cruise_control_model import CruiseControlModel
from cc_q_agent import QLearningAgent

def run_experiments(episode_length_settings, param_grid, init_state=(5.0,0.0), eval_episodes=200):
    keys = list(param_grid.keys())
    records = []

    # iterate over three (episodes, max_steps) settings
    for setting in episode_length_settings:
        episodes  = setting['episodes']
        max_steps = setting['max_steps']
        # now the Cartesian product over hyper parameters
        for vals in itertools.product(*param_grid.values()):
            params = dict(zip(keys, vals))
            # build environment with cruise control model
            env = CruiseControlModel(
                delta=0.1,
                x_bounds=(-5,5),
                v_bounds=(-5,5),
                w_x=1.0, w_v=0.1, w_u=0.01,
                tol_x=0.1, tol_v=0.1,
                done_bonus=200.0
            )
            # build agent
            actions = list(np.linspace(-1.0, 1.0, params['action_count']))
            agent = QLearningAgent(
                env,
                num_x_bins=params['num_x_bins'],
                num_v_bins=params['num_v_bins'],
                action_list=actions,
                alpha=params['alpha'],
                gamma=params['gamma'],
                epsilon=1.0,
                epsilon_decay=0.999
            )
            # train with the episodes and max length
            _ = agent.train(
                episodes=episodes,
                max_steps=max_steps,
                init_state=init_state
            )
            # evaluate the metrics
            metrics = agent.evaluate(
                episodes=eval_episodes,
                max_steps=max_steps,
                init_state=init_state
            )
            # Record everything
            record = {
                'episodes':    episodes,
                'max_steps':   max_steps,
                **params,
                **metrics
            }
            records.append(record)
    # return the df
    return pd.DataFrame(records)

# Experiment form 1.5 of assignment
if __name__ == "__main__":
    episode_length_settings = [
        # three different combined settings
        {'episodes': 100,  'max_steps': 100},
        {'episodes': 400,  'max_steps': 400},
        {'episodes': 900, 'max_steps': 900},
    ]
    param_grid = {
        'num_x_bins':   [11, 31, 51],
        'num_v_bins':   [11, 31, 51],
        'action_count': [11, 31, 51],
        # Learning rates
        'alpha':        [0.005, 0.02],
        # Discount factors
        'gamma':        [0.85, 0.9],
    }

    df = run_experiments(episode_length_settings, param_grid)
    df.to_csv("experiment_results.csv", index=False)
    print(df)
