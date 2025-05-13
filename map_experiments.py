import itertools
import pandas as pd
import numpy as np

from map_solving_environment import GridMapEnvironment
from map_q_agent import MapQLearningAgent 

def run_experiments(maps,
                    episode_length_settings,
                    param_grid,
                    eval_episodes: int = 100):
    keys = list(param_grid.keys())
    records = []
    # for each map
    for bmp in maps:
        # for each setting
        for setting in episode_length_settings:
            E, M = setting['episodes'], setting['max_steps']
            # for each combination of alpha & gamma
            for vals in itertools.product(*param_grid.values()):
                params = dict(zip(keys, vals))
                # build environment on this map
                env = GridMapEnvironment(
                    bmp_path=bmp,
                    threshold=127,
                    start=(1,1),
                    goal=(10,10),
                    step_cost=-1.0,
                    obstacle_cost=-5.0,
                    goal_reward=100.0
                )
                # build q-learning agent
                agent = MapQLearningAgent(
                    env,
                    alpha=params['alpha'],
                    gamma=params['gamma'],
                    epsilon=1.0,
                    epsilon_decay=0.995
                )
                # train agent
                agent.train(
                    episodes=E,
                    max_steps=M
                )
                # evaluate metrics
                metrics = agent.evaluate(
                    episodes=eval_episodes,
                    max_steps=M
                )
                # record results
                record = {
                    'map':       bmp,
                    'episodes':  E,
                    'max_steps': M,
                    **params,
                    **metrics
                }
                records.append(record)
    return pd.DataFrame(records)

if __name__ == "__main__":
    # 4 matched tiers
    episode_length_settings = [
        {'episodes': 10, 'max_steps': 10},
        {'episodes': 50, 'max_steps': 50},
        {'episodes': 57, 'max_steps': 57},
        {'episodes': 100,'max_steps': 100},
    ]
    # 2 learning rates and 2 discounts
    param_grid = {
        'alpha': [0.1, 0.01],
        'gamma': [0.8, 0.99],
    }
    maps = ["maps/map1.bmp", "maps/map2.bmp"]
    # run and save to new csv
    df = run_experiments(maps, episode_length_settings, param_grid)
    df.to_csv("map_experiment_results.csv", index=False)
    print(df)
