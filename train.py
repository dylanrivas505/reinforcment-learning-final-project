import numpy as np
from cruise_control_model import CruiseControlModel
from q_learning_agent import QLearningAgent

def main():
    # Build cruise control environment
    env = CruiseControlModel(
        delta=0.1,
        x_bounds=(-5, 5),
        v_bounds=(-5, 5),
        w_x=1.0, w_v=0.1, w_u=0.01,
        tol_x=0.1, tol_v=0.1,
        done_bonus=200.0
    )
    # Make agent
    actions = list(np.linspace(-1.0, 1.0, 21))
    agent = QLearningAgent(
        env,
        num_x_bins=20,
        num_v_bins=20,
        action_list=actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999
    )
    # train the agent
    rewards = agent.train(
        episodes=1000,
        max_steps=200,
        init_state=(5.0, 0.0)
    )
    # Evaluate the performance
    metrics = agent.evaluate(
        episodes=200,
        max_steps=200,
        init_state=(5.0, 0.0)
    )
    print("Evaluation:", metrics)
if __name__ == "__main__":
    main()