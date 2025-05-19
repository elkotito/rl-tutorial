import argparse
from typing import Dict

import matplotlib.pyplot as plt
from gymnasium.envs.toy_text import FrozenLakeEnv

env = FrozenLakeEnv(render_mode="rgb_array", map_name="8x8", is_slippery=False)
env.reset()


def value_iteration(env: FrozenLakeEnv, gamma: float = 0.9, max_iterations: int = 1000, eps: float = 1e-20) -> Dict[
    int, int]:
    value_function = {state: 0.0 for state in range(env.observation_space.n)}

    # Step 1: Estimate the optimal value function
    for i in range(max_iterations):
        new_value_function = {state: 0.0 for state in range(env.observation_space.n)}
        for state in range(env.observation_space.n):
            value = float("-inf")
            for action in env.P[state]:
                for prob, next_state, reward, done in env.P[state][action]:
                    value = max(value, prob * (reward + gamma * value_function[next_state]))

            new_value_function[state] = value

        max_error = max([
            abs(value_function[state] - new_value_function[state])
            for state in range(env.observation_space.n)
        ])

        value_function = new_value_function
        if max_error < eps:
            break

    # Step 2: Extract policy
    terminal_states = set()
    policy = {state: 0 for state in range(env.observation_space.n)}
    for state in range(env.observation_space.n):
        best_action, best_action_value = -1, float("-inf")
        for action in env.P[state]:
            for prob, next_state, reward, done in env.P[state][action]:
                if done:
                    action_value = prob * reward
                    terminal_states.add(next_state)
                else:
                    action_value = prob * (reward + gamma * value_function[next_state])

                if action_value > best_action_value:
                    best_action, best_action_value = action, action_value

            policy[state] = best_action

    # Filter out terminal states
    for state in terminal_states:
        policy[state] = -1

    return policy


def policy_iteration(env: FrozenLakeEnv, gamma: float = 0.9, max_iterations: int = 1000,
                     max_iterations_evaluation: int = 10, eps: float = 1e-20) -> Dict[int, int]:
    value_function = {state: 0.0 for state in range(env.observation_space.n)}
    policy = {state: 0 for state in range(env.observation_space.n)}
    terminal_states = set()

    for i in range(max_iterations):
        # Step 1: Estimate the expected value function
        for _ in range(max_iterations_evaluation):
            new_value_function = {state: 0.0 for state in range(env.observation_space.n)}
            for state in range(env.observation_space.n):
                value = 0.0
                action = policy[state]
                for prob, next_state, reward, done in env.P[state][action]:
                    value += prob * (reward + gamma * value_function[next_state])

                new_value_function[state] = value

            max_error = max([
                abs(value_function[state] - new_value_function[state])
                for state in range(env.observation_space.n)
            ])

            value_function = new_value_function
            if max_error < eps:
                break

        # Step 2: Policy improvement
        new_policy = {state: 0 for state in range(env.observation_space.n)}
        for state in range(env.observation_space.n):
            best_action, best_action_value = -1, float("-inf")
            for action in env.P[state]:
                action_value = 0
                for prob, next_state, reward, done in env.P[state][action]:
                    if done:
                        action_value += prob * reward
                        terminal_states.add(next_state)
                    else:
                        action_value += prob * (reward + gamma * value_function[next_state])

                if action_value > best_action_value:
                    best_action, best_action_value = action, action_value

            new_policy[state] = best_action

        if all(policy[state] == new_policy[state] for state in range(env.observation_space.n)):
            break

        policy = new_policy

    # Filter out terminal states
    for state in terminal_states:
        policy[state] = -1

    return policy


def plot(env: FrozenLakeEnv, policy: Dict[int, int]):
    rgb_array = env.render()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb_array)
    ax.set_xticks([])
    ax.set_yticks([])
    arrow_map = {-1: 'X', 0: '←', 1: '↓', 2: '→', 3: '↑'}

    img_height, img_width, _ = rgb_array.shape
    cell_size_x = img_width / env.ncol
    cell_size_y = img_height / env.nrow

    # Draw policy arrows
    for row in range(env.nrow):
        for col in range(env.ncol):
            state = row * env.ncol + col
            action = policy[state]
            if action != -1:
                ax.text(
                    col * cell_size_x + cell_size_x / 2,
                    row * cell_size_y + cell_size_y / 2,
                    arrow_map[action],
                    ha='center', va='center',
                    color='red', fontsize=16,
                    fontweight='bold',
                    transform=ax.transData
                )

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Run model-based reinforcement learning algorithms.')
    parser.add_argument('--algorithm', type=str, choices=['value_iteration', 'policy_iteration'],
                        default='policy_iteration', help='Algorithm to use: value_iteration or policy_iteration')
    args = parser.parse_args()

    if args.algorithm == 'value_iteration':
        policy = value_iteration(env)
    else:
        policy = policy_iteration(env)

    plot(env, policy)


if __name__ == "__main__":
    main()
