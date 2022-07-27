import gym
import sys
import numpy as np
import matplotlib.pyplot as plt
from QLearningAgent import Agent


if __name__ == '__main__':
    env = gym.make("FrozenLake-v1")
    agent = Agent(lr=0.001, gamma=0.9, eps_start=1.0, eps_end=0.001, eps_decay=0.9999995, n_actions=4, n_states=16)
    scores = []
    win_pct = []
    n_games = 500000

    for i in range(1, n_games+1):
        state = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            score += reward
        scores.append(score)
        if i % 100 == 0:
            win_pct.append(np.mean(scores[-100:]) * 100)
            if i % 1000 == 0:
                sys.stdout.write(f"\rEpisode {i}/{n_games} | Win %: {np.mean(win_pct[-10:]):.2f} | Score: {score:.2f} | Epsilon: {agent.epsilon:.2f}")
                sys.stdout.flush()
    plt.plot(win_pct)
    plt.show() 
    agent.save("frozenlake_agent.pkl")