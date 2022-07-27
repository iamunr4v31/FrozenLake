import gym
import numpy as np
import matplotlib.pyplot as plt


# LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3
'''
    SFFF
    FHFH
    FFFH
    HFFG
'''

policy = {
    0: 1, 1: 2, 2: 1, 3: 0,
    4: 1,       6: 1,
    8: 2, 9: 2, 10: 1,
          13: 2, 14: 2
    }

env = gym.make("FrozenLake-v1")

n_games = 1000
win_pct = []
scores = []

for i in range(1, n_games+1):
    done = False
    obs = env.reset()
    score = 0
    while not done:
        action = policy[obs]
        obs, reward, done, info = env.step(action)
        score+=reward
    scores.append(score)

    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_pct.append(average)

plt.plot(win_pct)
plt.show()