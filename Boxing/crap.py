import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from skimage.transform import resize

env = gym.make('Boxing-v0')

done = False

print(env.action_space)
print(env.observation_space)

state = env.reset()
c = 0
while not done:
    c += 1
    env.render()
    state, reward, done, _ = env.step(0)
    # state = state[35:195:2, ::2, 0]
    state = state[25:185,23:136,0]
    state = resize(state,(30,30),mode = 'constant')

    state = state.astype(np.float)
    if c == 50:
        plt.imsave('{}.png'.format(c), state, cmap='gray')
        print(state.shape)