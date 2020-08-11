import numpy as np

def preprocess(image):
    image = image[35:195:2, ::2, 0]
    image[np.logical_or(image == 144, image == 109)] = 0
    image[image != 0] = 1
    return image.astype(np.float)


def discounted_rewards(rew, gamma=0.99):
    d_r = np.zeros_like(rew)
    p_r = 0
    for t in range(rew.size - 1, 0, -1):
        if rew[t] != 0: p_r = 0
        p_r = p_r * gamma + rew[t]
        d_r[t] = p_r
    d_r -= np.mean(d_r)
    d_r /= np.std(d_r)
    return d_r