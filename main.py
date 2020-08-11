import keras
from keras.layers import Dense, Flatten
import gym
import numpy as np
import utils
import model
import warnings

warnings.filterwarnings("ignore") # Ignores all warning messages

def train_agent():
    render = False
    lr = 0.1
    states = []
    action_probs = []
    action_prob_grads = []
    rewards = []
    reward_sum = 0
    reward_sums = []
    episode_counter = 0

    env = gym.make("Pong-v0")
    observation = env.reset()
    previous_observation = utils.preprocess(observation)

    agent = model.build_dqn_model()

    while True:
        if render:
            env.render()
        current_observation = utils.preprocess(observation)
        state = current_observation - previous_observation
        previous_observation = current_observation

        action_prob = agent.predict_on_batch(state.reshape(1, 80, 80, 1))[0,:]
        action = np.random.choice(6, p=action_prob)

        observation, reward, done, _ = env.step(action)
        reward_sum += reward

        states.append(state)
        action_probs.append(action_prob)
        rewards.append(reward)
        y = np.zeros(6)
        y[action] = 1
        action_prob_grads.append(y - action_prob)
    
        if done:
            # Game Over - One of the players has gotten 21 points
            episode_counter += 1
            reward_sums.append(reward_sum)
            if len(reward_sums) > 40:
                reward_sums.pop(0)

            print('Episode: %d ------- Total Episode Reward: %f ------- Mean %f' % (episode_counter, reward_sum, np.mean(reward_sums)))
                
            rewards = np.vstack(rewards)
            action_prob_grads = np.vstack(action_prob_grads)
            rewards = utils.discounted_rewards(rewards)

            X = np.vstack(states).reshape(-1, 80, 80, 1)
            Y = action_probs + lr * rewards * action_prob_grads
            
            agent.train_on_batch(X, Y)

            agent.save_weights('Pong_agent')
            
            states, action_prob_grads, rewards, action_probs = [], [], [], []
            reward_sum = 0
            observation = env.reset()

def main():
    train_agent()

main()