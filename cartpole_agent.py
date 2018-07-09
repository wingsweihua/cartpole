__author__ = 'Wingslet'
# from https://gist.github.com/simoninithomas/7a3357966eaebd58dddb6166c9135930#file-cartpole-reinforce-monte-carlo-policy-gradients-ipynb
import random
random.seed(5)
import numpy as np
np.random.seed(5)

import tensorflow as tf
tf.set_random_seed(5)

from keras import backend as K
from keras.layers import Dense,Input
from keras import Model
from keras.objectives import categorical_crossentropy
from keras.losses import mae
from keras.callbacks import EarlyStopping, TensorBoard

import gym
import time

class Episode():

    def __init__(self):
        self.episode_rewards_sum = 0
        # Reset the transition stores
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_states_next = []

    def remember(self,state, action, reward, state_next):
         # Store s, a, r
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_states_next.append(state_next)
        self.episode_rewards_sum = np.sum(self.episode_rewards)



class CartpoleAgent():


    def __init__(self, state_size=4, action_size=2):
        self.env = self.init_env()

        ## ENVIRONMENT Hyperparameters
        assert state_size == self.env.observation_space.shape[0]
        self.state_size = state_size
        assert action_size == self.env.action_space.n
        self.action_size = self.env.action_space.n

        self.state = None


    def convert_state_to_input(self, state):
        return np.array([state])

    def close_env(self):
        self.env.close()


    def take_action(self, action):
        new_state, reward, done, info = self.env.step(action)
        return new_state,reward,done,info


    def update_state(self,state):
        self.state = state

    def init_env(self):
        env = gym.make('CartPole-v0')
        env = env.unwrapped
        #seed for reproducability
        # env.seed(5)
        return env

    def reset(self):
        self.state = self.env.reset()
        self.env.render()

    def get_state(self):
        return self.state


def log_perf(file_name,reward_str):
    fp = open(file_name, "a")
    fp.write(reward_str)
    fp.close()

if __name__ == '__main__':
    allRewards = []
    episode = 0
    max_episodes = 200
    file_name = "./records/reward_%s.txt"%(time.strftime('%m_%d_%H_%M_%S_', time.localtime(time.time())))

    cartpole = CartpoleAgent()


    for episode in range(max_episodes):
        episode_data = cartpole.generate_episode()


        allRewards.append(episode_data.episode_rewards_sum)
        mean_reward = np.divide(np.sum(allRewards), episode+1)
        maximumRewardRecorded = np.amax(allRewards)
        #print("==========================================")
        #print("Episode: ", episode)
        #print("Reward: ", episode_data.episode_rewards_sum)
        #print("Mean Reward", mean_reward)
        #print("Max reward so far: ", np.amax(allRewards))
        reward_str = "==========================================\n" + "Episode: {0}\n".format(episode) + \
                     "Reward: {0}\n".format(episode_data.episode_rewards_sum) + \
                     "Mean Reward: {0}\n".format(mean_reward) + \
                     "Max reward so far: {0}\n".format(np.amax(allRewards))


        log_perf(file_name,reward_str)

        #print out episode info and update network



