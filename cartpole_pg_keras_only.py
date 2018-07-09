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
#from keras.objectives import categorical_crossentropy
from keras import losses
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

    def remember(self,state, action, reward):
         # Store s, a, r
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_rewards_sum = np.sum(self.episode_rewards)


class Cartpole_PG_Agent():

    ## TRAINING Hyperparameters
    LEARNINH_RATE = 0.01
    GAMMA = 0.95 # Discount rate

    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.env = self.env.unwrapped
        # Policy gradient has high variance, seed for reproducability
        self.env.seed(5)
        ## ENVIRONMENT Hyperparameters
        self.state_size = 4
        self.action_size = self.env.action_space.n

        self.state = None
        self.network = self.build_network()


    def discount_and_normalize_rewards(self,episode_rewards):
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0.0
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * self.GAMMA + episode_rewards[i]
            discounted_episode_rewards[i] = cumulative

        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

        return discounted_episode_rewards

    # def choose(self,sess):
    #     # compute action probility from NN
    #     action_probability_distribution = sess.run(action_distribution_k,feed_dict={input_: self.state.reshape([1,4])})
    #     #print(action_probability_distribution)
    #     #take action with probility
    #     action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
    #     return action

    def convert_state_to_input(self, state):
        return np.array([state])

    def choose(self):
        #ValueError: Error when checking input: expected input_ to have shape (4,) but got array with shape (1,)
        state = self.convert_state_to_input(self.state)
        q_values = self.network.predict(state)
        action_probability_distribution = q_values[0]
        action = np.random.choice(range(action_probability_distribution.shape[0]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
        return action

    def take_action(self, action):
        new_state, reward, done, info = self.env.step(action)
        return new_state,reward,done,info


    def update_state(self,state):
        self.state = state

    def discounted_loss(discount):
        def loss(y_true, y_pred):
            neg_log_prob = K.categorical_crossentropy(y_true,y_pred,from_logits=False)

            return K.mean(-neg_log_prob*discount, axis=-1)
        return loss

    def build_network(self):
        input_ = Input(shape=(self.state_size,), name="input_")
        #discounted_episode_rewards_ = Input(shape=(1,), name='discounted_episode_rewards')
        #actions = Input(shape=(1,), name='actions')

        fc1 = Dense(10, activation='relu', name="fc1")(input_)
        fc2 = Dense(self.action_size, activation='relu', name='fc2')(fc1)
        fc3 = Dense(self.action_size,activation='linear', name='fc3')(fc2)
        action_distribution = Dense(2, activation='softmax', name='output')(fc3)

        model = Model(inputs=[input_],outputs=[action_distribution])

        model.compile(optimizer='sgd',loss=losses.categorical_crossentropy)
        model.summary()

        return model

    def train_network(self,X,Y):
        # Fit the model
        early_stopping = EarlyStopping(
                monitor='val_loss', patience=30, verbose=2, mode='min')

        hist = self.network.fit(X, Y, batch_size=20, epochs=150, shuffle=True,
                                  verbose=2, validation_split=0.2, callbacks=[early_stopping])


    def update_network(self, episode_data):

        # Calculate discounted reward
        discounted_episode_rewards = self.discount_and_normalize_rewards(episode_data.episode_rewards)

        #actions = np.array([action[1] for action in episode_data.episode_actions])
        discounted_episode_action = np.multiply(np.transpose([discounted_episode_rewards]),np.array(episode_data.episode_actions))

        X = np.vstack(np.array(episode_data.episode_states))
        Y = np.vstack(discounted_episode_action)

        self.train_network(X,Y)


    def generate_episode(self):

        episode_rewards_sum = 0

        episode_data = Episode()
        # Launch the game
        self.reset()

        while True:
            # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
            action = self.choose()

            # Perform a
            new_state, reward, done, info = self.take_action(action)

            # Store s, a, r
            # For actions because we output only one (the index) we need 2 (1 is for the action taken)
            # We need [0., 1.] (if we take right) not just the index
            action_ = np.zeros(self.action_size)
            action_[action] = 1
            episode_data.remember(self.state,action_,reward)

            if done:
                break

            self.update_state(new_state)

        return episode_data

def log_perf(file_name,reward_str):
    fp = open(file_name, "a")
    fp.write(reward_str)
    fp.close()

if __name__ == '__main__':
    allRewards = []
    episode = 0
    max_episodes = 200
    file_name = "./records/reward_%s.txt"%(time.strftime('%m_%d_%H_%M_%S_', time.localtime(time.time())))

    cartpole = Cartpole_PG_Agent()


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
        cartpole.update_network(episode_data)


