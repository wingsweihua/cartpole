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


import gym

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


class CartpoleAgent():

    ## TRAINING Hyperparameters
    learning_rate = 0.01
    gamma = 0.95 # Discount rate

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
            cumulative = cumulative * self.gamma + episode_rewards[i]
            discounted_episode_rewards[i] = cumulative

        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

        return discounted_episode_rewards

    def choose(self,sess):
        # compute action probility from NN
        action_probability_distribution = sess.run(action_distribution_k,feed_dict={input_: self.state.reshape([1,4])})
        #print(action_probability_distribution)
        #take action with probility
        action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
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
        discounted_episode_rewards_ = Input(shape=(1,), name='discounted_episode_rewards')
        actions = Input(shape=(1,), name='actions')

        fc1 = Dense(10, activation='relu', name="fc1")(input_)
        fc2 = Dense(self.action_size, activation='relu', name='fc2')(fc1)
        fc3 = Dense(self.action_size,activation='linear', name='fc3')(fc2)
        action_distribution = Dense(2, activation='softmax')(fc3)

        model = Model(inputs=[input_,discounted_episode_rewards_],outputs=[action_distribution])



    def update_network(self, episode_data, mean_reward):

        # allRewards.append(episode_data.episode_rewards_sum)

        #total_rewards = np.sum(allRewards)

        # Mean reward
        #mean_reward = np.divide(total_rewards, episode_no+1)


        # Calculate discounted reward
        discounted_episode_rewards = self.discount_and_normalize_rewards(episode_data.episode_rewards)

        # Feedforward, gradient and backpropagation
        loss_, _ = sess.run([loss_k, train_opt_k], feed_dict={input_: np.vstack(np.array(episode_data.episode_states)),
                                                         actions: np.vstack(np.array(episode_data.episode_actions)),
                                                         discounted_episode_rewards_: discounted_episode_rewards
                                                        })



        # Write TF Summaries
        summary = sess.run(write_op, feed_dict={input_: np.vstack(np.array(episode_data.episode_states)),
                                                         actions: np.vstack(np.array(episode_data.episode_actions)),
                                                         discounted_episode_rewards_: discounted_episode_rewards,
                                                            mean_reward_: mean_reward
                                                        })

    def generate_episode(self):

        episode_rewards_sum = 0

        episode_data = Episode()
        # Launch the game
        self.state = self.env.reset()

        self.env.render()

        while True:
            # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
            action = self.choose(sess)

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


if __name__ == '__main__':
    allRewards = []
    episode = 0
    max_episodes = 100

    cartpole = CartpoleAgent()

    ######### construct tensorflow
    input_ = tf.placeholder(tf.float32, [None, cartpole.state_size], name="input_")
    actions = tf.placeholder(tf.int32, [None, cartpole.action_size], name="actions")
    discounted_episode_rewards_ = tf.placeholder(tf.float32, [None,], name="discounted_episode_rewards")
    # Add this placeholder for having this variable in tensorboard
    mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

    fc1_k = Dense(10,activation='relu')(input_)

    fc2_k = Dense(cartpole.action_size, activation='relu')(fc1_k)

    fc3_k = Dense(cartpole.action_size, activation=None)(fc2_k)

    action_distribution_k = Dense(2, activation='softmax')(fc3_k)

    from keras.objectives import categorical_crossentropy
    loss_k = tf.reduce_mean(categorical_crossentropy(tf.cast(actions, tf.float32), fc3_k) * discounted_episode_rewards_)
    train_opt_k = tf.train.AdamOptimizer(cartpole.learning_rate).minimize(loss_k)

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter("/Users/Wingslet/PycharmProjects/trafficLightRL/tensorboard/1")
    ## Losses
    tf.summary.scalar("Loss", loss_k)
    ## Reward mean
    tf.summary.scalar("Reward_mean", mean_reward_)
    write_op = tf.summary.merge_all()
    ###############


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for episode in range(max_episodes):
            episode_data = cartpole.generate_episode()

            allRewards.append(episode_data.episode_rewards_sum)
            mean_reward = np.divide(np.sum(allRewards), episode+1)
            maximumRewardRecorded = np.amax(allRewards)
            print("==========================================")
            print("Episode: ", episode)
            print("Reward: ", episode_data.episode_rewards_sum)
            print("Mean Reward", mean_reward)
            print("Max reward so far: ", np.amax(allRewards))

            #print out episode info and update network
            cartpole.update_network(episode_data, mean_reward)


