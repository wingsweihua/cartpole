__author__ = 'Wingslet'

import random
random.seed(5)
import numpy as np
np.random.seed(5)
import tensorflow as tf
tf.set_random_seed(5)

from keras import backend as K
from keras.optimizers import RMSprop
from keras.layers import Dense,Input
from keras import Model
from keras import losses
from keras.callbacks import EarlyStopping, TensorBoard

import gym
import time
from cartpole_agent import CartpoleAgent


class DQN_Agent:
    REPLAY_MEMORY_SIZE = 10000
    RANDOM_ACTION_PROB = 0.5
    RANDOM_ACTION_DECAY = 0.99
    HIDDEN1_SIZE = 10
    HIDDEN2_SIZE = 10
    NUM_EPISODES = 200
    MAX_STEPS = 1000
    LEARNING_RATE = 0.0001
    MIN_BATCH_SIZE = 20
    DISCOUNT_FACTOR = 0.9
    TARGET_UPDATE_FREQ = 100
    REG_FACTOR = 0.001
    LOG_DIR = '/Users/Wingslet/PycharmProjects/cartpole/tensorboard/dqn'
    DONE_REWARD = -100
    MIN_SAMPLE_SIZE = 100

    def __init__(self, state_size=4, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        self.network = self.build_network()
        self.memory = []

        self.state = None
        self.action = None
        self.average_reward = None
        self.update_outdated = 0

    def build_network(self):
        input_ = Input(shape=(self.state_size,), name="input_")
        fc1 = Dense(self.HIDDEN1_SIZE, activation='tanh', name="fc1")(input_)
        fc2 = Dense(self.HIDDEN2_SIZE, activation='tanh', name='fc2')(fc1)
        Q_value = Dense(2, activation='linear', name='output')(fc2)

        model = Model(inputs=[input_],outputs=[Q_value])

        model.compile(optimizer=RMSprop(lr=self.LEARNING_RATE),loss=losses.mean_squared_error)
        model.summary()

        return model

    def choose(self):
        action = None
        if random.random() < self.RANDOM_ACTION_PROB:
            action = np.random.choice(self.action_size)
        else:
            q_values = self.network.predict(self.state)
            action = q_values.argmax()
        self.RANDOM_ACTION_PROB *= self.RANDOM_ACTION_DECAY
        return action

    def convert_state_to_input(self, state):
        return np.array([state])

    def convert_action_to_output(self, action_index):
        action = np.zeros(self.action_size)
        action[action_index] = 1
        return action

    def remember(self, state, action, reward, state_next):
        self.memory.append([state, action, reward, state_next])

    def _get_next_estimated_reward(self, next_state):
        next_estimated_reward = np.max(self.network.predict(next_state)[0])
        return next_estimated_reward

    @staticmethod
    def _cal_priority(sample_weight):
        pos_constant = 0.0001
        alpha = 1
        sample_weight_np = np.array(sample_weight)
        sample_weight_np = np.power(sample_weight_np + pos_constant, alpha) / sample_weight_np.sum()
        return sample_weight_np

    def _sample_memory(self, with_priority):

        len_memory = len(self.memory)

        sample_size = min(self.MIN_SAMPLE_SIZE, len_memory)

        if with_priority:
            # sample with priority
            sample_weight = []
            for i in range(len_memory):
                state, action, reward, next_state = self.memory[i]

                if reward == self.DONE_REWARD:
                    next_estimated_reward = self.DONE_REWARD
                else:
                    next_estimated_reward = self._get_next_estimated_reward(next_state)

                total_reward = reward + self.DISCOUNT_FACTOR * next_estimated_reward
                target = self.network.predict(state)
                pre_target = np.copy(target)
                target[0][action] = total_reward

                # get the bias of current prediction
                weight = abs(pre_target[0][action] - total_reward)
                sample_weight.append(weight)

            priority = self._cal_priority(sample_weight)
            p = random.choices(range(len(priority)), weights=priority, k=sample_size)
            sampled_memory = np.array(self.memory)[p]
        else:
            sampled_memory = random.sample(self.memory, sample_size)

        return sampled_memory

    def get_sample(self, sampled_memory):
        X = []
        Y = []
        action_mask = []

        for i in range(len(sampled_memory)):
            state, action, reward, next_state = sampled_memory[i]

            if reward == self.DONE_REWARD:
                next_estimated_reward = self.DONE_REWARD
            else:
                next_estimated_reward = self._get_next_estimated_reward(next_state)

            total_reward = reward + self.DISCOUNT_FACTOR * next_estimated_reward
            target = self.network.predict(state)

            target[0][np.argmax(action)] = total_reward
            Y.append(target[0])
            X.append(state)
            action_mask.append(action)

        return np.array(X), np.array(Y), np.array(action_mask)

    @staticmethod
    def _unison_shuffled_copies(Xs, Y):
        p = np.random.permutation(len(Y))
        new_Xs = []
        for x in Xs:
            assert len(x) == len(Y)
            new_Xs.append(x[p])
        return new_Xs, Y[p]

    def train_network(self,X,Y):
        batch_size = min(self.MIN_BATCH_SIZE, len(Y))

        # Fit the model
        early_stopping = EarlyStopping(
                monitor='val_loss', patience=30, verbose=2, mode='min')

        hist = self.network.fit(X, Y, batch_size=batch_size, epochs=150, shuffle=True,
                                  verbose=2, validation_split=0.2, callbacks=[early_stopping])



    def update_network(self, done, current_step):

        #update network when there is enough sample, and reach end of episode or time to update
        if len(self.memory) >= self.MIN_BATCH_SIZE \
                and (done or (current_step - self.update_outdated < self.MAX_STEPS)):
            self.update_outdated = current_step

            sampled_memory = self._sample_memory(with_priority=False)
            X, Y, action_mask = self.get_sample(sampled_memory)

            # shuffle the training samples, especially for different phases and actions
            Xs, Y = self._unison_shuffled_copies([X,action_mask], Y)

            Y_masked = np.multiply(np.array(Xs[1]),Y)
            X = np.vstack(X)
            Y_masked = np.vstack(Y_masked)
            # train network
            self.train_network(X, Y_masked)
            self.forget()
        else:
            return

    def forget(self):
        if len(self.memory) > self.REPLAY_MEMORY_SIZE:
            print("length of memory: {0}, before forget".format(len(self.memory)))
            self.memory = self.memory[-self.REPLAY_MEMORY_SIZE:]
        print("length of memory: {0}, after forget".format(len(self.memory)))


    def update_state(self,new_state):
        self.state=new_state



    def train(self, log_path, num_episodes=NUM_EPISODES, if_pretrain = False):
        cartpole_agent = CartpoleAgent(state_size=self.state_size, action_size=self.action_size)
        current_step = 0

        for episode in range(num_episodes):
            reward_str = "Training: Episode = %d, Global step = %d\n" % (episode, current_step)
            log_perf(log_path,reward_str)
            cartpole_agent.reset()
            self.action = None

            while True:
                self.state = self.convert_state_to_input(cartpole_agent.get_state())
                # Choose action a, remember WE'RE IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT actions.
                action = self.choose()

                # Perform a
                new_state, reward, done, info = cartpole_agent.take_action(action)

                # Store s, a, r
                self.action = self.convert_action_to_output(action)

                new_state_ = self.convert_state_to_input(new_state)
                if done:
                    reward = self.DONE_REWARD
                self.remember(self.state, self.action, reward, new_state_)

                # Update network
                self.update_network(done,current_step)

                current_step += 1
                if done:
                    break

                cartpole_agent.update_state(new_state)

        self.update_outdated = 0
        print("END training")

    def play(self):
        cartpole_agent = CartpoleAgent(state_size=self.state_size, action_size=self.action_size)
        cartpole_agent.reset()
        done = False
        steps = 0
        while not done:
            self.state = self.convert_state_to_input(cartpole_agent.get_state())
            q_values = self.network.predict(self.state)
            action = q_values.argmax()
            state, _, done, _ = cartpole_agent.take_action(action)
            steps += 1
            cartpole_agent.update_state(state)
        cartpole_agent.close_env()
        return steps

def log_perf(file_name,reward_str):
    fp = open(file_name, "a")
    fp.write(reward_str)
    fp.close()

if __name__ == '__main__':

    file_name = "./records/%s_reward_dqn.txt"%(time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
    player = DQN_Agent()

    ############!!!!!!!dqn.env.render()
    player.train(file_name)
    #dqn.env.monitor.close()

    res = []
    for i in range(100):
        steps = player.play()
        perf_str = "Test steps = {0}\n".format(steps)
        log_perf(file_name,perf_str)
        res.append(steps)
    perf_str = "Mean steps = {0}\n".format(sum(res) / len(res))
    log_perf(file_name,perf_str)