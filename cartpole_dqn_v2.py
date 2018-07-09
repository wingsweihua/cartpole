import gym

import numpy as np
import tensorflow as tf


class DeepQLearningAgent(object):
    def __init__(self, state_space, action_space):
        """
        initialize the agent. Solve the problem without temporal difference.
        """
        self._action_space = action_space
        self._dim_state = state_space.shape[0]
        self._dim_action = action_space.n
        self._batch_size = 200
        self._gamma = 0.95

        self._prev_state = None
        self._prev_action = None
        self._prev_reward = 0

        # Build a neural network with tensorflow. The goal is to mapping an
        # observation to action-values (the Q). If the Q is optimized, ask the
        # action-values before each step, then pick the action with higher
        # value. The action with higher value is the action would give us more
        # rewards.
        #
        # value(prev_state) = reward + gamma * value*(next_state)
        w1 = tf.random_uniform([self._dim_state, 128], -1.0, 1.0)
        w1 = tf.Variable(w1)
        b1 = tf.random_uniform([128], -1.0, 1.0)
        b1 = tf.Variable(b1)

        w2 = tf.random_uniform([128, 128], -1.0, 1.0)
        w2 = tf.Variable(w2)
        b2 = tf.random_uniform([128], -1.0, 1.0)
        b2 = tf.Variable(b2)

        w3 = tf.random_uniform([128, 128], -1.0, 1.0)
        w3 = tf.Variable(w3)
        b3 = tf.random_uniform([128], -1.0, 1.0)
        b3 = tf.Variable(b3)

        w4 = tf.random_uniform([128, self._dim_action], -1.0, 1.0)
        w4 = tf.Variable(w4)
        b4 = tf.random_uniform([self._dim_action], -1.0, 1.0)
        b4 = tf.Variable(b4)

        prev_states = tf.placeholder(tf.float32, [None, self._dim_state])
        hidden_1 = tf.nn.relu(tf.matmul(prev_states, w1) + b1)
        hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + b2)
        hidden_3 = tf.nn.relu(tf.matmul(hidden_2, w3) + b3)
        prev_action_values = tf.squeeze(tf.matmul(hidden_3, w4) + b4)
        prev_action_masks = \
            tf.placeholder(tf.float32, [None, self._dim_action])
        prev_values = tf.reduce_sum(
            tf.multiply(prev_action_values, prev_action_masks), reduction_indices=1)

        prev_rewards = tf.placeholder(tf.float32, [None, ])
        next_states = tf.placeholder(tf.float32, [None, self._dim_state])
        hidden_1 = tf.nn.relu(tf.matmul(next_states, w1) + b1)
        hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + b2)
        hidden_3 = tf.nn.relu(tf.matmul(hidden_2, w3) + b3)
        next_action_values = tf.squeeze(tf.matmul(hidden_3, w4) + b4)
        next_values = prev_rewards + self._gamma * \
            tf.reduce_max(next_action_values, reduction_indices=1)

        loss = tf.reduce_mean(tf.square(prev_values - next_values))
        training = tf.train.AdamOptimizer(1e-4).minimize(loss)

        self._tf_action_value_predict = prev_action_values
        self._tf_prev_states = prev_states
        self._tf_prev_action_masks = prev_action_masks
        self._tf_prev_rewards = prev_rewards
        self._tf_next_states = next_states
        self._tf_training = training
        self._tf_loss = loss
        self._tf_session = tf.InteractiveSession()

        self._tf_session.run(tf.initialize_all_variables())

        # Build the D which keeps experiences.
        self._time = 0
        self._epislon = 1.0
        self._epislon_decay_time = 100
        self._epislon_decay_rate = 0.9
        self._experiences_max = 1000
        self._experiences_num = 0
        self._experiences_prev_states = \
            np.zeros((self._experiences_max, self._dim_state))
        self._experiences_next_states = \
            np.zeros((self._experiences_max, self._dim_state))
        self._experiences_rewards = \
            np.zeros((self._experiences_max))
        self._experiences_actions_mask = \
            np.zeros((self._experiences_max, self._dim_action))

    def create_experience(self, prev_state, prev_action, reward, next_state):
        """
        keep an experience for later training.
        """
        if self._experiences_num >= self._experiences_max:
            # make sure we always have some painalities.
            a = self._experiences_max * 9 / 10
            b = self._experiences_max - a

            if reward > 0.0:
                idx = np.random.choice(a)
            else:
                idx = np.random.choice(b) + a
        else:
            idx = self._experiences_num

        self._experiences_num += 1

        self._experiences_prev_states[idx] = np.array(prev_state)
        self._experiences_next_states[idx] = np.array(next_state)
        self._experiences_rewards[idx] = reward
        self._experiences_actions_mask[idx] = np.zeros(self._dim_action)
        self._experiences_actions_mask[idx, prev_action] = 1.0

    def train(self):
        """
        train the deep q-learning network.
        """
        # start training only when there are enough experiences.
        if self._experiences_num < self._experiences_max:
            return

        ixs = np.random.choice(
            self._experiences_max, self._batch_size, replace=True)

        fatches = [self._tf_loss, self._tf_training]

        feed = {
            self._tf_prev_states: self._experiences_prev_states[ixs],
            self._tf_prev_action_masks: self._experiences_actions_mask[ixs],
            self._tf_prev_rewards: self._experiences_rewards[ixs],
            self._tf_next_states: self._experiences_next_states[ixs]
        }

        loss, _ = self._tf_session.run(fatches, feed_dict=feed)

    def act(self, observation, reward, done):
        """
        ask the next action from the agent
        """
        self._time += 1

        if self._time % self._epislon_decay_time == 0:
            self._epislon *= self._epislon_decay_rate

        if np.random.rand() > self._epislon:
            states = np.array([observation])

            action_values = self._tf_action_value_predict.eval(
                feed_dict={self._tf_prev_states: states})

            action = np.argmax(action_values)
        else:
            action = self._action_space.sample()

        if self._prev_state is not None:
            self.create_experience(
                self._prev_state, self._prev_action, reward, observation)

        self._prev_state = None if done else observation
        self._prev_action = None if done else action
        self._prev_reward = 0 if done else self._prev_reward + reward

        self.train()

        return action

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped

    max_episodes = 2000
    max_steps = 200
    running_reward = []

    agent = DeepQLearningAgent(env.observation_space, env.action_space)

    for episode in range(max_episodes):
        observation, reward, done = env.reset(), 0.0, False

        for step in range(max_steps):
            if done and step + 1 < max_steps:
                reward = -500.0
                observation = np.zeros_like(observation)

            action = agent.act(observation, reward, done)

            if done or step + 1 == max_steps:
                running_reward.append(step)

                if len(running_reward) > 100:
                    running_reward = running_reward[-100:]

                avg_reward = sum(running_reward) / float(len(running_reward))

                print("{} - {} - {}".format(episode, step, avg_reward))

                break

            observation, reward, done, _ = env.step(action)

    env.monitor.close()