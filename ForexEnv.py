import tensorflow as tf
import numpy as np


class ForexEnv:
    def __init__(self,
                 agent_obs_arrays,
                 bidask_arrays,
                 initial_balance=1,
                 bid_col=3,
                 ask_col=7,
                 p=0.1,
                 n_actions=1):
        """
        Forex Environment that the agent interacts with in training and validation


        Parameters
        ----------
        agent_obs_arrays: numpy.array, shape = (timestamps, num_features)
         Agent observable arrays, representation of market's states

        bidask_arrays: numpy.array, must include Bid and Ask prices for reward calculation
         Unprocessed price data for reward calculation

        initial_balance : float, default 1 (i.e. 100%)
         Initial balance available to the agent

        bid_col: int, default 3
         Column/index of Bid price in pandas.DataFrame/numpy.array format used for reward calculation

            e.g. bid_col = 3 when BidClose is used for reward calculation, where
            bidask_arrays.shape = (timestamps, ['BidOpen', 'BidHigh', 'BidLow', 'BidClose',
                                                'AskOpen', 'AskHigh', 'AskLow', 'AskClose'])

        ask_col: int, default 7
         Column/index of Ask price in pandas.DataFrame/numpy.array format used for reward calculation

            e.g. ask_col = 7 when AskClose is used for reward calculation, where
            bidask_arrays.shape = (timestamps, ['BidOpen', 'BidHigh', 'BidLow', 'BidClose',
                                                'AskOpen', 'AskHigh', 'AskLow', 'AskClose'])

        p: float, default 0.1
         p% of the entire balance that is controllable by the agent at each timestamp
        """

        self.dataset = tf.data.Dataset.from_tensor_slices(
            (tf.convert_to_tensor(agent_obs_arrays[:, np.newaxis], tf.float32),
             tf.convert_to_tensor(bidask_arrays[:, np.newaxis], tf.float32)))
        self.dataset_len = self.dataset.cardinality().numpy()
        self.num_features = agent_obs_arrays.shape[-1] + n_actions + 1

        self.initial_balance = initial_balance
        self.bid_col = bid_col
        self.ask_col = ask_col
        self.p = p
        self.n_actions = n_actions

    def is_done(self):
        """
        Determines if the agent has arrived at a terminal state
        """

        return (self.current_pos == self.dataset_len) or (self.balance < 0)

    def get_state(self):
        """
        Return the state of the timestamp
        """

        state, ba = self.iterator.get_next()
        state = np.concatenate([self.current_action, state], axis=1)
        self.current_pos += 1

        return state, ba

    def get_reward(self, reward_state, reward_state_):
        """
        Reward calculation and agent's episodic balance update
        """

        short_reward = tf.reshape(tf.gather(reward_state_[0], indices=tf.constant(self.ask_col))
                                  - tf.gather(reward_state[0], indices=tf.constant(self.bid_col)),
                                  (1, -1))
        long_reward = tf.reshape(tf.gather(reward_state_[0], indices=tf.constant(self.bid_col))
                                 - tf.gather(reward_state[0], indices=tf.constant(self.ask_col)),
                                 (1, -1))
        reward = ((tf.cast(tf.less(self.current_action, 0), dtype=tf.float32)
                   * self.current_action
                   * 0.01 * self.p * self.balance
                   * 10000 * short_reward)

                  + (tf.cast(tf.greater(self.current_action, 0), dtype=tf.float32)
                     * self.current_action
                     * 0.01 * self.p * self.balance
                     * 10000 * long_reward))

        reward = tf.reshape(tf.reduce_sum(reward), (1, -1))
        self.balance += reward

        return reward

    def step(self, action, reward_state):
        """
        Move to the next timestamp
        """

        self.current_action = action
        state_, reward_state_ = self.get_state()
        reward = self.get_reward(reward_state, reward_state_)
        state_ = np.concatenate([self.balance, state_], axis=1)

        return state_, reward, self.is_done(), reward_state_, None
    
    
    def reset(self, evaluate=False):
        """
        Reset to a new episode
        """
        
        if not evaluate:
            steps = np.random.randint(self.dataset_len-2)
        else:
            steps = 0
        self.iterator = iter(self.dataset.skip(steps))
        self.current_pos = steps
        self.balance = tf.convert_to_tensor([[self.initial_balance]], dtype=tf.float32)
        self.current_action = tf.zeros(shape=(1, self.n_actions), dtype=tf.float32)
        observation, reward_state = self.get_state()
        observation = tf.concat([self.balance, observation], axis=1)

        return observation, reward_state
