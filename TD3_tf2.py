import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import random


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size  # The maximum size of the replay buffer
        self.mem_counter = 0  # The amount of saved experience
        self.state_memory = np.empty((self.mem_size, *input_shape)) * np.nan
        self.new_state_memory = np.empty((self.mem_size, *input_shape)) * np.nan
        self.action_memory = np.empty((self.mem_size, n_actions)) * np.nan
        self.reward_memory = np.empty(self.mem_size) * np.nan
        self.terminal_memory = np.empty(self.mem_size, dtype=np.bool) * np.nan

    def store_transition(self, state, action, reward, new_state, done):
        """
        Store an experience into the experience replay buffer
        """

        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        """
        Sample experience batches to train our model
        """

        current_mem_size = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(current_mem_size, batch_size, replace=False)
        state_batch = tf.convert_to_tensor(self.state_memory[batch])
        next_state_batch = tf.convert_to_tensor(self.new_state_memory[batch])
        action_batch = tf.convert_to_tensor(self.action_memory[batch])
        reward_batch = tf.convert_to_tensor(self.reward_memory[batch])
        done_batch = tf.convert_to_tensor(self.terminal_memory[batch])

        return state_batch, next_state_batch, action_batch, reward_batch, done_batch


class ActorLayer(layers.Layer):
    """
    Hidden layer in the Actor network
    """

    def __init__(self, fc_dim, activation='relu'):
        super(ActorLayer, self).__init__()

        self.dense = layers.Dense(fc_dim, activation=activation)

    def call(self, state):
        prob = self.dense(state)
        return prob


class ActorNetwork(keras.Model):
    """
    Approximation for the optimal actions given the observations
    """

    def __init__(self, fc_dim=512, num_layers=2, activation='relu',
                 n_actions=1):
        super(ActorNetwork, self).__init__()
        self.num_layers = num_layers
        self.actorlayers = [ActorLayer(fc_dim, activation) for _ in range(num_layers)]
        self.mu = layers.Dense(n_actions, activation='tanh',
                               kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))

    def call(self, state):
        for i in range(self.num_layers):
            state = self.actorlayers[i](state)

        actions = self.mu(state)

        if tf.reduce_sum(abs(actions)) == 0:   # Practically will not happen, but technically can
            bal_allocation = tf.cast(0, tf.float32)
        else:
            bal_allocation = tf.cast(tf.reduce_sum(actions ** 2) / (tf.reduce_sum(abs(actions)) ** 2), dtype=tf.float32)

        return actions * bal_allocation


class CriticLayer(layers.Layer):
    """
    Hidden layer in the Critic network
    """

    def __init__(self, fc_dim, activation='relu'):
        super(CriticLayer, self).__init__()

        self.dense = layers.Dense(fc_dim, activation=activation)

    def call(self, state_action):
        value = self.dense(state_action)
        return value


class CriticNetwork(keras.Model):
    """
    Approximation for the Q-values
    """

    def __init__(self, fc_dim=512, num_layers=2, activation='relu'):
        super(CriticNetwork, self).__init__()

        self.num_layers = num_layers
        self.q1_criticlayers = [CriticLayer(fc_dim, activation) for _ in range(num_layers)]
        self.q1_output_layer = layers.Dense(1, activation=None)

        self.q2_criticlayers = [CriticLayer(fc_dim, activation) for _ in range(num_layers)]
        self.q2_output_layer = layers.Dense(1, activation=None)

    def call(self, state_input, action_input):
        q1_state_action = layers.concatenate([state_input, action_input])
        q2_state_action = layers.concatenate([state_input, action_input])

        for i in range(self.num_layers):
            q1_state_action = self.q1_criticlayers[i](q1_state_action)
            q2_state_action = self.q2_criticlayers[i](q2_state_action)

        q1 = self.q1_output_layer(q1_state_action)
        q2 = self.q2_output_layer(q2_state_action)
        return q1, q2

    def Q1(self, state_input, action_input):
        q1_state_action = layers.concatenate([state_input, action_input])

        for i in range(self.num_layers):
            q1_state_action = self.q1_criticlayers[i](q1_state_action)

        q1 = self.q1_output_layer(q1_state_action)
        return q1


class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradient
    """

    def __init__(self,
                 state_dim,
                 n_actions=1,
                 max_action=1,
                 min_action=-1,
                 gamma=0.995,
                 tau=0.002,
                 exploration_noise=0.3,
                 policy_noise=0.15,
                 noise_clip=0.3,
                 policy_freq=4,
                 fc_dim=1024,
                 num_actor_layers=4,
                 num_critic_layers=4,
                 activation="relu",
                 ac_lr=1e-5,
                 cr_lr=1e-5,
                 batch_size=64,
                 max_memory_size=1000000,
                 uniform_action_steps=10000,
                 ckpt_name="ckpt",
                 model_name="TD3"):

        """
        Parameters
        ----------
        state_dim : int
         Number of features in the agent-observable dataset

        n_actions : int
         Number of action the agent can take

        max_action : int, default 1 (i.e. 100%)
         Maximum action from the actor

        min_action : int, default -1 (i.e. -100%)
         Minimum action from the actor

        gamma : float, default 0.995
         Farsightedness, how much the agent values future return

        tau : float, default 0.002
         Target networks update rate

        exploration_noise : float, default 0.3
         Standard deviation of noise added to actor's action in training

        policy_noise: float, default 0.15
         Standard deviation of noise added to target_actor's action batch

        noise_clip : float, default 0.3
         Clip values of added noise to target_actor's action batch

        policy_freq : int, default 4
         Policy update frequency

        fc_dim : int, default 1024
         Number of nodes in one hidden dense layer

        num_actor_layers : int, default 4
         Number of hidden layers in each actor networks

        num_critic_layers : int, default 4
         Number of hidden layers in each critic networks

        activation : tf.keras.activations, default "relu"
         Hidden layers activation function in all actors and critics (keras dense) networks

        ac_lr : float, default 1e-5
         Learning rate of the actors

        cr_lr : float, default 1e-5
         Learning rate of the critics

        batch_size : int, default 64
         Batch size of each experience sample

        max_memory_size : int, default 1000000
         Maximum replay buffer memory size

        uniform_action_steps : default 10000
         Steps of consecutive uniformly sampled actions in the beginning of training

        ckpt_name : str, default "ckpt"
         Folder name of the checkpoint directory for model weights

        model_name : str, default "TD3"
         Name of the Model
        """

        self.gamma = tf.cast(gamma, dtype=tf.float32)
        self.tau = tau
        self.memory = ReplayBuffer(max_memory_size, (state_dim,), n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_action = max_action
        self.min_action = min_action
        self.uniform_action_steps = uniform_action_steps
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_steps = 0
        self.ckpt_dir = f"{model_name}/{ckpt_name}"

        self.actor = ActorNetwork(fc_dim=fc_dim,
                                  num_layers=num_actor_layers,
                                  activation=activation,
                                  n_actions=n_actions)

        self.target_actor = ActorNetwork(fc_dim=fc_dim,
                                         num_layers=num_actor_layers,
                                         activation=activation,
                                         n_actions=n_actions)

        self.critic = CriticNetwork(fc_dim=fc_dim,
                                    num_layers=num_critic_layers,
                                    activation=activation)

        self.target_critic = CriticNetwork(fc_dim=fc_dim,
                                           num_layers=num_critic_layers,
                                           activation=activation)

        self.actor.compile(optimizer=Adam(learning_rate=ac_lr))
        self.critic.compile(optimizer=Adam(learning_rate=cr_lr))
        self.target_actor.compile(optimizer=Adam(learning_rate=ac_lr))
        self.target_critic.compile(optimizer=Adam(learning_rate=cr_lr))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        """
        Target networks weights update
        """

        if tau is None:
            tau = self.tau

        updates = [current_w * tau + target_w * (1 - tau) for current_w, target_w in
                   zip(self.actor.weights, self.target_actor.weights)]
        self.target_actor.set_weights(updates)

        updates = [current_w * tau + target_w * (1 - tau) for current_w, target_w in
                   zip(self.critic.weights, self.target_critic.weights)]
        self.target_critic.set_weights(updates)

    def choose_action(self, observation=None, training=False, explore=False):
        """
        Return actions from the actor network
        """

        if not explore:
            actions = self.actor(observation)
            if training:
                actions = tf.clip_by_value(
                    tf.add(actions, tf.multiply(tf.random.normal(shape=actions.shape),
                                                self.exploration_noise)),
                    clip_value_min=self.min_action,
                    clip_value_max=self.max_action)
        else:
            actions = random.uniform(self.min_action,
                                     self.max_action)
            actions = np.reshape(actions, (1, self.n_actions))
        return actions

    @tf.function
    def update(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch):
        """
        Update weights of the networks
        """

        with tf.GradientTape() as tape:
            noise = tf.clip_by_value(tf.multiply(tf.random.normal(shape=action_batch.shape),
                                                 self.policy_noise),
                                     clip_value_min=-self.noise_clip,
                                     clip_value_max=self.noise_clip)

            target_action_batch = tf.clip_by_value(tf.add(self.target_actor(next_state_batch), noise),
                                                   clip_value_min=self.min_action,
                                                   clip_value_max=self.max_action)

            target_Q1, target_Q2 = self.target_critic(next_state_batch, target_action_batch)
            target_Q = tf.squeeze(tf.math.minimum(target_Q1, target_Q2), 1)
            done_batch = tf.cast(done_batch, dtype=tf.float32)
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            target_Q = reward_batch + (1 - done_batch) * self.gamma * target_Q

            current_Q1, current_Q2 = self.critic(state_batch, action_batch)
            current_Q1 = tf.squeeze(current_Q1, 1)
            current_Q2 = tf.squeeze(current_Q2, 1)

            critic_loss = keras.losses.MSE(current_Q1, target_Q) + keras.losses.MSE(current_Q2, target_Q)

        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        if tf.equal(tf.math.floormod(self.total_steps, self.policy_freq), 0):
            with tf.GradientTape() as tape:
                actor_loss = tf.math.reduce_mean(-self.critic.Q1(state_batch, self.actor(state_batch)))

            actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

    def learn(self):
        """
        Perform a complete learning step
        """

        self.total_steps += 1
        if self.memory.mem_counter < self.batch_size:
            return

        (state_batch, next_state_batch, action_batch,
         reward_batch, done_batch) = self.memory.sample_buffer(self.batch_size)
        self.update(state_batch, next_state_batch, action_batch, reward_batch, done_batch)
        self.update_network_parameters()

    def remember(self, state, action, reward, new_state, done):
        """
        Store an experience into the experience replay buffer
        """

        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self, game_num):
        """
        Save models weights to the checkpoint directory
        """

        self.actor.save_weights(f"{self.ckpt_dir}/actor_game{game_num}.tf")
        self.target_actor.save_weights(f"{self.ckpt_dir}/target_actor_game{game_num}.tf")
        self.critic.save_weights(f"{self.ckpt_dir}/critic_game{game_num}.tf")
        self.target_critic.save_weights(f"{self.ckpt_dir}/target_critic_game{game_num}.tf")

    def load_models(self, game_num):
        """
        Load models weights from the checkpoint directory
        """

        self.actor.load_weights(f"{self.ckpt_dir}/actor_game{game_num}.tf")
        self.target_actor.load_weights(f"{self.ckpt_dir}/target_actor_game{game_num}.tf")
        self.critic.load_weights(f"{self.ckpt_dir}/critic_game{game_num}.tf")
        self.target_critic.load_weights(f"{self.ckpt_dir}/target_critic_game{game_num}.tf")
