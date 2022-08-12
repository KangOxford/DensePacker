import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, initializers, regularizers, optimizers
from tensorflow.keras.losses import MSE
import logging

from model.pure_ddpg.replay_buffer import ReplayBuffer
from model.pure_ddpg.action_noise import *


class DDPGAgent(object):
    """
    DDPG agent
    """
    def __init__(self, state_dim, action_dim, action_boundaries,
                actor_lr = 1e-5, critic_lr = 1e-4, batch_size = 64, noise='ou', gamma = 0.99,
                rand_steps = 1, buffer_size = 10000, tau = 1e-3):
        # action and state size
        self.n_states = state_dim[0]
        self.n_actions = action_dim[0]

        self.batch_size = batch_size

        # environmental action boundaries
        self.lower_bound = action_boundaries[0]
        self.upper_bound = action_boundaries[1]

        if noise == 'ou':
            self.noise = OrnsteinUhlenbeckNoise(mu=np.zeros(action_dim))
        else:
            self.noise = NormalNoise(mu=np.zeros(action_dim))

        # Randomly initialize actor network and critic network and their target networks
        self.actor = ActorNetwork(state_dim = state_dim, action_dim = action_dim,
                            learning_rate = actor_lr, batch_size = batch_size, 
                            l2_weight = 0.01, tau = tau, upper_bound = self.upper_bound)
        self.critic = CriticNetwork(state_dim = state_dim, action_dim = action_dim,
                            learning_rate = critic_lr, batch_size = batch_size, tau = tau)

        # initialize experience replay buffer
        self._memory = ReplayBuffer(buffer_size, state_dim, action_dim)

        # Bellman discount factor
        self.gamma = gamma

        # number of episodes for random action exploration
        self.rand_steps = rand_steps - 1

        # turn off most logging
        logging.getLogger("tensorflow").setLevel(logging.FATAL)

        # date = datetime.now().strftime("%m%d%Y_%H%M%S")
        # path_actor = "./models/actor/actor" + date + ".h5"
        # path_critic = "./models/critic/actor" + date + ".h5"

    def get_action(self, state, step):
        """
        Return the best action in the previous state, according to the model
        in training. Noise added for exploration
        """
        #if(step > self.rand_steps):
        noise = self.noise()
        state = state.reshape(self.n_states, 1).T
        action = self.actor.model.predict(state)[0]
        # print(action)
        action_prev = action + noise
        #else:
            # random actions for the first episode to explore the action space
            #action_prev = np.random.uniform(self.lower_bound, self.upper_bound, self.n_actions)

        #clip the resulting action with the bounds
        action_prev = np.clip(action_prev, self.lower_bound, self.upper_bound)
        return action_prev


    def learn(self):
        """
        Fill the buffer up to the batch size, then train both networks with
        experience from the replay buffer.
        """
        if self._memory.isReady(self.batch_size):
            self.train_helper()

    def train_helper(self):
        """
        train critic and actor
        """

        # sample a random minibatch of N transitions from replay buffer
        states, actions, rewards, terminal, states_n = self._memory.sample(self.batch_size)
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        rewards = tf.cast(rewards, dtype=tf.float32)
        states_n = tf.convert_to_tensor(states_n)

        # train the critic before the actor
        # 交替未必就好，可能要按比例
        self.train_critic(states, actions, rewards, terminal, states_n)
        self.train_actor(states)

        # update the target models
        self.critic.update_target()
        self.actor.update_target()

    def train_critic(self, states, actions, rewards, terminal, states_n):
        """
        Use updated Q targets to train the critic network
        """
        self.critic.train(states, actions, rewards, terminal, states_n, self.actor.target_model, self.gamma)

    def train_actor(self, states):
        """
        Train the actor network with the critic evaluation
        """
        self.actor.train(states, self.critic.model)

    def remember(self, state, state_new, action, reward, terminal):
        """
        replay buffer interfate to the outsize
        """
        self._memory.remember(state, state_new, action, reward, terminal)

    def forward(self, state):
        action = self.actor.model.predict(state)[0]
        # print(action)
        return action


class ActorNetwork(object):
    """
    stochastic funcion approximator for the deterministic policy map u : S -> A
    (with S set of states, A set of actions)
    """

    def __init__(self, state_dim, action_dim, learning_rate, batch_size, l2_weight, tau, upper_bound):
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.l2_weight = l2_weight

        # soft replacement (polyak averaging)
        self.tau = tau
        self.upper_bound = upper_bound

        # create actor network
        self.model = self.build_network()
        self.model.summary()

        # duplicate model for target actor network
        self.target_model = self.build_network()
        self.target_model.set_weights(self.model.get_weights())

        # 布尔量 要不要bn

        self.optimizer = optimizers.Adam(self.learning_rate)
        
    def build_network(self):
        """
        actor network: state -> action
        Consists of two fully connected layers with batch norm
        """
        input_layer = layers.Input(shape=self.state_dim, name='ActorInputLayer')

        # two fully connected layer with batch norm
        layer1 = layers.Dense(units=256)(input_layer)
        layer1 = layers.BatchNormalization()(layer1)
        layer1 = layers.Activation("relu")(layer1)
    
        layer2 = layers.Dense(units=128)(layer1)
        layer2 = layers.BatchNormalization()(layer2)
        layer2 = layers.Activation("relu")(layer2)

        # scale the output to [-1, 1]
        output_layer = layers.Dense(*self.action_dim, activation="tanh",
                                    kernel_initializer = initializers.RandomUniform(-3e-3, 3e-3), 
                                    # bias_initializer = initializers.RandomUniform(-f3, f3),
                                    kernel_regularizer=tf.keras.regularizers.l2(0.006))(layer2)
        
        output_layer = layers.Lambda(lambda x : x * self.upper_bound)(output_layer)
        model = Model(input_layer, output_layer, name = 'CellPolicyNet')
        return model

    @tf.function
    def train(self, states, critic_model):
        """
        Update the weights with the new critic evaluation 
        """
        with tf.GradientTape() as tape:
            actions = self.model(states, training=True)
            q_value = critic_model([states, actions], training=True)
            loss = -tf.math.reduce_mean(q_value)
        gradient = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

    def update_target(self):
        # faster updates woth graph mode
        self._transfer(self.model.variables, self.target_model.variables)

    @tf.function
    def _transfer(self, model_weights, target_weights):
        """
        apply Polyak averaging on the target weights
        """
        assert len(target_weights) == len(model_weights)

        for (weight, target) in zip(model_weights, target_weights):
            # target = tau * weights + (1 - tau) * target
            target.assign(weight * self.tau + target * (1. - self.tau))


class CriticNetwork(object):
    """
    stochastic funcion approximator for the Q value function C : SxA -> R
    (with S set of states, A set of actions)
    """
    def __init__(self, state_dim, action_dim, learning_rate, batch_size, tau):
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.fc1_size = 256
        self.fc2_size = 128

        # soft replacement (polyak averaging)
        self.tau = tau

        self.model = self.build_network()
        self.model.summary()

        # duplicate model for target actor network
        self.target_model = self.build_network()
        self.target_model.set_weights(self.model.get_weights())

        #generate gradient function
        self.optimizer = optimizers.Adam(self.learning_rate)

    def build_network(self):
        """
        critic network: state + action -> Q
        Consists of two state fully connected layers and one action fully connected layer
        """
        state_input_layer = layers.Input(shape=(self.state_dim))
        action_input_layer = layers.Input(shape=(self.action_dim))

        # two state fully connected layers
        v1 = 1. / np.sqrt(self.fc1_size)
        fc1 = layers.Dense(self.fc1_size, 
                           kernel_initializer = initializers.RandomUniform(-v1, v1),
                           bias_initializer = initializers.RandomUniform(-v1, v1))(state_input_layer)
        #fc1 = layers.BatchNormalization()(fc1)
        fc1 = layers.Activation("relu")(fc1)

        v2 = 1. / np.sqrt(self.fc2_size)
        fc2 = layers.Dense(self.fc2_size, 
                           kernel_initializer = initializers.RandomUniform(-v2, v2),
                           bias_initializer = initializers.RandomUniform(-v2, v2))(fc1)
        #fc2 = layers.BatchNormalization()(fc2)
        # fcl2 = layers.Activation("linear")(fcl2)

        # one action fully connected layer
        action_layer =  layers.Dense(self.fc2_size, 
                                     kernel_initializer = initializers.RandomUniform(-v2, v2),
                                     bias_initializer = initializers.RandomUniform(-v2, v2))(action_input_layer)
        action_layer = layers.Activation("relu")(action_layer)

        concat = layers.Add()([fc2, action_layer])
        concat = layers.Activation("relu")(concat)

        # output Q value for a given (s_t, a_t)
        output = layers.Dense(units=1, 
                              kernel_initializer = initializers.RandomUniform(-3e-3, 3e-3),
                              kernel_regularizer=tf.keras.regularizers.l2(0.006))(concat)

        model = Model([state_input_layer, action_input_layer], output, name = 'CellValueNet')
        return model

    @tf.function
    def train(self, states, actions, rewards, terminals, states_n, actor_target, gamma):
        """
        Update the weights with the Q targets. Graphed function for more
        efficient Tensor operations
        """
        with tf.GradientTape() as tape:
            target_actions = actor_target(states_n, training=True)
            q_n = self.target_model([states_n, target_actions], training=True)
            # Bellman equation for the q value
            q_target = rewards + gamma * q_n * (1 - terminals)
            q_value = self.model([states, actions], training=True)
            loss = MSE(q_target, q_value)

        gradient = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))

    def update_target(self):
        # faster updates with graph mode
        self._transfer(self.model.variables, self.target_model.variables)

    @tf.function
    def _transfer(self, model_weights, target_weights):
        """
        apply Polyak averaging on the target weights
        """
        assert len(target_weights) == len(model_weights)

        for (weight, target) in zip(model_weights, target_weights):
            # target = tau * weights + (1 - tau) * target
            target.assign(weight * self.tau + target * (1 - self.tau))

