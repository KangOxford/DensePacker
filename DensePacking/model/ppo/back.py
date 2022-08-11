class PPO():
    """
    Class that implements PPO training
        Args:
            obs_shape (tuple): Shape of the observations.
            action_n (int): Number of possible actions inr the enviroment.
            entropy_coef (float): Entropy coefficient.
            value_loss_coef (float):  value loss coefficient.
            gamma (float): Discount rate.
            num_processes (int):
            num_steps (int): Number of parallel enviroments.
            ppo_epoch (int): Number of ppo updates
            num_mini_batch (int): Number of batches used in each ppo update
            learning_rate (float): The initial learning rate.
            clip_param (float): Clipped Surrogate Objective parameter
            gae_lambda (float): Generalized Advantage Estimation lambda
        Returns:
            tuple:  value, action, and entropy losses
    """
    def __init__(self, obs_shape, action_n, entropy_coef, value_loss_coef,
                 gamma, num_processes, num_steps, ppo_epoch, num_mini_batch,
                 learning_rate, clip_param, gae_lambda):
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.gamma = gamma
        self.obs_shape = obs_shape
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.num_mini_batch = num_mini_batch
        self.model = Model(action_n)
        self.clip_param = clip_param
        self.gae_lambda = gae_lambda
        self.ppo_epoch = ppo_epoch
        self.learning_rate = tf.Variable(learning_rate)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, epsilon=1e-5)

    def set_learning_rate(self, learning_rate):
        """Update learning rate."""
        self.learning_rate.assign(learning_rate)

    @tf.function
    def update(self, env_step, get_obs):
        """
            Implements PPO update.
            Args:
                env_step (function): Numpy enviroment function that performs enviroment step.
                get_obs (function): Numpy enviroment function that returns current observations.

            Returns:
                 Value loss, Action_loss and Entropy_loss.
        """
        observations = []
        masks = []
        rewards = []
        values = []
        old_log_probs = []
        actions = []
        obs = tf.numpy_function(func=get_obs, inp=[], Tout=tf.float32)

        # Shape inference is lost due to numpy_function.
        obs = tf.reshape(obs, self.obs_shape)
        observations.append(obs)

        for _ in range(self.num_steps):
            action, old_log_prob, _, value = self.model(obs)
            obs, reward, done = tf.numpy_function(func=env_step,
                                                  inp=[action],
                                                  Tout=(tf.float32, tf.float32,
                                                        tf.float32))

            obs = tf.reshape(obs, self.obs_shape)
            mask = 1.0 - done

            observations.append(obs)
            old_log_probs.append(old_log_prob)
            rewards.append(reward)
            masks.append(mask)
            values.append(value)
            actions.append(action)

        next_value = self.model(obs)[-1]

        returns = []
        values.append(next_value)
        gae = 0
        for step in reversed(range(self.num_steps)):
            delta = rewards[step] + self.gamma * values[
                step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])

        tf_returns = tf.concat(returns, axis=0)
        tf_observations = tf.concat(observations[:-1], axis=0)

        tf_actions = tf.concat(actions, axis=0)
        tf_old_log_probs = tf.concat(old_log_probs, axis=0)
        tf_values = tf.concat(values[:-1], axis=0)

        tf_adv_target = tf_returns - tf_values
        tf_adv_target = (tf_adv_target - tf.reduce_mean(tf_adv_target)) / (
            tf.math.reduce_std(tf_adv_target) + 1e-5)

        batch_size = self.num_processes * self.num_steps
        mini_batch_size = batch_size // self.num_mini_batch

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        entropy_loss_epoch = 0.0

        for _ in range(self.ppo_epoch):
            indx = tf.random.shuffle(tf.range(batch_size))
            indx = tf.reshape(indx, (-1, mini_batch_size))
            for sample in indx:
                obs_batch = tf.gather(tf_observations, sample)
                returns_batch = tf.gather(tf_returns, sample)
                adv_target_batch = tf.gather(tf_adv_target, sample)
                action_batch = tf.gather(tf_actions, sample)
                old_log_probs_batch = tf.gather(tf_old_log_probs, sample)
                values_batch = tf.gather(tf_values, sample)

                with tf.GradientTape() as tape:
                    _, action_log_probs, dist_entropy, value = self.model(
                        obs_batch, action_batch)

                    ratio = tf.exp(action_log_probs - old_log_probs_batch)
                    surr1 = -ratio * adv_target_batch
                    surr2 = -tf.clip_by_value(
                        ratio, 1.0 - self.clip_param,
                        1.0 + self.clip_param) * adv_target_batch
                    action_loss = tf.reduce_mean(tf.maximum(surr1, surr2))

                    value_pred_clipped = values_batch + tf.clip_by_value(
                        value - values_batch, -self.clip_param,
                        self.clip_param)
                    value_losses = tf.square(value - returns_batch)
                    value_losses_clipped = tf.square(value_pred_clipped -
                                                     returns_batch)
                    value_loss = 0.5 * tf.reduce_mean(
                        tf.maximum(value_losses, value_losses_clipped))

                    entropy_loss = tf.reduce_mean(dist_entropy)

                    loss = (self.value_loss_coef * value_loss + action_loss -
                            entropy_loss * self.entropy_coef)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, 0.5)

                self.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables))

                value_loss_epoch += value_loss
                action_loss_epoch += action_loss
                entropy_loss_epoch += entropy_loss

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        entropy_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, entropy_loss_epoch



import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model, layers

import copy

tf.keras.backend.set_floatx('float64')

EPSILON = 1e-8


class Actor(Model):

    def __init__(self, output_dim, hidden_dims):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_layers = [layers.Dense(layer, activation=tf.nn.relu) for layer in hidden_dims]
        self.mean_layer = layers.Dense(self.output_dim)
        # Standard deviation doesn't depend on state
        self.log_stddev = tf.Variable(initial_value=1.0, trainable=True,
                                      dtype=tf.float64)

    def call(self, state, action=None):
        # Pass input through all hidden layers
        inp = state
        for layer in self.hidden_layers:
            inp = layer(inp)

        # Generate mean output
        mu = self.mean_layer(inp)

        # Convert log stddev to stddev
        sigma = tf.exp(self.log_stddev)

        # Use re-parameterization trick to stochastically sample action from
        # the policy network. First, sample from a Normal distribution of
        # sample size as the action and multiply it with stdev
        dist = tfp.distributions.Normal(mu, sigma)

        if action is None:
            action = dist.sample()

        # Calculate log probability of the action
        log_pi = dist.log_prob(action)

        return action, log_pi, mu, sigma

    @property
    def trainable_variables(self):
        variables = []
        for layer in self.hidden_layers:
            variables.extend(layer.trainable_variables)
        variables.extend(self.mean_layer.trainable_variables)
        variables.append(self.log_stddev)
        return variables


class Critic(Model):

    def __init__(self, hidden_dims):
        super().__init__()
        self.hidden_layers = [layers.Dense(layer, activation=tf.nn.relu) for layer in hidden_dims]
        self.output_layer = layers.Dense(1)

    def call(self, state):
        # Pass input through all hidden layers
        inp = state
        for layer in self.hidden_layers:
            inp = layer(inp)

        # Generate mean output
        value = self.output_layer(inp)

        return value

    @property
    def trainable_variables(self):
        variables = []
        for layer in self.hidden_layers:
            variables.extend(layer.trainable_variables)
        variables.extend(self.output_layer.trainable_variables)
        return variables


class PPOClipped:

    def __init__(self,
                 writer,
                 action_dim,
                 policy_hidden_dims=[64, 64],
                 value_hidden_dims=[64, 64],
                 learning_rate=1e-4,
                 gamma=0.99,
                 lambd=0.9,
                 epsilon=0.2):
        """Implementation of PPO with Clipped object"""
        self.policy = Actor(action_dim, policy_hidden_dims)
        self.value = Critic(value_hidden_dims)

        self.writer = writer
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambd = lambd
        self.epsilon = epsilon

        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate)

    def update(self, transitions, epoch):
        """Does a backprop on policy and value networks"""

        with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:

            R = 0.0
            policy_loss_ = []
            value_loss_ = []
            advantage = 0
            for transition in reversed(transitions):
                current_state, action, log_pi_old, _, _, reward, next_state, done = transition

                R = self.gamma * R + reward

                # GAE
                V_s = self.value(current_state)
                V_s_next = self.value(next_state)
                td_residual = reward + self.gamma * V_s_next * done - V_s

                advantage = self.gamma * self.lambd * advantage + td_residual

                # Compute policy loss
                _, log_pi, _, _ = self.policy(current_state, action)

                is_ratio = tf.exp(log_pi - log_pi_old)


                unclipped = is_ratio * advantage
                clipped = tf.cond(advantage < 0,
                                  lambda: tf.multiply(1-self.epsilon, advantage),
                                  lambda: tf.multiply(1+self.epsilon, advantage))

                policy_loss_.append(tf.math.minimum(clipped, unclipped))
                value_loss_.append(tf.pow(V_s - R, 2))


            policy_loss = tf.reduce_mean(policy_loss_)
            value_loss = tf.reduce_mean(value_loss_)
            value_objective = value_loss
            policy_objective = - policy_loss

        policy_vars = self.policy.trainable_variables
        policy_grads = tape.gradient(policy_objective, policy_vars)
        self.policy_optimizer.apply_gradients(zip(policy_grads, policy_vars))

        with self.writer.as_default():
            tf.summary.scalar("advantage", advantage.numpy()[0][0], epoch)
            for var, grad in zip(policy_vars, policy_grads):
                tf.summary.histogram(f"policy_grad-{var.name}", grad, epoch)
                tf.summary.histogram(f"policy_var-{var.name}", var, epoch)

        value_vars = self.value.trainable_variables
        value_grads = tape.gradient(value_objective, value_vars)
        self.value_optimizer.apply_gradients(zip(value_grads, value_vars))

        with self.writer.as_default():
            for var, grad in zip(value_vars, value_grads):
                tf.summary.histogram(f"value_grad-{var.name}", grad, epoch)
                tf.summary.histogram(f"value_var-{var.name}", var, epoch)

        return policy_loss, value_loss