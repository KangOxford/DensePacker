import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers

import yaml
from collections import deque
from re import S

from ppo_lag.net import DeepNetwork
from ppo_lag.memorybuffer import Buffer

with open('config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    seed = cfg['setup']['seed']
    ymlfile.close()

np.random.seed(seed)
tf.random.set_seed(seed)

# Create environment

class LPPO:

    def __init__(self, env, info):


        self.env = env
        self.c_limit = 0.001

        self.pi = DeepNetwork.build(self.env, info['actor'], actor=True, name='actor')
        self.pi_opt = optimizers.Adam(learning_rate=info['pi_lr'])
        self.v= DeepNetwork.build(self.env, info['critic'], name='critic')
        self.v_opt = optimizers.Adam(learning_rate=info['vf_lr'])
        
        self.vc = DeepNetwork.build(self.env, info['critic'], name='critic')
        self.vc_opt = optimizers.Adam(learning_rate=info['vf_lr'])

        penalty_init = info['penalty']
        self.penalty = tf.Variable(np.log(max(np.exp(penalty_init)-1, 1e-8)), trainable=True, dtype=tf.float32)
        self.penalty_opt = optimizers.Adam(learning_rate=info['penalty_lr'])

        self.buffer = Buffer(info['steps_per_epoch'])

    def run_actor(self, s, logstd=-0.5):
        # Standard deviation of Gaussian distribution (we prefer log)
        std = np.exp(logstd)

        s = np.array([s])
        mu = self.pi(s).numpy()[0]
        a = np.random.normal(loc=mu, scale=std)
        a = np.clip(a, -1, 1)

        v = self.v(s).numpy().squeeze()
        vc = self.vc(s).numpy().squeeze()

        logp = -0.5 * ( ((a - mu)/(std+1e-10))**2 + 2 * logstd + np.log(2 * np.pi))
        logp = np.sum(logp)

        return a, mu, v, vc, logp


    def update(self, info, mean_cost, logstd=-0.5):
        """Prepare the samples and the cumulative reward to update the network

        Args:
           
        Returns:
            None
        """

        with tf.GradientTape() as tape_m:
            penalty_loss = tf.multiply(-self.penalty, (mean_cost - self.c_limit)) 
            penalty_grad = tape_m.gradient(penalty_loss, [self.penalty])
            self.penalty_opt.apply_gradients(zip(penalty_grad, [self.penalty]))

        s, a_old, mu_old, logp_old, adv, cadv, ret, cret = self.buffer.sample()
     
        clip = info['clip']
        target_kl = 0.01

        for i in range(info['pi_iters']):
            with tf.GradientTape() as tape_pi:
                std = np.exp(logstd)

                mu = self.pi(s)
                logp = -0.5 * ( ((a_old - mu)/(std+1e-10))**2 + 2 * logstd + np.log(2 * np.pi))
                logp = tf.reduce_sum(logp, axis=1)
                
                ratio = tf.exp(logp - logp_old)
                
                clip_adv = tf.where(adv > 0, (1+clip)*adv, (1-clip)*adv)
                surr_adv = tf.reduce_mean(tf.minimum(ratio*adv, clip_adv))

                surr_cadv = tf.reduce_mean(ratio*cadv)
                
                penalty = tf.nn.softplus(self.penalty).numpy()
                pi_obj = (surr_adv - penalty * surr_cadv) / (1 + penalty)

                pi_loss = -pi_obj
                pi_grad = tape_pi.gradient(pi_loss, self.pi.trainable_variables)
                self.pi_opt.apply_gradients(zip(pi_grad, self.pi.trainable_variables))

                var = tf.exp(2 * logstd)
                mu_ = self.pi(s)
                pre_sum = 0.5 * ( ((mu_old - mu_)**2 + var) / (var + 1e-10) - 1)
                kls = tf.reduce_sum(pre_sum, axis=1)
                kl = tf.reduce_mean(kls).numpy()
                
                if kl > 1.2 * target_kl:
                    print(f"Early stopping at iteration {i} due to reaching max kl")
                    break
                     
        for i in range(info['vf_iters']):
            with tf.GradientTape() as tape_v, tf.GradientTape() as tape_vc:

                v = self.v(s)
                v_loss = tf.reduce_mean((ret - v)**2)
                v_grad = tape_v.gradient(v_loss, self.v.trainable_variables)
                self.v_opt.apply_gradients(zip(v_grad, self.v.trainable_variables))

                vc = self.vc(s)
                vc_loss = tf.reduce_mean((cret - vc)**2)
                vc_grad = tape_vc.gradient(vc_loss, self.vc.trainable_variables)
                self.vc_opt.apply_gradients(zip(vc_grad, self.vc.trainable_variables))

        self.buffer.clear()

    def round_obs(self, obs):
        obs[:3] *= 0.1  # Normalize the Accelerometer inputs
        return np.around(obs, decimals=3)

    def train(self, tracker, info):
        r_mean, c_mean = deque(maxlen=100), deque(maxlen=100)
        c_tracker = deque(maxlen=info['steps_per_epoch'])
             
        n_step, steps_per_epoch = info['n_step'], info['steps_per_epoch']
        epochs = int(n_step / steps_per_epoch)
        ep_len = 1000

        ep_r, ep_c, steps, tot_steps = 0, 0, 0, 0
        s = self.env.reset()
        s = self.round_obs(s)

        for _ in range(epochs):
        
            for t in range(steps_per_epoch):
                a, mu, v, vc, logp = self.run_actor(s)
            
                s_, r, d, i = self.env.step(a)
                s_ = self.round_obs(s_)
                
                c = i
                
                self.buffer.store(s, a, mu, r, v, c, vc, logp)

                ep_r += r
                ep_c += c
                steps += 1
                tot_steps += 1
                
                s = s_

                if d or steps == ep_len:

                    e = int(tot_steps / ep_len)
                    r_mean.append(ep_r)
                    c_mean.append(ep_c)
                    c_tracker.append(ep_c)
                    tracker.update([e, ep_r, ep_c, self.penalty.numpy()])

                    s = np.array([s])
                    last_v = self.v(s).numpy().squeeze()
                    last_vc = self.vc(s).numpy().squeeze()

                    cost = self.env.cost()

                    #print(f'E: {e}, R: {ep_r:.3f}, C: {ep_c}, P: {tf.nn.softplus(self.penalty).numpy():.4f}, MeanR: {np.mean(r_mean):.3f}, MeanC: {np.mean(c_mean):.3f}')
                    print(f'Epoch: {e}, PD: {self.env.packing.fraction:.3f}, PW: {self.env.cost()}, P: {tf.nn.softplus(self.penalty).numpy():.4f}, MeanR: {np.mean(r_mean):.3f}, MeanC: {np.mean(c_mean):.3f}')


                    self.buffer.compute_mc(steps, last_v, last_vc)
                    ep_r, ep_c, steps = 0, 0, 0
                    s = self.env.reset()
                    s = self.round_obs(s)
      
            self.update(info, np.mean(c_tracker))
            tracker.save_metrics()



