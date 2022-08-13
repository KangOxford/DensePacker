import copy
from cell_gym import CellEnv

import numpy as np
from collections import deque

from core import Packing

from keras import backend as K

import gym, os
from policy import *
from rl.agents.ddpg import DDPGAgent
from rl.agents.dqn import DQNAgent
from gym import spaces
from gym.utils import seeding
from env import Cell
from collections import deque

import tensorflow as tf

class MultiParticleENV(gym.Env):

    def __init__(self, packing, reset_callback=None, reward_callback=None,
               observation_callback=None, info_callback=None,
               done_callback=None):

        self.packing = packing
        self.particles = self.packing.particles

        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        # configure spaces
        self.action_space = []
        self.observation_space = []

        for particle in enumerate(self.particles):
            total_action_space = []
            tran_space = spaces.Box(low=particle.tran_low, high=particle.tran_high, shape=(3,), dtype=np.float32)
            total_action_space.append(tran_space)
            rot_space = spaces.Box(low=-1., high=1., shape=(4,), dtype=np.float32)
            total_action_space.append(rot_space)

            act_space = spaces.Tuple(total_action_space)
            self.action_space.append(act_space)
            
        for particle in enumerate(self.particles):
            total_observation_space = []
            centroid_space = spaces.Box(low=self.packing.low_bound, high=self.packing.high_bound, shape=(3,), dtype=np.float32)
            total_observation_space.append(centroid_space)
            orientation_space = spaces.Box(low=-1., high=1., shape=(4,), dtype=np.float32)
            total_observation_space.append(orientation_space)

            obs_space = spaces.Tuple(total_observation_space)
            self.observation_space.append(obs_space)

        self._seed()

    def _step(self, action_n):
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents

        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n



        assert self.action_space.contains(action)

        invalid = self._act(self.packing, action)
        packing = copy.deepcopy(self.packing)

        # 颗粒本身的-PW^2也可以作为reward一部分
        reward = self._get_reward_from_agent(packing)
        return self.packing, reward

    
    
    def _reset(self):
        # reset packing
        self.reset_callback(self.packing)
        # reset renderer
        self._reset_render()

        return self.packing

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, particle):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(particle, self.packing)

    # set env action for a particular agent
    def _set_action(self, action, particle, time=None):
        particle.action.tran = action[0]
        particle.action.rot = action[1]

 

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _act(self, packing, action):
        # 都不合法要mask吗
        invalid = [False for i in range(packing.num_particles)]
        
        for i, particle in enumerate(packing.particles):
            old_particle = copy.deepcopy(particle)
            delta_centroid = action[i][0]
            particle.state.centroid += delta_centroid

            que = action[i][1]
            que /= np.linalg.norm(que) # normalization
            particle.state.orientation = que.apply(particle.state.orientation)

            if not self.any_overlap(packing, particle):
                particle.state = old_particle.state
                invalid[i] = True

        return invalid


    

    def _get_reward_from_agent(self, mazemap):

        self.used_agent = True

        cell = CellEnv(mazemap)
        cell.agent = self.agent
        cell.reset()





  

    def get_agent_action(self, state):
        # 直接调用cell的策略网络，输入状态给一个动作

        # 给一定概率随机走，但不太适合我这个？
        if np.random.random() < 0.1:
            return np.random.choice(self.agent_action_size, 1)[0]
        state = copy.deepcopy(state)
        action = self.agent.forward(state)
        # print('action: ', action)
        return action



    def _reset(self):
        # 随机生成一组无重叠的颗粒结构
        self.gamestep = 0
        self.invalid_count = 0
        self.conflict_count = 0
        self.mazemap = utils.initMazeMap()
        [sx, sy, tx, ty] = utils.findSourceAndTarget(self.mazemap)
        self.source = np.array([sx, sy])
        self.target = np.array([tx, ty])
        return self.mazemap

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def agent_step(self, action):
        done = False
        if self.gamestep >= config.Game.MaxGameStep:
            done = True

        reward = -1

        # 先获得所有颗粒的位置和四元数，包括颗粒信息
        # cell get action 
        # 更新cell信息

        # 判断什么时候done，收敛？还是比上一次小

        
        return copy.deepcopy(self.mazemap), reward, done, new_pos, noise, current_pos_onehot, potential_pos_onehot

    


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n

