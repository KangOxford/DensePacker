from re import T
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from sklearn.preprocessing import minmax_scale 

from packing.scenario import Scenario
from utils import data_scale


scenario = Scenario()

# minimal fundamental cell
class CellEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, 
                 packing=scenario.build_packing(), 
                 reset_callback=scenario.reset_packing, 
                 reward_callback=scenario.reward,
                 observation_callback=scenario.observation, 
                 done_callback=scenario.done,
                 penalty_callback=scenario.cell_penalty,
                 method:str="rotation"):

        """
        packing: multi-particles
        """
        self.packing = packing
        self.agent = self.packing.cell
        self.dim = self.packing.dim

        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.done_callback = done_callback
        self.penalty_callback = penalty_callback

        # action style
        self.method = method

        # action space
        if self.method == "strain_tensor":
            # (symmetric) strain tensor controlling the deformation of cell
            dim = self.dim*(self.dim+1)/2
            self.action_space = spaces.Box(low=-1., high=1., shape=(6, ), dtype=np.float32)
        elif self.method == "rotation":
            # euler angles + cell length
            self.action_space = spaces.Box(low=-1., high=1., shape=(4*self.dim, ), dtype=np.float32)

        # observation space
        obs_dim = len(observation_callback(self.packing))
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim, ), dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # assert len(action) == 6

        if self.method == "strain_tensor":
            assert len(action) == 6

            strain = np.zeros((self.dim, self.dim))
            id = -1
            for i in range(self.dim):
                for j in range(self.dim):
                    if i>j: continue
                    id += 1
                    strain[j][i] = strain[i][j] = action[id]

            self.agent.action.strain = 1e-1 * strain
            
        elif self.method == "rotation":
            assert len(action) == 12

            action = action.reshape(3, -1)

            self.agent.action.angle = data_scale(action[:, 0:3], from_range=(-1, 1), to_range=(0., 2.*np.pi))
            self.agent.action.angle[:, 1] /= 2.
            self.agent.action.length = data_scale(action[:, 3], from_range=(-1, 1), to_range=self.packing.cell_bound)
            

        # advance cell state in a packing
        self.packing.cell_step(self.method)

        # reward and observation
        obs = self.observation_callback(self.packing)
        reward = self.reward_callback(self.packing)
        done = self.done_callback(self.packing)

        info = {}
        info.update(self.cost())

        # 颗粒动的时候才调用
        # self.info_callback(agent, self.world)

        return obs, reward, done, info

    def reset(self):
        # reset packing
        self.reset_callback(self.packing)
        # reset renderer
        #self._reset_render()
        
        # record observation
        obs = self.observation_callback(self.packing)
        return obs


    def cost(self):
        ''' Penalty  '''
        cost = {}
        cost['cost'] = self.penalty_callback(self.packing)

        return cost