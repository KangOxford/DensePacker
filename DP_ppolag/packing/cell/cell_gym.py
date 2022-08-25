from copy import deepcopy
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from collections import OrderedDict
from myutils import *
from packing.scenario import Scenario

scenario = Scenario()

# environment for unit cell agent in the packing
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
                 cost_callback=scenario.cell_penalty,
                 mode:str="rotation"):

        self.packing = packing
        self.agent = self.packing.cell
        self.dim = self.packing.dim
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.done_callback = done_callback
        self.cost_callback = cost_callback
        # motion mode
        self.mode = mode

        # action space
        if self.mode == "strain_tensor":
            # (symmetric) strain tensor controlling the deformation of cell
            dim = self.dim*(self.dim+1)/2
            self.action_space = spaces.Box(low=-1., high=1., shape=(6, ), dtype=np.float32)
        elif self.mode == "rotation":
            # euler angles + cell length
            self.action_space = spaces.Box(low=-1., high=1., shape=(4*self.dim, ), dtype=np.float32)

        # observation space
        self.build_observation_space()

        self.seed()

        # perfomance
        self.performance = 1.0

    def build_observation_space(self):
        ''' Construct observtion space.  Happens only once at during __init__ '''
        obs_space_dict = OrderedDict()  # See self.obs()

        # Particle info
        for particle in self.packing.particles:
          obs_space_dict[particle] = gym.spaces.Box(-1.0, 1.0, (particle.obs_dim,), dtype=np.float32)
        # Cell info
        obs_space_dict['cell'] = gym.spaces.Box(-np.inf, np.inf, (self.dim, self.dim), dtype=np.float32)
        # Flatten it ourselves
        self.obs_space_dict = obs_space_dict
        self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)
        



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        ''' Take a step and return observation, reward, done, and info '''
        info = {}

        self._set_action(action)
        # advance cell state in a packing
        self.packing.cell_step(self.mode)

        # reward and observation
        reward = self.reward()
        done = self.done_callback(self.packing)
        info.update(self.cost())

        return self.obs, reward, done, info

    def reward(self):
      gap_prev = self.agent.gap
      self.agent.gap = max(self.agent.volume-self.packing.volume_allp, 0.)
        
      reward = (gap_prev - self.agent.gap) / self.packing.volume_allp
      return reward
    

    def reset(self):
        # reset packing
        self.reset_callback(self.packing)
        # reset renderer
        #self._reset_render()
        
        # record observation
        obs = self.observation_callback(self.packing)
        return obs

    def render(self):
        print("is_overlap {:d} overlap_potential {:2f} packing_fraction {:2f}".format(self.packing.is_overlap, self.packing.overlap_potential,self.packing.fraction))

    def _set_action(self, action):

        if self.mode == "strain_tensor":
            assert len(action) == 6

            strain = np.zeros((self.dim, self.dim))
            id = -1
            for i in range(self.dim):
                for j in range(self.dim):
                    if i>j: continue
                    id += 1
                    strain[j][i] = strain[i][j] = action[id]

            self.agent.action.strain = 1e-1 * strain
            
        elif self.mode == "rotation":
            assert len(action) == 12

            action = action.reshape(3, -1)

            self.agent.action.angle = data_scale(action[:, 0:3], from_range=(-1, 1), to_range=(0., 2.*np.pi))
            self.agent.action.angle[:, 1] /= 2.
            self.agent.action.length = data_scale(action[:, 3], from_range=(-1, 1), to_range=self.packing.cell_bound)

    def obs(self):
      ''' Return the observation of cell agent '''
      particle_info = []

      if self.packing.particle_type == 'ellipsoid':
        for particle in self.packing.particles:
          p = deepcopy(particle)
          p.periodic_check(self.agent.state.lattice.T)
          scaled_pos = p.scaled_centroid(self.agent.state.lattice.T)
          quaternion = Transform().euler2qua(p.state.orientation, 'JPL')
          particle_info.append(np.concatenate([scaled_pos] + [quaternion] + [p.semi_axis]))
        
      elif self.packing.particle_type == 'sphere':
        for particle in self.packing.particles:
          p = deepcopy(particle)
          p.periodic_check(self.agent.cell.state.lattice.T)
          scaled_pos = p.scaled_centroid(self.agent.state.lattice.T)
          particle_info.append(np.concatenate([scaled_pos] + [np.asarray([p.radius])]))

      # cell basis
      cell_info = (self.agent.state.lattice).tolist()

      obs = np.concatenate(particle_info + cell_info)

      return obs
      

    def cost(self):
        ''' Calculate the current costs and return a dict '''
        cost = {}
        # Overlap processing
        cost['cost_overlap'] = self.cost_callback(self.packing)

        # Sum all costs into single total cost
        cost['cost'] = sum(v for k, v in cost.items() if k.startswith('cost_'))

        # # Optionally remove shaping from reward functions.
        # if self.constrain_indicator:
        #     for k in list(cost.keys()):
        #         cost[k] = float(cost[k] > 0.0)  # Indicator function

        self._cost = cost

        return cost
