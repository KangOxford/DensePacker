import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from copy import deepcopy
from collections import OrderedDict
from myutils import *
from packing.scenario import Scenario

scenario = Scenario()

# environment for unit cell agent in the packing


class CellEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    # Default configuration (this should not be nested since it gets copied)
    DEFAULT = {
        'mode': 'rotation',  # motion mode of fundamental cell
        'random_initialization': False,  # Random initialize fundamental cell
    }

    def __init__(self, packing=scenario.build_packing(),
                 mode: str = "rotation"):

        self.packing = packing
        self.agent = self.packing.cell
        self.dim = self.packing.dim

        self.mode = mode
        self.random_initialization = False

        # action space
        if self.mode == "strain_tensor":
            # (symmetric) strain tensor controlling the deformation of cell
            dim = self.dim*(self.dim+1)/2
            self.action_space = spaces.Box(
                low=-1., high=1., shape=(6, ), dtype=np.float32)
        elif self.mode == "rotation":
            # euler angles + cell length
            self.action_space = spaces.Box(
                low=-1., high=1., shape=(4*self.dim, ), dtype=np.float32)

        # observation space
        self.build_observation_space()
        self.seed()

    def build_observation_space(self):
        ''' Construct observtion space.  Happens only once at during __init__ '''
        obs_space_dict = OrderedDict()  # See self.obs()

        # Particle info
        for particle in self.packing.particles:
            obs_space_dict[particle] = gym.spaces.Box(
                -1.0, np.inf, (particle.obs_dim,), dtype=np.float32)
        # Cell info
        obs_space_dict['cell'] = gym.spaces.Box(
            -np.inf, np.inf, (self.dim, self.dim), dtype=np.float32)
        # Flatten it ourselves
        self.obs_space_dict = obs_space_dict
        self.obs_flat_size = sum([np.prod(i.shape)
                                 for i in self.obs_space_dict.values()])
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.obs_flat_size,), dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        ''' Reset the physics simulation and return observation '''
        self.done = False
        self.steps = 0  # Count of steps taken in this episode

        if self.random_initialization:
            # 1st vector: along the x axis
            # 2nd vector: in the positive part of the xy-plane
            # 3rd vector: in the z > 0 half space
            variable = np.random.rand(6)

            lattice = np.zeros((3, 3))
            id = -1
            for i in range(3):
                for j in range(3):
                    if (i < j):
                        continue
                    id += 1

                    if id in [0, 3, 4]:
                        variable[id] = 2.*variable[id] - 1.
                    lattice[i][j] = variable[id] * self.packing.cell_bound[1]
            self.agent.state.lattice = lattice
        else:
            self.agent.state.lattice = np.array(
                [[4., 0,  0], [0,  2., 0], [0,  0,  2.]])

        self.agent.lattice_reduction()
        self.agent.gap = max(self.agent.volume-self.packing.volume_allp, 0.)

        return self.obs()

    def obs(self):
        ''' Return the observation of cell agent '''
        particle_info = []

        particles = deepcopy(self.packing.particles)
        if self.packing.particle_type == 'ellipsoid':
            for particle in particles:
                particle.periodic_check(self.agent.state.lattice.T)
                scaled_pos = particle.scaled_centroid(
                    self.agent.state.lattice.T)
                quaternion = Transform().euler2qua(particle.state.orientation, 'JPL')
                particle_info.append(np.concatenate(
                    [scaled_pos] + [quaternion] + [particle.semi_axis]))

        elif self.packing.particle_type == 'sphere':
            for particle in particles:
                particle.periodic_check(self.agent.state.lattice.T)
                scaled_pos = particle.scaled_centroid(
                    self.agent.state.lattice.T)
                particle_info.append(np.concatenate(
                    [scaled_pos] + [np.asarray([particle.radius])]))

        # cell basis
        cell_info = (self.agent.state.lattice).tolist()

        obs = np.concatenate(particle_info + cell_info)

        return obs

    def cost(self):
        ''' Calculate the current costs and return a dict '''
        cost = {}
        # Overlap processing
        if (self.packing.fraction > 1.):
            cost['cost_overlap'] = 1. + (1. - 1./self.packing.fraction)**2
        else:
            # need further calculation
            # overlap potential is roughly less than 6 (empiricial)
            cost['cost_overlap'] = self.packing.overlap_potential / 6.

        # Sum all costs into single total cost
        cost['cost'] = sum(v for k, v in cost.items() if k.startswith('cost_'))

        # # Optionally remove shaping from reward functions.
        # if self.constrain_indicator:
        #     for k in list(cost.keys()):
        #         cost[k] = float(cost[k] > 0.0)  # Indicator function

        self._cost = cost

        return cost

    def step(self, action):
        ''' Take a step and return observation, reward, done, and info '''
        info = {}

        self._set_action(action)

        # set action (small deformation)
        if self.mode == "strain_tensor":
            deformation = np.multiply(
                self.agent.state.base, self.agent.action.strain)
            self.agent.state.base += deformation
            self.agent.state.length = [np.linalg.norm(
                x) for x in self.agent.state.base]
            self.agent.state.basis = [
                x / np.linalg.norm(x) for x in self.agent.state.base]

            self.agent.action.num += 1
        elif self.mode == "rotation":
            self.agent.state.lattice = Transform().euler_rotate(
                self.agent.action.angle, self.agent.state.lattice)
            self.agent.set_length(self.cell.action.length)

        self.agent.lattice_reduction()
        self.packing.fraction_delta = np.fabs(
            self.packing.fraction - self.packing.fraction_prev)

        # reward and observation
        reward = self.reward()

        # Constraint violations
        info.update(self.cost())

        if self.packing.fraction_delta < 0.01:
            self.done = True
        else:
            self.done = False

        return self.obs(), reward, self.done, info

    def reward(self):
        gap_prev = self.agent.gap
        self.agent.gap = max(self.agent.volume-self.packing.volume_allp, 0.)

        reward = (gap_prev - self.agent.gap) / self.packing.volume_allp
        return reward

    def render(self):
        print("is_overlap {:d} overlap_potential {:2f} packing_fraction {:2f}".format(
            self.packing.is_overlap, self.packing.overlap_potential, self.packing.fraction))

    def _set_action(self, action):

        if self.mode == "strain_tensor":
            assert len(action) == 6

            strain = np.zeros((self.dim, self.dim))
            id = -1
            for i in range(self.dim):
                for j in range(self.dim):
                    if i > j:
                        continue
                    id += 1
                    strain[j][i] = strain[i][j] = action[id]

            self.agent.action.strain = 1e-1 * strain

        elif self.mode == "rotation":
            assert len(action) == 12

            action = action.reshape(3, -1)

            self.agent.action.angle = action[:, 0:3] * np.pi
            self.agent.action.angle[:, 1] /= 2.
            self.agent.action.length = data_scale(
                action[:, 3], from_range=(-1, 1), to_range=self.packing.cell_bound)
