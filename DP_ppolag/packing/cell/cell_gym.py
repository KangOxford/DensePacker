import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from myutils import data_scale
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
        obs_dim = len(observation_callback(self.packing))
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim, ), dtype=np.float32)

        self.seed()

        # perfomance
        self.performance = 1.0

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
        obs = self.observation_callback(self.packing)
        reward = self.reward_callback(self.packing)
        done = self.done_callback(self.packing)
        info.update(self.cost())

        return obs, reward, done, info

    def get_reward(self):
        # TODO get the reward wrt the self.is_done
        pass
    

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
