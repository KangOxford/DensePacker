import math
import numpy as np

from copy import deepcopy
from utils import Transform
from packing.core import Packing, Cell, Ellipsoid, Sphere

class Scenario(object):
    def build_packing(self):
        """
        build packings with fundamental cell
        """
        packing = Packing()

        packing.particle_type = 'ellipsoid'
        packing.num_particles = 2

        packing.fixed_particles = True
        packing.random_cell = False

        # add particles
        if packing.particle_type == 'ellipsoid':
            packing.dim = 3
            packing.particles = [Ellipsoid() for i in range(packing.num_particles)]
            for i, particle in enumerate(packing.particles):
                particle.name = 'ellipsoid %d' % i
                particle.color = np.array([0.51,0.792,0.992])
                particle.alpha, particle.beta = math.sqrt(3), 0.

        # add cell
        packing.cell = Cell(packing.dim, "strain_tensor")
        packing.cell.color = np.array([0.25,0.25,0.25])

        self.reset_packing(packing)
        return packing

    def reset_packing(self, packing):
        """
        initial conditions of the packing
        """
        # if packing.fixed_particles:
        #     p = packing.particles
        #     p[0].state.centroid = np.array([0., 0., 0.], dtype=np.float32)
        #     p[0].state.orientation = np.array([0, 0, 0], dtype=np.float32)

        #     p[1].state.centroid = np.array([2, 0., 0.], dtype=np.float32)
        #     p[1].state.orientation = np.array([0, math.sqrt(0.5), 0], dtype=np.float32)

        # if packing.random_cell:
        #     # 1st vector: along the x axis
        #     # 2nd vector: in the positive part of the xy-plane
        #     # 3rd vector: in the z > 0 half space
        #     variable = np.random.rand(6)

        #     lattice = np.zeros((3, 3))
        #     id = -1
        #     for i in range(3):
        #         for j in range(3):
        #             if (i < j): continue
        #             id += 1
            
        #             if id in [0, 3, 4]: variable[id] = 2.*variable[id] - 1.
        #             lattice[i][j] = variable[id] * packing.cell_bound[1]
        #     packing.cell.state.lattice = lattice
        # else:
        #     packing.cell.state.lattice = np.array([[4., 0,  0], [0,  2., 0], [0,  0,  2.]])

        packing.dilute_initialize()
        packing.cell.volume_elite = packing.cell.volume

    def reward(self, packing):
        if packing.cell.mode == "rotation":
            prev_performance = packing.cell.performance
            penalty = packing.cell_penalty
            packing.cell.performance = packing.fraction**2 / (penalty + 1e-10)
            # calculat the trend for judging if done
            packing.cell.trend = (packing.cell.performance-prev_performance)/prev_performance

            penalty_coefficient = 1.0
            reward_coefficient = 5.0

            if penalty > 0:
                reward = - penalty_coefficient * math.exp(penalty)
                # TODO revise this part to achieve the precision of -1e-10
                # print(f">>>Penalty: {reward}") ##
            else:
                # the reduction of cell volume between two steps
                agent = packing.cell
                if (agent.volume > agent.volume_elite): 
                    reward = 0.
                else:
                    reward = reward_coefficient * (agent.volume_elite - agent.volume)/agent.dv_prev
                    agent.dv_prev = agent.volume_elite - agent.volume
                    # the save the difference of the agent.volume_elite and agent.volume in the class
                    # return diff_this/diff_last
                    packing.cell.volume_elite = agent.volume
                    # print(f">>>Reward: {reward}") ##

        elif packing.cell.mode == "strain_tensor":
            # reward_coefficient = 14.79 # 0.74
            # reward = 1. - reward_coefficient * (packing.fraction - 1.)**2
            # reward = max(reward, 0)

            reward = packing.fraction_delta

        return reward
    
    def cell_penalty(self, packing):
        return packing.cell_penalty

    def observation(self, packing):
        # the same as XYZ file
        particle_info = []

        particles = deepcopy(packing.particles)
        if packing.particle_type == 'ellipsoid':
            for particle in particles:
                particle.periodic_check(packing.cell.state.lattice.T)
                # scaled_pos = partiicle.scaled_centroid(packing.cell.state.lattice.T)
                quaternion = Transform().euler2qua(particle.state.orientation, 'JPL')
                particle_info.append(np.concatenate([particle.state.centroid] + [quaternion] + [particle.semi_axis]))
        
        elif packing.particle_type == 'sphere':
            for particle in particles:
                particle.periodic_check(packing.cell.state.lattice.T)
                scaled_pos = particle.scaled_centroid(packing.cell.state.lattice.T)
                particle_info.append(np.concatenate([scaled_pos] + [np.asarray([particle.radius])]))

        # cell basis
        cell_info = (packing.cell.state.lattice).tolist()
        
        return np.concatenate(particle_info + cell_info)

    def done(self, packing):
        if packing.cell.mode == "strain_tensor":
            if packing.is_overlap:
                return True
            else: return False
        
        elif packing.cell.mode == "rotation":

            #if packing.cell_penalty > 0.:
            # if packing.fraction_delta < 0.01:
            #     return True
            # return False

            threshold_value = 1e-5
            if packing.cell.trend <= threshold_value: return True
            else: return False
