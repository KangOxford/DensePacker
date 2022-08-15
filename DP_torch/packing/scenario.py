import math
from statistics import variance
import numpy as np

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
            for i, ellipsoid in enumerate(packing.particles):
                ellipsoid.name = 'ellipsoid %d' % i
                ellipsoid.color = np.array([0.51,0.792,0.992])

                ellipsoid.alpha = 1.
                ellipsoid.beta = 0.

        # add cell
        packing.cell = Cell(packing.dim)
        packing.cell.color = np.array([0.25,0.25,0.25])

        # make initial conditions
        if packing.fixed_particles:
            p = packing.particles
            p[0].state.centroid = np.array([0., 0., 0.], dtype=np.float32)
            p[0].state.orientation = np.array([0, 0, 0], dtype=np.float32)

            p[1].state.centroid = np.array([2, 0., 0.], dtype=np.float32)
            p[1].state.orientation = np.array([0, math.sqrt(0.5), 0], dtype=np.float32)


        self.reset_packing(packing)
        return packing

    def reset_packing(self, packing):
        """
        initial conditions of the packing
        """
        if packing.random_cell:
            # 1st vector: along the x axis
            # 2nd vector: in the positive part of the xy-plane
            # 3rd vector: in the z > 0 half space
            variable = np.random.rand(6)

            lattice = np.zeros((3, 3))
            id = -1
            for i in range(3):
                for j in range(3):
                    if (i < j): continue
                    id += 1
            
                    if id in [0, 3, 4]: variable[id] = 2.*variable[id] - 1.
                    lattice[i][j] = variable[id] * packing.cell_bound[1]
            packing.cell.state.lattice = lattice
        else:
            packing.cell.state.lattice = np.array([[4., 0,  0], [0,  2., 0], [0,  0,  2.]])

        packing.cell.lattice_reduction()
        for particle in packing.particles:
            particle.periodic_check(packing.cell.state.lattice.T)
        packing.cell.volume_elite = packing.cell.volume

    def reward(self, packing):

        # the reduction of cell volume between two steps
        penalty_coefficient = 2.0
        if packing.cell_penalty > 0:
            reward = - penalty_coefficient * math.exp(packing.cell_penalty)

        else:
            # the reduction of cell volume between two steps
            agent = packing.cell
            reward = (agent.volume_elite - agent.volume) / packing.volume_allp
            if reward > 0.: 
                packing.cell.volume_elite = agent.volume
                #reward += 1.

        return reward
    
    def cell_penalty(self, packing):
        return packing.cell_penalty

    def particle_energy(self, particle, packing):
        # potential energy for certain particle
        u_local = 0.
        for particle2 in enumerate(packing.particles):
            if not particle2 == particle:
                potential = packing.overlap_fun(particle, particle2)
                if potential < 0.:
                    pass
                else:
                    u_local += 0.5 * potential**2
        return -u_local

    def observation(self, packing):
        # the same as XYZ file
        particle_info = []
        for p in packing.particles:
            scaled_pos = p.scaled_centroid(packing.cell.state.lattice.T)
            quaternion = Transform().euler2qua(angle = p.state.orientation)
            if packing.particle_type == 'ellipsoid':
                particle_info.append(np.concatenate([scaled_pos] + [quaternion] + [p.semi_axis]))
        
            elif packing.particle_type == 'sphere':
                particle_info.append(np.concatenate([scaled_pos] + [np.asarray([p.radius])]))

        # cell basis
        cell_info = (packing.cell.state.lattice).tolist()
        
        return np.concatenate(particle_info + cell_info)

    def done(self, packing):
        #if packing.cell_penalty > 0.:
        if packing.fraction_delta < 0.01:
            return True
        return False
