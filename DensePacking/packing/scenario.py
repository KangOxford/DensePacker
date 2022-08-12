import math
import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry import transformation
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
        packing.cell = Cell()
        packing.cell.color = np.array([0.25,0.25,0.25])

        # make initial conditions
        self.reset_packing(packing)
        return packing

    def reset_packing(self, packing):
        """
        initial conditions of the packing
        """
                   
    
        # set random initial states
        #basis = np.random.uniform(-1., 1., (packing.dim, packing.dim))
        #for i in range(packing.dim):
        #    basis[i] /= np.linalg.norm(basis[i])
        #packing.agent.state.basis = basis

        #packing.agent.state.lattice = np.eye(packing.dim)*packing.high_bound
        packing.cell.state.lattice = np.array([[4., 0,  0],
                                                [0,  2., 0],
                                                [0,  0,  2.]])

        if packing.fixed_particles:
            p = packing.particles
            p[0].state.centroid = np.array([0., 0., 0.], dtype=np.float32)
            p[0].state.orientation = np.array([0, 0, 0], dtype=np.float32)

            p[1].state.centroid = np.array([2, 0., 0.], dtype=np.float32)
            p[1].state.orientation = np.array([0, math.sqrt(0.5), 0], dtype=np.float32)
        else:
            is_valid = False
            for i, ellipsoid in enumerate(packing.particles):
                while not is_valid:
                    length = packing.num_particles*2.*ellipsoid.semi_axis[0]
                    centroid = np.random.uniform(0., length, packing.dim)

                    if packing.dim == 3:
                        orientation = np.random.uniform(-1., 1., 4)
                    else:
                        orientation = np.random.uniform(0., np.pi, 1)

                    # settled ellipsoids
                    for j, ellipsoid_s in enumerate(packing.particles):
                        if not j<i: continue

                        if packing.overlap_fun(ellipsoid, ellipsoid_s) < 0.: 
                            is_valid = False
                            break
                        else: is_valid = True

                ellipsoid.state.centroid = centroid
                ellipsoid.state.orientation = orientation

        packing.cell.volume_elite = packing.cell.volume

    def reward(self, packing):
        # the reduction of cell volume between two steps
        if packing.cell_penalty > 0:
            reward = - math.exp(packing.cell_penalty)
        else:
            agent = packing.cell
            reward = (agent.volume_elite - agent.volume) / packing.volume_allp
            if reward > 0.: 
                packing.agent.volume_elite = agent.volume
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

        particle_info = []
        for p in packing.particles:
            scaled_pos = p.scaled_centroid(packing.cell.state.lattice.T)
            orientation = tf.convert_to_tensor(p.state.orientation, dtype=np.double)
            quaternion = transformation.quaternion.from_euler(orientation)
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