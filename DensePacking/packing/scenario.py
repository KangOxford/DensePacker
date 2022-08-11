import math
import numpy as np
from packing.core import Packing, Cell, Ellipsoid, Sphere
from utils import affine_coordinate


class Scenario(object):
    def build_packing(self):
        """
        build packings with fundamental cell
        """
        packing = Packing()
        packing.dim = 3

        packing.fixed_particles = True
        packing.particle_type = 'sphere'
        packing.num_particles = 2

        # add cell
        packing.agent = Cell()

        # add particles
        #packing.particles = [Ellipsoid() for i in range(packing.num_particles)]
        #for i, ellipsoid in enumerate(packing.particles):
        #    ellipsoid.name = 'ellipsoid %d' % i
        #    ellipsoid.alpha = 1.
        #    ellipsoid.beta = 0
        
        packing.particles = [Sphere() for i in range(packing.num_particles)]
        for i, sphere in enumerate(packing.particles):
            sphere.name = 'sphere %d' % i
            sphere.radius = 1.

        # make initial conditions
        self.reset_packing(packing)
        return packing

    def reset_packing(self, packing):
        """
        initial conditions of the packing
        """
        packing.agent.color = np.array([0.25,0.25,0.25])             
    
        # set random initial states
        #basis = np.random.uniform(-1., 1., (packing.dim, packing.dim))
        #for i in range(packing.dim):
        #    basis[i] /= np.linalg.norm(basis[i])
        #packing.agent.state.basis = basis

        #packing.agent.state.lattice = np.eye(packing.dim)*packing.high_bound
        packing.agent.state.lattice = np.array([[4., 0,  0],
                                                [0,  2., 0],
                                                [0,  0,  2.]])

        if packing.fixed_particles:
            p = packing.particles
            p[0].state.centroid = np.array([0., 0., 0.], dtype=np.float32)
            p[0].state.orientation = np.array([0, 0, 0, 1], dtype=np.float32)

            p[1].state.centroid = np.array([2, 0., 0.], dtype=np.float32)
            p[1].state.orientation = np.array([0, math.sqrt(0.5), 0, math.sqrt(0.5)], dtype=np.float32)
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

        packing.agent.volume_elite = packing.agent.volume
        packing.fraction = packing.volume_allp / packing.agent.volume

    def reward(self, packing):
        # the reduction of cell volume between two steps
        if packing.cell_penalty > 0:
            reward = - math.exp(packing.cell_penalty)
        else:
            agent = packing.agent
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
        if packing.particle_type == 'ellipsoid':
            for p in packing.particles:
                r = p.state.centroid - packing.particles[0].state.centroid
                local_c = affine_coordinate(r, packing.agent.state.lattice.T)
                particle_info.append(np.concatenate([p.semi_axis] + [p.state.centroid] + [p.state.orientation] + [local_c]))
        
        elif packing.particle_type == 'sphere':
            for p in packing.particles:
                r = p.state.centroid - packing.particles[0].state.centroid
                local_c = affine_coordinate(r, packing.agent.state.lattice.T)
                particle_info.append(np.concatenate([np.asarray([p.radius])] + [p.state.centroid] + [local_c]))
        
        #for bulk in packing.script_particles:
        #    for particle in bulk:
        #        particle_info.append(np.concatenate([particle.state.centroid] + [particle.state.orientation]))
        #for particle in packing.script_particles:
        #    particle_info.append(np.concatenate([particle.state.centroid] + [particle.state.orientation]))

        # cell basis
        cell_info = (packing.agent.parallelepiped).tolist()
        
        return np.concatenate(particle_info + [cell_info] + [np.asarray([1.-packing.fraction])])

    def done(self, packing):
        #if packing.cell_penalty > 0.:
        if packing.fraction_delta < 0.01:
            return True
        return False