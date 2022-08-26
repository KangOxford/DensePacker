import math
import numpy as np

from copy import deepcopy
from myutils import Transform
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
        packing.cell = Cell(packing.dim)
        packing.cell.color = np.array([0.25,0.25,0.25])

        self.reset_packing(packing)
        return packing

    def reset_packing(self, packing):
        """
        initial conditions of the packing
        """
        if packing.fixed_particles:
            p = packing.particles
            p[0].state.centroid = np.array([0., 0., 0.], dtype=np.float32)
            p[0].state.orientation = np.array([0, 0, 0], dtype=np.float32)

            p[1].state.centroid = np.array([2, 0., 0.], dtype=np.float32)
            p[1].state.orientation = np.array([0, math.sqrt(0.5), 0], dtype=np.float32)

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
        packing.cell.volume_elite = packing.cell.volume_prev = packing.cell.volume


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

    
