import numpy as np

import math
from copy import deepcopy
from utils import *

class Particle(object):

    def __init__(self):
        self.name = ''
        # spatial dimensions
        self.dim = None
        self.state = ParticleState()

        # color
        self.color = None

        # translational degree of freedom
        self.tran = None

    def scaled_centroid(self, lattice):
        """
        Convert absolute centroid to the one in scaled coordinate frame.
        # Note: lattice = [v1, v2, v3] (Column-based Storage)
        """
        temp = np.linalg.pinv(lattice)
        new_pos = np.matmul(temp, self.state.centroid.T).T
        return new_pos

    def periodic_image(self, vector):
        """
        Translate the target particle by the vector.
        """
        image = deepcopy(self)
        image.state.centroid += vector
        return image
    
    def periodic_check(self, lattice):
        """
        Check whether the centroid is in the unit cell (legal),
        otherwise return its legal image.
        # Note: lattice = [v1, v2, v3] (Column-based Storage)
        """
        scaled_pos = self.scaled_centroid(lattice)

        for i in range(self.dim):
            while (scaled_pos[i] >= 1):
                scaled_pos[i] -= 1
            while (scaled_pos[i] < 0):
                scaled_pos[i] += 1
	
        self.state.centroid = np.matmul(lattice, scaled_pos.T).T

    def random_orientation(self):
        self.state.orientation = Transform().euler_random()

class ParticleState(object):
    """
    Base state of all particles with (absolute) coordinate and euler angle.
    """
    def __init__(self):
        self.centroid = None
        # the z-y-x rotation convention (Tait-Bryan angles)
        self.orientation = None


class ParticleAction(object):
    def __init__(self):
        # translation & rotation
        self.tran = None
        self.rot = None


class Sphere(Particle):
    def __init__(self):
        super().__init__()
        self.dim = 3
        
        # shape parameters
        self.radius = None
        
        # script behavior to execute
        self.action_callback = None

    @property
    def volume(self):
        return 4./3.*np.pi*self.radius**3

    @property
    def inscribed_d(self):
        """
        diameter of the inscribed sphere
        """
        return 2.*self.radius

    @property
    def outscribed_d(self):
        """
        diameter of the outscribed sphere
        """
        return 2.*self.radius


class Ellipsoid(Particle):
    def __init__(self):
        super().__init__()
        self.dim = 3

        # shape parameters (alpha: alpha^beta : 1)
        self.alpha = None
        self.beta = None

        # rotational degree of freedom
        self.rot = None
	
        # control action range
        self.tran_low = np.array([0., 0., 0.])
        self.tran_high = np.array([0.1, 0.1, 0.1])
        
        # script behavior to execute
        self.action_callback = None
    
    @property
    def semi_axis(self):
        """
        Here we note some different notations:
        # Donev: alpha: beta: 1 (beta=1)
        """
        return np.array([self.alpha, self.alpha**self.beta, 1.])

    @property
    def volume(self):
        return 4./3.*np.pi*np.cumprod(self.semi_axis)[2]

    @property
    def inscribed_d(self):
        """
        diameter of the outscribed sphere
        """
        return 2.*np.min(self.semi_axis)

    @property
    def outscribed_d(self):
        """
        diameter of the outscribed sphere
        """
        return 2.*np.max(self.semi_axis)

    @property
    def char_mat(self):
        """
        characteristic ellipsoid matrix describing the 
        shape and orientation of the ellipsoid
        """
        # note that: rot_mat = Rz*Ry*Rx = Q^T
        rot_mat = Transform().euler2mat(self.state.orientation)
        
        # O: a diagonal matrix containing the major semi-axes
        O = np.diag(self.semi_axis)
        temp_mat = np.linalg.pinv(O)**2

        matrix = np.matmul(rot_mat, temp_mat)
        matrix = np.matmul(matrix, np.transpose(rot_mat))

        return matrix

    @property
    def grad(self) -> np.array:
      grad_t = - self.tran
      grad_r = - 2. * self.rot / self.outscribed_d

      return np.concatenate([grad_t, grad_r], axis=0)



class Cell(object):
    def __init__(self, dim, mode):
        self.dim = dim
        # origin of lattice (located in the origin by default)
        self.origin = None
        
        # color
        self.color = None

        # initial and previous volume
        self.volume_elite = None
        self.dv_prev = 1.
        self.performance = 1.
        self.trend = None

        self.mode = mode
        self.strainMod = 1e-2

        self.state = CellState()
        self.action = CellAction()
    
    @property
    def volume(self):
        if self.dim == 3:
            v = np.dot(np.cross(self.state.lattice[0], self.state.lattice[1]), self.state.lattice[2])
        elif self.dim == 2:
            v = np.cross(self.state.lattice[0], self.state.lattice[1])
        return math.fabs(v)
    
    @property
    def distortion(self):
        """
        Measure the distortion of the simulation cell, for 3D now only 
        """
        norm = 0.
        for i in range(3):
            norm += np.linalg.norm(self.state.lattice[i])
    
        fun = norm * surface_area(self.state.lattice) / self.volume / 18.
        return fun

    def new_combination(self):
        """
        Replace the cell by an equivalent set of basis vectors, 
        which are shorter and more orthogonal
        """
        new_lattice = self.state.lattice.copy()
        is_terminated = True
        for i in range(3):
            for j in range(3):
                if (j == i): continue

                for k in range(2):
                    lattice = self.state.lattice.copy()
                    lattice[i] = self.state.lattice[i] + (-1)**k * self.state.lattice[j]

                    if surface_area(lattice) < surface_area(new_lattice):
                        is_terminated = False
                        new_lattice = lattice.copy()
        
        self.state.lattice = new_lattice
        return is_terminated

    def lattice_reduction(self):
        """
        Repeat the above procedure.
        """
        if self.distortion > 1.5:
            terminal = False
            iter = 0
            while (not terminal):
                terminal = self.new_combination()
                iter += 1

                # it is prudent to impose a cutoff at roughly 10 iterations
                if (iter > 10): break

    def set_length(self, length):
        """
        Set length for cell in each direction
        """
        for i in range(self.dim):
            norm = np.linalg.norm(self.state.lattice[i])
            self.state.lattice[i] /= norm
            self.state.lattice[i] *= length[i]


class CellState(object):
    def __init__(self):
        # cell vectors
        self.lattice = None

class CellAction(object):
    def __init__(self):
        # euler angle for rotation
        self.angle = None
        # cell length
        self.length = None

        self.strain = None


class Packing(object):
    """
    combination of multi-particles and cell
    """
    def __init__(self):
        # spatial dimension
        self.dim = None

        # particle info
        self.particle_type = None
        self.particles = []
        self.num_particles = None

        # cell info
        self.cell = None

        # color dimensionality
        self.dim_color = 3

        # log the change of fraction for "done"
        self.fraction_delta = 0.01  
    
    @property
    def volume_allp(self):
        """
        volume of all particles
        """
        volume = 0.
        for particle in self.particles:
            volume += particle.volume
        return volume

    @property
    def fraction(self):
        v = 0.
        for particle in self.particles:
            v += particle.volume
        return v / self.cell.volume

    @property
    def cell_bound(self):
        """
        Lower and upper bound for cell vectors
        """
        lbound = ubound = 0.
        for particle in self.particles:
            lbound = max(particle.inscribed_d, lbound)
            ubound += particle.outscribed_d

        return [lbound, ubound]

    @property
    def max_od(self):
        """
        The diameter of the largest outscribed sphere of all particles
        """
        d = 0.
        for particle in self.particles:
            d = max(particle.outscribed_d, d)
        return d
    
    @property
    def upbound_image(self):
        """
        The upper bounds to the number of images that need to be
        checked in each direction.
        """
        cube, num_image = [], []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    initial_vertice = np.array([(-1)**i, (-1)**j, (-1)**k])*self.max_od
                    vertice = scaled_coordinate(initial_vertice, self.cell.state.lattice.T)
                    cube.append(vertice)
        
        for i in range(3):
            num_image.append(math.ceil(max(v[i] for v in cube)))
        return num_image

    @property
    def visable_particles(self):
        """
        Return particles in all vertices of unit cell (parallelpiiped)
        """
        copy_particles = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if(i==j==k==0): continue
                    
                    pos = [i, j, k]
                    base = np.matmul(pos, self.cell.lattice)
                    
                    for particle in self.particles:
                        image = particle.periodic_image(base)
                        image.color = np.array([0.992,0.525,0.529])
                        copy_particles.append(image)                 

        return self.particles + copy_particles

    @property
    def potential_energy(self, cal_force=False):
        image_list, extended_list = self.build_list()

        potential = 0.
        for a, particle_a in enumerate(self.particles):
            for b, particle_b in enumerate(self.particles):
                # internal part (in the unit cell)
                if (b > a): potential += overlap_fun(self.particle_type, particle_a, particle_b, cal_force)

                if (b == a):
                    # external part I (with one's own periodic images)
                    for index in image_list:
                        vector = np.matmul(index, self.cell.state.lattice)
                        particle_i = particle_b.periodic_image(vector)
                        potential += overlap_fun(self.particle_type, particle_a, particle_i, cal_force)
                else:
                    # external part II (with other particles’ periodic images)
                    # make sure that particle_b located in the origin
                    pa_new = particle_a.periodic_image(-particle_b.state.centroid)
                    pa_new.periodic_check(self.cell.state.lattice.T)
                    
                    for index in extended_list:
                        vector = np.matmul(index, self.cell.state.lattice)
                        particle_i = particle_b.periodic_image(vector-particle_b.state.centroid)
                        distance = np.linalg.norm(particle_i.state.centroid - pa_new.state.centroid)
                        if distance < self.max_od:
                            potential += overlap_fun(self.particle_type, pa_new, particle_i, cal_force)

        return potential
        
    @property  
    def is_overlap(self) -> bool:
        for i in range(3):
            if ((np.linalg.norm(self.cell.state.lattice[i])-self.cell_bound[0]) < -1e-10): return True

        image_list, extended_list = self.build_list()

        for a, particle_a in enumerate(self.particles):
            for b, particle_b in enumerate(self.particles):
                # internal part (in the unit cell)
                if (b > a): 
                    potential = overlap_fun(self.particle_type, particle_a, particle_b)
                    if (potential > 1e-20): return True

                if (b == a):
                    # external part I (with one's own periodic images)
                    for index in image_list:
                        vector = np.matmul(index, self.cell.state.lattice)
                        particle_i = particle_b.periodic_image(vector)
                        potential = overlap_fun(self.particle_type, particle_a, particle_i)
                        if (potential > 1e-20): return True
                else:
                    # external part II (with other particles’ periodic images)
                    # make sure that particle_b located in the origin
                    pa_new = particle_a.periodic_image(-particle_b.state.centroid)
                    pa_new.periodic_check(self.cell.state.lattice.T)
                    
                    for index in extended_list:
                        vector = np.matmul(index, self.cell.state.lattice)
                        particle_i = particle_b.periodic_image(vector-particle_b.state.centroid)
                        distance = np.linalg.norm(particle_i.state.centroid - pa_new.state.centroid)
                        if distance < self.max_od:
                            potential = overlap_fun(self.particle_type, pa_new, particle_i)
                            if (potential > 1e-20): return True

        return False

    @property
    def cell_penalty(self):
        """
        Penalty function for constraint violation.
        """
        # obvious overlap: within the range of (1, 2)
        if (self.fraction > 1.):
            penalty = 1. + (1. - 1./self.fraction)**2
        else:
            # need further calculation
	          # overlap potential is roughly less than 6 (empiricial)
            penalty = self.potential_energy / 6.
	
        return penalty

    def dilute_initialize(self):
        relative_pos = []
        for i in range(self.num_particles):
          relative_pos.append(np.random.rand(self.dim))
        
        min_d = 4.*np.sqrt(3)
        for a in range(self.num_particles):
          for b in range(self.num_particles):
            if a >= b: continue

            for i in range(-1, 2):
              for j in range(-1, 2):
                for k in range(-1, 2): 
                  pos = np.array([i, j, k], dtype=int)
                  temp_d = np.linalg.norm(relative_pos[b] - relative_pos[a] - pos)
                  min_d = min(min_d, temp_d)
        
        length = 1.001*self.max_od/min_d

        for i, particle in enumerate(self.particles):
          particle.state.centroid = length * relative_pos[i]
          particle.random_orientation()
        
        self.cell.state.lattice = length * np.eye(self.dim)

    def build_list(self):
        """
        Construct the image list in the scaled coordinate frame
        """
        num_image = self.upbound_image
        normal = np.sum(self.cell.state.lattice, axis=0)

        # establish the equivalent set of images in the regular coordinate frame
        image_list = []
        for i in range(-num_image[0], num_image[0]+1):
            for j in range(-num_image[1], num_image[1]+1):
                for k in range(-num_image[2], num_image[2]+1): 
                    if (i==j==k==0): continue

                    index = [i, j, k]
                    vec = np.matmul(index, self.cell.state.lattice)
                    # reduced self-image list
                    if np.dot(vec, normal)<0. or np.linalg.norm(vec)>self.max_od: continue
                    image_list.append(index)     

        index_bound = np.max(image_list + [[0, 0, 0]], axis=0).tolist()

        # add 1 layer of images in the positive vi directions to the set
        extended_list = image_list.copy()
        for i in range(index_bound[0]+3):
            for j in range(index_bound[1]+3):
                for k in range(index_bound[2]+3):
                    if (i==j==k==0): continue

                    index = [i, j, k]
                    if (index not in image_list): extended_list.append(index)

        # concentric approach
        if (len(image_list) > 0): image_list.sort(key=abs_norm)
        extended_list.sort(key=abs_norm)

        return image_list, extended_list

    def get_cell_origin(self):
        """
        Set the origin of lattice as the mass of all particles
        """
        origin = np.zeros(3)
        for particle in self.particles:
            origin += particle.state.centroid
        
        self.cell.origin = origin / self.num_particles

    def cell_step(self):
        """
        Update cell in the packing.
        """
        # store previous information
        self.cell.volume_prev = self.cell.volume
        fraction_prev = self.fraction

        # set action (small deformation)
        if self.cell.mode == "strain_tensor":
            deformation = np.multiply(self.cell.state.lattice, self.cell.action.strain)
            self.cell.state.lattice += deformation

        elif self.cell.mode == "rotation":
            self.cell.state.lattice = Transform().euler_rotate(
                self.cell.action.angle, self.cell.state.lattice)
            self.cell.set_length(self.cell.action.length)

        self.cell.lattice_reduction()
        self.fraction_delta = self.fraction - fraction_prev
        # math.fabs() #/ self.fraction_old
