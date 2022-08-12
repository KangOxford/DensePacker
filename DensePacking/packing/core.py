import numpy as np
import tensorflow as tf
import math

from copy import deepcopy
from utils import *
from scipy.spatial.transform import Rotation as R
from tensorflow_graphics.geometry import transformation


# base particle
class Particle(object):
    """
    properties and state of particles
    """
    def __init__(self):
        self.name = ''
        # spatial dimensions
        self.dim = None

        # state: coordinate and euler angle
        self.state = ParticleState()

        # color
        self.color = None

    def scaled_centroid(self, lattice):
        """
        return centroid in scaled coordinate frame 
        lattice = [v1, v2, v3] (Column-based Storage)
        """
        temp = np.linalg.pinv(lattice)
        new_pos = np.matmul(temp, self.state.centroid.T).T
        return new_pos

    def periodic_image(self, vector):
        """
        translate the target particle by the vector
        """
        image = deepcopy(self)
        image.state.centroid += vector
        return image
    
    def periodic_check(self, lattice):
        """
        Check whether the centroid is in the unit cell (legal),
        otherwise return its legal image.
        lattice = [v1, v2, v3] (Column-based Storage)
        """
        scaled_pos = self.scaled_centroid(lattice)

        for i in range(3):
            while (scaled_pos[i] >= 1):
                scaled_pos[i] -= 1
            while (scaled_pos[i] < 0):
                scaled_pos[i] += 1
	
        self.state.centroid = np.matmul(lattice, scaled_pos.T).T

class ParticleState(object):
    """
    base state of all particles with (absolute) coordinate and
    euler angle 
    """
    def __init__(self):
        self.centroid = None
        # the z-y-x rotation convention (Tait-Bryan angles)
        # (fai 0-2*PI; cita 0-PI; pesai 0-2*PI)
        self.orientation = None

class ParticleAction(object):
    def __init__(self):
        # translation & rotation
        self.tran = None
        self.rot = None


class Sphere(Particle):
    def __init__(self):
        self.dim = 3

        # shape parameters
        self.radius = None

        # control range
        self.tran_low = np.array([0., 0., 0.])
        self.tran_high = np.array([0.1, 0.1, 0.1])

        # state
        self.state = ParticleState()
        
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

# specific particles
class Ellipsoid(Particle):
    def __init__(self):
        self.dim = 3

        # shape parameters
        self.alpha = None
        self.beta = None

        # control range
        self.tran_low = np.array([0., 0., 0.])
        self.tran_high = np.array([0.1, 0.1, 0.1])

        # state
        self.state = ParticleState()
        
        # script behavior to execute
        self.action_callback = None
    
    @property
    def semi_axis(self):
        return np.array([self.alpha, self.alpha**self.beta, 1.0])

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
        angle = tf.convert_to_tensor(self.state.orientation, dtype=np.double)
        rot_mat = transformation.rotation_matrix_3d.from_euler(angle)
        
        # O: a diagonal matrix containing the major semi-axes
        O = np.diag(self.semi_axis)
        temp_mat = np.linalg.pinv(O)**2

        matrix = np.matmul(rot_mat, temp_mat)
        matrix = np.matmul(matrix, np.transpose(rot_mat))

        return matrix


class Cell(object):
    """
    unit cell class
    """
    def __init__(self):
        # origin of lattice (located in the origin by default)
        self.origin = None
        
        # color
        self.color = None

        # initial and previous volume
        self.volume_elite = None
        self.volume_prev = None

        self.state = CellState()
        self.action = CellAction()
    
    @property
    def volume(self):
        v = np.dot(np.cross(self.state.lattice[0], self.state.lattice[1]), self.state.lattice[2])
        v = math.fabs(v)
        return v
    
    @property
    def distortion(self):
        """
        measure the distortion of the simulation cell
        """
        norm = 0.
        for i in range(3):
            norm += np.linalg.norm(self.state.lattice[i])
    
        fun = norm * surface_area(self.state.lattice) / self.volume / 18.
        return fun

    def new_combination(self):
        """
        repeatedly generate a set of 12 lattice combinations
        and select the one with the smallest surface area
        """
        new_lattice = self.state.lattice.copy()
        flag = 1
        for i in range(3):
            for j in range(3):
                if (j == i): continue

                for k in range(2):
                    lattice = self.state.lattice.copy()
                    lattice[i] = self.state.lattice[i] + (-1)**k * self.state.lattice[j]

                    if surface_area(lattice) < surface_area(new_lattice):
                        flag = 0
                        new_lattice = lattice.copy()
        
        self.state.lattice = new_lattice
        return flag
    
    def set_length(self, length):
        """
        set length for cell in each direction
        """
        for i in range(3):
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
        self.dim = 3

        # particle info
        self.particle_type = None
        self.particles = []
        self.num_particles = None

        # cell info
        self.cell = None

        # color dimensionality
        self.dim_color = 3

        # spatial boundary for packing generation
        self.low_bound = np.array([0., 0., 0.])

        self.fraction_delta = 0.01  # log the change of fraction for "done"
    
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
        """
        packing fraction
        """
        v = 0.
        for particle in self.particles:
            v += particle.volume
        return v / self.cell.volume

    @property
    def cell_bound(self):
        """
        lower and upper bound for cell vectors
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
        the upper bounds to the number of images that need to be
        checked in each direction
        """
        cube = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    initial_vertice = np.array([(-1)**i, (-1)**j, (-1)**k])*self.max_od
                    vertice = scaled_coordinate(initial_vertice, self.cell.state.lattice.T)
                    cube.append(vertice)
        
        num_image = []
        for i in range(3):
            num_image.append(math.ceil(max(v[i] for v in cube)))
        
        return num_image

    @property
    def visable_particles(self):
        """
        return particles in all vertices of unit cell (parallelpiiped)
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
    def cell_penalty(self):
        """
        the external part of overlap potential for all overlaping pairs
        """
        image_list, extended_list = self.build_list()

        # calculate penalty
        penalty = 0.
        for a, particle_a in enumerate(self.particles):
            for b, particle_b in enumerate(self.particles):
                if (b == a):
                    # external part I (with one's own periodic images)
                    for index in image_list:
                        vector = np.matmul(index, self.cell.state.lattice)
                        particle_i = particle_b.periodic_image(vector)
                        penalty += overlap_fun(self.particle_type, particle_a, particle_i)
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
                            penalty += overlap_fun(self.particle_type, pa_new, particle_i)

        return penalty

    @property  
    def is_overlap(self):
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
                        vector = np.matmul(index, self.cell.lattice)
                        particle_i = particle_b.periodic_image(vector-particle_b.state.centroid)
                        distance = np.linalg.norm(particle_i.state.centroid - pa_new.state.centroid)
                        if distance < self.max_od:
                            potential = overlap_fun(self.particle_type, pa_new, particle_i)
                            if (potential > 1e-20): return True

        return False

    def build_list(self):
        """
        construct the image list in the scaled coordinate frame
        """
        num_image = self.upbound_image
        normal = np.sum(self.cell.state.lattice, axis=0)

        # establish the equivalent set of images in the regular coordinate frame
        image_list = []
        num_layer = 0
        for i in range(-num_image[0], num_image[0]+1):
            for j in range(-num_image[1], num_image[1]+1):
                for k in range(-num_image[2], num_image[2]+1): 
                    if (i==j==k==0): continue

                    # reduced self-image list
                    index = [i, j, k]
                    vec = np.matmul(index, self.cell.state.lattice)
                    if np.dot(vec, normal)<0. or np.linalg.norm(vec)>self.max_od: continue
                    image_list.append(index) 
                    
                    # calculate max(i+j+k), and i,j,k should be non-negative simultaneously
                    num_layer = int(max(num_layer, abs_norm(index, Relu=True)))      

        # add 1 layer of images in the positive vi directions to the set
        extended_list = image_list.copy()
        # 1 <= i+j+k <= layer+6
        for i in range(num_layer+7):
            for j in range(num_layer+7-i):
                for k in range(relu(1-i-j), num_layer+7-i-j):
                    index = [i, j, k]
                    if (index not in image_list): extended_list.append(index)

        # concentric approach
        if (len(image_list)>0): image_list.sort(key=abs_norm)
        extended_list.sort(key=abs_norm)

        return image_list, extended_list

    def lattice_reduction(self):
        if self.cell.distortion > 1.5:
            terminal = False

            iter = 0
            while (not terminal):
                terminal = self.cell.new_combination()
                iter += 1

                # it is prudent to impose a cutoff at roughly 10 iterations
                if (iter > 10): break

    def get_cell_origin(self):
        """
        Set the origin of lattice as the mass of all particles
        """
        origin = np.zeros(3)
        for particle in self.particles:
            origin += particle.state.centroid
        
        self.cell.origin = origin / self.num_particles


    # update state of the packing
    def cell_step(self, method):
        # store previous information
        self.cell.volume_prev = self.cell.volume
        self.fraction_prev = self.fraction

        # set action (small deformation)
        if method == "strain_tensor":
            deformation = np.multiply(self.agent.state.base, self.agent.action.strain)
            self.agent.state.base += deformation
            self.agent.state.length = [np.linalg.norm(x) for x in self.agent.state.base]
            self.agent.state.basis = [x / np.linalg.norm(x) for x in self.agent.state.base]
            
            self.agent.action.num += 1

        elif method == "rotation":
            a = self.cell
            qua = transformation.quaternion.from_euler(a.action.angle)
            lattice = tf.convert_to_tensor(a.state.lattice, dtype=np.double)
            new_base = transformation.quaternion.rotate(lattice, qua)
            self.cell.state.lattice = new_base.numpy()
            self.cell.set_length(self.cell.action.length)
            # print(self.cell.action.angle)
            # print(self.cell.action.length)

        self.lattice_reduction()
        # print(self.cell.state.lattice)
        for particle in self.particles:
            particle.periodic_check(self.cell.state.lattice.T)

        self.fraction_delta = math.fabs(self.fraction - self.fraction_prev) #/ self.fraction_old


    # update state of the packing
    def particle_step(self):
        # set actions for the agent
        self.agent.action = self.agent.action_callback(self)

        # get information (lattice constant)
        self.agent.length = self.get_length()

        # update the packing fraction
        fraction_old = self.fraction

        self.fraction = self.volume_allp / self.agent.volume
        self.fraction_delta = (self.fraction - fraction_old) / fraction_old


