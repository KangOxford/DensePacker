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
        # name
        self.name = ''
        # spatial dimensions
        self.dim = None
        # color
        self.color = None
        # state: coordinate and quaternion
        self.state = ParticleState()

    def copy_translate(self, vector, scale):
        tmp = deepcopy(self)
        tmp.state.centroid += scale * vector
        return tmp

class ParticleState(object):
    """
    base state of all particles with (absolute) coordinate (x, y, z) and
    JPL quaternion (x,y,z,w)   
    """
    def __init__(self):
        self.centroid = None
        self.orientation = None

class ParticleAction(object):
    def __init__(self):
        # translation & rotation
        self.tran = None
        self.rot = None


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
    def bounding_diameter(self):
        return 2.*np.max(self.semi_axis)

    @property
    def char_mat(self):
        """
        a characteristic ellipsoid matrix describing the 
        shape and orientation of the ellipsoid.
        """
        rot_mat = transformation.rotation_matrix_3d.from_quaternion(self.state.orientation)
        #qua = R.from_quat(self.state.orientation)
        #rot_mat = qua.as_matrix()

        # O: a diagonal matrix containing the major semi-axes
        O = np.diag(self.semi_axis)
        temp_mat = np.matmul(np.linalg.pinv(O), np.linalg.pinv(O))

        matrix = np.matmul(np.transpose(rot_mat), temp_mat)
        matrix = np.matmul(matrix, rot_mat)
        return matrix

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
    def bounding_diameter(self):
        return 2.*self.radius


class Cell(object):
    """
    unit cell class
    """
    def __init__(self):
        # origin of coordinate (not necessary)
        self.origin = None
        
        # color
        self.color = None

        # initial and previous volume
        self.volume_elite = None
        self.volume_prev = None

        self.state = CellState()
        self.action = CellAction()
        self.action_callback = None
    
    @property
    def parallelepiped(self):
        """
        seven vector pointing towards all vertex from lattice origin
        """
        base = la = self.state.lattice
        base = np.append(base, la[0]+la[1])
        base = np.append(base, la[0]+la[2])
        base = np.append(base, la[1]+la[2])
        base = np.append(base, la[0]+la[1]+la[2])

        return base

    @property
    def volume(self):
        v = np.dot(np.cross(self.state.lattice[0], self.state.lattice[1]), self.state.lattice[2])
        v = math.fabs(v)
        return v

class CellState(object):
    def __init__(self):
        # affine coordinate system with normalized basis (e1, e2, e3)
        self.lattice = None

class CellAction(object):
    def __init__(self):
        # rotate all three vector in a base
        self.base = None
        self.angle = None

        self.rot = None

        self.strain = None
        self.num = None
        self.num_accept = None
        self.acceptence = None


class Packing(object):
    """
    combination of multi-particles and cell
    """
    def __init__(self):
        self.agent = None

        self.particle_type = None
        self.particles = []
        self.num_particles = None

        # spatial dimension
        self.dim = 3
        # color dimensionality
        self.dim_color = 3

        # spatial boundary for packing generation
        self.low_bound = np.array([0., 0., 0.])

        self.fraction = 0.0
        self.fraction_delta = 0.01  # log the change of fraction for "done"

    @property
    def high_bound(self):
        """
        generation boundary for multi_particles
        """
        particle = self.particles[0]
        return self.num_particles*particle.bounding_diameter

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
    def script_particles(self):
        """
        return all particles and their periodcially copyed ones
        """
        id=[]
        copy_particles = []
        lattice = self.agent.state.lattice
        for i in range(-1, self.dim-1):
            for j in range(-1, self.dim-1):
                for k in range(0, self.dim-1):
                    if(i==j==k==0): continue
                    
                    index = 100*i+10*j+k
                    if (-index not in id):
                        id.append(index)
                        vec = i*lattice[0] + j*lattice[1] + k*lattice[2]
                        part = []
                        for particle in self.particles:
                            part.append(particle.copy_translate(vec, 1.))

                        copy_particles.append(part)
        
        return copy_particles

    @property
    def visable_particles(self):
        """
        return all particles for plot
        """
        id=[]
        copy_particles = []
        lattice = self.agent.state.lattice
        for i in range(0, self.dim-1):
            for j in range(0, self.dim-1):
                for k in range(0, self.dim-1):
                    if(i==j==k==0):continue
                    
                    base = i*lattice[0] + j*lattice[1] + k*lattice[2]
                    part =[]
                    for particle in self.particles:
                        part.append(particle.copy_translate(base, 1.))

                    copy_particles.append(part)                  

        return [self.particles] + copy_particles

    @property
    def is_valid(self):
        for a, ellipsoid_a in enumerate(self.particles):
            for b, ellipsoid_b in enumerate(self.particles):
                if(b <= a): continue
                if self.overlap_fun(ellipsoid_a, ellipsoid_b) < 0.:
                    return False
        return True

    @property
    def cell_penalty(self):
        """
        we ignore the validity of particles within a fundamental cell,
        but concentrate on validity of our cell
        """
        penalty = 0.
        type = self.particle_type
        for i in range(len(self.script_particles)):
            for particle_a in self.script_particles[i]:
                for particle_b in self.particles:
                    potential = overlap_fun(type, particle_a, particle_b)
                    # penalty for overlap: soft potential with energy scale
                    penalty += potential

        return penalty
    
    # update state of the packing
    def cell_step(self, method):
        # store previous information
        self.agent.volume_prev = self.agent.volume
        self.fraction_old = self.fraction

        # set action (small deformation)
        if method == "strain_tensor":
            deformation = np.multiply(self.agent.state.base, self.agent.action.strain)
            self.agent.state.base += deformation
            self.agent.state.length = [np.linalg.norm(x) for x in self.agent.state.base]
            self.agent.state.basis = [x / np.linalg.norm(x) for x in self.agent.state.base]
            
            self.agent.action.num += 1

        elif method == "rotation":
            a = self.agent
            qua = transformation.quaternion.from_euler(a.action.angle)
            base = tf.convert_to_tensor(a.action.base, dtype=np.double)
            new_base = transformation.quaternion.rotate(base, qua)
            self.agent.state.lattice = new_base.numpy()

            #self.agent
            #for i in range(self.dim):
            #    qua = transformation.quaternion.normalize(a.action.rot[i])
            #    vec = tf.convert_to_tensor(a.state.basis[i], dtype=np.double)
            #    new_vec = transformation.quaternion.rotate(vec, qua)
            #    self.agent.state.basis[i] = new_vec.numpy()

        #print(self.agent.action.base)
        #print(self.agent.action.angle)
        self.fraction = self.volume_allp / (self.agent.volume + 1e-8)
        self.fraction_delta = math.fabs(self.fraction - self.fraction_old) #/ self.fraction_old
        
        
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


