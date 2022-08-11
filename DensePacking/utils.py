from ast import Num
import numpy as np
from gym.spaces import Box, Discrete
import tensorflow as tf
from scipy import optimize

def space_n_to_shape_n(space_n):
    """
    Takes a list of gym spaces and returns a list of their shapes
    """
    return np.array([space_to_shape(space) for space in space_n])

def space_to_shape(space):
    """
    Takes a gym.space and returns its shape
    """
    if isinstance(space, Box):
        return space.shape
    elif isinstance(space, Discrete):
        return [space.n]
    else:
        raise RuntimeError("Unknown space type. Can't return shape.")

def clip_by_local_norm(gradients, norm):
    """
    Clips gradients by their own norm, NOT by the global norm
    as it should be done (according to TF documentation).
    This here is the way MADDPG does it.
    """
    for idx, grad in enumerate(gradients):
        # avoid exploding gradient problem
        gradients[idx] = tf.clip_by_norm(grad, norm)
    return gradients

class LinearSchedule(object):
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.
    FROM STABLE BASELINES
    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.current_step = 0
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, step):
        fraction = min(float(step) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def scr(filename, type, particles, lattice):

	with open(filename, 'w') as f:
            f.write('-osnap off\n')
            f.write("erase all \n")
            f.write("vscurrent 2\n")

            if type == 'sphere':
                for i in range(len(particles)):
                    for a, p in enumerate(particles[i]):
                        center = p.state.centroid
                        radius = p.radius
                        f.write(f'sphere {center[0]},{center[1]},{center[2]} {radius}\n')
            
                o = np.zeros(3)
                for k in range(len(particles[0])):
                    o += particles[0][k].state.centroid / len(particles[0])
            
                direction = [0, 1, 2]
                for j in range(3):

                    vector = lattice[j]
                    direction = [0, 1, 2]
                    direction.remove(j)

                    [d1, d2] = direction
                    for m in range(2):
                        for n in range(2):
                            origin = o + m*lattice[d1] + n*lattice[d2]
                            axis = origin + vector
                            f.write(f'line {origin[0]},{origin[1]},{origin[2]} {axis[0]},{axis[1]},{axis[2]} \n')
            
                f.write("zoom e ")   

def Heaviside(x): 
    # step function
    if x>0: return 1
    else: return 0

# overlap potential
def overlap_fun(type, particle_a, particle_b):
    """
    calculate overlap potential (energy) between two particles
    """
    r_AB = particle_b.state.centroid - particle_a.state.centroid

    if type == 'ellipsoid':
        # maximum of PW function Fun_AB
        X_A = particle_a.char_mat
        X_B = particle_b.char_mat

        t_c = optimize.fminbound(lambda t: - Fun_AB(t, X_A, X_B, r_AB), 0, 1)
        f = Fun_AB(t_c, X_A, X_B, r_AB)
        overlap_p = 0.5 * Heaviside(-f) * f**2

    elif type =='sphere':
        r = np.linalg.norm(r_AB)
        sigma = particle_a.radius + particle_b.radius
        x = 1. - r/sigma
        overlap_p = 0.5 * Heaviside(x) * x**2

    return overlap_p

def Fun_AB(t, XA, XB, r):
    """
    Calculation of Perram-Wertheim function:
    # F_AB = t(1-t)r^T Y^{-1} r, where Y = t*XB^{-1}+(1-t)*XA^{-1}
    """
    # F_AB = t(1-t)r^T Y^{-1} r
    Y = t*np.linalg.pinv(XB) + (1.-t)*np.linalg.pinv(XA)
    Y = np.linalg.pinv(Y)

    F_AB = np.matmul(r.reshape(1,-1), Y)
    F_AB = t*(1.-t)*np.matmul(F_AB, r.reshape(-1,1)) - 1.

    assert len(F_AB) == 1
    return F_AB[0][0]

def affine_coordinate(vec, lattice):
    """
    coordinate transformation
    """
    lat = np.linalg.pinv(lattice)
    new_vec = np.matmul(lat, vec.reshape(-1,1))
    return new_vec.reshape(1,-1)[0]