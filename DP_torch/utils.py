import numpy as np
import torch
from pytorch3d import transforms
from scipy import optimize


def relu(x):
    return np.maximum(0, x)


def Heaviside(x):
    """
    step function
    """
    if x > 0:
        return 1
    else:
        return 0


def abs_norm(x, Relu=False):
    temp = relu(x) if Relu else x
    return int(np.linalg.norm(temp, ord=1))


class Transform():
    def __init__(self):
        self.euler_convention = "ZYX"

    def euler2mat(self, angle):
        """
        Convert rotations given as Euler angles in radians to rotation matrices.
        """
        x = torch.as_tensor(angle, dtype=torch.double)
        mat = transforms.euler_angles_to_matrix(x, "ZYX")
        return mat

    def euler2qua(self, angle, convention='Hamilton'):
        """
        Convert euler angle to quaternion (two type of convention):
                        Hamilton = (w, x, y, z), by default
                        JPL = (x, y, z, w)  
        """
        x = self.euler2mat(angle)
        qua = transforms.matrix_to_quaternion(x)

        if convention == 'Hamilton':
            return qua
        elif convention == 'JPL':
            temp = torch.tensor(np.expand_dims(qua[0], 0))
            return torch.cat([qua[1:], temp])

    def euler_rotate(self, angle, point):
        """ Apply the rotation given by a quaternion to a 3D point. """
        p = torch.as_tensor(point, dtype=torch.double)
        x = self.euler2mat(angle)
        qua = transforms.matrix_to_quaternion(x)

        y = transforms.quaternion_apply(qua, p)
        return y.numpy()

    def euler_random(self):
        qua = transforms.random_quaternions(n=1, dtype=torch.double)
        x = transforms.quaternion_to_matrix(qua)

        y = transforms.matrix_to_euler_angles(x, "ZYX")
        return y[0].numpy()



def data_scale(unscaled, from_range, to_range):
    x = (unscaled - from_range[0]) / (from_range[1] - from_range[0])
    x = x * (to_range[1] - to_range[0]) + to_range[0]
    return x


def surface_area(lattice):
    """
    Calculate the surface area for simulation cell
    """
    area = 0.
    for i in range(3):
        direction = [0, 1, 2]
        direction.remove(i)

        [d1, d2] = direction
        area += 2.*np.linalg.norm(np.cross(lattice[d1], lattice[d2]))
    return area


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
                    f.write(
                        f'sphere {center[0]},{center[1]},{center[2]} {radius}\n')

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
                        f.write(
                            f'line {origin[0]},{origin[1]},{origin[2]} {axis[0]},{axis[1]},{axis[2]} \n')

            f.write("zoom e ")


def scaled_coordinate(position, frame):
    """
    frame = [v1 v2 v3] (Column-based Storage)
    """
    temp = np.linalg.pinv(frame)
    new_pos = np.matmul(temp, position.T).T
    return new_pos


# overlap potential
def overlap_fun(type, particle_a, particle_b, contact_force=False):
    """
    calculate overlap potential (energy) between two particles
    """
    r_AB = particle_b.state.centroid - particle_a.state.centroid

    if type in ('ellipsoid', 'ellipse'):
        # maximum of PW function Fun_AB
        X_A = particle_a.char_mat
        X_B = particle_b.char_mat

        t_c = optimize.fminbound(lambda t: - Fun_AB(t, X_A, X_B, r_AB), 0, 1)
        f = Fun_AB(t_c, X_A, X_B, r_AB)
        delta = np.sqrt(1. + f)
        # overlap_p = 0.5 * Heaviside(-f) * f**2
        overlap_p = 0.5 * Heaviside(-f) * (1.-delta)**2

    elif type in ('sphere', 'disk'):
        r = np.linalg.norm(r_AB)
        sigma = particle_a.radius + particle_b.radius
        x = 1. - r / sigma
        overlap_p = 0.5 * Heaviside(x) * x**2

    return overlap_p


def Fun_AB(t, XA, XB, r):
    """
    Calculation of Perram-Wertheim function:

    F_AB = t(1-t)r^T Y^{-1} r - 1, where Y = t*XB^{-1}+(1-t)*XA^{-1}
    """
    Y = t*np.linalg.pinv(XB) + (1.-t)*np.linalg.pinv(XA)
    Y = np.linalg.pinv(Y)

    F_AB = np.matmul(r.reshape(1, -1), Y)
    F_AB = t*(1.-t)*np.matmul(F_AB, r.reshape(-1, 1)) - 1.

    assert len(F_AB) == 1
    return F_AB[0][0]


def output_xyz(filename, packing):
    """
    For visulaization in ovito
    """
    centroid = [particle.centroid for particle in packing.visable_particles]
    quaternion = [Transform().euler2qua(particle.orientation, 'JPL')
                  for particle in packing.visable_particles]
    semi_axis = [particle.semi_axis for particle in packing.visable_particles]
    color = [particle.color for particle in packing.visable_particles]

    # colors = self.color_palette(labels, probabilities)
    n = len(packing.visable_particles)
    with open(filename, 'w') as f:
        # The keys should be strings
        f.write(str(n) + '\n')
        f.write('Lattice="' + ' '.join([str(vector)
                                        for vector in packing.cell.lattice.flat]) + '" ')
        # f.write('Origin="' + ' '.join(str(index) for index in packing.cell.origin) + '" ')
        f.write('Properties=pos:R:3:orientation:R:4:aspherical_shape:R:3:color:R:3 \n')

        if (packing.particle_type == 'ellipsoid'):
            np.savetxt(f, np.column_stack(
                [centroid, quaternion, semi_axis, color]))
