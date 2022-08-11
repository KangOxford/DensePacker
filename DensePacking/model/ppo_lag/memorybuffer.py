"""Memory buffer script

This manages the memory buffer. 
"""

import random
from collections import deque

import numpy as np
import scipy.signal

class Buffer:
    """
    Class for the Buffer creation
    """

    def __init__(self, size, gamma=.99, lambd=.97, gamma_c=.99, lambd_c=.97):
        self.s_buf = deque()
        self.a_buf = deque()
        self.mu_buf = deque()

        self.r_buf = np.zeros(size, dtype=np.float32)
        self.v_buf = np.zeros(size, dtype=np.float32)
        self.c_buf = np.zeros(size, dtype=np.float32)
        self.vc_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.cadv_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.cret_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lambd = gamma, lambd
        self.gamma_c, self.lambd_c = gamma_c, lambd_c

    def store(self, s, a, mu, r, v, c, vc, logp):
        self.s_buf.append(s)
        self.a_buf.append(a)
        self.mu_buf.append(mu)

        idx = self.size - 1
        self.r_buf[idx] = r
        self.v_buf[idx] = v
        self.c_buf[idx] = c
        self.vc_buf[idx] = vc
        self.logp_buf[idx] = logp

    def _discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def compute_mc(self, steps, last_v=0, last_vc=0):
        path_slice = slice(self.size - steps, self.size)

        r = np.append(self.r_buf[path_slice], last_v)
        v = np.append(self.v_buf[path_slice], last_v)
        deltas = r[:-1] + self.gamma * v[1:] - v[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lambd)
        self.ret_buf[path_slice] = self._discount_cumsum(r, self.gamma)[:-1]
       
        c = np.append(self.c_buf[path_slice], last_vc)
        vc = np.append(self.vc_buf[path_slice], last_vc)
        cdeltas = c[:-1] + self.gamma_c * vc[1:] - vc[:-1]
        self.cadv_buf[path_slice] = self._discount_cumsum(cdeltas, self.gamma_c * self.lambd_c)
        self.cret_buf[path_slice] = self._discount_cumsum(c, self.gamma_c)[:-1]

    def sample(self):
        self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / np.std(self.adv_buf)
        self.cadv_buf -= np.mean(self.cadv_buf)

        return [np.array(self.s_buf), np.array(self.a_buf), np.array(self.mu_buf), self.logp_buf, self.adv_buf, self.cadv_buf, self.ret_buf.reshape(-1, 1), self.cret_buf.reshape(-1, 1)]

    def clear(self):
        self.s_buf.clear()
        self.a_buf.clear()
        self.mu_buf.clear()

    @property
    def size(self):
        return len(self.s_buf)

