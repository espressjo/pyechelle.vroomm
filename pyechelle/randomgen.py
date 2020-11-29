import math

import numpy as np
from numba import int32, float32, njit, jit, float64
from numba.experimental import jitclass


@njit(float64[:, :](int32), parallel=True, nogil=True, cache=True)
def generate_slit_xy(N):
    """  Generate uniform distributed XY position within a unit box

    Args:
        N (int):  number of random numbers

    Returns:
        np.ndarray: random XY position
    """
    return np.random.random((N, 2))
    # x = np.random.random(N)
    # y = np.random.random(N)
    # return np.vstack((x, y))


@njit(float64[:, :](int32), parallel=True, nogil=True, cache=True)
def generate_slit_round(N):
    """  Generate uniform distributed XY position within a unit circle

    Args:
        N (int):  number of random numbers

    Returns:
        np.ndarray: random XY position
    """
    r = np.sqrt(np.random.random(N))
    phi = np.random.random(N) * np.pi * 2

    return np.vstack((r * np.cos(phi), r * np.sin(phi)))


# @njit(float64[:, :](int32, int32, float32), parallel=True, nogil=True, cache=True)
def generate_slit_polygon(npoly=5, N=1000, phi=0.):
    """
    Given three vertices A, B, C,
    sample point uniformly in the triangle
    """
    phi = np.deg2rad(phi)
    r1, r2 = np.random.random((2, N))
    s1 = np.sqrt(r1)
    phi_segment = 2. * np.pi / npoly

    B = [1., 0.]
    C = [math.cos(phi_segment), math.sin(phi_segment)]
    x = B[0] * (1.0 - r2) * s1 + C[0] * r2 * s1
    y = B[1] * (1.0 - r2) * s1 + C[1] * r2 * s1

    segments = np.random.randint(0, npoly, N)
    arg_values = phi_segment * segments + phi
    cos_values = np.cos(arg_values)
    sin_values = np.sin(arg_values)
    xnew = x * cos_values - y * sin_values
    ynew = x * sin_values + y * cos_values
    return np.vstack((xnew / 2. + 0.5, ynew / 2. + 0.5))


@njit()
def unravel_index(index, shape):
    out = []
    for dim in shape[::-1]:
        out.append(index % dim)
        index = index // dim
    return out[::-1]


spec = [("K", int32), ("q", float32[:]), ("J", int32[:])]

@jit()
def samplealias2d(a, n=1):
    a = np.asarray(a)
    aa = AliasSample(a.T.ravel())
    index = aa.sample(n)
    # return np.unravel_index(index, dims=a.shape)
    return unravel_index(index, np.array(a.shape, dtype=np.int32))

@njit(int32[:](int32[:], float32[:], int32), parallel=True, nogil=True)
def draw(J,q,n):
    r1, r2 = np.random.rand(n), np.random.rand(n)
    res = np.zeros(n, dtype=np.int32)
    lj = len(J)
    for i in range(n):
        kk = int(np.floor(r1[i] * lj))
        if r2[i] < q[kk]:
            res[i] = kk
        else:
            res[i] = J[kk]
    return res


@jitclass(spec)
class AliasSample:
    """ The AliasSample class allows to draw random numbers from discrete distributions.

        As described `here <https://www.keithschwarz.com/darts-dice-coins/>`_, the most efficient way to draw random
         numbers from a discrete probability distribution are alias sampling methods.
         Here, we use a slightly adapted implementation of the Vose sampling method from
         `here <https://gist.github.com/jph00/30cfed589a8008325eae8f36e2c5b087>`_.
    """

    def __init__(self, probability: np.ndarray):
        """
        Constructor

        Args:
            probability: discrete probability density to draw from. Total sum needs to be 1.0
        """
        # probability = probability / np.sum(probability)
        self.K = len(probability)
        self.q = np.zeros(self.K, dtype=np.float32)
        self.J = np.zeros(self.K, dtype=np.int32)

        smaller, larger = [], []
        for kk, prob in enumerate(probability):
            self.q[kk] = self.K * prob
            if self.q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small, large = smaller.pop(), larger.pop()
            self.J[small] = large
            self.q[large] = self.q[large] - (1.0 - self.q[small])
            if self.q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

    def draw_one(self) -> float32:
        """
        Draw single number from given probability function.

        Returns:
            random sample
        """
        K, q, J = self.K, self.q, self.J
        kk = int(np.floor(np.random.rand() * len(J)))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]

    def sample(self, n: int) -> np.ndarray:
        """
        Draw n random numbers from distribution.

        Args:
            n: number of samples to draw

        Returns:
            array of random numbers
        """
        # r1, r2 = np.random.rand(n), np.random.rand(n)
        # res = np.zeros(n, dtype=np.int32)
        # lj = len(self.J)
        # for i in range(n):
        #     kk = int(np.floor(r1[i] * lj))
        #     if r2[i] < self.q[kk]:
        #         res[i] = kk
        #     else:
        #         res[i] = self.J[kk]
        # return res
        return draw(self.J, self.q, n)


def piecewise_constant(x, y, N: int):
    """
    Piecewise constant
    Args:
        x:
        y:
        N: number of samples to draw

    Returns:

    """
    diff = np.ediff1d(x)
    x0 = np.random.choice(y, N, p=y)
    dist = np.random.rand(N)
    # x0 += dist*
