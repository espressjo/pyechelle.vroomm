import numpy as np
from numba import int32, float32, njit, jit
from numba.types import UniTuple
from numba.experimental import jitclass

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
    aa = AliasSample(a.ravel())
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