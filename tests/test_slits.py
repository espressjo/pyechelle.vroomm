import inspect
import random

from numba.cuda import FakeCUDAKernel
from numba.cuda.random import create_xoroshiro128p_states

import pyechelle.slit

available_slits = [f for n, f in vars(pyechelle.slit).items() if inspect.isfunction(f) if n != 'njit']
available_cuda_slits = [f for n, f in vars(pyechelle.slit).items() if isinstance(f, FakeCUDAKernel)]


def test_slits():
    for slit_func in available_slits:
        x, y = slit_func(random.random(), random.random())
        assert 0. <= x <= 1.0
        assert 0. <= y <= 1.0


def test_cuda_slits():
    for slit_func in available_cuda_slits:
        threads_per_block = 1
        blocks = 1
        rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=random.randint(0, 1000))
        x, y = slit_func[threads_per_block, blocks](random.random(), random.random(), rng_states, 0)
        assert 0. <= x <= 1.0
        assert 0. <= y <= 1.0
