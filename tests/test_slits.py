import inspect
import random

import pyechelle.slit

available_slits = [f for n, f in vars(pyechelle.slit).items() if inspect.isfunction(f) if n != 'njit']


# available_cuda_slits = [m[0] for m in inspect.getmembers(pyechelle.slit) if isinstance(m[1], CUDADispatcher)]


def test_slits():
    for slit_func in available_slits:
        x, y = slit_func(random.random(), random.random())
        assert 0. <= x <= 1.0
        assert 0. <= y <= 1.0

# TODO: add test for cuda slits
