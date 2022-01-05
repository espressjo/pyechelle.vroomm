import inspect
import random

from numba.core.registry import CPUDispatcher

import pyechelle.slit

available_slits = [m[0] for m in inspect.getmembers(pyechelle.slit) if isinstance(m[1], CPUDispatcher)]


def test_slits_numba():
    for s in available_slits:
        slit = getattr(pyechelle.slit, s)
        x, y = slit(random.random(), random.random())
        assert 0. <= x <= 1.0
        assert 0. <= y <= 1.0


def test_slits_native():
    for s in available_slits:
        slit = getattr(pyechelle.slit, s)
        x, y = slit.py_func(random.random(), random.random())
        assert 0. <= x <= 1.0
        assert 0. <= y <= 1.0
