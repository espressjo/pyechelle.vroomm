import math

import pyechelle.telescope as telescope


def test_telescope():
    t = telescope.Telescope(1., 0.)
    assert t.get_area() == math.pi / 4.

    t = telescope.Telescope(10., 10.)
    assert t.get_area() == 0.
