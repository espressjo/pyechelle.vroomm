import math

import pyechelle.telescope as telescope


def test_telescope():
    t = telescope.Telescope(1.0, 0.0)
    assert t.area == math.pi / 4.0

    t = telescope.Telescope(10.0, 10.0)
    assert t.area == 0.0
