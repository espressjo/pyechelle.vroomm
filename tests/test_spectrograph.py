import numpy as np

from pyechelle import simulator, spectrograph
from spectrograph import LocalDisturber


def test_zemax_spectrograph():
    # get spectrograph if not available
    spec = spectrograph.ZEMAX(simulator.available_models[0])
    for o in spec.get_orders():
        min_wl, max_wl = spec.get_wavelength_range(order=o)
        assert min_wl < max_wl


def test_distruber():
    spec = spectrograph.ZEMAX(simulator.available_models[0])
    distruber = LocalDisturber(spec, 0.1)
    o = spec.get_orders()[0]
    wl = sum(spec.get_wavelength_range(o, 1, 1)) / 2.
    aff1 = spec.get_transformation(wl, o, 1, 1)
    aff2 = distruber.get_transformation(wl, o, 1, 1)
    assert np.isclose(aff1.tx, aff2.tx - 0.1)
