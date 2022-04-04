import numpy as np

from pyechelle import simulator
from pyechelle.spectrograph import LocalDisturber, SimpleSpectrograph, GlobalDisturber, ZEMAX


def test_zemax_spectrograph():
    # get spectrograph if not available
    spec = ZEMAX(simulator.available_models[0])
    for o in spec.get_orders():
        min_wl, max_wl = spec.get_wavelength_range(order=o)
        assert min_wl < max_wl


def test_simple_spectrograph():
    simple = SimpleSpectrograph()
    for o in simple.get_orders(1, 1):
        wl = np.linspace(*simple.get_wavelength_range(o, 1, 1), 100)
        transformations = simple.get_transformation(wl, o, 1, 1)
        tx = transformations[4]
        assert np.min(np.ediff1d(tx)) > 0.


def test_LocaDisturber():
    spec = ZEMAX(simulator.available_models[0])
    o = spec.get_orders()[0]
    wl = sum(spec.get_wavelength_range(o, 1, 1)) / 2.
    aff1 = spec.get_transformation(wl, o, 1, 1)

    # test tx
    disturber = LocalDisturber(spec, d_tx=0.1)
    aff2 = disturber.get_transformation(wl, o, 1, 1)
    assert np.isclose(aff1.tx, aff2.tx - 0.1)

    # test ty
    disturber = LocalDisturber(spec, d_ty=0.1)
    aff2 = disturber.get_transformation(wl, o, 1, 1)
    assert np.isclose(aff1.ty, aff2.ty - 0.1)


def test_GlobalDisturber():
    spec = ZEMAX(simulator.available_models[0])
    o = spec.get_orders()[0]
    wl = sum(spec.get_wavelength_range(o, 1, 1)) / 2.
    aff1 = spec.get_transformation(wl, o, 1, 1)

    # test tx
    disturber = GlobalDisturber(spec, tx=0.1)
    aff2 = disturber.get_transformation(wl, o, 1, 1)
    assert np.isclose(aff1.tx, aff2.tx - 0.1)

    # test ty
    disturber = GlobalDisturber(spec, ty=0.1)
    aff2 = disturber.get_transformation(wl, o, 1, 1)
    assert np.isclose(aff1.ty, aff2.ty - 0.1)


def test_zemax_models():
    """
    This test looks whether the .HDF file based models make sense.
    In particular, it is looked for jumps in the 'rotation' and 'shear' parameter. This can happen, since those
    parameters are not uniquely defined and can have +- 2*pi jumps from one wavelength to the next.
    Because pyechelle interpolates between affine transformation values, it is required that the model is fixed by
    removing those jumps before using the model for simulations.
    """

    for s in simulator.available_models:
        spec = ZEMAX(s)
        for ccd in spec.get_ccd():
            for f in spec.get_fibers(ccd):
                for o in spec.get_orders(f, ccd):
                    shear = [af.shear for af in spec.transformations(o, f, ccd)]
                    assert max(abs(np.ediff1d(shear))) < 1, f'There is a jump in the shear parameter:' \
                                                            f'model file: {s}. ' \
                                                            f'CCD index: {ccd} ' \
                                                            f'fiber index: {f} ' \
                                                            f'order: {o}'

                    rot = [af.rot for af in spec.transformations(o, f, ccd)]
                    assert max(abs(np.ediff1d(rot))) < 1, f'There is a jump in the shear parameter:' \
                                                          f'model file: {s}. ' \
                                                          f'CCD index: {ccd} ' \
                                                          f'fiber index: {f} ' \
                                                          f'order: {o}'
