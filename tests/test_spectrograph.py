import numpy as np

from pyechelle import simulator
from pyechelle.spectrograph import (
    LocalDisturber,
    SimpleSpectrograph,
    GlobalDisturber,
    ZEMAX,
    AtmosphericDispersion,
)


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
        assert np.min(np.ediff1d(tx)) > 0.0

def test_get_spot_positions():
    simple = SimpleSpectrograph()
    for o in simple.get_orders(1, 1):
        wl = np.linspace(*simple.get_wavelength_range(o, 1, 1), 10)
        spot_positions = simple.get_spot_positions(wl, o, 1, 1)
        # check that the result is a tuple with two arrays each 10 entries long
        assert isinstance(spot_positions, tuple), "Spot positions should be a tuple"
        assert len(spot_positions) == 2, "Spot positions should contain two arrays"
        assert spot_positions[0].shape == (10,), "X positions should have shape (N,)"
        assert spot_positions[1].shape == (10,), "Y positions should have shape (N,)"


    # also test for single wavelength
    o = simple.get_orders(1, 1)[0]
    wl_single = simple.get_wavelength_range(o, 1, 1)[0]
    spot_positions_single = simple.get_spot_positions(wl_single, o, 1, 1)
    assert isinstance(spot_positions_single, tuple), "Spot positions should be a tuple"
    assert len(spot_positions_single) == 2, "Spot positions should contain two arrays"
    # should be two float values
    assert np.isscalar(spot_positions_single[0]), "X position should be a scalar"
    assert np.isscalar(spot_positions_single[1]), "Y position should be a scalar"

def test_comparing_spectrographs():
    simple1 = SimpleSpectrograph()
    simple2 = SimpleSpectrograph()
    assert simple1 == simple2, "Two SimpleSpectrograph instances should be equal"

def test_psf_maps():
    simple = SimpleSpectrograph()
    order = simple.get_orders(1, 1)[0]
    psf_maps = simple.get_psf(None, order, 1, 1)
    # compare to single psf
    wl_start = simple.get_wavelength_range(order, 1, 1)[0]
    psf_single = simple.get_psf(wl_start, order, 1, 1)
    assert np.all(psf_maps[0].data == psf_single.data), "PSF maps should match single PSF for the first wavelength"


def test_LocaDisturber():
    spec = ZEMAX(simulator.available_models[0])
    o = spec.get_orders()[0]
    wl = sum(spec.get_wavelength_range(o, 1, 1)) / 2.0
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
    wl = sum(spec.get_wavelength_range(o, 1, 1)) / 2.0
    aff1 = spec.get_transformation(wl, o, 1, 1)

    # test tx
    disturber = GlobalDisturber(spec, tx=0.1)
    aff2 = disturber.get_transformation(wl, o, 1, 1)
    assert np.isclose(aff1.tx, aff2.tx - 0.1)

    # test ty
    disturber = GlobalDisturber(spec, ty=0.1)
    aff2 = disturber.get_transformation(wl, o, 1, 1)
    assert np.isclose(aff1.ty, aff2.ty - 0.1)


# def test_zemax_models():
#     """
#     This test looks whether the .HDF file based models make sense.
#     In particular, it is looked for jumps in the 'rotation' and 'shear' parameter. This can happen, since those
#     parameters are not uniquely defined and can have +- 2*pi jumps from one wavelength to the next.
#     Because pyechelle interpolates between affine transformation values, it is required that the model is fixed by
#     removing those jumps before using the model for simulations.
#     """
#
#     for s in simulator.available_models:
#         spec = ZEMAX(s)
#         for ccd in spec.get_ccd():
#             for f in spec.get_fibers(ccd):
#                 for o in spec.get_orders(f, ccd):
#                     shear = [af.shear for af in spec.transformations(o, f, ccd)]
#                     assert max(abs(np.ediff1d(shear))) < 1, f'There is a jump in the shear parameter:' \
#                                                             f'model file: {s}. ' \
#                                                             f'CCD index: {ccd} ' \
#                                                             f'fiber index: {f} ' \
#                                                             f'order: {o}'
#
#                     rot = [af.rot for af in spec.transformations(o, f, ccd)]
#                     assert max(abs(np.ediff1d(rot))) < 1, f'There is a jump in the shear parameter:' \
#                                                           f'model file: {s}. ' \
#                                                           f'CCD index: {ccd} ' \
#                                                           f'fiber index: {f} ' \
#                                                           f'order: {o}'


def test_atmospheric_dispersion():
    spec = AtmosphericDispersion(30)
    for ccd in spec.get_ccd():
        for f in spec.get_fibers(ccd):
            for o in spec.get_orders(f, ccd):
                assert np.any(spec.get_psf(0.5, o, f, ccd).data > 0), (
                    "the returned PSF contains negative values"
                )
                assert spec.get_field_shape(f, ccd) == "singlemode", (
                    "the field shape should be singlemode"
                )
                minwl, maxwl = spec.get_wavelength_range(o, f, ccd)
                assert spec.get_transformation(minwl, o, f, ccd).ty > 0
                assert spec.get_transformation(maxwl, o, f, ccd).ty > 0
