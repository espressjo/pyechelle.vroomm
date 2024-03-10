import pathlib

import numpy as np

from pyechelle.hdfbuilder import HDFBuilder
from pyechelle.spectrograph import SimpleSpectrograph, ZEMAX


def test_builder_simple_spectrograph():
    spec = SimpleSpectrograph()
    builder = HDFBuilder(spec, "test.hdf")
    builder.save_to_hdf(50, 15)

    spec_hdf = ZEMAX("test.hdf")
    for ccd, _ in spec.get_ccd().items():
        for f in spec.get_fibers(ccd):
            for o in spec.get_orders(f, ccd):
                wlr = spec_hdf.get_wavelength_range(o, f, ccd)
                wlr_orig = spec.get_wavelength_range(o, f, ccd)
                assert np.isclose(wlr[0], wlr_orig[0])
                assert np.isclose(wlr[1], wlr_orig[1])
                for wl in np.linspace(wlr[0], wlr[1]):
                    assert spec.get_transformation(
                        wl, o, f, ccd
                    ) == spec_hdf.get_transformation(wl, o, f, ccd)
                    assert np.isclose(
                        spec.get_psf(wl, o, f, ccd).data,
                        spec_hdf.get_psf(wl, o, f, ccd).data,
                        atol=1e-4,
                    ).all()

    pathlib.Path.cwd().joinpath("test.hdf").unlink(missing_ok=True)

#
# def test_builder_full_simulation():
#     TODO: once random generator seed is implemented we should compare entire simulated spectra to
#      make sure they are identical
#     spec = SimpleSpectrograph()
#     builder = HDFBuilder(spec, "test.hdf")
#     builder.save_to_hdf(50, 15)
#
#     spec_hdf = ZEMAX("test.hdf")
#
#     sim_spec = Simulator(spec)
#     sim_hdf = Simulator(spec_hdf)
#
#     for i, s in enumerate([sim_spec, sim_hdf]):
#         s.set_ccd(1)
#         s.set_fibers(1)
#         s.set_sources(Etalon(d=0.5))
#         s.set_output(f"output_{i}.fits")
#         s.run()
#
#
