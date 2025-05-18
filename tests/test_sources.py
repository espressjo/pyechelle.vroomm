import pathlib
import time

import astropy.units as u
from astropy.io import fits
import hypothesis
import numpy as np
from hypothesis import given, strategies as st

import pyechelle.sources as sources
from pyechelle.simulator import available_sources
from pyechelle.spectrograph import check_url_exists
from synphot import SourceSpectrum, units
from synphot.models import Empirical1D, GaussianFlux1D


@given(
    st.floats(min_value=0.3, max_value=1.5, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0001, max_value=0.005, allow_nan=False, allow_infinity=False),
)
@hypothesis.settings(deadline=None, max_examples=5)
def test_sources(wl, bw):
    for source_name in available_sources:
        wavelength = np.linspace(wl, wl + bw, 1000, dtype=float)
        if not (
            source_name == "CSVSource"
            or source_name == "ArcLamp"
            or source_name == "Phoenix"
            or source_name == "SynphotSource"
        ):
            t1 = time.time()
            print(f"Test {source_name}... ")
            source = getattr(sources, source_name)()
            sd = source.get_counts(wavelength, 1 * u.s)
            assert isinstance(sd, tuple) or isinstance(sd, np.ndarray)
            print(f"Test {source_name} took {time.time() - t1:.2f} s")


def test_arclamp():
    source = sources.ArcLamp()
    sd = source.get_counts(np.linspace(0.5, 0.7, 1000) * u.micron, 1 * u.s)
    assert isinstance(sd, np.ndarray) or isinstance(sd, tuple)
    assert sd[0].unit == u.micron
    assert sd[1].unit == u.ph


def test_phoenix():
    source = sources.Phoenix()
    sd = source.get_counts(np.linspace(0.5, 0.7, 1000) * u.micron, 1 * u.s)
    assert isinstance(sd, np.ndarray) or isinstance(sd, tuple)
    assert sd[0].unit == u.micron
    assert sd[1].unit == u.ph


def test_resolved_etalon():
    # test default values
    source = sources.ResolvedEtalon()
    sd = source.get_counts(np.linspace(0.5, 0.7, 1000) * u.micron, 1 * u.s)
    assert isinstance(sd, np.ndarray) or isinstance(sd, tuple)
    assert sd[0].unit == u.micron
    assert sd[1].unit == u.ph

    # test creation where the reflectivity of one surface is 95%
    source = sources.ResolvedEtalon(
        d=10.0 * u.mm, theta=0.5 * u.deg, R1=0.95, name="Reflectivity 95%"
    )
    sd = source.get_counts(np.linspace(0.5, 0.7, 1000) * u.micron, 1 * u.s)
    assert isinstance(sd, np.ndarray) or isinstance(sd, tuple)
    assert sd[0].unit == u.micron
    assert sd[1].unit == u.ph

    # test creation where the reflectivity of one surface is 95% and the other 90%
    source = sources.ResolvedEtalon(
        d=10.0 * u.mm,
        theta=0.5 * u.deg,
        R1=0.95,
        R2=0.9,
        name="Reflectivity 95% and 90%",
    )
    sd = source.get_counts(np.linspace(0.5, 0.7, 1000) * u.micron, 1 * u.s)
    assert isinstance(sd, np.ndarray) or isinstance(sd, tuple)
    assert sd[0].unit == u.micron
    assert sd[1].unit == u.ph

    # test where finesse is given
    source = sources.ResolvedEtalon(
        d=10.0 * u.mm, theta=0.5 * u.deg, finesse=100, name="Finesse 100"
    )
    sd = source.get_counts(np.linspace(0.5, 0.7, 1000) * u.micron, 1 * u.s)
    assert isinstance(sd, np.ndarray) or isinstance(sd, tuple)
    assert sd[0].unit == u.micron
    assert sd[1].unit == u.ph


def test_csv_source():
    # test a continuous source
    source = sources.CSVSource(
        pathlib.Path(__file__).parent.joinpath("test_data/test_eff.csv"),
        wavelength_units="micron",
        flux_units=u.uW / u.AA,
        list_like=False,
    )
    assert source.data["wavelength"].values.unit == u.micron
    sd = source.get_counts(np.linspace(0.5, 0.501, 1000) * u.micron, 1 * u.s)
    assert isinstance(sd, np.ndarray) or isinstance(sd, tuple)
    assert sd[0].unit == u.micron
    assert sd[1].unit == u.ph

    # test a line list
    source = sources.CSVSource(
        pathlib.Path(__file__).parent.joinpath("test_data/test_source_listlike.csv"),
        flux_units=u.ph / u.s,
        wavelength_units="nm",
        list_like=True,
    )
    sd = source.get_counts(np.linspace(0.5, 0.6, 1000) * u.micron, 1 * u.s)
    assert isinstance(sd, np.ndarray) or isinstance(sd, tuple)
    assert sd[0].unit == u.micron
    assert sd[1].unit == u.ph

    # test a continuous source where the flux is given in u.ph/u.s
    source = sources.CSVSource(
        pathlib.Path(__file__).parent.joinpath("test_data/test_eff.csv"),
        wavelength_units="micron",
        flux_units=u.ph / u.s,
        list_like=False,
    )
    assert source.data["wavelength"].values.unit == u.micron
    sd = source.get_counts(np.linspace(0.5, 0.7, 1000) * u.micron, 1 * u.s)
    assert isinstance(sd, np.ndarray) or isinstance(sd, tuple)
    assert sd[0].unit == u.micron
    assert sd[1].unit == u.ph

    # test a continuous source where the flux is given in u.ph/u.s/AA
    source = sources.CSVSource(
        pathlib.Path(__file__).parent.joinpath("test_data/test_eff.csv"),
        wavelength_units="micron",
        flux_units=u.ph / u.s / u.AA,
        list_like=False,
    )
    assert source.data["wavelength"].values.unit == u.micron
    sd = source.get_counts(np.linspace(0.5, 0.7, 1000) * u.micron, 1)
    assert isinstance(sd, np.ndarray) or isinstance(sd, tuple)
    assert sd[0].unit == u.micron
    assert sd[1].unit == u.ph


def test_synphot_source():
    wave = [1000, 2000, 3000, 4000, 5000]  # Angstrom
    flux = [1e-15, -2.3e-16, 1.8e-15, 4.5e-15, 9e-16] * units.FLAM
    sp = SourceSpectrum(Empirical1D, points=wave, lookup_table=flux, keep_neg=False)
    source = sources.SynphotSource(sp)
    sd = source.get_counts(np.linspace(0.3, 0.4, 1000) * u.micron, 1000 * u.s)
    assert isinstance(sd, np.ndarray) or isinstance(sd, tuple)
    assert sd[0].unit == u.micron
    assert sd[1].unit == u.ph


def test_source_str():
    # generate temporary fits file
    hdu = fits.PrimaryHDU()

    for i, source_name in enumerate(available_sources):
        if source_name == "CSVSource":
            source = getattr(sources, source_name)(
                pathlib.Path(__file__).parent.joinpath("test_data/test_eff.csv"),
                wavelength_units="micron",
                flux_units=u.uW / u.AA,
                list_like=False,
            )
        elif source_name == "SynphotSource":
            g_em = SourceSpectrum(
                GaussianFlux1D,
                total_flux=3.5e-13 * u.erg / (u.cm**2 * u.s),
                mean=3000,
                fwhm=100,
            )
            source = sources.SynphotSource(g_em)
        else:
            source = getattr(sources, source_name)()

        assert len(str(source)) > 0
        # also try to write as keyword into a fits header
        hdu.header.set(f"SOURCE{i}", str(source))

    hdul = fits.HDUList([hdu])
    hdul.writeto("test.fits", overwrite=True)


#
#
# @given(
#     st.floats(min_value=0.1, max_value=50000., allow_nan=False, allow_infinity=False)
# )
# def test_rv_shift(rv):
#     wl = np.linspace(0.5, 0.5005, 1000)
#     c = 299792458.
#
#     et = sources.Etalon()
#     sd0 = et.get_spectral_density_rv(wl, 0.)
#     sd1 = et.get_spectral_density_rv(wl, rv)
#     assert sd0[0][0] > sd1[0][0]
#     # TODO: fix this test. Right now the problem is that some edge lines are in sd0 which are not in sd1
#     # if len(sd0[0]) == len(sd1[0]):
#     #    assert np.allclose(sd0[0], sd1[0] * ((c+rv)/c))


# test LineList using different units for the wavelength
def test_LineList():
    # test with nm and single value for the intensity
    ll = sources.LineList([500, 501, 502] * u.nm, 1)
    wls, intensities = ll.get_counts([400, 600] * u.nm, 10)
    assert np.allclose(wls, [0.500, 0.501, 0.502] * u.micron)
    assert np.allclose(intensities, [10, 10, 10] * u.ph)

    # test with micron and array for the intensities
    ll = sources.LineList([0.5, 0.501, 0.502] * u.micron, [1, 2, 3])
    wls, intensities = ll.get_counts([0.4, 0.6] * u.micron, 1 * u.min)
    assert np.allclose(wls, [0.5, 0.501, 0.502] * u.micron)
    assert np.allclose(intensities, [60, 120, 180] * u.ph)

    # test with default units without specifying them
    ll = sources.LineList([0.5, 0.501, 0.502], [1, 2, 3])
    wls, intensities = ll.get_counts([0.4, 0.6], 1)
    assert np.allclose(wls, [0.5, 0.501, 0.502] * u.micron)
    assert np.allclose(intensities, [1, 2, 3] * u.ph)

    # only specify intensities units
    ll = sources.LineList([0.5, 0.501, 0.502], 60 * u.ph / u.min)
    wls, intensities = ll.get_counts([0.4, 0.6], 1)
    assert np.allclose(wls, [0.5, 0.501, 0.502] * u.micron)
    assert np.allclose(intensities, [1, 1, 1] * u.ph)


def test_phoenix_base_url():
    assert check_url_exists(sources.Phoenix().get_wavelength_url())


# @given(
#     st.sampled_from(sources.Phoenix.valid_t),
#     st.sampled_from(sources.Phoenix.valid_a),
#     st.sampled_from(sources.Phoenix.valid_g),
#     st.sampled_from(sources.Phoenix.valid_z),
# )
# @hypothesis.settings(deadline=1000)
# TODO: this test needs fixing. The get_spectrum_url() seems correct, but the PHOENIX grid might not be as complete as
# described in the paper
# def test_phoenix(t, a, g, z):
#     if np.isclose(a, 0.) and z < 4:
#         assert check_url_exists(sources.Phoenix.get_spectrum_url(t, a, g, z))
#     elif 3500. <= t <= 8000. and -3. < z <= 0. and g > 0.:
#         assert check_url_exists(sources.Phoenix.get_spectrum_url(t, a, g, z))
#     else:
#         print(f"Skip {t}, {a}, {g}, {z}")
