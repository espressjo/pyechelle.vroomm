import hypothesis
import numpy as np
from hypothesis import given, strategies as st

import pyechelle.sources as sources
from pyechelle.simulator import available_sources


@given(
    st.floats(min_value=0.3, max_value=1.5, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0001, max_value=0.005, allow_nan=False, allow_infinity=False),
    st.sampled_from(available_sources)
)
@hypothesis.settings(deadline=None)
def test_sources(wl, bw, source_name):
    wavelength = np.linspace(wl, wl + bw, 1000, dtype=np.float)
    print(f"Test {source_name}...")
    source = getattr(sources, source_name)()
    sd = source.get_spectral_density(wavelength)
    assert isinstance(sd, tuple) or isinstance(sd, np.ndarray)


@given(
    st.floats(min_value=0.1, max_value=50000., allow_nan=False, allow_infinity=False)
)
def test_rv_shift(rv):
    wl = np.linspace(0.5, 0.5005, 1000)
    c = 299792458.

    et = sources.Etalon()
    sd0 = et.get_spectral_density_rv(wl, 0.)
    sd1 = et.get_spectral_density_rv(wl, rv)
    print(rv)
    assert sd0[0][0] > sd1[0][0]
    # TODO: fix this test. Right now the problem is that some edge lines are in sd0 which are not in sd1
    # if len(sd0[0]) == len(sd1[0]):
    #    assert np.allclose(sd0[0], sd1[0] * ((c+rv)/c))

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
