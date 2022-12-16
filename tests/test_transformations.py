import numpy as np
from hypothesis import given, strategies as st
from optics import AffineTransformation, PSF


@given(
    st.floats(
        min_value=0.0,
        exclude_min=True,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    st.floats(
        min_value=0.0,
        exclude_min=True,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_transformation(wl1, wl2, x, y):
    at1 = AffineTransformation(0., 1., 1., 0., 0., 0., wl1)
    at2 = AffineTransformation(0., 1., 1., 0., 0., 0., wl2)
    at3 = AffineTransformation(0., 1., 1., 0., 0., 0., wl1)
    assert at1.as_matrix() == (0., 1., 1., 0., 0., 0.)
    # test identity
    rx, ry = at1 * (x, y)
    assert np.isclose(rx, x) and np.isclose(ry, y)

    # rotation by 90 deg
    at4 = AffineTransformation(np.pi / 2., 1., 1., 0., 0., 0., None)
    rx, ry = at4 * (x, y)
    assert np.isclose(rx, -y) and np.isclose(ry, x)

    assert isinstance(at1 < at2, bool)
    assert isinstance(at1 >= at2, bool)
    assert (at1 + at3).sx == 2.
    assert (at1 + at3).sy == 2.
    at1 += at3
    assert at1.sy == 2.
    at1 -= at3
    assert at1.sy == 1.
    assert (at1 - at3).sx == 0.


def test_psf():
    psf1 = PSF(0.3, np.random.random((50, 50)) + 1., 3.)
    psf2 = PSF(0.4, np.random.random((50, 50)) + 1., 3.)
    assert psf1 < psf2
    assert psf1 <= psf2
    assert np.all(psf1.data > 0)
    assert (len(psf1.__str__()) > 49 * 49)
