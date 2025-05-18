import numpy as np
from hypothesis import given, strategies as st, settings

from pyechelle.CCD import CCD


@given(
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=0, max_value=1000),
    st.floats(min_value=0, max_value=10, allow_nan=False),
)
@settings(deadline=None)
def test_ccd(max_x, max_y, bias, read_noise):
    ccd = CCD(n_pix_x=max_x, n_pix_y=max_y)
    assert ccd.data.shape == (max_y, max_x)
    ccd.add_bias(bias)
    assert np.mean(ccd.data) == bias
    ccd.data *= 0
    ccd.add_readnoise(read_noise)
    ccd.clip()
    assert np.all(ccd.data >= 0)
    ccd.data += ccd.maxval
    ccd.add_readnoise(3)
    ccd.clip()
    assert np.all(ccd.data <= ccd.maxval)
