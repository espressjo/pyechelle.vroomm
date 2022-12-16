from hypothesis import given, strategies as st, settings
from pyechelle.CCD import CCD
import numpy as np


@given(st.integers(min_value=1, max_value=1000),
       st.integers(min_value=1, max_value=1000),
       st.integers(min_value=0, max_value=1000),
       st.floats(min_value=0, max_value=10, allow_nan=False))
@settings(deadline=None)
def test_ccd(maxx, maxy, bias, readnoise):
    ccd = CCD(n_pix_x=maxx, n_pix_y=maxy)
    assert ccd.data.shape == (maxy, maxx)
    ccd.add_bias(bias)
    assert np.mean(ccd.data) == bias
    ccd.data *= 0
    ccd.add_readnoise(readnoise)
    ccd.clip()
    assert np.all(ccd.data >= 0)
    ccd.data += ccd.maxval
    ccd.add_readnoise(3)
    ccd.clip()
    assert np.all(ccd.data <= ccd.maxval)
