import os

# disable jit, because it is not
os.environ['NUMBA_DISABLE_JIT'] = '4'
from hypothesis import given, strategies as st, settings
from pyechelle.CCD import CCD
import numpy as np


@given(st.integers(min_value=1, max_value=1000),
       st.integers(min_value=1, max_value=1000),
       st.integers(min_value=0, max_value=1000),
       st.floats(min_value=0, max_value=10, allow_nan=False))
@settings(deadline=None)
def test_ccd(maxx, maxy, bias, readnoise):
    ccd = CCD(xmax=maxx, ymax=maxy)
    assert ccd.data.shape == (maxy, maxx)
    ccd.add_bias(bias)
    assert np.mean(ccd.data) == bias
    ccd.add_readnoise(readnoise)
    assert np.any(ccd.data <= ccd.maxval)
    assert np.any(ccd.data >= 0)
    ccd.data *= 0
    ccd.add_readnoise(3)
    assert np.any(ccd.data >= 0)
    ccd.data += ccd.maxval
    ccd.add_readnoise(3)
    ccd.clip()
    assert np.any(ccd.data <= ccd.maxval)


@given(
    st.floats(min_value=0, max_value=4096, allow_nan=False, exclude_max=True),
    st.floats(min_value=0, max_value=4096, allow_nan=False, exclude_max=True)
)
def test_binning(xval, yval):
    ccd = CCD()
    ccd.add_photons(np.array(xval), np.array(yval))
    assert ccd.data[int(yval), int(xval)] == 1
