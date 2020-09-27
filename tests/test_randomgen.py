import os

# disable jit, because it is not
os.environ['NUMBA_DISABLE_JIT'] = '4'
from hypothesis import given, strategies as st, settings
from pyechelle.randomgen import AliasSample
import numpy as np


@settings(deadline=None)
@given(
    st.lists(st.floats(min_value=0.1, exclude_min=True, allow_nan=False, allow_infinity=False, max_value=1E50),
             min_size=1, max_size=1000),
)
def test_alias_sampling(probabilities):
    probabilities = np.asarray(probabilities / np.sum(probabilities), dtype=np.float32)
    sampler = AliasSample(probabilities)
    f = sampler.draw_one()
    assert f < len(probabilities)
    assert probabilities[f] > 0


@settings(deadline=None)
@given(
    st.integers(min_value=1, max_value=10000),
    st.lists(st.floats(min_value=0.1, exclude_min=True, allow_nan=False, allow_infinity=False, max_value=1E50),
             min_size=1,
             max_size=1000),
)
def test_alias_sample_n(number_of_samples, probabilities):
    sampler = AliasSample(probabilities)
    g = sampler.sample(number_of_samples)
    assert len(g) == number_of_samples
    assert g.dtype == np.int32
