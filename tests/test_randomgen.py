import os

# disable jit, because it is not
os.environ['NUMBA_DISABLE_JIT'] = '4'
from hypothesis import given, strategies as st, settings
from pyechelle.randomgen import AliasSample, generate_slit_polygon, generate_slit_xy, generate_slit_round, unravel_index
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


@settings(deadline=None)
@given(
    st.integers(min_value=1, max_value=10000),
)
def test_slit_round(number_of_samples):
    sx, sy = generate_slit_round(number_of_samples)
    assert len(sx) == number_of_samples
    assert len(sy) == number_of_samples

    assert np.min(sx) >= 0.
    assert np.min(sy) >= 0.

    assert np.max(sx) <= 1.
    assert np.max(sy) <= 1.


@settings(deadline=None)
@given(
    st.integers(min_value=1, max_value=10000),
)
def test_slit_xy(number_of_samples):
    sx, sy = generate_slit_xy(number_of_samples)
    assert len(sx) == number_of_samples
    assert len(sy) == number_of_samples

    assert np.min(sx) >= 0.
    assert np.min(sy) >= 0.

    assert np.max(sx) <= 1.
    assert np.max(sy) <= 1.


@settings(deadline=None)
@given(
    st.integers(min_value=3, max_value=12),
    st.integers(min_value=1, max_value=10000),
    st.floats(min_value=0., max_value=360.),
)
def test_slit_polygon(n_poly, number_of_samples, phi):
    sx, sy = generate_slit_polygon(n_poly, number_of_samples, phi)
    assert len(sx) == number_of_samples
    assert len(sy) == number_of_samples

    assert np.min(sx) >= 0.
    assert np.min(sy) >= 0.

    assert np.max(sx) <= 1.
    assert np.max(sy) <= 1.


@settings(deadline=None)
@given(
    st.lists(st.integers(min_value=1, max_value=10000),
             min_size=1, max_size=100),
    st.lists(st.integers(min_value=1, max_value=1000),
             min_size=2, max_size=3),

)
def test_unravel_index(indices, shape):
    indices = np.array(indices)
    try:
        numpy_answer = np.array(np.unravel_index(indices, shape))
    except:
        numpy_answer = None

    if isinstance(numpy_answer, np.ndarray):
        randomgen_answer = np.array(unravel_index(indices, shape))
        assert (numpy_answer == randomgen_answer).all()
