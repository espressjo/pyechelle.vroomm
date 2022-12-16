import math
import random
from hypothesis import given, strategies as st, settings
from pyechelle.randomgen import make_alias_sampling_arrays, unravel_index
import numpy as np


@settings(deadline=None)
@given(
    st.lists(st.floats(min_value=0.1, exclude_min=True, allow_nan=False, allow_infinity=False, max_value=1E50),
             min_size=1, max_size=1000),
)
def test_alias_sampling(probabilities):
    probabilities = np.asarray(probabilities / np.sum(probabilities), dtype=np.float32)
    q, j = make_alias_sampling_arrays(probabilities)
    assert len(q) == len(j)

    k = int(math.floor(random.random() * len(j)))
    if not random.random() < q[k]:
        k = j[k]
    assert probabilities[k] > 0


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
