from hypothesis import given, strategies as st

import pyechelle.efficiency as eff


@given(
    st.floats(min_value=0.0, exclude_min=True, allow_nan=False, allow_infinity=False, ),
    st.floats(
        min_value=0.0,
        exclude_min=True,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    st.integers(),
)
def test_constant(wl, efficiency, order):
    e = eff.ConstantEfficiency("test", eff=efficiency)
    assert e.get_efficiency(wl) == efficiency
    assert e.get_efficiency_per_order(wl, order)


@given(
    st.floats(min_value=1.0, max_value=90.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.0, max_value=90.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.01, max_value=3000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.text(),
    st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    st.integers(min_value=1, max_value=200),
)
def test_grating(alpha, beta, gpmm, peak_efficiency, name, wl, order):
    e = eff.GratingEfficiency(alpha, beta, gpmm, peak_efficiency, name)
    assert e.get_efficiency(wl) >= 0.0
    assert e.get_efficiency_per_order(wl, order) >= 0.0
