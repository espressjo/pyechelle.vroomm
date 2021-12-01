import inspect

import numpy as np

import pyechelle.sources as sources


# @given(
#     st.floats(min_value=0.0, exclude_min=True, allow_nan=False, allow_infinity=False, ),
#     st.floats(
#         min_value=0.0,
#         exclude_min=True,
#         max_value=1.0,
#         allow_nan=False,
#         allow_infinity=False,
#     ),
#     st.integers(),
# )
def test_sources():
    wl = np.linspace(300, 305, 1000, dtype=np.float)
    available_sources = [m[0] for m in inspect.getmembers(sources, inspect.isclass) if
                         issubclass(m[1], sources.Source) and m[0] != "Source"]
    assert len(available_sources) > 0
    for s in available_sources:
        source = getattr(sources, s)()
        sd = source.get_spectral_density(wl)
        print(sd)

    # s = sources.Constant()
    # e = eff.ConstantEfficiency("test", eff=efficiency)
    # assert e.get_efficiency(wl) == efficiency
    # assert e.get_efficiency_per_order(wl, order)
