import os
import pathlib

# disable jit, because it is not
os.environ['NUMBA_DISABLE_JIT'] = '4'

from pyechelle import simulator


def test_simulation(capsys, benchmark):
    benchmark.pedantic(simulator.main,
                       args=([["-s", "MaroonX", "--sources", "Constant", "-t", "0.01", "--orders", "100-102", "-o",
                               "test.fits"]]),
                       iterations=1, rounds=1)
    captured = capsys.readouterr()
    result = captured.out
    assert "Simulation took" in result
    # cleanup files
    pathlib.Path.cwd().joinpath('test.fits').unlink(missing_ok=True)
