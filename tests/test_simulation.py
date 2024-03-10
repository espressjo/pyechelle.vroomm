import pathlib

from astropy.io import fits

from pyechelle import simulator


def test_simulation(capsys, benchmark):
    benchmark.pedantic(simulator.main,
                       args=([["-s", "MaroonX", "--sources", "ConstantFlux", "-t", "0.01", "--orders", "100-102", "-o",
                               "test.fits", "--overwrite"]]),
                       iterations=1, rounds=1)
    captured = capsys.readouterr()
    result = captured.out
    assert "Simulation took" in result
    # check that metadata has been written to fits file
    assert 'pyechelle.' in ''.join([h for h in fits.getheader('test.fits')])
    # cleanup files
    pathlib.Path.cwd().joinpath('test.fits').unlink(missing_ok=True)


def test_simulation_multicore(capsys, benchmark):
    benchmark.pedantic(simulator.main,
                       args=([
                           ["-s", "MaroonX", "--sources", "ConstantFlux", "-t", "0.01", "--orders", "100-102",
                            "--max_cpu",
                            "3", "--overwrite"]]),
                       iterations=1, rounds=1)
    captured = capsys.readouterr()
    result = captured.out
    assert "Simulation took" in result
    # cleanup files
    pathlib.Path.cwd().joinpath('test.fits').unlink(missing_ok=True)


def test_benchmark():
    # import os
    # os.environ['NUMBA_DISABLE_JIT'] = '0'
    from pyechelle import benchmark

    benchmark.run_benchmark_cpu([1], [0.00001])
    # benchmark.run_benchmark_cuda([0.01])
    pathlib.Path.cwd().joinpath('test.fits').unlink(missing_ok=True)
