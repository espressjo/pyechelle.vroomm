import pathlib

from pyechelle import simulator, spectrograph
from pyechelle.sources import Phoenix, ConstantPhotons
from pyechelle.telescope import Telescope


def test_setup_simulator():
    spec = spectrograph.ZEMAX(simulator.available_models[0])
    sim = simulator.Simulator(spec)
    sim.set_ccd(1)
    sim.set_fibers([2])
    sim.set_orders([spec.get_orders(2)[0]])
    sim.set_sources(Phoenix(t_eff=4200, log_g=4.0))
    sim.set_telescope(Telescope(d_primary=1.0, d_secondary=0.7))
    sim.set_cuda(True)
    sim.set_radial_velocities(10.)
    sim.set_atmospheres(True, {'airmass': 2.0})
    sim.set_bias(500)
    sim.set_read_noise(10.)
    sim.set_exposure_time(0.01)
    sim.set_output(path='test.fits', overwrite=True)
    sim.validate()
    # sim.run()


def test_AtmosphericDispersion():
    spec = spectrograph.AtmosphericDispersion(30.)
    sim = simulator.Simulator(spec)
    sim.set_ccd(1)
    sim.set_fibers(1)
    sim.set_sources(ConstantPhotons(0.01))
    sim.set_exposure_time(0.01)
    sim.set_output(path='test.fits', overwrite=True)
    sim.validate()
    sim.run()

    # cleanup files
    pathlib.Path.cwd().joinpath('test.fits').unlink(missing_ok=True)

# def test_AtmosphericDispersion_CUDA():
#     with MonkeyPatch.context() as m:
# TODO: CUDASIM causes problems with random number generation. seems like a numba bug.
#         m.setenv('NUMBA_ENABLE_CUDASIM', 'True')
#         importlib.reload(pyechelle.simulator)
#         importlib.reload(pyechelle.spectrograph)
#
#         spec = spectrograph.AtmosphericDispersion(30.)
#         sim = simulator.Simulator(spec)
#         sim.set_ccd(1)
#         sim.set_fibers(1)
#         sim.set_sources(ConstantPhotons(0.01))
#         sim.set_exposure_time(0.01)
#         sim.set_cuda(True)
#         sim.set_output(path='test.fits', overwrite=True)
#         sim.validate()
#
#         sim.run()
#
#         # cleanup files
#         pathlib.Path.cwd().joinpath('test.fits').unlink(missing_ok=True)
