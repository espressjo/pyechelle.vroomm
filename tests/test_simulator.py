from pyechelle import simulator, spectrograph
from pyechelle.sources import Phoenix
from pyechelle.telescope import Telescope


def test_setup_simulator():
    spec = spectrograph.ZEMAX(simulator.available_models[0])

    sim = simulator.Simulator(spec)
    sim.set_ccd(1)
    sim.set_fibers([2, 3, 4])
    sim.set_sources(Phoenix(t_eff=4200, log_g=4.0))
    sim.set_telescope(Telescope(d_primary=1.0, d_secondary=0.7))
    # sim.set_atmosphere(True)
    # sim.set_radial_velocities(50.45)
    sim.set_cuda(True)
    sim.set_exposure_time(0.01)
    sim.set_output(path='test.fits', overwrite=True)
    # sim.run()
