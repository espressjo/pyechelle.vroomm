""" Atmosphere and RVs
======================

This script shows how to include atmospheric transmission for the stellar targets and also adjust their
radial velocity."""

if __name__ == "__main__":
    from pyechelle.simulator import Simulator
    from pyechelle.sources import IdealEtalon, Phoenix
    from pyechelle.spectrograph import ZEMAX
    from pyechelle.telescope import Telescope

    sim = Simulator(ZEMAX("MaroonX"))
    sim.set_fibers([1, 2, 3, 4])
    # set telescope size to match Gemini observatory
    sim.set_telescope(Telescope(8.1, 0.8))
    sim.set_sources([IdealEtalon(d=10, n_photons=1E5),
                     Phoenix(t_eff=4000, log_g=4.0),
                     Phoenix(t_eff=4000, log_g=4.0),
                     Phoenix(t_eff=4000, log_g=4.0)])
    # activate atmospheric transmission for the three stellar targets and set airmass to 1.4
    sim.set_atmospheres([False, True, True, True], sky_calc_kwargs={'airmass': 1.4})
    # set radial velocity of stellar target to 42 m/s
    sim.set_radial_velocities([0., 42., 42., 42.])
    sim.set_exposure_time(1.)
    # Enable cuda
    sim.set_cuda(True)
    sim.set_output('03_atmosphere_and_rvs.fits', overwrite=True)
    sim.run()

    from pyechelle.simulator import export_to_html

    export_to_html(sim.spectrograph.get_ccd(1).data,
                   f'docs/source/_static/plots/example_results/{__file__.split("/")[-1][:-3]}.html')
