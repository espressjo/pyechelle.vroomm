""" Atmospheric dispersion
==========================

This script shows how to simulate atmospheric dispersion using the
:class:`~pyechelle.spectrograph.AtmosphericDispersion` model.

We simulate atmospheric dispersion for an object at 50° zenith distance, with a reference wavelength of 350nm
through a bandpass filter between 300nm and 400nm.
"""

if __name__ == "__main__":
    from pyechelle.simulator import Simulator
    from pyechelle.sources import ConstantPhotonFlux
    from pyechelle.spectrograph import AtmosphericDispersion
    from pyechelle.efficiency import BandpassFilter
    import numpy as np

    sim = Simulator(AtmosphericDispersion(zd=50, reference_wavelength=0.35))
    sim.set_ccd(1)
    sim.set_fibers(1)
    sim.set_sources(ConstantPhotonFlux(100))
    sim.set_exposure_time(1)
    # We reduce the resolution of the spectrum retrieved by skycalc to 100
    sim.set_atmospheres(True, sky_calc_kwargs={'airmass': 1. / (np.deg2rad(50)), 'wres': 100})
    sim.set_efficiency(BandpassFilter(0.3, 0.4))
    sim.set_output('06_atmospheric_dispersion.fits', overwrite=True)

    sim.run()

    from pyechelle.simulator import export_to_html

    export_to_html(sim.spectrograph.get_ccd(1).data,
                   f'docs/source/_static/plots/example_results/{__file__.split("/")[-1][:-3]}.html', False, 1000, 300,
                   None, None)
