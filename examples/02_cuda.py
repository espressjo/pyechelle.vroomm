""" CUDA example
================

This script shows how to activate CUDA support and use a specific random seed for the random generator.
Normally, you don't want to fix the random seed, but for testing it can be helpful to have reproducible
results.
"""

if __name__ == "__main__":
    from pyechelle.simulator import Simulator
    from pyechelle.sources import Constant
    from pyechelle.spectrograph import ZEMAX

    sim = Simulator(ZEMAX("MaroonX"))
    sim.set_ccd(1)
    sim.set_fibers(1)
    sim.set_sources(Constant())
    sim.set_exposure_time(1.)
    # Enable cuda and set a specific random seed.
    sim.set_cuda(True, 42)
    sim.set_output('02_cuda.fits', overwrite=True)
    sim.run()

    from pyechelle.simulator import export_to_html

    export_to_html(sim.spectrograph.get_ccd(1).data,
                   f'docs/source/_static/plots/example_results/{__file__.split("/")[-1][:-3]}.html')
