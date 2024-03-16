""" CUDA example
================

By default, PyEchelle uses CUDA if available. You can however also explicitly ask PyEchelle to use CUDA.
In case there is no CUDA device available, this will raise an Exception. You can also use the function
to set a specific random seed for the random generator.
Normally, you don't want to fix the random seed, but for testing it can be helpful to have reproducible
results.
"""

if __name__ == "__main__":
    from pyechelle.simulator import Simulator
    from pyechelle.sources import ConstantFlux
    from pyechelle.spectrograph import ZEMAX

    sim = Simulator(ZEMAX("MaroonX"))
    sim.set_ccd(1)
    sim.set_fibers(1)
    sim.set_sources(ConstantFlux())
    sim.set_exposure_time(1.)
    # Enable cuda and set a specific random seed.
    sim.set_cuda(True, 42)  # raises an Exception if no CUDA device is available
    sim.set_output('02_cuda.fits', overwrite=True)
    sim.run()

    from pyechelle.simulator import export_to_html

    export_to_html(sim.spectrograph.get_ccd(1).data,
                   f'docs/source/_static/plots/example_results/{__file__.split("/")[-1][:-3]}.html')
