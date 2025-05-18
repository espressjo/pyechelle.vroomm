"""Minimal example
====================

This script shows a minimal example for using pyechelle in python.
It simulates a spectrum with constant spectral density for the MaroonX
spectrograph."""

if __name__ == "__main__":
    from pyechelle.simulator import Simulator
    from pyechelle.sources import ConstantPhotonFlux
    from pyechelle.spectrograph import ZEMAX

    sim = Simulator(ZEMAX("MaroonX"))
    sim.set_fibers(1)
    sim.set_sources(ConstantPhotonFlux())
    sim.set_exposure_time(10.0)
    sim.set_output("00_minimal_example.fits", overwrite=True)

    sim.run()

    from pyechelle.simulator import export_to_html

    export_to_html(
        sim.spectrograph.get_ccd(1).data,
        f"docs/source/_static/plots/example_results/{__file__.split('/')[-1][:-3]}.html",
    )
