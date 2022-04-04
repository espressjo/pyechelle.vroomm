"""Disturbing the spectrograph model globally
=============================================

This script shows how to use the GlobalDisturber class on top of a 'regular' spectrograph model.
It allows applying global distortions to the entire spectrum. This is useful e.g. to test the robustness
of pipelines to changes in the instrument model (without the need of actually changing the model).

The simulation produces a spectrum, where the position of the spectral lines is rotated with respect to the CCD center.
"""

if __name__ == "__main__":
    from pyechelle.simulator import Simulator
    from pyechelle.sources import Etalon
    from pyechelle.spectrograph import ZEMAX, GlobalDisturber

    spec = GlobalDisturber(ZEMAX("MaroonX"), rot=0.1)
    sim = Simulator(spec)
    sim.set_ccd(1)
    sim.set_fibers(1)
    sim.set_sources(Etalon(d=10.))
    sim.set_exposure_time(10.)
    sim.set_output(f'05_GlobalDisturber.fits', overwrite=True)
    sim.set_cuda(True)

    sim.run()

    from pyechelle.simulator import export_to_html

    export_to_html(sim.spectrograph.get_ccd(1).data,
                   f'docs/source/_static/plots/example_results/{__file__.split("/")[-1][:-3]}.html')
