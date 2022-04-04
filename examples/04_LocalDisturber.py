"""Disturbing the spectrograph model
====================================

This script shows how to use the LocalDisturber class on top of a 'regular' spectrograph model.
It allows applying local distortions to the fiber/slit projection. This is useful e.g. to test the robustness
of pipelines to changes in the instrument model (without the need of actually changing the model).

The simulation produces a fits file, where the fiber is rotated in place.
"""

if __name__ == "__main__":
    from pyechelle.simulator import Simulator
    from pyechelle.sources import Etalon
    from pyechelle.spectrograph import ZEMAX, LocalDisturber

    spec = LocalDisturber(ZEMAX("MaroonX"), d_rot=0.01)
    sim = Simulator(spec)
    sim.set_ccd(1)
    sim.set_fibers(1)
    sim.set_sources(Etalon(d=10.))
    sim.set_exposure_time(10.)
    sim.set_output(f'04_LocalDisturber.fits', overwrite=True)
    sim.set_cuda(True)

    sim.run()

    from pyechelle.simulator import export_to_html

    export_to_html(sim.spectrograph.get_ccd(1).data,
                   f'docs/source/_static/plots/example_results/{__file__.split("/")[-1][:-3]}.html')
