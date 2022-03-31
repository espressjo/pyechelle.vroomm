"""Disturbing the spectrograph model
====================================

This script shows how to use the Disturber class on top of a 'regular' spectrograph model.
It allows to apply global distortions to the entire spectrum. This is useful e.g. to test the robustness
of pipelines to changes in the instrument model (without the need of actually changing the model).

The simulation produces two fits files, where the second spectrum is slightly rotated wrt. the first.
"""

if __name__ == "__main__":

    from pyechelle.simulator import Simulator
    from pyechelle.sources import Etalon
    from pyechelle.spectrograph import ZEMAX, LocalDisturber

    for rotation in [0.0, 0.1]:
        spec = LocalDisturber(ZEMAX("MaroonX"), d_rot=rotation)
        sim = Simulator(spec)
        sim.set_ccd(1)
        sim.set_fibers(1)
        sim.set_sources(Etalon(d=10.))
        sim.set_exposure_time(10.)
        sim.set_output(f'04_disturber_rot{rotation}.fits', overwrite=True)
        sim.set_cuda(True)

        sim.run()

    from pyechelle.simulator import export_to_html

    export_to_html(sim.spectrograph.get_ccd(1).data,
                   f'docs/source/_static/plots/example_results/{__file__.split("/")[-1][:-3]}.html')
