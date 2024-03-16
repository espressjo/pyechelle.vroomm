"""Multiple fibers and multiple sources
=======================================

 This example shows how to simulate multiple fibers and source in a single go.
 """

if __name__ == "__main__":
    from pyechelle.simulator import Simulator
    from pyechelle.sources import ConstantFlux, IdealEtalon
    from pyechelle.spectrograph import ZEMAX

    sim = Simulator(ZEMAX("MaroonX"))
    sim.set_ccd(1)
    # multiple fibers/fields can be specified at once
    sim.set_fibers([2, 3, 4])
    # now, the number of sources specified needs either to match the number of fibers,
    # or it needs to be a single source (that is then used for each fiber)
    sim.set_sources([ConstantFlux(), IdealEtalon(d=5, n_photons=30000), IdealEtalon(d=10, n_photons=30000)])
    sim.set_exposure_time(0.01)
    sim.set_output('01_multiple_fibers_and_sources.fits', overwrite=True)
    sim.run()

    from pyechelle.simulator import export_to_html

    export_to_html(sim.spectrograph.get_ccd(1).data,
                   f'docs/source/_static/plots/example_results/{__file__.split("/")[-1][:-3]}.html')
