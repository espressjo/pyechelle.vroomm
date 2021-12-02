#!/usr/bin/env python
import argparse
import inspect
import logging
import re
import sys
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import pyechelle
from pyechelle import spectrograph, sources
from pyechelle.CCD import read_ccd_from_hdf
from pyechelle.efficiency import GratingEfficiency, TabulatedEfficiency, SystemEfficiency, Atmosphere
from pyechelle.randomgen import AliasSample, generate_slit_polygon, generate_slit_xy, generate_slit_round
from pyechelle.sources import Phoenix
from pyechelle.spectrograph import trace
from pyechelle.telescope import Telescope

logger = logging.getLogger('Simulator')


def parse_num_list(string_list: str) -> list:
    """
    Converts a string specifying a range of numbers (e.g. '1-3') into a list of these numbers ([1,2,3])
    Args:
        string_list: string like "[start value]-[end value]"

    Returns:
        List of integers
    """

    m = re.match(r'(\d+)(?:-(\d+))?$', string_list)
    # ^ (or use .split('-'). anyway you like.)
    if not m:
        raise argparse.ArgumentTypeError(
            "'" + string_list + "' is not a range of number. Expected forms like '0-5' or '2'.")
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start, 10), int(end, 10) + 1))


def export_to_html(data, filename: str = 'test.html'):
    """
    Exports a 2D image into a 'standalone' HTML file. This is used e.g. for some of the examples in the documentation.
    Args:
        data: 2d numpy array
        filename: output filename

    Returns:
        None
    """
    import plotly.express as px
    fig = px.imshow(data, binary_string=True, aspect='equal')

    fig.update_traces(
        hovertemplate=None,
        hoverinfo='skip'
    )
    w = 1000
    h = 300
    fig.update_layout(autosize=True, width=w, height=h, margin=dict(l=0, r=0, b=0, t=0))
    fig.update_yaxes(range=[2000, 3000])
    fig.write_html(filename, include_plotlyjs=False)


def check_for_spectrogrpah_model(modelname):
    file_path = Path.cwd().joinpath("models").joinpath(f"{modelname}.hdf")
    if not file_path.is_file():
        # download file
        print(f"Spectrograph model {modelname} not found locally. Trying to download...")
        Path(Path.cwd().joinpath('models')).mkdir(parents=False, exist_ok=True)
        url = f"https://stuermer.science/nextcloud/index.php/s/zLAw7L5NPEqp7aB/download?path=/&files={modelname}.hdf"
        print(f"Spectrograph model {modelname} not found locally. Trying to download from {url}...")
        with urllib.request.urlopen(url) as response, open(file_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
    return file_path


def simulate(args):
    spec_path = check_for_spectrogrpah_model(args.spectrograph)

    # generate flat list for all fields to simulate
    if any(isinstance(el, list) for el in args.fiber):
        fibers = [item for sublist in args.fiber for item in sublist]
    else:
        fibers = args.fiber

    # generate flat list of all sources to simulate
    source_names = args.sources
    if len(source_names) == 1:
        source_names = [source_names[0]] * len(
            fibers)  # generate list of same length as 'fields' if only one source given

    assert len(fibers) == len(source_names), 'Number of sources needs to match number of fields (or be 1).'

    # generate flat list of whether atmosphere is added
    atmosphere = args.atmosphere
    if len(atmosphere) == 1:
        atmosphere = [atmosphere[0]] * len(
            fibers)  # generate list of same length as 'fields' if only one flag is given

    assert len(fibers) == len(
        atmosphere), f'You specified {len(atmosphere)} atmosphere flags, but we have {len(fibers)} fields/fibers.'

    # generate flat list of whether atmosphere is added
    rvs = args.rv
    if len(rvs) == 1:
        rvs = [rvs[0]] * len(
            fibers)  # generate list of same length as 'fields' if only one flag is given

    assert len(fibers) == len(
        rvs), f'You specified {len(rvs)} radial velocity flags, but we have {len(rvs)} fields/fibers.'

    ccd = read_ccd_from_hdf(spec_path)
    t1 = time.time()
    for f, s, atmo, rv in zip(fibers, source_names, atmosphere, rvs):
        spec = spectrograph.ZEMAX(spec_path, f, args.n_lookup)
        telescope = Telescope(args.d_primary, args.d_secondary)
        # extract kwords specific to selected source
        source_args = [ss for ss in vars(args) if s.lower() in ss]
        # create dict consisting of kword arguments and values specific to selected source
        source_kwargs = dict(zip([ss.replace(f"{s.lower()}_", "") for ss in source_args],
                                 [getattr(args, ss) for ss in source_args]))
        source = getattr(sources, s)(**source_kwargs)
        if args.no_blaze:
            grating_efficiency = None
        else:
            grating_efficiency = GratingEfficiency(spec.blaze, spec.blaze, spec.gpmm)

        if args.no_efficiency:
            spectrograph_efficiency = None
        else:
            if spec.efficiency is not None:
                spectrograph_efficiency = TabulatedEfficiency("Spectrograph efficiency", *spec.efficiency)
            else:
                spectrograph_efficiency = None
        if atmo:
            atmosphere = Atmosphere("Atmosphere", sky_calc_kwargs={'airmass': args.airmass})
        else:
            atmosphere = None
        all_efficiencies = [e for e in [grating_efficiency, spectrograph_efficiency, atmosphere] if e is not None]

        if all_efficiencies:
            efficiency = SystemEfficiency(all_efficiencies, "Total efficiency")
        else:
            efficiency = None

        if args.orders is None:
            orders = spec.orders
        else:
            requested_orders = [item for sublist in args.orders for item in sublist]
            orders = []
            for o in requested_orders:
                if o in spec.orders:
                    orders.append(o)
                else:
                    logger.warning(f'Order {o} is requested, but it is not in the Spectrograph model.')

        for o in np.sort(orders):
            # default wavelength 'grid' per order
            wavelength = np.linspace(*spec.get_wavelength_range(o), num=100000)

            # get spectral density per order
            spectral_density = source.get_spectral_density_rv(wavelength, rv)
            # if source returns own wavelength vector, use that for further calculations instead of default grid
            if isinstance(spectral_density, tuple):
                wavelength, spectral_density = spectral_density

            # for stellar targets calculate collected flux by telescope area
            if source.stellar_target:
                spectral_density *= telescope.area

            # get efficiency per order
            if efficiency is not None:
                eff = efficiency.get_efficiency_per_order(wavelength=wavelength, order=o)
                effective_density = eff * spectral_density
            else:
                effective_density = spectral_density

            # calculate photon flux
            if source.flux_in_photons:
                flux = effective_density
            else:
                ch_factor = 5.03E12  # convert microwatts / micrometer to photons / s per wavelength interval
                wl_diffs = np.ediff1d(wavelength, wavelength[-1] - wavelength[-2])
                flux = effective_density * wavelength * wl_diffs * ch_factor

            flux_photons = flux * args.integration_time
            total_photons = int(np.sum(flux_photons))
            print(f'Order {o:3d}:    {(np.min(wavelength) * 1000.):7.1f} - {(np.max(wavelength) * 1000.):7.1f} nm.     '
                  f'Number of photons: {total_photons}')

            n_simulated = 0
            while n_simulated < total_photons:
                n_photons = min(total_photons - n_simulated, 100000000)
                n_simulated += n_photons
                # get XY list for field
                # x, y = generate_slit_round(n_photons)
                if spec.field_shape == "rectangular":
                    x, y = generate_slit_xy(n_photons)
                elif spec.field_shape == "octagonal":
                    x, y = generate_slit_polygon(8, n_photons, 0.)
                elif spec.field_shape == "hexagonal":
                    x, y = generate_slit_polygon(6, n_photons, 0.)
                elif spec.field_shape == "circular":
                    x, y = generate_slit_round(n_photons)
                else:
                    raise NotImplementedError(f"Field shape {spec.field_shape} is not implemented.")

                # draw wavelength from effective spectrum
                sampler = AliasSample(np.asarray(flux_photons / np.sum(flux_photons), dtype=np.float32))

                wl_sample = wavelength[sampler.sample(n_photons)]

                # trace
                sx, sy, rot, shear, tx, ty = spec.transformations[f'order{o}'].get_matrices_lookup(wl_sample)
                xt, yt = trace(x, y, sx, sy, rot, shear, tx, ty)

                x_psf, y_psf = spec.psfs[f"psf_order_{o}"].draw_xy(wl_sample)

                xt += x_psf / ccd.pixelsize
                yt += y_psf / ccd.pixelsize

                # add photons to ccd
                ccd.add_photons(xt, yt)
    ccd.clip()

    # add bias / global ccd effects
    if args.bias:
        ccd.add_bias(args.bias)
    if args.read_noise:
        ccd.add_readnoise(args.read_noise)
    t2 = time.time()
    logger.info(f"Total time for simulation: {t2 - t1}s.")

    # save simulation to .fits file
    hdu = fits.PrimaryHDU(data=ccd.data)
    hdu_list = fits.HDUList([hdu])
    hdu_list.writeto(args.output, overwrite=args.overwrite)

    if args.html_export:
        export_to_html(ccd.data, args.html_export)
    if args.show:
        plt.figure()
        plt.imshow(ccd.data)
        plt.show()


def main(args=None):
    if not args:
        args = sys.argv[1:]

    dir_path = Path(__file__).resolve().parent.parent.joinpath("models").joinpath("available_models.txt")
    with open(dir_path, 'r') as am:
        models = [line.strip() for line in am.readlines() if line.strip() != '']

    available_sources = [m[0] for m in inspect.getmembers(pyechelle.sources, inspect.isclass) if
                         issubclass(m[1], pyechelle.sources.Source) and m[0] != "Source"]

    parser = argparse.ArgumentParser(description='PyEchelle Simulator')
    parser.add_argument('-s', '--spectrograph', choices=models, type=str, default="MaroonX", required=True,
                        help=f"Filename of spectrograph model. Model file needs to be located in models/ folder. ")
    parser.add_argument('-t', '--integration_time', type=float, default=1.0, required=False,
                        help=f"Integration time for the simulation in seconds [s].")
    parser.add_argument('--fiber', type=parse_num_list, default='1', required=False)
    parser.add_argument('--n_lookup', type=int, default=10000, required=False)
    parser.add_argument('--no_blaze', action='store_true')
    parser.add_argument('--no_efficiency', action='store_true')

    atmosphere_group = parser.add_argument_group('Atmosphere')
    atmosphere_group.add_argument('--atmosphere', nargs='+', required=False, help='Add telluric lines to spectrum.',
                                  type=bool, default=[False])
    atmosphere_group.add_argument('--airmass', default=1.0, type=float, required=False,
                                  help='airmass for atmospheric model')

    telescope_group = parser.add_argument_group('Telescope settings')
    telescope_group.add_argument('--d_primary', type=float, required=False, default=1.0)
    telescope_group.add_argument('--d_secondary', type=float, required=False, default=0)

    parser.add_argument('--orders', type=parse_num_list, nargs='+', required=False,
                        help='Echelle order numbers to simulate... '
                             'if not specified, all orders of the spectrograph are simulated')

    parser.add_argument('--sources', nargs='+', choices=available_sources, required=True)
    parser.add_argument('--rv', nargs='+', type=float, required=False, default=[0.],
                        help="radial velocity shift of source")
    const_source_group = parser.add_argument_group('Constant source')
    const_source_group.add_argument('--constant_intensity', type=float, default=0.0001, required=False,
                                    help="Flux in microWatts / nanometer for constant flux spectral source")
    arclamps_group = parser.add_argument_group('Arc Lamps')
    arclamps_group.add_argument('--scale', default=10.0, required=False,
                                help='Intensity scale of gas lines (e.g. Ag or Ne) vs metal (Th)')
    phoenix_group = parser.add_argument_group('Phoenix')
    phoenix_group.add_argument('--phoenix_t_eff', default=3600,
                               choices=Phoenix.valid_t,
                               type=int, required=False,
                               help="Effective temperature in Kelvins [K].")
    phoenix_group.add_argument('--phoenix_log_g', default=5.,
                               choices=Phoenix.valid_g,
                               type=float, required=False,
                               help="Surface gravity log g.")
    phoenix_group.add_argument('--phoenix_z',
                               choices=Phoenix.valid_z,
                               type=float, required=False, default=0.,
                               help="Overall metallicity.")
    phoenix_group.add_argument('--phoenix_alpha',
                               choices=Phoenix.valid_a,
                               type=float, required=False, default=0.,
                               help="Alpha element abundance.")
    phoenix_group.add_argument('--phoenix_magnitude', default=10., required=False, type=float,
                               help='V Magnitude of stellar object.')

    etalon_group = parser.add_argument_group('Etalon')
    etalon_group.add_argument('--etalon_d', type=float, default=5., required=False,
                              help='Mirror distance of Fabry Perot etalon in [mm]. Default: 5.0')
    etalon_group.add_argument('--etalon_n', type=float, default=1.0, required=False,
                              help='Refractive index of medium between etalon mirrors. Default: 1.0')
    etalon_group.add_argument('--etalon_theta', type=float, default=0., required=False,
                              help='angle of incidence of light in radians. Default: 0.')
    etalon_group.add_argument('--etalon_n_photons', default=1000, required=False,
                              help='Number of photons per seconds per peak of the etalon spectrum. Default: 1000')

    ccd_group = parser.add_argument_group('CCD')
    ccd_group.add_argument('--bias', type=int, required=False, default=0)
    ccd_group.add_argument('--read_noise', type=float, required=False, default=0)

    parser.add_argument('--show', default=False, action='store_true')
    parser.add_argument('-o', '--output', type=argparse.FileType('wb', 0), required=False, default='test.fits',
                        help='A .fits file where the simulation is saved.')
    parser.add_argument('--overwrite', default=False, action='store_true')

    parser.add_argument('--html_export', type=str, default='',
                        help="If given, the spectrum will be exported to an interactive image using plotly. It's not a"
                             "standalone html file, but requires plotly.js to be loaded.")
    args = parser.parse_args(args)
    t1 = time.time()
    simulate(args)
    t2 = time.time()
    print(f"Simulation took {t2 - t1} s")


if __name__ == "__main__":
    main()
