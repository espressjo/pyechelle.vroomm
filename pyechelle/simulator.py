#!/usr/bin/env python
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import pyechelle
from pyechelle import spectrograph, sources
from pyechelle.CCD import read_ccd_from_hdf
from pyechelle.efficiency import GratingEfficiency
from pyechelle.randomgen import generate_slit_round, AliasSample
from pyechelle.spectrograph import trace
from pyechelle.telescope import Telescope


def parse_num_list(string_list: str) -> list:
    m = re.match(r'(\d+)(?:-(\d+))?$', string_list)
    # ^ (or use .split('-'). anyway you like.)
    if not m:
        raise argparse.ArgumentTypeError("'" + string_list + "' is not a range of number. Expected forms like '0-5' or '2'.")
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start, 10), int(end, 10) + 1))


def model_name_to_path(model_name: str) -> Path:
    """
    Converts a spectrograph model name into a full path to the corresponding .hdf file
    Args:
        model_name (str): Name of the spectorgraph model e.g. 'MaroonX'

    Returns:
        full path to .hdf file
    """
    script_dir = Path(__file__).resolve().parent.parent.joinpath("models")
    return script_dir.joinpath(f"{model_name}.hdf")


def main(args):
    # generate flat list for all fields to simulate
    if any(isinstance(el, list) for el in args.fiber):
        fibers = [item for sublist in args.fiber for item in sublist]
    else:
        fibers = args.fiber

    # generate flat list of all sources to simulate
    source_names = args.sources
    if len(source_names) == 1:
        source_names = [source_names[0]] * len(
            fibers)  # generate list of same length than 'fields' if only one source given

    assert len(fibers) == len(source_names), 'Number of sources needs to match number of fields (or be 1).'

    ccd = read_ccd_from_hdf(args.spectrograph)
    for f, s in zip(fibers, source_names):
        spec = spectrograph.ZEMAX(args.spectrograph, f, args.n_lookup)
        telescope = Telescope(args.d_primary, args.d_secondary)
        source = getattr(sources, s)()
        if args.use_blaze_efficiency:
            efficiency = GratingEfficiency(spec.blaze, spec.blaze, spec.gpmm)

        if args.orders is None:
            orders = spec.orders
        else:
            orders = [item for sublist in args.orders for item in sublist]
            # TODO: Check that order exists

        for o in orders:
            wavelength = np.linspace(*spec.get_wavelength_range(o), num=10000)
            # get spectral density per order
            if source.list_like_source:
                wavelength, spectral_density = source.get_spectral_density(wavelength)
            else:
                spectral_density = source.get_spectral_density(wavelength)

            # get efficiency per order
            eff = efficiency.get_efficiency_per_order(wavelength=wavelength, order=o)

            # calculate efficiency * spectral density
            effective_density = eff * spectral_density

            # calculate photon flux
            if not source.list_like_source:
                ch_factor = 5.03E12  # convert microwatts / micrometer to photons / s per wavelength intervall
                wl_diffs = np.ediff1d(wavelength, wavelength[-2] - wavelength[-1])
                flux = effective_density * wavelength * wl_diffs * ch_factor
            else:
                flux = spectral_density

            flux_photons = flux * args.integration_time
            n_photons = int(np.sum(flux_photons))
            print(f'Order {o}: Number of photons: {n_photons}')

            # get XY list for field
            x, y = generate_slit_round(n_photons)

            # draw wavelength from effective spectrum
            sampler = AliasSample(np.asarray(flux_photons / np.sum(flux_photons), dtype=np.float32))

            wltest = wavelength[sampler.sample(n_photons)]
            # wltest = (np.max(wavelength) - np.min(wavelength)) * np.random.random(n_photons) + np.min(wavelength)

            # trace
            sx, sy, rot, shear, tx, ty = spec.transformations[f'order{o}'].get_matrices_lookup(wltest)
            xt, yt = trace(x, y, sx, sy, rot, shear, tx, ty)

            X, Y = spec.psfs[f"psf_order_{o}"].draw_xy(wltest)

            xt += X / ccd.pixelsize
            yt += Y / ccd.pixelsize

            # add photons to ccd
            ccd.add_photons(xt, yt)

    # add bias / global ccd effects
    if args.bias:
        ccd.add_bias(args.bias)
    if args.read_noise:
        ccd.add_readnoise(args.read_noise)
    plt.figure()
    plt.imshow(ccd.data)
    plt.show()


if __name__ == "__main__":
    import sys
    import inspect

    dir_path = Path(__file__).resolve().parent.parent.joinpath("models")
    models = [x.stem for x in dir_path.glob('*.hdf')]

    available_sources = [m[0] for m in inspect.getmembers(pyechelle.sources, inspect.isclass) if
                         issubclass(m[1], pyechelle.sources.Source)]

    parser = argparse.ArgumentParser(description='PyEchelle Simulator')
    parser.add_argument('-s', '--spectrograph', nargs='?', type=model_name_to_path, default=sys.stdin, required=True,
                        help=f"Filename of spectrograph model. Model file needs to be located in models/ folder. "
                             f"Options are {','.join(models)}")
    parser.add_argument('-t', '--integration_time', type=float, default=1.0, required=False,
                        help=f"Integration time for the simulation in seconds [s].")
    parser.add_argument('--fiber', type=parse_num_list, default='1', required=False)
    parser.add_argument('--n_lookup', type=int, default=10000, required=False)

    parser.add_argument('--d_primary', type=float, required=False, default=1.0)
    parser.add_argument('--d_secondary', type=float, required=False, default=0)

    parser.add_argument('--orders', type=parse_num_list, nargs='+', required=False,
                        help='Echelle order numbers to simulate... '
                             'if not specified, all orders of the spectrograph are simulated')
    parser.add_argument('--sources', nargs='+', choices=available_sources, required=True)
    parser.add_argument('--use_blaze_efficiency', default=True, action='store_true')
    parser.add_argument('--bias', type=int, required=False, default=1000)
    parser.add_argument('--read_noise', type=float, required=False, default=3.0)

    arguments = parser.parse_args()
    main(arguments)
