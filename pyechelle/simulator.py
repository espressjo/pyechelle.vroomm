#!/usr/bin/env python
from __future__ import annotations

import argparse
import distutils.util
import inspect
import logging
import re
import sys
import time
import urllib.request
from pathlib import Path
from urllib.error import URLError

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed
from numba import cuda

import pyechelle
import pyechelle.slit
from CCD import CCD
from pyechelle import spectrograph, sources
from pyechelle.efficiency import GratingEfficiency, TabulatedEfficiency, SystemEfficiency, Atmosphere
from pyechelle.raytrace_cuda import make_cuda_kernel
from pyechelle.raytrace_cuda import raytrace_order_cuda
from pyechelle.raytracing import raytrace_order_cpu
from pyechelle.sources import Phoenix, Source
from pyechelle.telescope import Telescope
from spectrograph import Spectrograph, ZEMAX

logger = logging.getLogger('Simulator')
logger.setLevel(level=logging.INFO)

# get list of available spectrograph models
dir_path = Path(__file__).resolve().parent.joinpath("models").joinpath("available_models.txt")
with open(dir_path, 'r') as am:
    available_models = [line.strip() for line in am.readlines() if line.strip() != '']

# get list of available sources
available_sources = [m[0] for m in inspect.getmembers(pyechelle.sources, inspect.isclass) if
                     issubclass(m[1], pyechelle.sources.Source) and m[0] != "Source"]


def parse_num_list(string_list: str) -> list:
    """
    Converts a string specifying a range of numbers (e.g. '1-3') into a list of these numbers ([1,2,3])
    Args:
        string_list: string like "[start value]-[end value]"

    Returns:
        List of integers
    """

    m = re.match(r'(\d+)(?:-(\d+))?$', string_list)
    if not m:
        raise argparse.ArgumentTypeError(
            "'" + string_list + "' is not a range of number. Expected forms like '0-5' or '2'.")
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start, 10), int(end, 10) + 1))


def check_url_exists(url: str) -> bool:
    """
    Check if URL exists.
    Args:
        url: url to be tested

    Returns:
        if URL exists
    """
    try:
        with urllib.request.urlopen(url) as response:
            return float(response.headers['Content-length']) > 0
    except URLError:
        return False


def export_to_html(data, filename, include_plotlyjs=False):
    """
    Exports a 2D image into a 'standalone' HTML file. This is used e.g. for some examples in the documentation.
    Args:
        data: 2d numpy array
        filename: output filename
        include_plotlyjs: whether plotlyjs is included in html file or not

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
    fig.write_html(filename, include_plotlyjs=include_plotlyjs)


def check_for_spectrograph_model(model_name: str, download=True):
    """
    Check if spectrograph model exists locally. Otherwise: Download if download is true (default) or check if URL to
    spectrograph model is valid (this is mainly for testing purpose).

    Args:
        model_name: name of spectrograph model. See models/available_models.txt for valid names
        download: download flag

    Returns:

    """
    file_path = Path(__file__).resolve().parent.joinpath("models").joinpath(f"{model_name}.hdf")
    if not file_path.is_file():
        url = f"https://stuermer.science/nextcloud/index.php/s/zLAw7L5NPEqp7aB/download?path=/&files={model_name}.hdf"
        if download:
            print(f"Spectrograph model {model_name} not found locally. Trying to download from {url}...")
            Path(Path(__file__).resolve().parent.joinpath("models")).mkdir(parents=False, exist_ok=True)
            with urllib.request.urlopen(url) as response, open(file_path, "wb") as out_file:
                data = response.read()
                out_file.write(data)
        else:
            check_url_exists(url)
    return file_path


def log_elapsed_time(msg: str, t0: float) -> float:
    t1 = time.time()
    logger.info(msg + f' (took {t1 - t0:2f} s )')
    return t1


class Simulator:
    """ PyEchelle simulator

    Simulator class that contains everything needed to generate spectra.
    Attributes:
        spectrograph (Spectrograph): spectrograph used for simulations
        fibers (list[int]): fiber / list of fibers to be simulated
        sources (list[Source]): spectral source / list of spectral sources per fiber (same length as fibers)
        orders (list[int]): order / list of diffraction orders to be simulated
        atmosphere (list[bool]): whether to include atmospheric transmission per fiber (same length as fibers)
        max_cpu (int): number of CPUs used for simulation (if -1, max_cpu is number of available cores)

    """

    def __init__(self, spectrograph: Spectrograph, sources: Source | list[Source],
                 fibers: int | list[int] | None, orders: int | list[int] | None, telescope: Telescope,
                 atmosphere: bool | list[bool] = False, rvs: float | list[float] = 0.,
                 show_after_simulation: bool = False,
                 ccds: int | list[int] | None = None, cuda: bool = False, cuda_seed: int = -1, max_cpu: int = 1):

        self.spectrograph = spectrograph
        self.fibers = [fibers] if isinstance(fibers, int) else fibers

        self.orders = [orders] if isinstance(orders, int) else orders

        self.sources = [sources] if isinstance(sources, Source) else sources
        self.atmosphere = [atmosphere] if isinstance(atmosphere, bool) else atmosphere
        self.rvs = [rvs] if isinstance(rvs, float) else rvs
        self.telescope = telescope
        self.show_after_simulation = show_after_simulation
        self.ccds = [ccds] if isinstance(ccds, int) else ccds
        self.cuda = cuda
        self.cuda_seed = -1
        self.max_cpu = max_cpu

    def validate(self):
        assert len(self.fibers) == len(
            self.sources), 'Number of sources needs to match the number of fields/fibers (or be 1)'
        for ccd in self.ccds:
            assert ccd in self.spectrograph.get_ccd().keys(), f'You requested simulation of CCD {ccd}, which is not ' \
                                                              f'available '

    def get_valid_orders(self, fiber: int, ccd_index: int):
        valid_orders = []

        requested_orders = self.spectrograph.get_orders(fiber, ccd_index) if self.orders is None else self.orders
        for o in requested_orders:
            if o in self.spectrograph.get_orders(fiber, ccd_index):
                valid_orders.append(o)
            else:
                logger.warning(f'Order {o} is requested, but it is not in the Spectrograph model.')
        return valid_orders

    def get_slit_function(self, fiber: int, ccd_index: int, cuda: bool = False):
        try:
            if cuda:
                slit_fun = getattr(pyechelle.slit, f"cuda_{self.spectrograph.get_field_shape(fiber, ccd_index)}")
            else:
                slit_fun = getattr(pyechelle.slit, self.spectrograph.get_field_shape(fiber, ccd_index))
        except AttributeError:
            raise NotImplementedError(
                f"Field shape {self.spectrograph.get_field_shape(fiber, ccd_index)} is not implemented.")
        return slit_fun

    def simulate_multi_CPU(self, orders, fiber, ccd_index, slit_fun, s, rv, integration_time, c, efficiency):
        simulated_photons = []
        t0 = time.time()
        results = Parallel(n_jobs=min(self.max_cpu, len(orders)))(
            delayed(raytrace_order_cpu)(o, self.spectrograph.get_wavelength_range(o, fiber, ccd_index), s, slit_fun,
                                        self.telescope, rv, integration_time,
                                        c,
                                        efficiency,
                                        self.max_cpu) for o in np.sort(orders))
        logger.info('Add up orders...')
        ccd_results = [r[0] for r in results]
        simulated_photons.extend([r[1] for r in results])
        c.data = np.sum(ccd_results, axis=0)
        log_elapsed_time('done.', t0)
        return simulated_photons

    def simulate_single_CPU(self, orders, fiber, ccd_index, s, slit_fun, rv, integration_time, c, efficiency):
        simulated_photons = []
        for o in np.sort(orders):
            nphot = raytrace_order_cpu(o, self.spectrograph, s, slit_fun, self.telescope, rv,
                                       integration_time,
                                       c, fiber, ccd_index,
                                       efficiency,
                                       1)
            simulated_photons.append(nphot)
        return simulated_photons

    def simulate_cuda(self, orders, slit_fun, rv, integration_time, dccd, efficiency, s, c, fiber, ccd_index):
        cuda_kernel = make_cuda_kernel(slit_fun)
        simulated_photons = []
        for o in np.sort(orders):
            nphot = raytrace_order_cuda(o, self.spectrograph, s, self.telescope, rv, integration_time, dccd,
                                        float(c.pixelsize), fiber, ccd_index, efficiency, seed=self.cuda_seed,
                                        cuda_kernel=cuda_kernel)
            simulated_photons.append(nphot)
        return simulated_photons

    def run(self, max_cpu, output, overwrite, integration_time=1.):
        requested_ccds = self.spectrograph.get_ccd().keys() if self.ccds is None else self.ccds
        for i in requested_ccds:
            c = self.spectrograph.get_ccd(i)
            total_simulated_photons = []

            # copy empty array to CUDA device
            if self.cuda:
                dccd = cuda.to_device(np.zeros_like(c.data, dtype=np.uint32))

            for f, s, atmo, rv in zip(self.fibers, self.sources, self.atmosphere, self.rvs):
                orders = self.get_valid_orders(f, i)
                slit_fun = self.get_slit_function(f, i, self.cuda)
                e = self.spectrograph.get_efficiency(i)

                if not self.cuda:
                    if max_cpu > 1:
                        self.simulate_multi_CPU(orders, f, i, slit_fun, s, rv, integration_time, c, e)
                    else:
                        self.simulate_single_CPU(orders, f, i, s, slit_fun, rv, integration_time, c, e)
                else:
                    self.simulate_cuda(orders, slit_fun, rv, integration_time, dccd, e, s, c, f, i)

            if self.cuda:
                dccd.copy_to_host(c.data)
            logger.info('Finish up simulation and save...')
            c.clip()

            # add bias / global ccd effects
            # if args.bias:
            #     ccd.add_bias(args.bias)
            # if args.read_noise:
            #     ccd.add_readnoise(args.read_noise)
            t2 = time.time()

            self.write_to_fits(c, output, overwrite)

            # if args.html_export:
            #     export_to_html(ccd.data, args.html_export)
            # if args.show:
            #     plt.figure()
            #     plt.imshow(ccd.data)
            #     plt.show()
            # t0 = log_elapsed_time('done.', t0)
            # logger.info(f"Total time for simulation: {t2 - t1:.3f}s.")
            logger.info(f"Total simulated photons: {sum(total_simulated_photons)}")
            return sum(total_simulated_photons)

    def write_to_fits(self, c: CCD, filename: str | Path, overwrite: bool = True):
        hdu = fits.PrimaryHDU(data=np.array(c.data, dtype=int))
        hdu_list = fits.HDUList([hdu])
        hdu_list.writeto(filename, overwrite=overwrite)


def simulate(args):
    t0 = time.time()
    logger.info(f'Check/download spectrograph model...')
    spec_path = check_for_spectrograph_model(args.spectrograph)
    spec = ZEMAX(spec_path)
    t0 = log_elapsed_time('done.', t0)

    logger.info(f'Prepare simulation arguments...')
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

    # generate flat list for RV values
    rvs = args.rv
    if len(rvs) == 1:
        rvs = [rvs[0]] * len(
            fibers)  # generate list of same length as 'fields' if only one flag is given

    assert len(fibers) == len(
        rvs), f'You specified {len(rvs)} radial velocity flags, but we have {len(rvs)} fields/fibers.'

    ccd = read_ccd_from_hdf(spec_path)
    ccd = ZEMAX.get_spectrograph_unit()
    t0 = log_elapsed_time('done.', t0)
    t1 = time.time()

    n_cpu_max = args.max_cpu

    total_simulated_photons = []

    if args.cuda:
        dccd = cuda.to_device(np.zeros_like(ccd.data, dtype=np.uint32))

    for f, s, atmo, rv in zip(fibers, source_names, atmosphere, rvs):
        logger.info('Read in spectrograph model')
        spec = spectrograph.ZEMAX(spec_path, f)
        t0 = log_elapsed_time('done.', t0)
        logger.info('Prepare simulation...')
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

        t0 = log_elapsed_time('done.', t0)
        logger.info('Do raytracing...')
        if not args.cuda:
            # get slit function
            try:
                slit_fun = getattr(pyechelle.slit, spec.field_shape)
            except AttributeError:
                raise NotImplementedError(f"Field shape {spec.field_shape} is not implemented.")

            if n_cpu_max > 1:
                results = Parallel(n_jobs=min(n_cpu_max, len(orders)))(
                    delayed(raytrace_order_cpu)(o, spec, source, slit_fun, telescope, rv, args.integration_time, ccd,
                                                efficiency,
                                                n_cpu_max) for o in np.sort(orders))
                logger.info('Add up orders...')
                ccd_results = [r[0] for r in results]
                total_simulated_photons.extend([r[1] for r in results])
                ccd.data = np.sum(ccd_results, axis=0)
                t0 = log_elapsed_time('done.', t0)
            else:
                for o in np.sort(orders):
                    nphot = raytrace_order_cpu(o, spec, source, slit_fun, telescope, rv, args.integration_time, ccd,
                                               efficiency,
                                               1)
                    total_simulated_photons.append(nphot)
        else:
            try:
                slit_fun = getattr(pyechelle.slit, f"cuda_{spec.field_shape}")
            except AttributeError:
                raise NotImplementedError(f"Field shape {spec.field_shape} is not implemented.")

            cuda_kernel = make_cuda_kernel(slit_fun)
            for o in np.sort(orders):
                nphot = raytrace_order_cuda(o, spec, source, telescope, rv, args.integration_time, dccd,
                                            float(ccd.pixelsize), efficiency, seed=args.cuda_seed,
                                            cuda_kernel=cuda_kernel)
                total_simulated_photons.append(nphot)
        t0 = log_elapsed_time('done.', t0)

    if args.cuda:
        dccd.copy_to_host(ccd.data)
    logger.info('Finish up simulation and save...')
    ccd.clip()

    # add bias / global ccd effects
    if args.bias:
        ccd.add_bias(args.bias)
    if args.read_noise:
        ccd.add_readnoise(args.read_noise)
    t2 = time.time()

    # save simulation to .fits file
    hdu = fits.PrimaryHDU(data=np.array(ccd.data, dtype=int))
    hdu_list = fits.HDUList([hdu])
    hdu_list.writeto(args.output, overwrite=args.overwrite)

    if args.html_export:
        export_to_html(ccd.data, args.html_export)
    if args.show:
        plt.figure()
        plt.imshow(ccd.data)
        plt.show()
    t0 = log_elapsed_time('done.', t0)
    logger.info(f"Total time for simulation: {t2 - t1:.3f}s.")
    logger.info(f"Total simulated photons: {sum(total_simulated_photons)}")
    return sum(total_simulated_photons)


def generate_parser():
    parser = argparse.ArgumentParser(description='PyEchelle Simulator')
    parser.add_argument('-s', '--spectrograph', choices=available_models, type=str, default="MaroonX", required=True,
                        help=f"Filename of spectrograph model. Model file needs to be located in models/ folder. ")
    parser.add_argument('-t', '--integration_time', type=float, default=1.0, required=False,
                        help=f"Integration time for the simulation in seconds [s].")
    parser.add_argument('--fiber', type=parse_num_list, default='1', required=False,
                        help='Fiber/Field number(s) to be simulated. Can either be a single integer, or an integer'
                             'range (e.g. 1-3) ')
    parser.add_argument('--no_blaze', action='store_true',
                        help='If set, the blaze efficiency per order will be ignored.')
    parser.add_argument('--no_efficiency', action='store_true',
                        help='If set, all instrument/atmosphere efficiencies will be ignored.')

    parser.add_argument('--cuda', action='store_true',
                        help='If set, CUDA will be used for raytracing. Note: the max_cpu flag is then obsolete.')

    parser.add_argument('--cuda_seed', type=int, default=-1,
                        help='Random seed for generating CUDA RNG states. If <0, then the seed is choosen randomly.')

    parser.add_argument('--max_cpu', type=int, default=1,
                        help="Maximum number of CPU cores used. Note: The parallelization happens 'per order'."
                             " Order-wise images are added up. This requires a large amount of memory at the moment."
                             "If planning on simulating multiple images, consider using only 1 CPU per simulation "
                             "and starting multiple simulations instead.")

    atmosphere_group = parser.add_argument_group('Atmosphere')
    atmosphere_group.add_argument('--atmosphere', nargs='+', required=False,
                                  help='Add telluric lines to spectrum. For adding tellurics to all spectra just use'
                                       '--atmosphere Y, for specifying per fiber user e.g. --atmosphere Y N Y',
                                  type=lambda x: bool(distutils.util.strtobool(x)), default=[False])

    atmosphere_group.add_argument('--airmass', default=1.0, type=float, required=False,
                                  help='airmass for atmospheric model')

    telescope_group = parser.add_argument_group('Telescope settings')
    telescope_group.add_argument('--d_primary', type=float, required=False, default=1.0,
                                 help='Diameter of the primary telescope mirror.')
    telescope_group.add_argument('--d_secondary', type=float, required=False, default=0,
                                 help='Diameter of the secondary telescope mirror.')

    parser.add_argument('--orders', type=parse_num_list, nargs='+', required=False,
                        help='Echelle/Grating order numbers to simulate... '
                             'if not specified, all orders of the spectrograph are simulated.'
                             'Can either be a single integer, or a range (e.g. 80-90)')

    parser.add_argument('--sources', nargs='+', choices=available_sources, required=True,
                        help='Spectral source for the simulation. Can either be a single string, e.g. "Etalon",'
                             ' or a comma separated list of sources (e.g. "Etalon, Constant, Etalon") which length must'
                             'match the number of fields/fibers.')
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

    parser.add_argument('--show', default=False, action='store_true',
                        help='If set, the simulated frame will be shown in a matplotlib imshow frame at the end.')
    parser.add_argument('-o', '--output', type=argparse.FileType('wb', 0), required=False, default='test.fits',
                        help='A .fits file where the simulation is saved.')
    parser.add_argument('--overwrite', default=False, action='store_true')

    parser.add_argument('--html_export', type=str, default='',
                        help="If given, the spectrum will be exported to an interactive image using plotly. It's not a"
                             "standalone html file, but requires plotly.js to be loaded.")
    return parser


def main(args=None):
    if not args:
        args = sys.argv[1:]
    parser = generate_parser()
    args = parser.parse_args(args)
    t1 = time.time()
    # n_total_photons = simulate(args)
    spec_path = check_for_spectrograph_model(args.spectrograph)

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

    # generate flat list for RV values
    rvs = args.rv
    if len(rvs) == 1:
        rvs = [rvs[0]] * len(
            fibers)  # generate list of same length as 'fields' if only one flag is given

    sim = Simulator(ZEMAX("/home/stuermer/PycharmProjects/new_Models/models/MaroonX.hdf"),
                    [getattr(sources, source)() for source in source_names], fibers,
                    [item for sublist in args.orders for item in sublist] if args.orders is not None else None,
                    Telescope(args.d_primary, args.d_secondary), cuda=args.cuda, cuda_seed=args.cuda_seed, rvs=rvs,
                    atmosphere=atmosphere)
    sim.run(args.max_cpu, args.output, True, args.integration_time)
    t2 = time.time()
    print(f"Simulation took {t2 - t1:.3f} s")

    # return n_total_photons


if __name__ == "__main__":
    main()
