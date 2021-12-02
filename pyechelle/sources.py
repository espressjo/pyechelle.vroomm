import pathlib
import urllib.request

import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import pandas as pd
import scipy.interpolate

try:
    from astroquery.nist import Nist
except ImportError:
    Nist = None


def pull_catalogue_lines(min_wl, max_wl, catalogue='Th', wavelength_type='vacuum'):
    """
    Reads NIST catalogue lines between min_wl and max_wl of catalogue.

    Args:
        min_wl (float): minimum wavelength bound [micron]
        max_wl (float): maximum wavelength bound [micron]
        catalogue (str): catalogue appreciation label, e.g. 'Th', 'Ar', etc.
        wavelength_type (str): either 'var+air' or 'vacuum'

    Returns:
        (tuple) line catalogue wavelength and relative intensities. wavelength is in [angstrom]
    """
    try:
        table_lines = Nist.query(min_wl * u.micron, max_wl * u.micron, linename=catalogue, output_order='wavelength',
                                 wavelength_type=wavelength_type)[['Ritz', 'Rel.']]
        df = table_lines.filled(0).to_pandas()
        df['Rel.'] = pd.to_numeric(df['Rel.'], downcast='float', errors='coerce')
        df['Ritz'] = pd.to_numeric(df['Ritz'], downcast='float', errors='coerce')
        df.dropna(inplace=True)
        idx = np.logical_and(df['Rel.'] > 0, df['Ritz'] > 0)
        return df['Ritz'].values[idx], df['Rel.'].values[idx]
    except Exception as e:
        print(e)
        print(f"Warning: Couldn't retrieve {catalogue} catalogue data between {min_wl} and {max_wl} micron")
        return np.array([]), np.array([])


def calc_flux_scale(source_wavelength, source_spectral_density, mag):
    # V - band-filter
    v_filter_wl = [0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6,
                   0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7]
    v_filter_tp = [0, 0.03, 0.163, 0.458, 0.78, 0.967, 1, 0.973, 0.898, 0.792, 0.684, 0.574, 0.461,
                   0.359, 0.27, 0.197, 0.135, 0.081, 0.045, 0.025, 0.017, 0.013, 0.009, 0]

    # Reference flux obtained from integration of vega over bessel filter (units are microwatts/m^2*micrometer)
    v_zp = 3.68E-02

    v_filter_interp = scipy.interpolate.interp1d(v_filter_wl, v_filter_tp)

    # get total flux in filter/source range
    lower_wl_limit = max(np.min(source_wavelength), np.min(v_filter_wl))
    upper_wl_limit = min(np.max(source_wavelength), np.max(v_filter_wl))

    idx = np.logical_and(source_wavelength > lower_wl_limit, source_wavelength < upper_wl_limit)

    step = np.ediff1d(source_wavelength[idx], source_wavelength[idx][-1] - source_wavelength[idx][-2])
    total_flux = np.sum(source_spectral_density[idx] * v_filter_interp(source_wavelength[idx]) * step)

    return pow(10, mag / (-2.5)) * v_zp / total_flux


class Source:
    """ A spectral source.

    This class should be subclassed to implement different spectral sources.

    Attributes:
        name (str): name of the source. This will end up in the .fits header.
        min_wl (float): lower wavelength limit [nm] (for normalization purposes)
        max_wl (float): upper wavelength limit [nm] (for normalization purposes)

    """

    def __init__(self, min_wl=599.8, max_wl=600.42, name=""):
        self.name = name
        self.min_wl = min_wl
        self.max_wl = max_wl
        self.stellar_target = False
        self.flux_in_photons = False

    def get_spectral_density(self, wavelength):
        raise NotImplementedError()

    def get_spectral_density_rv(self, wavelength, rv=0.):
        c = 299792458.  # m/s
        rv_shifted = wavelength * ((c - rv) / c)

        spec_density = self.get_spectral_density(rv_shifted)
        # first case: source returns own wavelength vector:
        if isinstance(spec_density, tuple):
            wl, sd = spec_density
            return wl * ((c - rv) / c), sd
        else:
            return spec_density


class Constant(Source):
    def __init__(self, intensity=0.001, **kwargs):
        super().__init__(**kwargs, name="Constant")
        self.intensity = intensity

    def get_spectral_density(self, wavelength):
        return np.ones_like(wavelength) * self.intensity


class ThAr(Source):
    def __init__(self, argon_to_thorium_factor=10):
        super().__init__(name='ThAr')
        if Nist is None:
            raise SystemError(
                "You have to install pyechelle's optional dependency astroquery. "
                "Please refer to the documentation how to do this.")
        self.list_like_source = True
        self.flux_in_photons = True
        self.scale = argon_to_thorium_factor

    def get_spectral_density(self, wavelength):
        minwl = np.min(wavelength)
        maxwl = np.max(wavelength)
        thwl, thint = pull_catalogue_lines(minwl, maxwl, 'Th')
        arwl, arint = pull_catalogue_lines(minwl, maxwl, 'Ar')
        arint *= self.scale
        return np.hstack((thwl / 10000., arwl / 10000.)), np.hstack((thint, arint))


class ThNe(Source):
    def __init__(self, neon_to_thorium_factor=10):
        super().__init__(name='ThNe')
        if Nist is None:
            raise SystemError(
                "You have to install pyechelle's optional dependency astroquery. "
                "Please refer to the documentation how to do this.")
        self.list_like_source = True
        self.flux_in_photons = True
        self.scale = neon_to_thorium_factor

    def get_spectral_density(self, wavelength):
        minwl = np.min(wavelength)
        maxwl = np.max(wavelength)
        thwl, thint = pull_catalogue_lines(minwl, maxwl, 'Th')
        newl, neint = pull_catalogue_lines(minwl, maxwl, 'Ne')
        neint *= self.scale

        return np.hstack((thwl / 10000., newl)), np.hstack((thint / 10000., neint))


class Etalon(Source):
    def __init__(self, d=5.0, n=1.0, theta=0.0, n_photons=1000, **kwargs):
        super().__init__(**kwargs, name="Etalon")
        self.d = d
        self.n = n
        self.theta = theta
        self.min_m = np.ceil(2e3 * d * np.cos(theta) / self.max_wl)
        self.max_m = np.floor(2e3 * d * np.cos(theta) / self.min_wl)
        self.n_photons = n_photons
        self.list_like_source = True
        self.flux_in_photons = True

    @staticmethod
    def peak_wavelength_etalon(m, d=10.0, n=1.0, theta=0.0):
        return 2e3 * d * n * np.cos(theta) / m

    def get_spectral_density(self, wavelength):
        self.min_m = np.ceil(2e3 * self.d * np.cos(self.theta) / np.max(wavelength))
        self.max_m = np.floor(2e3 * self.d * np.cos(self.theta) / np.min(wavelength))
        intensity = np.ones_like(np.arange(self.min_m, self.max_m), dtype=float)
        intensity = intensity * float(self.n_photons)
        return self.peak_wavelength_etalon(
            np.arange(self.min_m, self.max_m), self.d, self.n, self.theta
        ), np.asarray(intensity, dtype=int)

    def draw_wavelength(self, N):
        return np.random.choice(
            self.peak_wavelength_etalon(
                np.arange(self.min_m, self.max_m), self.d, self.n, self.theta
            ),
            N,
        )


class Phoenix(Source):
    """
    Phoenix M-dwarf spectra.

             .-'  |
            / M <\|
           /dwarf\'
           |_.- o-o
           / C  -._)\
          /',        |
         |   `-,_,__,'
         (,,)====[_]=|
           '.   ____/
            | -|-|_
            |____)_)

    This class provides a convenient handling of PHOENIX M-dwarf spectra.
    For a given set of effective Temperature, log g, metalicity and alpha, it downloads the spectrum from PHOENIX ftp
    server.

    """
    valid_t = [*list(range(2300, 7000, 100)), *list((range(7000, 12200, 200)))]
    valid_g = [*list(np.arange(0, 6, 0.5))]
    valid_z = [*list(np.arange(-4, -2, 1)), *list(np.arange(-2.0, 1.5, 0.5))]
    valid_a = [-0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]

    def __init__(
            self, t_eff=3600, log_g=5.0, z=0, alpha=0.0, magnitude=10, **kwargs
    ):
        self.t_eff = t_eff
        self.log_g = log_g
        self.z = z
        self.alpha = alpha
        self.magnitude = magnitude
        super().__init__(**kwargs, name="phoenix")
        self.stellar_target = True

        path = pathlib.Path(__file__).parent.parent.resolve()

        # create data directory if it doesn't exist:
        pathlib.Path(path.joinpath('data')).mkdir(parents=False, exist_ok=True)

        wavelength_path = path.joinpath('data').joinpath('WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')

        if t_eff in self.valid_t and log_g in self.valid_g and z in self.valid_z and alpha in self.valid_a:
            if not wavelength_path.is_file():
                print("Download Phoenix wavelength file...")
                url = "ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
                with urllib.request.urlopen(url) as response, open(wavelength_path, "wb") as out_file:
                    data = response.read()
                    out_file.write(data)

            self.wl_data = fits.getdata(wavelength_path) / 10000.0

            baseurl = (
                "ftp://phoenix.astro.physik.uni-goettingen.de/"
                "HiResFITS/PHOENIX-ACES-AGSS-COND-2011/"
                "Z-{0:{1}2.1f}{2}{3}/".format(
                    z,
                    "+" if z > 0 else "-",
                    "" if alpha == 0 else ".Alpha=",
                    "" if alpha == 0 else "{:+2.2f}".format(alpha),
                )
            )
            url = (
                    baseurl + "lte{0:05}-{1:2.2f}-{2:{3}2.1f}{4}{5}."
                              "PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format(
                t_eff,
                log_g,
                z,
                "+" if z > 0 else "-",
                "" if alpha == 0 else ".Alpha=",
                "" if alpha == 0 else "{:+2.2f}".format(alpha),
            )
            )
            # TODO: this fix is needed to avoid double --, so something in the above statement is
            # not quite right
            url = url.replace("--", "-")
            spectrum_path = path.joinpath('data').joinpath(url.split("/")[-1])

            if not spectrum_path.is_file():
                print(f"Download Phoenix spectrum from {url}...")
                with urllib.request.urlopen(url) as response, open(spectrum_path, "wb") as out_file:
                    print("Trying to download:" + url)
                    data = response.read()
                    out_file.write(data)

            self.spectrum_data = 0.1 * fits.getdata(spectrum_path)  # convert ergs/s/cm^2/cm to uW/m^2/um
            self.spectrum_data *= calc_flux_scale(self.wl_data, self.spectrum_data, self.magnitude)
            self.ip_spectra = scipy.interpolate.interp1d(self.wl_data, self.spectrum_data)
        else:
            print("Valid values are:")
            print("T: ", *self.valid_t)
            print("log g: ", *self.valid_g)
            print("Z: ", *self.valid_z)
            print("alpha: ", *self.valid_a)
            raise ValueError("Invalid parameter for M-dwarf spectrum ")

    def get_spectral_density(self, wavelength):
        idx = np.logical_and(self.wl_data > np.min(wavelength), self.wl_data < np.max(wavelength))
        return self.wl_data[idx], self.ip_spectra(self.wl_data[idx])
