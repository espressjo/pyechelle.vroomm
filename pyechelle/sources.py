""" Spectral sources

Implementing various spectral sources that can be used in pyechelle.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np
    import pyechelle.sources as sources
    from pyechelle.simulator import available_sources

    available_sources.remove('CSVSource')
    available_sources.remove('SynphotSource')

    fig, ax = plt.subplots(len(available_sources), 1, figsize=(9, len(available_sources) * 2.5), sharex=True)
    fig.suptitle('Supported source functions')
    for i, source_name in enumerate(available_sources):
        wavelength = np.linspace(0.4999, 0.501, 1000, dtype=float)
        source = getattr(sources, source_name)()
        sd = source.get_counts(wavelength, 1)
        if source.list_like:
            if isinstance(sd, tuple):
                ax[i].vlines(sd[0], [0]*len(sd[1]), sd[1])
            else:
                ax[i].vlines(wavelength, [0]*len(sd), sd)
        else:
            if isinstance(sd, tuple):
                ax[i].plot(*sd)
            else:
                ax[i].plot(wavelength, sd)
        ax[i].set_title(source_name)
        ax[i].set_ylabel("")
        ax[i].set_yticks([])
    ax[-1].set_xlabel("Wavelength [microns]")
    plt.tight_layout()
    plt.show()
"""
from __future__ import annotations

import numbers
import pathlib
import time
import urllib.request
from typing import Sequence

import astropy.io.fits as fits
import astropy.units as u
import scipy
from astropy import constants
import numpy as np
import pandas as pd
import synphot
import synphot.binning
import synphot.units
from joblib import Memory
from numpy.typing import ArrayLike

try:
    from astroquery.nist import Nist
except ImportError:
    Nist = None

path = pathlib.Path(__file__).parent.resolve()
cache_path = path.joinpath(".cache")
# create data directory if it doesn't exist:
pathlib.Path(cache_path).mkdir(parents=False, exist_ok=True)
memory = Memory(cache_path, verbose=0)


@memory.cache
def pull_catalogue_lines(
        min_wl: float | u.Quantity["length"],
        max_wl: float | u.Quantity["length"],
        catalogue: str = "Th",
        wavelength_type: str = "vacuum",
) -> tuple[u.Quantity[u.micron], np.ndarray]:
    """Reads NIST catalogue lines between min_wl and max_wl of catalogue.

    Args:
        min_wl: minimum wavelength bound [micron] or Quantity with length units
        max_wl: maximum wavelength bound [micron] or Quantity with length units
        catalogue: abbreviation of element, e.g. 'Th', 'Ar', etc.
        wavelength_type: either 'var+air' or 'vacuum'

    Returns:
        line catalogue wavelength and relative intensities. wavelength is in [micron] and dimensionless relative
        intensities.

    """
    try:
        min_wl = min_wl << u.micron
        max_wl = max_wl << u.micron
        table_lines = Nist.query(
            min_wl,
            max_wl,
            linename=catalogue,
            output_order="wavelength",
            wavelength_type=wavelength_type,
        )[["Ritz", "Rel."]]
        # replace non-numeric numbers from Rel. column
        df = table_lines.filled(0).to_pandas()
        df["Rel."] = df["Rel."].str.replace(r"[^0-9.]", "", regex=True)
        df["Rel."] = pd.to_numeric(df["Rel."], downcast="float", errors="coerce")
        df["Ritz"] = pd.to_numeric(df["Ritz"], downcast="float", errors="coerce")
        df.dropna(inplace=True)
        idx = np.logical_and(df["Rel."] > 0, df["Ritz"] > 0)
        return df["Ritz"].values[idx] << u.micron, df["Rel."].values[idx]
    except Exception as e:
        print(e)
        print(
            f"Warning: Couldn't retrieve {catalogue} catalogue data between {min_wl} and {max_wl} micron"
        )
        return np.array([]) << u.micron, np.array([])


class Source:
    """A spectral source, e.g. a star or a calibration lamp.

    There are two types of sources: list-like and continuous. List-like sources have a discrete number of lines, while
    continuous sources have a continuous spectrum and can typically be evaluated at any wavelength.

    Attributes:
        name (str): Name of the source.
        list_like (bool): True if the source is list-like, False if it is continuous.

    Notes:
        To implement a new source, subclass this class and implement the get_counts method.
    """

    def __init__(self, name: str, list_like: bool = False):
        """Constructor

        Args:
            name: Name of the source.
            list_like: True if the source is list-like, False if it is continuous.
        """
        self.name = name
        self.list_like = list_like

    def get_counts(
            self,
            wl: ArrayLike | u.Quantity["length"][float],
            integration_time: float | u.Quantity["time"],
    ) -> u.Quantity[u.ph] | tuple[u.Quantity[u.micron], u.Quantity[u.ph]]:
        """
        Get the number of photons per wavelength bin.

        Args:
            wl (Quantity or ArrayLike): Wavelengths to evaluate the spectrum at. Only the range is used.
                If ArrayLike is provided, it is assumed to be in microns.
            integration_time (Quantity or float):
                Integration time either as Quantity with time units or as float in seconds.

        Returns:
            Quantity or tuple: If the source is list-like, it returns a tuple containing two elements. The first
            element is an array of wavelengths in microns for which the emission lines exist within the provided
            wavelength range. The second element is an array of the number of photons per line for the corresponding
            wavelengths, calculated as the product of the line intensities and the integration time.
            If the source is not list-like, it returns the number of photons per wavelength bin.
        """
        raise NotImplementedError()

    def get_counts_rv(
            self,
            wl: ArrayLike | u.Quantity["length"],
            integration_time: float | u.Quantity["time"],
            rv_shift: u.Quantity["speed"] | float,
    ) -> u.Quantity[u.ph] | tuple[u.Quantity[u.micron], u.Quantity[u.ph]]:
        """
        Get the number of photons per wavelength bin, considering a radial velocity shift.

        Args:
            wl (Quantity or ArrayLike): Wavelengths to evaluate the spectrum at. Only the range is used.
                If ArrayLike is provided, it is assumed to be in microns.
            integration_time (Quantity or float): Integration time either as Quantity with time units or as float
                in seconds.
            rv_shift (Quantity or float): Radial velocity shift either as Quantity with speed units or as float in m/s.

        Returns:
            Quantity or tuple: If the source is list-like, it returns a tuple containing two elements.
            The first element is an array of wavelengths in microns for which the
            emission lines exist within the provided wavelength range, considering the radial velocity shift.
            The second element is an array of the number of photons
            per line for the corresponding wavelengths, calculated as the product of the line intensities and the
            integration time. If the source is not list-like, it returns the number of photons per wavelength bin,
            considering the radial velocity shift.
        """
        # if list-like move the returned wavelengths
        if self.list_like:
            source_wl, counts = self.get_counts(wl, integration_time)
            return (
                           1 + (rv_shift << u.m / u.s) / constants.c
                   ).value * source_wl << u.micron, counts
        else:
            return self.get_counts(
                (1 + (rv_shift << u.m / u.s) / constants.c).value * wl << u.micron,
                integration_time,
            )


class ConstantPhotonFlux(Source):
    """A source with a constant photon flux.

    Attributes:
        flux (u.Quantity): Flux in [ph/s/A].

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from pyechelle.sources import ConstantPhotonFlux
        from pyechelle.simulator import available_sources

        fig = plt.figure()
        wavelength = np.linspace(0.4, 0.8, 1000, dtype=float)
        source = ConstantPhotonFlux()
        sd = source.get_counts(wavelength, 1)
        if isinstance(sd, tuple):
            plt.plot(*sd)
        else:
            plt.plot(wavelength, sd)
        plt.suptitle('ConstantPhotonFlux')
        # ax[i].set_ylabel("")
        # ax[i].set_yticks([])
        plt.xlabel("Wavelength [microns]")
        plt.tight_layout()
        plt.show()
    """

    def __init__(
            self,
            flux: float | u.Quantity[u.ph / u.s / u.AA] = 1.0,
            name: str = "ConstantPhotonFlux",
    ):
        """
        Initialize the ConstantPhotonFlux class.

        Args:
            flux (float | Quantity): Photon flux in [ph/s/A].
            name (str, optional): Name of the source. Defaults to "ConstantPhotonFlux".
        """
        super().__init__(name=name, list_like=False)
        self.flux = flux << u.ph / u.s / u.AA

    def get_counts(
            self,
            wl: ArrayLike | u.Quantity["length"],
            integration_time: u.Quantity[u.s] | float,
    ) -> u.Quantity[u.ph]:
        """Get the number of photons per wavelength bin.

        Args:
            wl (Quantity or ArrayLike): Wavelengths to evaluate the spectrum at. If ArrayLike is provided,
                it is assumed to be in microns.
            integration_time (Quantity or float): Integration time either as Quantity with time units or as float [s].

        Returns:
            Quantity: Number of photons per wavelength bin.
        """
        wl_edges = synphot.binning.calculate_bin_edges(wl)
        wl_widths = synphot.binning.calculate_bin_widths(wl_edges)
        return (self.flux * (wl_widths << u.micron) * (integration_time << u.s)).to(
            u.ph
        )

    def __str__(self):
        return f"{self.name}: {self.flux} [ph/s/A]"


class ConstantFlux(Source):
    """A source with a constant flux measured in [microW/micron].

    Attributes:
        flux (u.Quantity): Flux in [microW/micron].
    """

    def __init__(
            self, flux: float | u.Quantity[u.erg / u.AA] = 0.001, name: str = "ConstantFlux"
    ):
        super().__init__(name, list_like=False)
        self.flux = flux << u.uW / u.micron

    def get_counts(
            self,
            wl: u.Quantity["lengths"] | ArrayLike[float],
            integration_time: u.Quantity[u.s] | float,
    ) -> u.Quantity[u.ph]:
        """Get the number of photons per wavelength bin.

        Args:
            wl (Quantity or ArrayLike): Wavelengths to evaluate the spectrum at. If ArrayLike is provided,
                it is assumed to be in microns.
            integration_time (Quantity or float): Integration time either as Quantity with time units or as float [s].

        Returns:
            Quantity: Number of photons per wavelength bin.
        """
        wl_edges = synphot.binning.calculate_bin_edges(wl << u.micron)
        wl_widths = synphot.binning.calculate_bin_widths(wl_edges)
        return (self.flux * (wl_widths << u.micron) * (integration_time << u.s)).to(
            u.ph, u.spectral_density(wl << u.micron)
        )

    def __str__(self):
        return f"{self.name}: {self.flux} [microW/micron]"


class BlackBody(Source):
    """A stellar black body spectrum.

    Attributes:
        temperature (u.Quantity): Temperature in [K].
        v_mag (u.Quantity): V-band magnitude.
        collection_area (u.Quantity): Collection area in [cm^2].
    """

    def __init__(
            self,
            temperature: float | u.Quantity[u.K] = 5780 << u.K,
            v_mag: float | u.Quantity[u.VEGAMAG] = 10 << synphot.units.VEGAMAG,
            collection_area: float | u.Quantity[u.cm ** 2] = 100.0 << u.cm ** 2,
            name: str = "BlackBody",
    ):
        super().__init__(name, list_like=False)
        self.temperature = temperature << u.K
        self.v_mag = v_mag << synphot.units.VEGAMAG
        self.collection_area = collection_area << u.cm ** 2
        self.sp = synphot.SourceSpectrum(
            synphot.BlackBody1D, temperature=self.temperature
        )
        self.bp = synphot.SpectralElement.from_filter("johnson_v")
        self.vega = synphot.SourceSpectrum.from_vega()  # For unit conversion
        self.sp_norm = self.sp.normalize(self.v_mag, self.bp, vegaspec=self.vega)

    def get_counts(
            self,
            wl: u.Quantity["lengths"] | ArrayLike,
            integration_time: u.Quantity[u.s] | float,
    ) -> u.Quantity[u.ph]:
        """Get the number of photons per wavelength bin.

        Args:
            wl (Quantity or ArrayLike): Wavelengths to evaluate the spectrum at. If ArrayLike is provided,
                it is assumed to be in microns.
            integration_time (Quantity or float): Integration time either as Quantity with time units or as float [s].

        Returns:
            Quantity: Number of photons per wavelength bin.
        """
        wl_edges = synphot.binning.calculate_bin_edges(wl << u.micron)
        wl_widths = synphot.binning.calculate_bin_widths(wl_edges)
        return (
                self.sp_norm(wl << u.micron)
                * wl_widths
                * (integration_time << u.s)
                * self.collection_area
        ).to(u.ph)

    def __str__(self):
        return f"{self.name}: {self.temperature} K, {self.v_mag} mag"


class LFC(Source):
    r"""Laser Frequency Comb.

    In frequency, the spectrum of a LFC is:

    .. math::
        f = f_0 + m \cdot f_{rep}

    where f_0 is the carrier frequency, f_rep is the repetition rate and m is the peak interference number.


    Attributes:
        f_0 (u.Quantity): Carrier frequency in [GHz].
        f_rep (u.Quantity): Repetition rate in [GHz].
        brightness (u.Quantity): Brightness in [ph/s].

    """

    def __init__(
            self,
            f_0: float | u.Quantity["frequency"] = 7.35 * u.GHz,
            f_rep: float | u.Quantity["frequency"] = 10.0 * u.GHz,
            brightness: float | u.Quantity[u.ph / u.s] = 1e5,
            name="LFC",
    ):
        """
        Initializes the LFC class.

        Args:
            f_0: Carrier/Offset frequency either as a float (assumed to be in GHz) or as a Quantity with frequency units.
            f_rep: Repetition rate either as a float (assumed to be in GHz) or as a Quantity with frequency units.
            brightness: Brightness per line either as a float (assumed to be in photons per second) or as a Quantity with photons per time units.
            name: name of the source, defaults to "LFC".
        """
        super().__init__(name, list_like=True)
        self.f_0 = f_0 << u.GHz
        self.f_rep = f_rep << u.GHz
        self.brightness = brightness << u.ph / u.s

    def peak_wavelength_lfc(self, m) -> u.Quantity[u.micron]:
        """Calculate the peak wavelength of the LFC.

        Args:
            m: peak interference number

        Returns:
            peak wavelength in [micron]
        """
        return (self.f_0 + m * self.f_rep).to(u.micron, equivalencies=u.spectral())

    def get_counts(
            self,
            wl: ArrayLike | u.Quantity["length"],
            integration_time: float | u.Quantity["time"],
    ) -> tuple[u.Quantity[u.micron], u.Quantity[u.ph]]:
        """Get the number of photons per line.

        Args:
            wl (Quantity, float): Wavelengths to evaluate the spectrum at. Only the range is used.
            float [micron]. Here, only the range is used.
            integration_time (Quantity, float): Integration time either as Quantity with time units or as float [s].

        Returns:
            tuple: wavelengths and number of photons per line.
        """
        # calculate the minimum and maximum m for the given wavelength range
        wl_min = wl.min() << u.micron
        wl_max = wl.max() << u.micron

        min_m = np.ceil(
            (wl_max.to(u.GHz, equivalencies=u.spectral()) - self.f_0) / self.f_rep
        ).value
        max_m = np.floor(
            (wl_min.to(u.GHz, equivalencies=u.spectral()) - self.f_0) / self.f_rep
        ).value
        # generate corresponding wavelengths and intensities
        return (
            self.peak_wavelength_lfc(np.arange(min_m, max_m)).to(u.micron),
            np.ones_like(np.arange(min_m, max_m), dtype=float)
            * (self.brightness << u.ph / u.s)
            * (integration_time << u.s),
        )

    def __str__(self):
        return f"{self.name}: f0={self.f_0} GHz , f_rep={self.f_rep} GHz, {self.brightness} ph/s/line"


class IdealEtalon(Source):
    r"""Fabry-Perot etalon.

    Implements spectrum of an ideal (i.e. dispersion-free and unresolved) Fabry-Perot etalon.
    This means, the peak wavelength are at:

    .. math::
        \lambda_{peak} = \frac{d \cdot n \cdot \cos{(\theta)}}{m}

    Attributes:
        d (u.Quantity): Mirror distance in [mm].
        n (float): Refractive index between mirrors.
        theta (u.Quantity): Angle of incidence onto mirrors in [rad].
        n_photons (u.Quantity): Number of photons per peak per second [u.ph/u.s].
    """

    def __init__(
            self,
            d: float | u.Quantity["length"] = 5.0,
            n: float = 1.0,
            theta: float | u.Quantity["angle"] = 0.0,
            n_photons: float | u.Quantity[u.ph / u.s] = 1000.0,
            name: str = "IdealEtalon",
    ):
        """
        Initializes the IdealEtalon class.

        Args:
            d: Mirror distance. Either a float or a Quantity with length units. When a float is given, it is
            assumed to be in mm. Defaults to 5.0.
            n: Refractive index between mirrors. Defaults to 1.0.
            theta: Angle of incidence onto mirrors. Either a float or a Quantity with angle units.
            When a float is given, it is assumed to be in radians. Defaults to 0.0.
            n_photons: Number of photons per peak per second. Either a float or a Quantity with photon units.
            When a float is given, it is assumed to be in photons per second. Defaults to 1000.0.
            name: Name of the source. Defaults to "IdealEtalon".
        """
        super().__init__(name=name, list_like=True)
        self.d = d << u.mm
        self.n = n
        self.theta = theta << u.rad
        self.n_photons = n_photons << u.ph / u.s

    @staticmethod
    def peak_wavelength_etalon(
            m,
            d: float | u.Quantity["length"] = 10.0,
            n: float = 1.0,
            theta: float | u.Quantity["angle"] = 0.0,
    ) -> u.Quantity[u.nm]:
        """
        Calculate the peak wavelength of an etalon.
        Args:
            m: peak interference number
            d: mirror distance. Either a float or a Quantity with length units.
            When a float is given, it is assumed to be in [mm]. Defaults to 10.0.
            n: refractive index between mirrors. Defaults to 1.0.
            theta: angle of incidence onto mirrors. Either a float or a Quantity with angle units. Defaults to 0.0.

        Returns:
            peak wavelength in nm

        """
        return ((d << u.mm) * n * np.cos((theta << u.rad)) / m).to(u.nm)

    def get_counts(
            self,
            wl: ArrayLike | u.Quantity["length"],
            integration_time: float | u.Quantity["time"],
    ) -> tuple[u.Quantity[u.micron], u.Quantity[u.ph]]:
        min_m = np.ceil(
            (self.d * np.cos(self.theta) / np.max(wl << u.micron)).to_value(
                u.dimensionless_unscaled
            )
        )
        max_m = np.floor(
            (self.d * np.cos(self.theta) / np.min(wl << u.micron)).to_value(
                u.dimensionless_unscaled
            )
        )
        intensity = (
                np.ones_like(np.arange(min_m, max_m), dtype=float)
                * self.n_photons
                * (integration_time << u.s)
        )
        return (
            self.peak_wavelength_etalon(
                np.arange(min_m, max_m), self.d, self.n, self.theta
            ).to(u.micron),
            intensity,
        )

    def __str__(self):
        return f"{self.name}: {self.d}, {self.n}, {self.theta}, {self.n_photons}"


class ResolvedEtalon(Source):
    r"""Fabry-Perot etalon with resolved line widths.

    Implements spectrum of a Fabry-Perot etalon with resolved line widths.
    This means, the peak wavelength are at:

    .. math::
        \lambda_{peak} = \frac{d \cdot n \cdot \cos{(\theta)}}{m}

    and the line width is determined by the finesse of the etalon, which is defined as the ratio of the free spectral
    range (FSR) to the line width:

    .. math::
        \mathcal{F} = \frac{FSR}{\Delta \lambda}

    The transmission function of the etalon is given by:

    .. math::
        T = \frac{1}{1 + F \cdot \sin^2{(\frac{2 \cdot \pi \cdot n \cdot d \cdot cos(\theta)}{\lambda})}}

    where F is the *coefficient of finesse*, which is related to the *finesse* by:

    .. math::
        \mathcal{F} = \frac{\pi}{2 \cdot arcsin(1/\sqrt{F})}

    The coefficienct of finesse is related to the reflectivity of the mirrors:

    .. math::
        F = \frac{4 \cdot \sqrt{R_1 \cdot R_2}}{(1 - \sqrt{R_1 \cdot R_2})^2}


    .. TODO:: follow https://opg.optica.org/oe/fulltext.cfm?uri=oe-24-15-16366&id=346183#e31 for more precise calculation
    .. TODO :: add the possibility to add wavelength dependent reflectivity

    Attributes:
        d (u.Quantity): Mirror distance in [mm].
        n (float): Refractive index between mirrors.
        theta (u.Quantity): Angle of incidence onto mirrors in [rad].
        n_photons (u.Quantity): Number of photons per peak per second [u.ph/u.s].
        finesse (float): Finesse of the etalon.
        R1 (float): Reflectivity of mirror 1.
        R2 (float): Reflectivity of mirror 2.
    """

    def __init__(
            self,
            d: float | u.Quantity["length"] = 5.0,
            n: float = 1.0,
            theta: float | u.Quantity["angle"] = 0.0,
            n_photons: float | u.Quantity[u.ph / u.s] = 1000.0,
            finesse: float | None = 100.0,
            R1: float | None = None,
            R2: float | None = None,
            name: str = "ResolvedEtalon",
    ):
        """
        Initializes the ResolvedEtalon class.

        Either R1 and R2 or finesse must be given. If R1 and R2 are given, finesse is calculated from them. If only R1
        is given, R2 is assumed to be the same. If finesse is given, R1 and R2 are assumed to be the same.

        Args:
            d: Mirror distance. Either a float or a Quantity with length units. When a float is given, it is
            assumed to be in mm. Defaults to 5.0.
            n: Refractive index between mirrors. Defaults to 1.0.
            theta: Angle of incidence onto mirrors. Either a float or a Quantity with angle units.
            When a float is given, it is assumed to be in radians. Defaults to 0.0.
            n_photons: Number of photons per peak per second. Either a float or a Quantity with photon units.
            When a float is given, it is assumed to be in photons per second. Defaults to 1000.0.
            finesse: Finesse of the etalon. Defaults to 100.0.
            R1: Reflectivity of mirror 1. Defaults to None
            R2: Reflectivity of mirror 2. Defaults to None
            name: Name of the source. Defaults to "ResolvedEtalon".
        """
        super().__init__(name=name, list_like=False)
        self.d = d << u.mm
        self.n = n
        self.theta = theta << u.rad
        self.n_photons = n_photons << u.ph / u.s
        if finesse is not None:
            self.finesse = finesse
            if R1 is not None:
                print("Warning: R1 and R2 are ignored, because finesse is given.")
            self.R1 = None
            self.R2 = None
        else:
            assert R1 is not None, "Either finesse or at least R1 must be given."
            self.R1 = R1
            self.R2 = R2 if R2 is not None else R1
            coeff_finesse = (
                    4 * np.sqrt(self.R1 * self.R2) / (1 - np.sqrt(self.R1 * self.R2)) ** 2
            )
            self.finesse = np.pi / (2 * np.arcsin(1 / np.sqrt(coeff_finesse)))

    def _transmission(self, wl: ArrayLike | u.Quantity["length"]) -> u.Quantity:
        """
        Calculate the transmission function of the etalon.

        Args:
            wl: Wavelength in [nm].

        Returns:
            Transmission function.
        """
        return 1 / (
                1
                + self.finesse
                * np.sin(
            (
                    2 * np.pi * self.n * self.d * np.cos(self.theta) / (wl << u.micron)
            ).value
        )
                ** 2
        )

    def get_counts(
            self,
            wl: ArrayLike | u.Quantity["length"][float],
            integration_time: float | u.Quantity["time"],
    ) -> u.Quantity[u.ph] | tuple[u.Quantity[u.micron], u.Quantity[u.ph]]:
        """Get the etalon spectrum

        Args:
            wl (Quantity, float): Wavelengths to evaluate the spectrum at, either as Quantity with length units or as
            float [micron]. Here, only the range is used.
            integration_time (Quantity, float): Integration time either as Quantity with time units or as float [s].

        Returns:
            tuple: wavelengths and number of photons per line.
        """
        return (wl << u.micron), (
                self._transmission(wl << u.micron)
                * self.n_photons
                * (integration_time << u.s)
        )

    def __str__(self):
        return f"{self.name}: d={self.d}, n={self.n}, theta={self.theta}, n_photons={self.n_photons}, finesse={self.finesse}, R1={self.R1}, R2={self.R2}"


class ArcLamp(Source):
    """An arc lamp spectrum.

    This class provides a convenient handling of arc lamp spectra.
    Specifying the element name(s), it downloads the line list(s) from NIST.
    Multiple elements can be specified together with a scaling factor for each element to get a combined spectrum.
    The scaling factor is needed, because the NIST database only contains relative intensities.
    An overall lamp brightness can also be specified which scales the brightest line to this value.

    Attributes:
        elements (list): List of element names.
        scaling_factors (list): List of scaling factors for each element.
        brightness (u.Quantity): Overall lamp brightness scaling factor in [ph/s] per line.

    Examples:
        A ThNe lamp and default scaling factors and brightness:
        >>> arc_thar = ArcLamp()
        >>> print(arc_thar)
        ArcLamp: ThAr, [1.0, 1.0], 100000.0 ph / s

        A ThAr lamp with a scaling factor of 1 for Th and 1 for Ar and a brightness of 1000000.0 photons per second.:
        >>> arc_thne = ArcLamp(["Th", "Ne"], scaling_factors=[1, 1], brightness=1E6)
        >>> print(arc_thne)
        ArcLamp: ThNe, [1, 1], 1000000.0 ph / s
    """

    def __init__(
            self,
            elements: Sequence[str] = ("Th", "Ar"),
            scaling_factors: Sequence[float] | float = 1.0,
            brightness: float | u.Quantity[u.ph / u.s] = 1e5 << u.ph / u.s,
            name: str = "ArcLamp",
    ):
        """Initialize the ArcLamp class.

        Args:
            elements: List of element names.
            scaling_factors: List of scaling factors for each element.
            brightness: Overall lamp brightness scaling factor.
            name: Name of the source.
        """
        super().__init__(name, list_like=True)
        if isinstance(scaling_factors, numbers.Number):
            scaling_factors = [scaling_factors] * len(elements)
        assert len(elements) == len(
            scaling_factors
        ), "Length of elements and scaling_factors must be the same."
        self.elements = elements
        self.scaling_factors = scaling_factors
        self.brightness = brightness << u.ph / u.s

    def get_counts(
            self,
            wl: ArrayLike | u.Quantity["length"],
            integration_time: u.Quantity[u.s] | float,
    ) -> tuple[u.Quantity[u.micron], u.Quantity[u.ph]]:
        """Get the number of photons per line.

        Args:
            wl (Quantity, float): Wavelengths to evaluate the spectrum at, either as Quantity with length units or as
            float [micron]. Here, only the range is used.
            integration_time (Quantity, float): Integration time either as Quantity with time units or as float [s].

        Returns:
            tuple: wavelengths and number of photons per line.
        """
        combined_wl = np.array([]) << u.micron
        combined_counts = np.array([]) << u.ph
        for element, scaling in zip(self.elements, self.scaling_factors):
            lines, intensities = pull_catalogue_lines(wl.min(), wl.max(), element)
            if len(lines) > 0:
                combined_wl = np.append(combined_wl, lines)
                combined_counts = np.append(
                    combined_counts,
                    intensities
                    / np.max(intensities)
                    * (integration_time << u.s)
                    * scaling
                    * self.brightness,
                )
        return combined_wl, combined_counts

    def __str__(self):
        return f"{self.name}: {''.join(self.elements)}, {self.scaling_factors}, {self.brightness}"


class Phoenix(Source):
    """Phoenix M-dwarf spectra.

    This class provides a convenient handling of PHOENIX M-dwarf spectra.
    For a given set of effective Temperature, log g, metalicity and alpha, it downloads the spectrum from PHOENIX ftp
    server.

    See the `original paper <http://dx.doi.org/10.1051/0004-6361/201219058>`_ for more details.


    Attributes:
        t_eff (u.Quantity[u.K]): effective temperature in [K].
        log_g (float): surface gravity
        z (float): metalicity [Fe/H]
        alpha (float): abundance of alpha elements [α/Fe]
        v_mag (u.Quantity[u.VEGAMAG]): V-band magnitude.
        collection_area (u.Quantity[u.cm**2]): Collection area in [cm^2].

    """

    valid_t = [*list(range(2300, 7000, 100)), *list((range(7000, 12200, 200)))]
    valid_g = [*list(np.arange(0, 6, 0.5))]
    valid_z = [*list(np.arange(-4, -2, 1)), *list(np.arange(-2.0, 1.5, 0.5))]
    valid_a = [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]

    def __init__(
            self,
            t_eff: int = 3600,
            log_g: float = 5.0,
            z: float = 0,
            alpha: float = 0.0,
            v_mag: float | u.Quantity[u.VEGAMAG] = 10.0,
            collection_area: float | u.Quantity["area"] = 100.0,
            name: str = "Phoenix",
    ):
        assert t_eff in self.valid_t, f"Not a valid effective Temperature {t_eff}"
        assert log_g in self.valid_g, f"Not a valid log g value {log_g}"
        assert alpha in self.valid_a, f"Not a valid alpha value {alpha}"
        assert z in self.valid_z, f"Not a valid metalicity value {z}"
        if not np.isclose(alpha, 0.0):
            assert 3500.0 <= t_eff <= 8000.0 and -3.0 <= z <= 0.0, (
                "PHOENIX parameters are not valid. Please check them again. "
            )
        self.t_eff = t_eff << u.K
        self.log_g = log_g
        self.z = z
        self.alpha = alpha
        self.v_mag = v_mag << synphot.units.VEGAMAG
        self.collection_area = collection_area << u.cm ** 2
        super().__init__(name=name, list_like=False)

        wavelength_path = cache_path.joinpath("WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")

        if not wavelength_path.is_file():
            print("Download Phoenix wavelength file...")
            with urllib.request.urlopen(self.get_wavelength_url()) as response, open(
                    wavelength_path, "wb"
            ) as out_file:
                data = response.read()
                out_file.write(data)

        self.wl_data = fits.getdata(wavelength_path) << u.AA

        url = self.get_spectrum_url(t_eff, alpha, log_g, z)
        spectrum_path = cache_path.joinpath(url.split("/")[-1])

        if not spectrum_path.is_file():
            print(f"Download Phoenix spectrum from {url}...")
            with urllib.request.urlopen(url) as response, open(
                    spectrum_path, "wb"
            ) as out_file:
                print("Trying to download:" + url)
                data = response.read()
                out_file.write(data)
        self.flux = fits.getdata(spectrum_path) << u.erg / u.s / u.cm ** 2 / u.cm

        spec = synphot.SourceSpectrum(
            synphot.Empirical1D, points=self.wl_data, lookup_table=self.flux
        )
        bp = synphot.SpectralElement.from_filter("johnson_v")
        vega = synphot.SourceSpectrum.from_vega()  # For unit conversion
        sp_norm = spec.normalize(
            self.v_mag, bp, vegaspec=vega, wavelengths=self.wl_data
        )
        self.spectrum_data = sp_norm

    @staticmethod
    def get_wavelength_url():
        return "ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"

    @staticmethod
    def get_spectrum_url(t_eff, alpha, log_g, z):
        zstring = f"{'+' if z > 0 else '-'}{abs(z):2.1f}"
        alphastring = "" if np.isclose(alpha, 0.0) else f".Alpha={alpha:+2.2f}"

        url = (
            f"ftp://phoenix.astro.physik.uni-goettingen.de/"
            f"HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z{zstring}{alphastring}/lte{t_eff:05}-{log_g:2.2f}{zstring}"
            f"{alphastring}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
        )
        return url

    def get_counts(
            self,
            wl: ArrayLike | u.Quantity["length"],
            integration_time: u.Quantity[u.s] | float,
    ) -> u.Quantity[u.ph] | tuple[u.Quantity[u.AA], u.Quantity[u.ph]]:
        """Get the number of photons per wavelength bin.

        Args:
            wl (Quantity): Wavelengths to evaluate the spectrum at.
            integration_time (Quantity): Integration time.

        Returns:
            Quantity: Number of photons per wavelength bin.
        """
        # idx = np.logical_and(self.wl_data > np.min(wl), self.wl_data < np.max(wl))
        wl_edges = synphot.binning.calculate_bin_edges(wl << u.micron)
        wl_widths = synphot.binning.calculate_bin_widths(wl_edges)
        return wl << u.micron, (
                self.spectrum_data(wl << u.micron)
                * wl_widths
                * (integration_time << u.s)
                * self.collection_area
        ).to(u.ph)

    def __str__(self):
        return (
                self.name
                + f": t={self.t_eff},g={self.log_g},z={self.z},a={self.alpha},mag={self.v_mag}"
        )


class CSVSource(Source):
    """A source with a spectrum from a CSV file.

    The .csv file must contain two columns: wavelength and flux.
    The spectrum can either be list-like or continuous - this must be specified.
    Also, the flux can either be given as a flux density (e.g. [erg/s/A], or [ph/s/A]) or as a flux
    (e.g. [erg/s], or [ph/s]). This will be inferred from the unit of the flux column.
    The units are specified using the astropy units package / notation.
    To read the .csv file, pandas is used, so any kwargs for pandas.read_csv can be passed to the constructor.

    If the source is a stellar target, in addition to the flux, also the collection area and a V-band magnitude have to be specified.

    Attributes:
        file_path (pathlib.Path): Path to the .csv file.
        list_like (bool): True if the source is list-like, False if it is continuous.
        wavelength_units (str): Units of the wavelength column.
        flux_units (str): Units of the flux column.

    Examples:
        >>> csv_source = CSVSource("../tests/test_data/test_source1.csv", list_like=False, wavelength_units="nm", flux_units="ph/s/AA", skiprows=1)
        >>> print(csv_source.get_counts([500, 600] * u.nm, 1 * u.s))
        (<Quantity [500., 600.] nm>, <Quantity [1000.,    0.] ph>)
        >>> print(csv_source)
        ../tests/test_data/test_source1.csv, False, nm, ph / (Angstrom s)

    """

    def __init__(
            self,
            file_path: str | pathlib.Path,
            wavelength_units: str | u.Unit = "nm",
            flux_units: str | u.Unit = "ph/s/AA",
            list_like: bool = False,
            v_mag: float = None,
            collection_area: float | u.Quantity = None,
            name: str = "CSVSource",
            **pandas_kwargs,
    ):
        super().__init__(name, list_like)
        self.file_path = pathlib.Path(file_path)
        self.wavelength_units = u.Unit(wavelength_units)
        self.flux_units = u.Unit(flux_units)
        self.pandas_kwargs = pandas_kwargs
        if v_mag is not None:
            self.v_mag = v_mag << synphot.units.VEGAMAG
        else:
            self.v_mag = None

        if collection_area is not None:
            self.collection_area = collection_area << u.cm ** 2
        else:
            self.collection_area = None

        # check that if v_mag is given, collection_area is also given and vice versa
        if self.v_mag is not None and self.collection_area is None:
            raise ValueError(
                "If V-band magnitude is given, collection area must also be given."
            )
        if self.collection_area is not None and self.v_mag is None:
            raise ValueError(
                "If collection area is given, V-band magnitude must also be given."
            )

        # read the csv file and convert the columns to astropy quantities
        self.data = pd.read_csv(
            file_path, names=["wavelength", "flux"], **pandas_kwargs
        )
        self.data["wavelength"] = (
                self.data["wavelength"].values << self.wavelength_units
        )
        self.data["flux"] = self.data["flux"] << self.flux_units
        if not self.list_like:
            self.interpolated_flux = scipy.interpolate.interp1d(
                self.data["wavelength"],
                self.data["flux"],
                fill_value=0.0,
                bounds_error=False,
            )

    def get_counts(
            self,
            wl: u.Quantity[u.micron] | ArrayLike[float],
            integration_time: u.Quantity[u.s] | float,
    ) -> tuple[u.Quantity[u.micron], u.Quantity[u.ph]]:
        """Get the number of photons per wavelength bin.

        Args:
            wl (Quantity, float): Wavelengths to evaluate the spectrum at. Only the range is used.
            integration_time (Quantity, float): Integration time either as Quantity with time units or as float [s].

        Returns:
            tuple: wavelengths and number of photons per line.
        """
        min_wl = np.min(wl) << u.micron
        max_wl = np.max(wl) << u.micron

        idx = np.logical_and(
            self.data["wavelength"].values > min_wl,
            self.data["wavelength"].values < max_wl,
        )

        if self.list_like:
            return (
                self.data["wavelength"][idx].values.to(u.micron),
                self.data["flux"][idx].values * (integration_time << u.s),
            )
        else:
            # stellar target
            if self.v_mag is not None:
                sp = synphot.SourceSpectrum(
                    synphot.Empirical1D,
                    points=self.data["wavelength"],
                    lookup_table=self.data["flux"],
                )
                bp = synphot.SpectralElement.from_filter("johnson_v")
                vega = synphot.SourceSpectrum.from_vega()
                sp_norm = sp.normalize(
                    self.v_mag, bp, vegaspec=vega, wavelengths=self.data["wavelength"]
                )
                return self.data["wavelength"].to(u.micron), (
                        sp_norm(self.data["wavelength"])
                        * (integration_time << u.s)
                        * self.collection_area
                ).to(u.ph)
            else:
                # flux is given as flux density
                if self.flux_units.is_equivalent(
                        u.ph / u.s / u.nm, equivalencies=u.spectral_density(1 * u.AA)
                ):
                    wl_edges = synphot.binning.calculate_bin_edges(
                        wl << self.wavelength_units
                    )
                    wl_widths = synphot.binning.calculate_bin_widths(wl_edges)

                    return wl, (
                            self.interpolated_flux(wl << self.wavelength_units)
                            * self.flux_units
                            * wl_widths
                            * (integration_time << u.s)
                    ).to(u.ph, equivalencies=u.spectral_density(wl))
                elif u.get_physical_type(self.flux_units) == "power/radiant flux":
                    return self.data["wavelength"].values.to(u.micron)[idx], (
                            self.data["flux"][idx].values * (integration_time << u.s)
                    ).to(u.ph)
                # if flux is given in ph/s
                elif self.flux_units.is_equivalent(u.ph / u.s):
                    return self.data["wavelength"].values.to(u.micron)[idx], self.data[
                                                                                 "flux"
                                                                             ][idx].values * (integration_time << u.s)
                else:
                    raise ValueError(f"Flux units {self.flux_units} not recognized.")

    def __str__(self):
        return f"{self.file_path}, {self.list_like}, {self.wavelength_units}, {self.flux_units}"


class LineList(Source):
    """A list of emission lines.

    This class provides a convenient handling of a list of emission lines.
    The spectrum is defined by a list of wavelengths and a list of intensities.

    Attributes:
        wavelengths (u.Quantity[u.micron]): List of wavelengths in [micron].
        intensities (u.Quantity[u.ph/u.s]): List of intensities in [ph/s].

    Examples:
        >>> balmer_lines = LineList([656.3, 486.1, 434.0, 410.2, 397.0, 388.9, 383.5, 364.5] * u.nm, 1 * u.ph/u.s, "BalmerLines")
        >>> print(balmer_lines)
        BalmerLines: [0.6563 0.4861 0.434  0.4102 0.397  0.3889 0.3835 0.3645] micron, [1. 1. 1. 1. 1. 1. 1. 1.] ph / s
    """

    def __init__(
            self,
            wavelengths: ArrayLike | u.Quantity["length"] = (
                                                                    656.3,
                                                                    486.1,
                                                                    434.0,
                                                                    410.2,
                                                                    397.0,
                                                                    388.9,
                                                                    383.5,
                                                                    364.5,
                                                            )
                                                            * u.nm,
            intensities: ArrayLike | u.Quantity[u.ph / u.s] | float = 1.0 * u.ph / u.s,
            name: str = "LineList",
    ):
        """
        Initializes the LineList class.

        Args:
            wavelengths (ArrayLike or Quantity): List of wavelengths of the emission lines.
                If ArrayLike is provided, it is assumed to be in microns.
            intensities (ArrayLike or Quantity or float): List of intensities of the emission lines.
                If ArrayLike is provided, it is assumed to be in photons per second.
                If a float or int is provided, it is assumed to be the same for all lines.
            name (str, optional): Name of the source. Defaults to "LineList".
        """
        super().__init__(name, list_like=True)
        # convert to microns
        self.wavelengths = wavelengths << u.micron
        # convert to photons per second
        if isinstance(intensities, numbers.Number):
            intensities = [intensities] * len(wavelengths)
        # check if intensities is a single Quantity
        if isinstance(intensities, u.Quantity):
            if intensities.isscalar:
                intensities = [intensities] * len(wavelengths)

        self.intensities = intensities << u.ph / u.s

    def get_counts(
            self,
            wl: ArrayLike | u.Quantity["length"],
            integration_time: float | u.Quantity["time"],
    ) -> tuple[u.Quantity[u.micron], u.Quantity[u.ph]]:
        """
        Get the number of photons per line.

        Args:
            wl (Quantity or ArrayLike): Wavelengths to evaluate the spectrum at. Only the range is used.
                If ArrayLike is provided, it is assumed to be in microns.
            integration_time (Quantity or float): Integration time either as Quantity with time units or as float in seconds.

        Returns:
            tuple: A tuple containing two elements. The first element is an array of wavelengths in microns for which the
            emission lines exist within the provided wavelength range. The second element is an array of the number of photons
            per line for the corresponding wavelengths, calculated as the product of the line intensities and the integration time.
        """
        idx = np.logical_and(
            self.wavelengths >= np.min(wl << u.micron),
            self.wavelengths <= np.max(wl << u.micron),
        )

        return self.wavelengths[idx] << u.micron, self.intensities[idx] * (
                integration_time << u.s
        )

    def __str__(self):
        return f"{self.name}: {self.wavelengths}, {self.intensities}"


class SynphotSource(Source):
    """A source with a spectrum from synphot."""

    def __init__(
            self,
            spectrum: synphot.SourceSpectrum,
            collection_area: float | u.Quantity[u.cm ** 2] = 100.0,
            name: str = "SynphotSource",
    ):
        super().__init__(name, list_like=False)
        self.collection_area = collection_area << u.cm ** 2
        self.spectrum = spectrum

    def get_counts(
            self,
            wl: ArrayLike | u.Quantity["length"],
            integration_time: u.Quantity[u.s] | float,
    ) -> tuple[u.Quantity[u.micron], u.Quantity[u.ph]]:
        wl_edges = synphot.binning.calculate_bin_edges(wl << u.micron)
        wl_widths = synphot.binning.calculate_bin_widths(wl_edges)
        return wl << u.micron, (
                self.spectrum(wl << u.micron)
                * wl_widths
                * (integration_time << u.s)
                * self.collection_area
        ).to(u.ph)

    def __str__(self):
        return (
            f"{self.name}: {''.join(c for c in str(self.spectrum) if c.isprintable())}"
        )
