import os
import urllib

import astropy.io.fits as fits
import numpy as np

from pyechelle.randomgen import AliasSample


class Source:
    """ A spectral source.

    This class should be subclassed to implement different spectral sources.

    Attributes:
        name (str): name of the source. This will end up in the .fits header.
        wavelength (np.ndarray, None): wavelength grid of source if applicable
        min_wl (float): lower wavelength limit [nm] (for normalization purposes)
        max_wl (float): upper wavelength limit [nm] (for normalization purposes)

    """

    def __init__(self, min_wl=599.8, max_wl=600.42, name=""):
        self.name = name
        self.wavelength = None
        self.min_wl = min_wl
        self.max_wl = max_wl
        self.list_like_source = False

    def get_spectral_density(self, wavelength):
        raise NotImplementedError()

    # def draw_wavelength(self, N):
    #     """
    #     Overwrite this function in child class !
    #     Args:
    #         N (int): number of wavelength to randomly draw
    #
    #     Returns:
    #
    #     """
    #     raise NotImplementedError()

    def apply_rv(self, rv):
        """ Apply radial velocity shift.

        Applies an RV shift to the formerly drawn wavelength.
        Args:
            rv (float): radial velocity shift [m/s]

        Returns:
            np.ndarray: shifted wavelength

        """
        self.wavelength = apply_rv(self.wavelength, rv)
        return self.wavelength

    def bin_to_wavelength(self, wl_vector):
        """ Bins random wavelength into wavelength vector.

        Args:
            wl_vector (np.ndarray): wavelength bin edges

        Returns:
            see np.histogram for details
        """
        return np.histogram(self.wavelength, wl_vector)


class Constant(Source):
    def __init__(self, intensity=0.001, **kwargs):
        super().__init__(**kwargs, name="Constant")
        self.intensity = intensity
        self.list_like_source = False

    def get_spectral_density(self, wavelength):
        return np.ones_like(wavelength) * self.intensity


class Etalon(Source):
    def __init__(self, d=5.0, n=1.0, theta=0.0, **kwargs):
        super().__init__(**kwargs, name="Etalon")
        self.d = d
        self.n = n
        self.theta = theta
        self.min_m = np.ceil(2e3 * d * np.cos(theta) / self.max_wl)
        self.max_m = np.floor(2e3 * d * np.cos(theta) / self.min_wl)
        self.list_like_source = True

    @staticmethod
    def peak_wavelength_etalon(m, d=10.0, n=1.0, theta=0.0):
        return 2e3 * d * n * np.cos(theta) / m

    def get_spectral_density(self, wavelength):
        self.min_m = np.ceil(2e3 * self.d * np.cos(self.theta) / np.max(wavelength))
        self.max_m = np.floor(2e3 * self.d * np.cos(self.theta) / np.min(wavelength))
        return self.peak_wavelength_etalon(
            np.arange(self.min_m, self.max_m), self.d, self.n, self.theta
        ), np.ones_like(np.arange(self.min_m, self.max_m))

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

    TODO:
    * recalculate spectral flux of original fits files to photons !!!!!
    """

    def __init__(
            self, t_eff=3600, log_g=5.0, z=0, alpha=0.0, data_folder="data", **kwargs
    ):
        self.t_eff = t_eff
        self.log_g = log_g
        self.z = z
        self.alpha = alpha
        super().__init__(**kwargs, name="phoenix")
        valid_T = [*list(range(2300, 7000, 100)), *list((range(7000, 12200, 200)))]
        valid_g = [*list(np.arange(0, 6, 0.5))]
        valid_z = [*list(np.arange(-4, -2, 1)), *list(np.arange(-2.0, 1.5, 0.5))]
        valid_a = [*list(np.arange(-0.2, 1.4, 0.2))]

        if t_eff in valid_T and log_g in valid_g and z in valid_z and alpha in valid_a:
            if not os.path.exists(
                    data_folder + "/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
            ):
                print("Download Phoenix wavelength file...")
                url = "ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
                with urllib.request.urlopen(url) as response, open(
                        data_folder + "/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits", "wb"
                ) as out_file:
                    data = response.read()
                    out_file.write(data)

            self.wl_data = (
                    fits.getdata(
                        str(data_folder + "/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
                    )
                    / 10000.0
            )

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

            filename = data_folder + "/" + url.split("/")[-1]

            if not os.path.exists(filename):
                print("Download Phoenix spectrum...")
                with urllib.request.urlopen(url) as response, open(
                        filename, "wb"
                ) as out_file:
                    print("Trying to download:" + url)
                    data = response.read()
                    out_file.write(data)

            self.spectrum_data = fits.getdata(filename)
            low_wl = np.argmax(self.wl_data > self.min_wl)
            high_wl = np.argmax(self.wl_data > self.max_wl)

            # low_wl = np.where(self.wl_data > self.min_wl)[0][0]
            # high_wl = np.where(self.wl_data > self.max_wl)[0][0]
            self.spectrum_data = self.spectrum_data[low_wl:high_wl]
            self.wl_data = self.wl_data[low_wl:high_wl]
            self.sampler = AliasSample(np.asarray(self.spectrum_data / np.sum(self.spectrum_data), dtype=np.float32))
        else:
            print("Valid values are:")
            print("T: ", *valid_T)
            print("log g: ", *valid_g)
            print("Z: ", *valid_z)
            print("alpha: ", *valid_a)
            raise ValueError("Invalid parameter for M-dwarf spectrum ")

    def get_spectral_density(self, wavelength):
        raise NotImplementedError

    def draw_wavelength(self, N):
        self.wavelength = self.wl_data[self.sampler.sample(N)]
        # self.wavelength = np.random.choice(
        #     self.wl_data, p=self.spectrum_data / np.sum(self.spectrum_data), size=N
        # )
        # self.wavelength += np.random.random(N) * (self.wl_data[1] - self.wl_data[0])
        return self.wavelength


class EchelleSpectrum:
    def __init__(self, source, spectrograph, efficiency=None):
        """

        Args:
            source(Source):
            efficiency(Union[None, Spectrograph)]:
        """
        self.source = source
        self.spectrograph = spectrograph
        self.spectra = {}
        for t in self.spectrograph.transformations:
            self.spectra[t] = source(
                min_wl=t.min_wavelength(), max_wl=t.max_wavelength()
            )
            print(t)

    def draw_wavelength_per_order(self, t, N):
        return self.spectra[t].draw_wavelength(N)


if __name__ == "__main__":
    from pyechelle.spectrograph import ZEMAX

    source = Phoenix()
    spec = ZEMAX(
        "/home/stuermer/Repos/cpp/EchelleSimulator/data/spectrographs/MaroonX.hdf"
    )
    EchelleSpectrum(Phoenix, spec)
