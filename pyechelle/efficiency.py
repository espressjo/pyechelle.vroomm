import numpy as np
from numpy import deg2rad
from numpy import sin, cos, tan, arcsin
from scipy.interpolate import interp1d


class Efficiency:
    def __init__(self, name):
        self.name = name

    def get_efficiency(self, wavelength):
        raise NotImplementedError

    def get_efficiency_per_order(self, wavelength, order):
        raise NotImplementedError


class ConstantEfficiency(Efficiency):
    def __init__(self, name, eff=1.):
        super().__init__(name)
        self.eff = eff

    def get_efficiency(self, wavelength):
        return np.ones_like(wavelength) * self.eff

    def get_efficiency_per_order(self, wavelength, order):
        return self.get_efficiency(wavelength)


class SystemEfficiency(Efficiency):
    def __init__(self, efficiencies, name):
        super().__init__(name)
        self.efficiencies = efficiencies

    def get_efficiency(self, wavelength):
        e = np.ones_like(wavelength)
        for ef in self.efficiencies:
            e *= ef.get_efficiency(wavelength)
        return e

    def get_efficiency_per_order(self, wavelength, order):
        e = np.ones_like(wavelength)
        for ef in self.efficiencies:
            e *= ef.get_efficiency_per_order(wavelength, o)
        return e


class GratingEfficiency(Efficiency):
    def __init__(self, alpha, blaze, gpmm, peak_efficiency=1.0, name="Grating"):
        super().__init__(name)
        self.alpha = deg2rad(alpha)
        self.blaze = deg2rad(blaze)
        self.gpmm = gpmm
        self.peak_efficiency = peak_efficiency

    def calc_efficiency(self, order, wl):
        bb = np.nan_to_num(
            arcsin(-sin(self.alpha) + order * wl * 1e-6 / (1.0 / self.gpmm / 1000.0))
        )
        # blaze_wavelength = 2.*self.gpmm * sin(self.blaze) / order
        # fsr = blaze_wavelength / order

        x = (
                order
                * (cos(self.alpha) / cos(self.alpha - self.blaze))
                * (cos(self.blaze) - sin(self.blaze) / tan((self.alpha + bb) / 2.0))
        )
        sinc = np.sinc(x)

        return self.peak_efficiency * sinc * sinc

    def get_efficiency(self, wavelength):
        e = np.zeros_like(wavelength)
        for o in range(50, 150):
            e += self.calc_efficiency(o, wavelength)
        return e

    def get_efficiency_per_order(self, wavelength, order):
        return self.calc_efficiency(order, wavelength)


class CSVEfficiency(Efficiency):
    def __init__(self, name, path, interpolation="cubic", delimiter=","):
        super().__init__(name)
        self.path = path
        self.ip = None
        self.ip_per_order = None

        data = np.genfromtxt(path, delimiter=delimiter)
        # file contains wavelength, efficiency
        if data.shape[1] == 2:
            self.ip = interp1d(
                data[:, 0], data[:, 1], kind=interpolation, fill_value=0.0, bounds_error=False
            )
            self.ip_per_order = self.ip

        # file contains order, wavelength, efficiency
        if data.shape[1] == 3:
            self.ip_per_order = {}
            orders = data[:, 0]

            for o in np.unique(orders):
                idx = orders == o
                self.ip_per_order[o] = interp1d(
                    data[:, 1][idx], data[:, 2][idx], fill_value=0.0, bounds_error=False
                )

            y = np.zeros_like(data[:, 1])
            for o in np.unique(orders):
                y += self.ip_per_order[o](data[:, 1])

            self.ip = interp1d(data[:, 1], y)

    def get_efficiency(self, wavelength):
        return self.ip(wavelength)

    def get_efficiency_per_order(self, wavelength, order):
        if isinstance(self.ip_per_order, dict):
            return self.ip_per_order[order](wavelength)
        else:
            return self.ip_per_order(wavelength)


if __name__ == "__main__":
    ge = GratingEfficiency(76.4, 76.4, 31.6)
    wl = np.linspace(0.35, 0.9, 10000)
    e = ge.get_efficiency(wl)
    thar = CSVEfficiency(
        "ThAr", "/home/stuermer/silver.csv", 'cubic'
    )
    se = SystemEfficiency([ge, thar], 'Total')
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(wl, thar.get_efficiency(wl), "g--")
    for o in range(50, 100):
        plt.plot(wl, se.get_efficiency_per_order(wl, o))
    plt.show()
