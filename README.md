# PyEchelle

PyEchelle is a simulation tool, to generate realistic 2D spectra, in particular cross-dispersed echelle spectra.
However, it is not limited to echelle spectrographs, but allows simulating arbitrary spectra for any fiber-fed or slit
spectrograph, where a model file is available. Optical aberrations are treated accurately, the simulated spectra include
photon and read-out noise.

### Example usage

```bash
pyechelle --spectrograph MaroonX --fiber 2-4 --sources Phoenix --phoenix_t_eff 3500 -t 10 --rv 100 -o mdwarf.fits
```

simulates a PHOENIX M-dwarf spectrum with the given stellar parameters, and a RV shift of 100m/s for the MAROON-X
spectrograph. The output is a 2D raw frame (.fits) and will look similar to:

![mdwarf spectrum](https://gitlab.com/Stuermer/pyechelle/-/blob/master/docs/source/_static/plots/mdwarf.jpg "")

Check out the [Documentation](https://stuermer.gitlab.io/pyechelle/usage.html) for more examples.

Pyechelle is the successor of [Echelle++](https://github.com/Stuermer/EchelleSimulator) which has a similar
functionality but was written in C++. This package was rewritten in python for better maintainability, easier package
distribution and for smoother cross-platform development.

# Installation

As simple as

```bash
pip install pyechelle[optional_dep]
```

Here, adding *[optional_dep]* will also install optional dependencies (e.g. astroquery, skycalc-ipy) for supporting all
features such as atmosphere tellurics.

Check out the [Documentation](https://stuermer.gitlab.io/pyechelle/) for alternative installation instruction.

# Usage

See

```bash
pyechelle -h
```

for all available command line options.

See [Documentation](https://stuermer.gitlab.io/pyechelle/) for more examples.

# Citation

Please cite this [paper](http://dx.doi.org/10.1088/1538-3873/aaec2e) if you find this work useful in your research.
