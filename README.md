# PyEchelle

PyEchelle is a simulation tool, to generate realistic 2D spectra, in particular cross-dispersed echelle spectra.
However, it is not limited to echelle spectrographs, but allows simulating arbitrary spectra for any fiber-fed or slit
spectrograph, where a model file is available. Optical aberrations are treated accurately, the simulated spectra include
photon and read-out noise.

PyEchelle uses numba for implementing fast Python-based simulation code. It also comes with **CUDA support** for major
speed improvements.

### Example usage

You can use PyEchelle directly from the console:

```bash
pyechelle --spectrograph MaroonX --fiber 2-4 --sources Phoenix --phoenix_t_eff 3500 -t 10 --rv 100 -o mdwarf.fit
```

If you rather script in python, you can do the same as above with the following python script:

```python

from pyechelle.simulator import Simulator
from pyechelle.sources import Phoenix
from pyechelle.spectrograph import ZEMAX

sim = Simulator(ZEMAX("MaroonX"))
sim.set_ccd(1)
sim.set_fibers([2, 3, 4])
sim.set_sources(Phoenix(t_eff=3500))
sim.set_exposure_time(10.)
sim.set_radial_velocities(100.)
sim.set_output('mdwarf.fits', overwrite=True)
sim.run()

```

Both times, a PHOENIX M-dwarf spectrum with the given stellar parameters, and a RV shift of 100m/s for the MAROON-X
spectrograph is simulated.

The output is a 2D raw frame (.fits) and will look similar to:

![](https://gitlab.com/Stuermer/pyechelle/-/raw/master/docs/source/_static/plots/mdwarf.jpg "")

Check out the [Documentation](https://stuermer.gitlab.io/pyechelle/usage.html) for more examples.

Pyechelle is the successor of [Echelle++](https://github.com/Stuermer/EchelleSimulator) which has a similar
functionality but was written in C++. This package was rewritten in python for better maintainability, easier package
distribution and for smoother cross-platform development.

# Installation

As simple as

```bash
pip install pyechelle
```

Check out the [Documentation](https://stuermer.gitlab.io/pyechelle/installation.html) for alternative installation instruction.

# Usage

See

```bash
pyechelle -h
```

for all available command line options.

See [Documentation](https://stuermer.gitlab.io/pyechelle/usage.html) for more examples.

# Concept:

The basic idea is that any spectrograph can be modelled with a set of wavelength-dependent transformation matrices and
point spread functions which describe the spectrographs' optics:

First, wavelength-dependent **affine transformation matrices** are extracted from the ZEMAX model of the spectrograph.
As the underlying geometric transformations (scaling, rotation, shearing, translation) vary smoothly across an echelle
order, these matrices can be interpolated for any intermediate wavelength.

Second, a wavelength-dependent **point spread functions (PSFs)** is applied on the transformed slit images to properly
account for optical aberrations. Again, the PSF is only slowly varying across an echelle order, allowing for
interpolation at intermediate wavelength.

![Echelle simulation](https://gitlab.com/Stuermer/pyechelle/-/raw/master/docs/source/_static/plots/intro.png "Echelle simulation")

**Both, the matrices and the PSFs have to be extracted from ZEMAX only once. It is therefore possible to simulate
spectra without access to ZEMAX**

# Citation

Please cite this [paper](http://dx.doi.org/10.1088/1538-3873/aaec2e) if you find this work useful in your research.
