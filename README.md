PyEchelle
===========
PyEchelle is a simulation tool, to generate realistic 2D spectra, in particular cross-dispersed echelle spectra. It
allows simulating arbitrary spectra for any fiber-fed or slit spectrograph, where a model file is available. Optical
aberrations are treated accurately, the simulated spectra include photon and read-out noise.

It is the successor of [Echelle++](https://github.com/Stuermer/EchelleSimulator) which has a similar functionality but
was written in C++. This package was rewritten in python for better maintainability, easier package distribution and for
smoother cross-platform development.

Installation
============
There are multiple ways of installing PyEchelle.
The recommended way for now is to clone the repository:

Install from source
-------------------
```bash
git clone https://gitlab.com/Stuermer/pyechelle.git
```

After that you can either install [Poetry](https://python-poetry.org/) and use it inside the pyechelle directoryto
automatically install the dependencies of PyEchelle:

```bash
poetry install
```

or you can use pip/conda and install the required python packages that are listed in pyproject.toml
under **[tool.poetry.dependencies]** directly:

If you want to install the optional dependencies (e.g. for including atmospheric conditions) you need to run

```bash
poetry install -E optional_dep
```

Install via pip
---------------
The simplest way for installing pyechelle is using *pip* (NOT WORKING YET):

```bash
pip install pyechelle
```

Usage
=====
See
```bash
python simulator.py -h
```

for all available command line options.

See [Documentation](https://stuermer.gitlab.io/pyechelle/) for more examples.

