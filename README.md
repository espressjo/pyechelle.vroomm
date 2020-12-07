PyEchelle
===========
PyEchelle is a simulation tool, to generate realistic 2D spectra, in particular cross-dispersed echelle spectra. It
allows to simulate arbitrary spectra for any fiber-fed or slit spectrograph, where a model file is available. Optical
aberrations are treated accurately, the simulated spectra include photon and read-out noise.

It is the successor of [Echelle++](https://github.com/Stuermer/EchelleSimulator) which has a similar functionality but
was written in C++. This package was rewritten in python for better maintainability, easier package distribution and for
smoother cross-plattform development.

Installation
============
The simplest way for installing pyechelle is using *pip*:

```bash
pip install pyechelle
```

Usage
=====