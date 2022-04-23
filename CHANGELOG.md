# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2022-04-23

### Fixed

- fixed bug that caused PSF to not be applied correctly

## [0.3] - 2022-04-20

NOTE: HDF model files from previous versions are not compatible anymore. In case you are using any of the supported
spectrograph models, simply delete the local copy,
and a new version will be downloaded. For you own models, check the HDF file structure of the supported
spectrographs and adjust your HFD file accordingly.

### Added

- Simulator class for easy scripting in python. See [examples](https://stuermer.gitlab.io/pyechelle/example_direct.html)
  .
- Two classes to add distortions to any spectrograph class (
  see [documentation](https://stuermer.gitlab.io/pyechelle/models.html#perturbations))

### Changed

- spectrograph HDF model files are slightly restructured to better work with new simulator class.

## [0.2.3] - 2022-02-20

### Added

- support for CSV based spectra
- support for CSV based spectrograph efficiency files
- added *--append* option, to add simulations with different command line arguments to existing .fits frame rather than
  overwrite the file content

### Changed

- updated project dependencies

### Removed

- *--overwrite* flag, since it wasn't working

## [0.2.2] - 2022-01-26

### Fixed

- project dependencies

## [0.2.1] - 2022-01-26

### Added

- CUBES spectrograph (red and blue arm)

### Changed
 - updated project dependencies

## [UNRELEASED] -

### Changed

Code restructuring for better readability/maintainability.

## [0.2.0] - 2022-01-03

### Added

- CUDA support (use the *--cuda* flag)
- more tests to improve code reliability and increase coverage
- better code documentation incl. API for modules and classes

### Changed

- restructured core algorithm, code now properly parallelized by diffraction order
- better scaling with number of CPUs for multi-order simulations
- speed improvements for CPU simulation
- more precise simulation by interpolating transformation matrices

### Removed

- dependency on autologging package

### Fixed

- model_viewer is working again
- ThNe source returned Ne wavelength in angstroms rather than microns

## [0.1.7] - 2021-12-20

### Added

- caching to local files for atmosphere and nist queries to minimize web/download requests

### Changed

- added skycalc and astroquery as dependencies rather than optional dependencies
- cleaner implementation of PHOENIX request url
- cached data (PHOENIX spectra, atmosphere data etc.) now gets saved in a *.cache* folder
- moved *.cache* folder and *models* folder into pyechelle folder so it doesn't mess up the site-packages folder

### Fixed

- atmosphere argparse arguments where not handled correctly. Now fiber/field specific keywords work as expected.

## [0.1.6] - 2021-12-01

### Added

- arclamps (ThAr and ThNe) sources.
- support for radial velocity shift option for all sources

### Changed

- removed .hdf files from repository and added downloader for model files instead
- minor improvements / added tests

## [0.1.5]

### Added

- atmosphere/tellurics via optional dependency skycal-ipy

## [0.1.0] - 2020-12-07

### Added

- Initial release of PyEchelle

