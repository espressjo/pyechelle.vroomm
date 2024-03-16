# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.7] - unreleased

### Added

- Simplified creation of new .hdf models: introduced two new classes
  [InteractiveZemax](https://stuermer.gitlab.io/pyechelle/_autosummary/pyechelle.spectrograph.InteractiveZEMAX.html#pyechelle.spectrograph.InteractiveZEMAX)
  and
  [HDFBuilder](https://stuermer.gitlab.io/pyechelle/_autosummary/pyechelle.hdfbuilder.HDFBuilder.html#pyechelle.hdfbuilder.HDFBuilder),
  which can be used to generate .hdf model files the ZEMAX design from any spectrograph
  (see: [How to create a new spectrograph model](https://stuermer.gitlab.io/pyechelle/new_model.html))
- Simulation metadata is now added to FITS header (.fits files now contain information about simulation parameters)
- added python 3.12 support, removed python 3.8 support

### Fixed
 - fixed #9 (issue with CUDA random seed)
- fixed #10 (issue with ZEMAX PSF model)
 - adapted to work with latest skycalc-ipy version


## [0.3.6] - 2023-08-07

### Added

- New spectral source types: Blackbody, LineList and ConstantPhotons
- New spectrograph
  model: [AtmosphericDispersion](https://stuermer.gitlab.io/pyechelle/example_direct.html#atmospheric-dispersion)
- added python 3.11 support

## [0.3.5] - 2022-11-30

### Added

- support for single mode spectrographs (use field type 'singlemode')

### Changed

- fiber/field shape attribute can now be either bytes or string
- updated project dependencies

### Fixed

- fixed #8 (atmospheric absorption was not applied)

## [0.3.4] - 2022-08-28

### Added

- support for python 3.10

### Changed

- fixed CI/CD pipeline

## [0.3.3] - 2022-08-27

### Fixed

- fixed #6

### Changed

- updated project dependencies
- added CSV source documentation

## [0.3.2] - 2022-08-24

### Fixed

- fixed issues #2 #3 #4 and #5

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

