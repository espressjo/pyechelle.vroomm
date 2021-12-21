# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- more tests to improve code reliability and increase coverage

### Removed

- dependency on autologging package

### Fixed

- model_viewer is working again

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

