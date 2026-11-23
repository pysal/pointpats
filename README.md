# pointpats: Point Pattern Analysis in PySAL

[![Continuous Integration](https://github.com/pysal/pointpats/actions/workflows/tests.yaml/badge.svg)](https://github.com/pysal/pointpats/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/pysal/pointpats/branch/main/graph/badge.svg)](https://codecov.io/gh/pysal/pointpats)
[![Documentation](https://img.shields.io/static/v1.svg?label=docs&message=current&color=9cf)](http://pysal.org/pointpats/)
[![PyPI version](https://badge.fury.io/py/pointpats.svg)](https://badge.fury.io/py/pointpats)
[![Discord](https://img.shields.io/badge/Discord-join%20chat-7289da?style=flat&logo=discord&logoColor=cccccc&link=https://discord.gg/he6Y8D2ap3)](https://discord.gg/he6Y8D2ap3)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7706219.svg)](https://doi.org/10.5281/zenodo.7706219)

Statistical analysis of planar point patterns.

This package is part of [PySAL](https://pysal.org): The Python Spatial Analysis Library.

## Introduction

This [pointpats](https://github.com/pysal/pointpats) package is intended to support the statistical analysis of planar point patterns.

It currently works on cartesian coordinates. Users with data in geographic coordinates need to project their data prior to using this module.

### What you can do with ``pointpats``

- Build and summarize **point pattern objects** from coordinate data.
- Compute **centrographic measures** (mean center, standard distance, ellipses).
- Perform **quadrat statistics** for tests of complete spatial randomness.
- Use **distance-based functions** (G, K, L) and simulation envelopes.
- Work with **marked patterns** and **simulated point processes**.

## Documentation

Online documentation is available [here](http://pysal.org/pointpats/).

## Installation

Install pointpats by running:

    $ pip install pointpats

## Development

pointpats development is hosted on [github](https://github.com/pysal/pointpats).

As part of the PySAL project, pointpats development follows  [PySAL's documentation for developers](https://pysal.org/docs/devs/) and the PySAL [development guidelines](https://github.com/pysal/pysal/wiki).

## Bug reports

To search for or report bugs, please see pointpats' [issues](https://github.com/pysal/pointpats/issues).

## Get in touch

If you have a question regarding pointpats, feel free to open an [issue](https://github.com/pysal/pointpats/issues) on GitHub or join a chat on [Discord](https://discord.gg/he6Y8D2ap3).

## BibTeX Citation

```
@software{sergio_rey_2026_20814583,
  author       = {Sergio Rey and
                  Wei Kang and
                  Levi John Wolf and
                  James Gaboardi and
                  eli knaap and
                  Martin Fleischmann and
                  Hu Shao and
                  Mridul Seth and
                  Sugam Srivastava and
                  R Virinchi and
                  privateAGI and
                  Giovanni Palla and
                  Dani Arribas-Bel and
                  Matt Richards and
                  Qiusheng Wu and
                  Taher Chegini and
                  stevee404},
  title        = {pysal/pointpats: v2.6.0},
  month        = jun,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v2.6.0},
  doi          = {10.5281/zenodo.20814583},
  url          = {https://doi.org/10.5281/zenodo.20814583},
}
```
