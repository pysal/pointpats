# pointpats: Point Pattern Analysis in PySAL

[![Continuous Integration](https://github.com/pysal/pointpats/actions/workflows/tests.yaml/badge.svg)](https://github.com/pysal/pointpats/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/pysal/pointpats/branch/main/graph/badge.svg)](https://codecov.io/gh/pysal/pointpats)
[![Documentation](https://img.shields.io/static/v1.svg?label=docs&message=current&color=9cf)](http://pysal.org/pointpats/)
[![PyPI version](https://badge.fury.io/py/pointpats.svg)](https://badge.fury.io/py/pointpats)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7706219.svg)](https://doi.org/10.5281/zenodo.7706219)

Statistical analysis of planar point patterns.

This package is part of [PySAL](https://pysal.org): The Python Spatial Analysis Library.

## Introduction

This [pointpats](https://github.com/pysal/pointpats) package is intended
to support the statistical analysis of planar point patterns.

It currently works on cartesian coordinates. Users with data in
geographic coordinates need to project their data prior to using this
module.

## Documentation

Online documentation is available
[here](http://pysal.org/pointpats/).

## Examples

- [Basic point pattern
    structure](https://github.com/pysal/pointpats/tree/main/notebooks/pointpattern.ipynb)
- [Centrography and
    visualization](https://github.com/pysal/pointpats/tree/main/notebooks/centrography.ipynb)
- [Marks](https://github.com/pysal/pointpats/tree/main/notebooks/marks.ipynb)
- [Simulation of point
    processes](https://github.com/pysal/pointpats/tree/main/notebooks/process.ipynb)
- [Distance based
    statistics](https://github.com/pysal/pointpats/tree/main/notebooks/distance_statistics-numpy-oriented.ipynb)

##  Installation

Install pointpats by running:

    $ pip install pointpats

## Development

pointpats development is hosted on
[github](https://github.com/pysal/pointpats).

As part of the PySAL project, pointpats development follows these
[guidelines](http://pysal.org/getting_started).

##  Bug reports

To search for or report bugs, please see pointpats'
[issues](https://github.com/pysal/pointpats/issues).

##  BibTeX Citation

```
@software{wei_kang_2023_7706219,
  author       = {Wei Kang and
                  Levi John Wolf and
                  Sergio Rey and
                  Hu Shao and
                  Mridul Seth and
                  Martin Fleischmann and
                  Sugam Srivastava and
                  James Gaboardi and
                  Giovanni Palla and
                  Dani Arribas-Bel and
                  Qiusheng Wu},
  title        = {pysal/pointpats: pointpats 2.3.0},
  month        = mar,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v2.3.0},
  doi          = {10.5281/zenodo.7706219},
  url          = {https://doi.org/10.5281/zenodo.7706219}
}
```
