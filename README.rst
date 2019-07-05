Point Pattern Analysis in PySAL
===============================
.. image:: https://api.travis-ci.org/pysal/pointpats.svg
   :target: https://travis-ci.org/pysal/pointpats

.. image:: https://readthedocs.org/projects/pointpats/badge/?version=latest
   :target: https://pointpats.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://badge.fury.io/py/pointpats.svg
    :target: https://badge.fury.io/py/pointpats

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3265637.svg
   :target: https://doi.org/10.5281/zenodo.3265637

Statistical analysis of planar point patterns.

This package is part of a `refactoring of PySAL
<https://github.com/pysal/pysal/wiki/PEP-13:-Refactor-PySAL-Using-Submodules>`_.


************
Introduction
************

This `pointpats <https://github.com/pysal/pointpats>`_ package is intended to support the statistical analysis of planar point patterns.

It currently works on cartesian coordinates. Users with data in geographic coordinates need to project their data prior to using this module.

*************
Documentation
*************

Online documentation is available `here <https://pointpats.readthedocs.io>`_.

********
Examples
********

- `Basic point pattern structure <https://github.com/pysal/pointpats/tree/master/notebooks/pointpattern.ipynb>`_
- `Centrography and visualization <https://github.com/pysal/pointpats/tree/master/notebooks/centrography.ipynb>`_
- `Marks <https://github.com/pysal/pointpats/tree/master/notebooks/marks.ipynb>`_
- `Simulation of point processes <https://github.com/pysal/pointpats/tree/master/notebooks/process.ipynb>`_
- `Distance based statistics <https://github.com/pysal/pointpats/tree/master/notebooks/distance_statistics.ipynb>`_

************
Installation
************

Install pointpats by running:

::

    $ pip install pointpats

***********
Development
***********

pointpats development is hosted on `github <https://github.com/pysal/pointpats>`_.

As part of the PySAL project, pointpats development follows these `guidelines <http://pysal.readthedocs.io/en/latest/developers/index.html>`_.

***********
Bug reports
***********

To search for or report bugs, please see pointpat's `issues <https://github.com/pysal/pointpats/issues>`_.

***************
BibTeX Citation
***************

.. code-block::
    @misc{sergio_rey_2019_3265637,
      author       = {Sergio Rey and
                      Wei Kang and
                      Hu Shao and
                      Levi John Wolf and
                      Mridul Seth and
                      James Gaboardi and
                      Dani Arribas-Bel},
      title        = {pysal/pointpats: pointpats 2.1.0},
      month        = jul,
      year         = 2019,
      doi          = {10.5281/zenodo.3265637},
      url          = {https://doi.org/10.5281/zenodo.3265637}
    }
