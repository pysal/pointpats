.. pointpats documentation master file.

Point Pattern Analysis (pointpats)
==================================

Statistical analysis of planar point patterns in Python.

`pointpats` is an open-source library for the analysis of planar point patterns
and a subpackage of the Python Spatial Analysis Library, `PySAL`_.


Key workflows (notebooks)
-------------------------

Explore the main workflows through executable notebooks:

.. raw:: html

    <div class="container-fluid">
      <div class="row equal-height">
        <div class="col-sm-1 col-xs-hidden"></div>

        <!-- Centrography -->
        <div class="col-md-3 col-xs-12">
          <a href="user-guide/centrography.html" class="thumbnail">
            <img src="_static/images/centrography.png"
                 class="img-responsive center-block" alt="Centrography notebook">
            <div class="caption text-center">
              <h6>Centrography</h6>
              <p>Centroids, dispersion measures, and visual summaries.</p>
            </div>
          </a>
        </div>

        <!-- Quadrat statistics -->
        <div class="col-md-3 col-xs-12">
          <a href="user-guide/Quadrat_statistics.html" class="thumbnail">
            <img src="_static/images/quadrat.png"
                 class="img-responsive center-block" alt="Quadrat statistics notebook">
            <div class="caption text-center">
              <h6>Quadrat statistics</h6>
              <p>Grid-based summaries and tests for spatial randomness.</p>
            </div>
          </a>
        </div>

        <!-- Distance statistics -->
        <div class="col-md-3 col-xs-12">
          <a href="user-guide/distance_statistics.html" class="thumbnail">
            <img src="_static/images/ripleyg.png"
                 class="img-responsive center-block" alt="Distance statistics notebook">
            <div class="caption text-center">
              <h6>Distance-based statistics</h6>
              <p>G, K, and L functions, envelopes, and interaction structure.</p>
            </div>
          </a>
        </div>

        <div class="col-sm-1 col-xs-hidden"></div>
      </div>
    </div>

Quickstart
----------

Install the latest release:

.. code-block:: bash

    pip install pointpats

or using conda-forge:

.. code-block:: bash

    conda install -c conda-forge pointpats

A minimal example:

.. code-block:: python

    import numpy as np
    from pointpats import PointPattern

    # toy coordinates
    coords = np.random.random((100, 2))
    pp = PointPattern(coords)

    print("n:", pp.n)
    print("mean center:", pp.mean_center)
    print("average nearest-neighbor distance:", pp.nnd.mean())

What you can do with ``pointpats``
----------------------------------

- Build and summarize **point pattern objects** from coordinate data.
- Compute **centrographic measures** (mean center, standard distance, ellipses).
- Perform **quadrat statistics** for tests of complete spatial randomness.
- Use **distance-based functions** (G, K, L) and simulation envelopes.
- Work with **marked patterns** and **simulated point processes**.

User Guide
----------

The full user guide is organized around executable notebooks:

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user-guide/intro

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   Installation <installation>
   API <api>
   References <references>

Part of the PySAL ecosystem
---------------------------

``pointpats`` is part of the `PySAL`_ family of spatial analysis libraries,
alongside components for spatial weights, regression, clustering, and more.

- Source code: https://github.com/pysal/pointpats
- Bug reports and feature requests: https://github.com/pysal/pointpats/issues

Citation
--------

If you use ``pointpats`` in your work, please cite the Zenodo record:

.. code-block:: bibtex

    @software{wei_kang_2023_7706219,
      author    = {Wei Kang and Levi John Wolf and Sergio Rey and Hu Shao
                   and Mridul Seth and Martin Fleischmann and Sugam Srivastava
                   and James Gaboardi and Giovanni Palla and Dani Arribas-Bel
                   and Qiusheng Wu},
      title     = {pysal/pointpats: pointpats 2.3.0},
      year      = {2023},
      publisher = {Zenodo},
      version   = {v2.3.0},
      doi       = {10.5281/zenodo.7706219},
      url       = {https://doi.org/10.5281/zenodo.7706219}
    }

.. _PySAL: https://github.com/pysal/pysal
