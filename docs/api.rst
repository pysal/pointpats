.. _api_ref:

.. currentmodule:: pointpats

API reference
=============

.. _pointpattern_api:

Point Pattern
--------------

.. autosummary::
   :toctree: generated/

    PointPattern

.. _pointprocess_api:

Point Processes
---------------

.. autosummary::
   :toctree: generated/

    PointProcess
    PoissonPointProcess
    PoissonClusterPointProcess

.. _centrgraphy_api:

Centrography
------------

.. autosummary::
   :toctree: generated/

    minimum_bounding_rectangle
    hull
    mean_center
    weighted_mean_center
    manhattan_median
    std_distance
    euclidean_median
    ellipse
    skyum
    dtot


.. _density_api:

Density
-------

.. autosummary::
   :toctree: generated/

    plot_density

.. _quadrat_api:

Quadrat Based Statistics
------------------------

.. autosummary::
   :toctree: generated/

    RectangleM
    HexagonM
    QStatistic


.. _distance_api:

Distance Based Statistics
--------------------------

.. autosummary::
   :toctree: generated/

    f
    g
    k
    j
    l
    f_test
    g_test
    k_test
    j_test
    l_test

.. _window_api:

Window functions
----------------

.. autosummary::
   :toctree: generated/

   Window
   as_window
   poly_from_bbox
   to_ccf

Random distributions
--------------------

.. autosummary::
   :toctree: generated/

   random.poisson
   random.normal
   random.cluster_poisson
   random.cluster_normal

Space-Time Interaction Tests
-----------------------------

.. autosummary::
   :toctree: generated/

   SpaceTimeEvents
   Knox
   KnoxLocal
   mantel
   jacquez
   modified_knox

Visualization
---------------

.. autosummary::
   :toctree: generated/

   plot_density