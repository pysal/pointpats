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
    minimum_area_rectangle
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
    _circle

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

    DStatistic
    G 
    F 
    J 
    K
    L 
    Envelopes
    Genv
    Fenv
    Jenv
    Kenv
    Lenv

.. _window_api:

Window functions
----------------

.. autosummary::
   :toctree: generated/

   Window
   as_window
   poly_from_bbox
   to_ccf



Space-Time Interaction Tests
-----------------------------

.. autosummary::
   :toctree: generated/

   SpaceTimeEvents
   knox
   mantel
   jacquez
   modified_knox