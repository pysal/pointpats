import contextlib
from importlib.metadata import PackageNotFoundError, version

from .pointpattern import PointPattern
from .window import as_window, poly_from_bbox, to_ccf, Window
from .centrography import *
from .process import *
from .quadrat_statistics import *
from .distance_statistics import *
from .spacetime import *
from .kde import *

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("pointpats")