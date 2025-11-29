"""
Centrographic measures for point patterns

- documentation

"""

__author__ = [
    "Serge Rey sjsrey@gmail.com",
    "Levi John Wolf levi.john.wolf@gmail.com"
]

__all__ = [
    "hull",
    "mean_center",
    "weighted_mean_center",
    "manhattan_median",
    "std_distance",
    "euclidean_median",
    "ellipse",
    "minimum_rotated_rectangle",
    "minimum_bounding_rectangle",
    "minimum_bounding_circle",
    "dtot",
    "_circle",
]


import copy
import math
import sys
import warnings
from collections.abc import Sequence
from functools import singledispatch

import numpy as np
import shapely
from geopandas.base import GeoPandasBase
from geopandas import GeoDataFrame, GeoSeries, points_from_xy
from libpysal.cg import is_clockwise
from numpy.typing import NDArray
from esda import shape
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

not_clockwise = lambda x: not is_clockwise(x)

MAXD = sys.float_info.max
MIND = sys.float_info.min


@singledispatch
def minimum_bounding_rectangle(points):
    """
    Find minimum bounding rectangle of a point array.

    Parameters
    ----------
    points : arraylike
             array representing a point pattern

    Returns
    -------
    rectangle
        minimum bounding rectangle of a given point pattern

    Examples
    --------
    >>> import numpy as np
    >>> import geopandas as gpd

    Create an array of point coordinates.

    >>> coords = np.array(
    ...     [
    ...         [66.22, 32.54],
    ...         [22.52, 22.39],
    ...         [31.01, 81.21],
    ...         [9.47, 31.02],
    ...         [30.78, 60.10],
    ...         [75.21, 58.93],
    ...         [79.26, 7.68],
    ...         [8.23, 39.93],
    ...         [98.73, 77.17],
    ...         [89.78, 42.53],
    ...         [65.19, 92.08],
    ...         [54.46, 8.48],
    ...     ]
    ... )

    Passing an array of coordinates returns a tuple capturing the bounds.

    >>> minimum_bounding_rectangle(coords)
    (np.float64(8.23), np.float64(7.68), np.float64(98.73), np.float64(92.08))

    Passing a GeoPandas object returns a shapely geometry.

    >>> geoms = gpd.GeoSeries.from_xy(*coords.T)
    >>> minimum_bounding_rectangle(geoms)
    <POLYGON ((8.23 7.68, 98.73 7.68, 98.73 92.08, 8.23 92.08, 8.23 7.68))>

    """
    try:
        points = np.asarray(points)
        return minimum_bounding_rectangle(points)
    except AttributeError as e:
        raise NotImplementedError from e


@minimum_bounding_rectangle.register
def _(points: np.ndarray) -> tuple[float, float, float, float]:
    points = np.asarray(points)
    min_x = min_y = MAXD
    max_x = max_y = MIND
    x, y = zip(*points, strict=True)
    min_x = min(x)
    min_y = min(y)
    max_x = max(x)
    max_y = max(y)
    return min_x, min_y, max_x, max_y


@minimum_bounding_rectangle.register
def _(points: GeoPandasBase) -> shapely.Polygon:
    return shapely.envelope(shapely.geometrycollections(points.geometry.values))


@singledispatch
def minimum_rotated_rectangle(points, return_angle=False):
    """
    Compute the minimum rotated rectangle for a point array.

    This is the smallest enclosing rectangle (possibly rotated)
    for the input point set. It is computed using Shapely.

    Parameters
    ----------
    points : arraylike
             array representing a point pattern
    return_angle : bool
        whether to return the angle (in degrees) of the angle between
        the horizontal axis of the rectanle and the first side (i.e. length).

    Returns
    -------
    rectangle | tuple(rectangle, angle)

    Examples
    --------
    >>> import numpy as np
    >>> import geopandas as gpd

    Create an array of point coordinates.

    >>> coords = np.array(
    ...     [
    ...         [66.22, 32.54],
    ...         [22.52, 22.39],
    ...         [31.01, 81.21],
    ...         [9.47, 31.02],
    ...         [30.78, 60.10],
    ...         [75.21, 58.93],
    ...         [79.26, 7.68],
    ...         [8.23, 39.93],
    ...         [98.73, 77.17],
    ...         [89.78, 42.53],
    ...         [65.19, 92.08],
    ...         [54.46, 8.48],
    ...     ]
    ... )


    Passing an array of coordinates returns an array capturing the corners.

    >>> minimum_rotated_rectangle(coords)
    array([[ 75.5990843 ,  -0.72615725],
           [  4.08727852,  30.41752523],
           [ 36.40164577, 104.61744544],
           [107.91345156,  73.47376296]])
    

    >>> minimum_rotated_rectangle(coords, return_angle=True)
    (array([[ 75.5990843 ,  -0.72615725],
           [  4.08727852,  30.41752523],
           [ 36.40164577, 104.61744544],
           [107.91345156,  73.47376296]]), 66.466678613503)
    
    Passing a GeoPandas object returns a shapely geometry.

    >>> geoms = gpd.GeoSeries.from_xy(*coords.T)
    >>> minimum_rotated_rectangle(geoms)
    <POLYGON ((107.913 73.474, 36.402 104.617, 4.087 30.418, 75.599 -0.726, 107....>

    >>> minimum_rotated_rectangle(geoms, return_angle=True)
    (<POLYGON ((107.913 73.474, 36.402 104.617, 4.087 30.418, 75.599 -0.726, 107....>, 66.466678613503)
    """  # noqa: E501
    try:
        points = np.asarray(points)
        return minimum_rotated_rectangle(points, return_angle=return_angle)
    except AttributeError as e:
        raise NotImplementedError from e


def _get_mrr_angle(out_points):
    angle = (
        math.degrees(
            math.atan2(
                out_points[1][1] - out_points[0][1],
                out_points[1][0] - out_points[0][0],
            )
        )
        % 90
    )
    return angle


@minimum_rotated_rectangle.register
def _(
    points: np.ndarray, return_angle: bool = False
) -> NDArray[np.float64] | tuple[NDArray[np.float64], float]:
    out_points = shapely.get_coordinates(
        shapely.minimum_rotated_rectangle(shapely.multipoints(points))
    )[:-1]
    if return_angle:
        return (out_points[::-1], _get_mrr_angle(out_points))
    return out_points[::-1]


@minimum_rotated_rectangle.register
def _(
    points: GeoPandasBase, return_angle: bool = False
) -> shapely.Polygon | tuple[shapely.Polygon, float]:
    rectangle = shapely.minimum_rotated_rectangle(shapely.multipoints(points.geometry))

    if return_angle:
        out_points = shapely.get_coordinates(rectangle)[:-1]
        return (rectangle, _get_mrr_angle(out_points))
    return rectangle


@singledispatch
def hull(points):
    """
    Find convex hull of a point array.

    Parameters
    ----------
    points : arraylike
             array representing a point pattern

    Returns
    -------
    rectangle
        convex hull of a given point pattern

    Examples
    --------
    >>> import numpy as np
    >>> import geopandas as gpd

    Create an array of point coordinates.

    >>> coords = np.array(
    ...     [
    ...         [66.22, 32.54],
    ...         [22.52, 22.39],
    ...         [31.01, 81.21],
    ...         [9.47, 31.02],
    ...         [30.78, 60.10],
    ...         [75.21, 58.93],
    ...         [79.26, 7.68],
    ...         [8.23, 39.93],
    ...         [98.73, 77.17],
    ...         [89.78, 42.53],
    ...         [65.19, 92.08],
    ...         [54.46, 8.48],
    ...     ]
    ... )

    Passing an array of coordinates returns an array capturing the vertices.

    >>> hull(coords)
    array([[31.01, 81.21],
           [ 8.23, 39.93],
           [ 9.47, 31.02],
           [22.52, 22.39],
           [54.46,  8.48],
           [79.26,  7.68],
           [89.78, 42.53],
           [98.73, 77.17],
           [65.19, 92.08]])

    Passing a GeoPandas object returns a shapely geometry.

    >>> geoms = gpd.GeoSeries.from_xy(*coords.T)
    >>> hull(geoms)
    <POLYGON ((79.26 7.68, 54.46 8.48, 22.52 22.39, 9.47 31.02, 8.23 39.93, 31.0...>
    """
    try:
        points = np.asarray(points)
        return hull(points)
    except AttributeError as e:
        raise NotImplementedError from e


@hull.register
def _(points: np.ndarray) -> NDArray[np.float64]:
    h = ConvexHull(points)
    return points[h.vertices]


@hull.register
def _(points: GeoPandasBase) -> shapely.Polygon:
    return shapely.convex_hull(shapely.multipoints(points.geometry))


@singledispatch
def mean_center(points):
    """
    Find mean center of a point array.

    Parameters
    ----------
    points : arraylike
             array representing a point pattern

    Returns
    -------
    center
        center of a given point pattern

    Examples
    --------
    >>> import numpy as np
    >>> import geopandas as gpd

    Create an array of point coordinates.

    >>> coords = np.array(
    ...     [
    ...         [66.22, 32.54],
    ...         [22.52, 22.39],
    ...         [31.01, 81.21],
    ...         [9.47, 31.02],
    ...         [30.78, 60.10],
    ...         [75.21, 58.93],
    ...         [79.26, 7.68],
    ...         [8.23, 39.93],
    ...         [98.73, 77.17],
    ...         [89.78, 42.53],
    ...         [65.19, 92.08],
    ...         [54.46, 8.48],
    ...     ]
    ... )

    Passing an array of coordinates returns an array capturing the center.

    >>> mean_center(coords)
    array([52.57166667, 46.17166667])

    Passing a GeoPandas object returns a shapely geometry.

    >>> geoms = gpd.GeoSeries.from_xy(*coords.T)
    >>> mean_center(geoms)
    <POINT (52.572 46.172)>
    """
    try:
        points = np.asarray(points)
        return mean_center(points)
    except AttributeError as e:
        raise NotImplementedError from e


@mean_center.register
def _(points: np.ndarray) -> NDArray[np.float64]:
    return points.mean(axis=0)


@mean_center.register
def _(points: GeoPandasBase) -> shapely.Point:
    coords = shapely.get_coordinates(points.geometry)
    return shapely.Point(mean_center(coords))


@singledispatch
def weighted_mean_center(points, weights):
    """
    Find weighted mean center of a marked point pattern.

    Parameters
    ----------
    points : arraylike
        array representing a point pattern
    weights : arraylike
        a series of attribute values of length n.

    Returns
    -------
    center
        center of a given point pattern

    Examples
    --------
    >>> import numpy as np
    >>> import geopandas as gpd

    Create an array of point coordinates.

    >>> coords = np.array(
    ...     [
    ...         [66.22, 32.54],
    ...         [22.52, 22.39],
    ...         [31.01, 81.21],
    ...         [9.47, 31.02],
    ...         [30.78, 60.10],
    ...         [75.21, 58.93],
    ...         [79.26, 7.68],
    ...         [8.23, 39.93],
    ...         [98.73, 77.17],
    ...         [89.78, 42.53],
    ...         [65.19, 92.08],
    ...         [54.46, 8.48],
    ...     ]
    ... )
    >>> weight = np.arange(1, 13, 1)

    Passing an array of coordinates returns an array capturing the center.

    >>> weighted_mean_center(coords, weight)
    array([59.29448718, 47.52282051])

    Passing a GeoPandas object returns a shapely geometry.

    >>> geoms = gpd.GeoSeries.from_xy(*coords.T)
    >>> weighted_mean_center(geoms, weight)
    <POINT (59.294 47.523)>
    """
    try:
        points = np.asarray(points)
        return weighted_mean_center(points, weights)
    except AttributeError as e:
        raise NotImplementedError from e


@weighted_mean_center.register
def _(points: np.ndarray, weights: Sequence) -> NDArray[np.float64]:
    points, weights = np.asarray(points), np.asarray(weights)
    w = weights * 1.0 / weights.sum()
    w.shape = (1, len(points))
    return np.dot(w, points)[0]


@weighted_mean_center.register
def _(points: GeoPandasBase, weights) -> shapely.Point:
    coords = shapely.get_coordinates(points.geometry)
    return shapely.Point(weighted_mean_center(coords, weights))


@singledispatch
def manhattan_median(points):
    """
    Find manhattan median of a point array.

    Parameters
    ----------
    points  : arraylike
              array representing a point pattern

    Returns
    -------
    median
        manhattan median of a given point pattern

    Examples
    --------
    >>> import numpy as np
    >>> import geopandas as gpd

    Create an array of point coordinates.

    >>> coords = np.array(
    ...     [
    ...         [66.22, 32.54],
    ...         [22.52, 22.39],
    ...         [31.01, 81.21],
    ...         [9.47, 31.02],
    ...         [30.78, 60.10],
    ...         [75.21, 58.93],
    ...         [79.26, 7.68],
    ...         [8.23, 39.93],
    ...         [98.73, 77.17],
    ...         [89.78, 42.53],
    ...         [65.19, 92.08],
    ...     ]
    ... )

    Passing an array of coordinates returns an array capturing the median.

    >>> manhattan_median(coords)
    array([65.19, 42.53])

    Passing a GeoPandas object returns a shapely geometry.

    >>> geoms = gpd.GeoSeries.from_xy(*coords.T)
    >>> manhattan_median(geoms)
    <POINT (65.19 42.53)>
    """
    try:
        points = np.asarray(points)
        return manhattan_median(points)
    except AttributeError as e:
        raise NotImplementedError from e


@manhattan_median.register
def _(points: np.ndarray) -> NDArray[np.float64]:
    if not len(points) % 2:
        warnings.warn(
            "Manhattan Median is not unique for even point patterns.", stacklevel=3
        )
    return np.median(points, axis=0)


@manhattan_median.register
def _(points: GeoPandasBase) -> shapely.Point:
    coords = shapely.get_coordinates(points.geometry)
    return shapely.Point(manhattan_median(coords))


@singledispatch
def std_distance(points) -> np.float64:
    """
    Calculate standard distance of a point array.

    Parameters
    ----------
    points  : arraylike
              array representing a point pattern

    Returns
    -------
    distance
       standard distance of a given point pattern

    Examples
    --------
    >>> import numpy as np
    >>> import geopandas as gpd

    Create an array of point coordinates.

    >>> coords = np.array(
    ...     [
    ...         [66.22, 32.54],
    ...         [22.52, 22.39],
    ...         [31.01, 81.21],
    ...         [9.47, 31.02],
    ...         [30.78, 60.10],
    ...         [75.21, 58.93],
    ...         [79.26, 7.68],
    ...         [8.23, 39.93],
    ...         [98.73, 77.17],
    ...         [89.78, 42.53],
    ...         [65.19, 92.08],
    ...     ]
    ... )

    Passing an array of coordinates returns a tuple capturing the bounds.

    >>> std_distance(coords)
    np.float64(40.21575987282547)

    The same applies to a GeoPandas object.

    >>> geoms = gpd.GeoSeries.from_xy(*coords.T)
    >>> std_distance(geoms)
    np.float64(40.21575987282547)
    """
    try:
        points = np.asarray(points)
        return std_distance(points)
    except AttributeError as e:
        raise NotImplementedError from e


@std_distance.register
def _(points: np.ndarray) -> np.float64:
    n, _ = points.shape
    m = points.mean(axis=0)
    return np.sqrt(((points * points).sum(axis=0) / n - m * m).sum())


@std_distance.register
def _(points: GeoPandasBase) -> np.float64:
    coords = shapely.get_coordinates(points.geometry)
    return std_distance(coords)



@singledispatch
def ellipse(points, weights=None, method="crimestat", 
              crimestatCorr=True, degfreedCorr=True):
    """
    Computes a weighted standard deviational ellipse for a set of point geometries.

    Parameters
    ----------
    points : array-like
        Array representing a point pattern.
    weights : array-like, optional
        Array of weights for each point.
    method : str
        Correction method to apply. Must be either 'crimestat' or 'yuill'.
    crimestatCorr : bool
        Whether to apply the CrimeStat correction (used only if `method == 'yuill'`).
    degfreedCorr : bool
        Whether to apply degrees-of-freedom correction (used only if `method == 'yuill'`).
        Apply degrees-of-freedom correction if method == 'yuill'.

    Returns
    -------
    tuple(major_axis_length, minor_axis_length, rotation_angle) | ellipse

    Notes
    -----
    Implements approach from:

    https://www.icpsr.umich.edu/CrimeStat/files/CrimeStatChapter.4.pdf

    Examples
    --------
    >>> import numpy as np
    >>> import geopandas as gpd

    Create an array of point coordinates.

    >>> coords = np.array(
    ...     [
    ...         [66.22, 32.54],
    ...         [22.52, 22.39],
    ...         [31.01, 81.21],
    ...         [9.47, 31.02],
    ...         [30.78, 60.10],
    ...         [75.21, 58.93],
    ...         [79.26, 7.68],
    ...         [8.23, 39.93],
    ...         [98.73, 77.17],
    ...         [89.78, 42.53],
    ...         [65.19, 92.08],
    ...     ]
    ... )

    Passing an array of coordinates returns a tuple capturing the semi-major axis,
    semi-minor axis and clockwise rotation angle of the ellipse.

    >>> ellipse(coords)
    (np.float64(50.13029459102783), np.float64(37.95222670267865), np.float64(0.3465582095042642))

    Passing a GeoPandas object returns a shapely geometry.

    >>> geoms = gpd.GeoSeries.from_xy(*coords.T)
    >>> ellipse(geoms)
    <POLYGON ((99.55 66.626, 100.586 63.045, 101.159 59.334, 101.262 55.53, 100....>
    """
    try:
        points = np.asarray(points)
        return ellipse(points, weights, method,
                          crimestatCorr, degfreedCorr)
    except AttributeError as e:
        raise NotImplementedError


@ellipse.register
def _(
    points: np.ndarray,
    weights=None,
    method='crimestat',
    crimestatCorr=True,
    degfreedCorr=True ) -> tuple[float, float, float]:
    method = method.lower()
    if method not in ("crimestat", "yuill"):
        raise ValueError("`method` must be either 'crimestat' or 'yuill'")

    x = points[:, 0]
    y = points[:, 1]

    if weights is None:
        weights = np.ones(len(points))
    else:
        weights = np.asarray(weights)
        if len(weights) != len(points):
            raise ValueError("weights must have same length as points")
        
    w = weights
    sumw = w.sum()
    meanx = np.average(x, weights=w)
    meany = np.average(y, weights=w)

    xm = x - meanx
    ym = y - meany

    xyw = np.sum(xm * ym * w)
    x2w = np.sum(xm**2 * w)
    y2w = np.sum(ym**2 * w)

    den = 2 * xyw
    left = x2w - y2w
    right = np.sqrt(left**2 + 4 * xyw**2)

    if den == 0:
        theta1 = 0
        theta2 = np.pi / 2
    else:
        theta1 = np.arctan(-(left + right) / den)
        theta2 = np.arctan(-(left - right) / den)

    term1 = np.sum(w * (ym * np.cos(theta1) - xm * np.sin(theta1))**2)
    term2 = np.sum(w * (ym * np.cos(theta2) - xm * np.sin(theta2))**2)

    sx = np.sqrt(term1 / sumw)
    sy = np.sqrt(term2 / sumw)

    n = len(points)
    if method == "crimestat":
        correction = (np.sqrt(2) * np.sqrt(n)) / np.sqrt(n - 2)
        sx *= correction
        sy *= correction
    elif method == "yuill":
        if crimestatCorr:
            sx *= np.sqrt(2)
            sy *= np.sqrt(2)
        if degfreedCorr:
            sx *= np.sqrt(n) / np.sqrt(n - 2)
            sy *= np.sqrt(n) / np.sqrt(n - 2)

    if sy > sx:
        major_axis, minor_axis = sy, sx
        major_angle = theta1
    else:
        major_axis, minor_axis = sx, sy
        major_angle = theta2

    return major_axis, minor_axis, major_angle

@ellipse.register
def _(points: GeoPandasBase,
      weights=None,
      method='crimestat',
      crimestatCorr=True,
      degfreedCorr=True) -> shapely.Polygon:
    coords = shapely.get_coordinates(points.geometry)
    major, minor, rotation = ellipse(coords,
                                       weights=weights,
                                       method=method,
                                       crimestatCorr=crimestatCorr,
                                       degfreedCorr=degfreedCorr)
    if weights is None:
        centre = mean_center(points).buffer(1)
    else:
        centre = weighted_mean_center(points, weights=weights).buffer(1)
        
    scaled = shapely.affinity.scale(centre, major, minor)
    rotated = shapely.affinity.rotate(scaled, rotation, use_radians=True)
    return rotated


@singledispatch
def dtot(coord, points) -> float:
    """
    Sum of Euclidean distances between event points and a selected point.

    Parameters
    ----------
    coord
              starting point
    points  : arraylike
              array representing a point pattern

    Returns
    -------
    distance
        sum of Euclidean distances.

    Examples
    --------
    >>> import numpy as np
    >>> import geopandas as gpd
    >>> import shapely

    Create an array of point coordinates.

    >>> coords = np.array(
    ...     [
    ...         [66.22, 32.54],
    ...         [22.52, 22.39],
    ...         [31.01, 81.21],
    ...         [9.47, 31.02],
    ...         [30.78, 60.10],
    ...         [75.21, 58.93],
    ...         [79.26, 7.68],
    ...         [8.23, 39.93],
    ...         [98.73, 77.17],
    ...         [89.78, 42.53],
    ...         [65.19, 92.08],
    ...     ]
    ... )

    Passing an array of coordinates returns a tuple capturing the bounds.

    >>> point = [30.78, 60.10]
    >>> dtot(point, coords)
    np.float64(465.3617957739617)

    The same applies to a GeoPandas object.

    >>> point = shapely.Point(point)
    >>> geoms = gpd.GeoSeries.from_xy(*coords.T)
    >>> dtot(point, geoms)
    np.float64(465.3617957739617)

    """
    try:
        coord = np.asarray(coord)
        points = np.asarray(points)
        return dtot(coord, points)
    except AttributeError as e:
        raise NotImplementedError from e


@dtot.register
def _(coord: np.ndarray, points: np.ndarray) -> float:
    xd = points[:, 0] - coord[0]
    yd = points[:, 1] - coord[1]
    d = np.sqrt(xd * xd + yd * yd).sum()
    return d


@dtot.register
def _(coord: shapely.Point, points: GeoPandasBase) -> float:
    coord = shapely.get_coordinates(coord).flatten()
    points = shapely.get_coordinates(points.geometry)
    return dtot(coord, points)


@singledispatch
def euclidean_median(points):
    """
    Calculate the Euclidean median for a point pattern.

    Parameters
    ----------
    points  : arraylike
        array representing a point pattern

    Returns
    -------
    median
        Euclidean median of a given point pattern

    Examples
    --------
    >>> import numpy as np
    >>> import geopandas as gpd

    Create an array of point coordinates.

    >>> coords = np.array(
    ...     [
    ...         [66.22, 32.54],
    ...         [22.52, 22.39],
    ...         [31.01, 81.21],
    ...         [9.47, 31.02],
    ...         [30.78, 60.10],
    ...         [75.21, 58.93],
    ...         [79.26, 7.68],
    ...         [8.23, 39.93],
    ...         [98.73, 77.17],
    ...         [89.78, 42.53],
    ...         [65.19, 92.08],
    ...     ]
    ... )

    Passing an array of coordinates returns an array capturing the median.

    >>> euclidean_median(coords)
    array([53.51770575, 49.6572671 ])

    Passing a GeoPandas object returns a shapely geometry.

    >>> geoms = gpd.GeoSeries.from_xy(*coords.T)
    >>> euclidean_median(geoms)
    <POINT (53.518 49.657)>

    """
    try:
        points = np.asarray(points)
        return euclidean_median(points)
    except AttributeError as e:
        raise NotImplementedError from e


@euclidean_median.register
def _(points: np.ndarray) -> NDArray[np.float64]:
    start = mean_center(points)
    res = minimize(dtot, start, args=(points,))
    return res["x"]

### ROBUST CENTROGRAPHY STATISTICS

def trim_pointset(points, p=1, peeling=True, hull='convex', **hull_args):
    """
    Pare a spatial point set down to some minimal pointset using a
    onion peeling/potato paring heuristic.

    Parameters
    ----------
    points : numpy.ndarray or geopandas.GeoSeries
        input points intended to be pared
    p : float
        value between 0 and 1 indicating how much of the point set should be left
        after paring. Values closer to 1 indicate more points will be retained,
        while values closer to 0 indicate more points will be removed. Note that
        in the worst case, slightly less than p will be returned.
    peeling: bool
        whether to use the peeling heuristic. By default, the peeling heuristic is used.
            - For the default peeling heuristic, we build a hull for the candidate point set,
            then remove *all points on the hull*.
            - For the alternative paring heuristic, we build a hull for the candidate point
            set, then remove the point on the hull that is at the center of the smallest angle.
    hull: str or callable
        algorithm to use for the hull in each iteration. If "convex" (default), the
        convex hull will be used in each iteration. If 'concave', then the concave
        hull will be used. If a function is provided, then it must take a shapely
        multipoint object and return a polygon whose boundary intersects the input
        points.
    **hull_args: arguments passed straight to the hulling function.
    Returns
    -------
    a numpy array aligned with the input points that is "True" when a point remains
    in the dataset, and "False" when the point has been pared.

    Notes
    -----
    If convex=False, the concave hull algorithm provided by geopandas is used by default.
    If you use convex=False, you should also provide a "ratio" argument, which controls
    the percentage of the input dataset that is contained within the concave hull in
    each iteration. Generally, ratio should be larger than p if peeling=True.
    """
    if isinstance(points, GeoSeries):
        pass
    elif isinstance(points, GeoDataFrame):
        points = points.geometry
    else:
        if points.shape[1] == 2:
            points = GeoSeries(points_from_xy(*points.T))
        else:
            points = GeoSeries(points)
    n_points = len(points)
    trimmed = pandas.Series(np.zeros(n_points), index=points.index).astype(bool)
    if hull=='convex':
        huller = lambda x: x.convex_hull
    elif hull == 'concave':
        huller = lambda x: GeoSeries(x).concave_hull(**hull_args).item()
    elif callable(hull):
        huller = lambda x: hull(x, **hull_args)
    else:
        raise ValueError("hull argument must be either `concave', `convex', or callable.")
    while True:
        pointset = points[~trimmed].union_all()
        hull_ = huller(pointset)
        on_hull = hull_.boundary.intersects(points)
        if not on_hull.any():
            raise ValueError("Hull algorithm must return shape that intersects at least some of the input points.")
        if peeling:
            q = (trimmed | on_hull).mean()
        else:
            q = ((trimmed).sum() + 1)/n_points
        if (1 - q) < p:
            break
        if peeling:
            trimmed[on_hull] = True
        else:
            # roll the angles by 1 because the angles are aligned on the LEFT EDGE. So, the first
            # element of get_angles() is the angle centered on 2, with 1 as its left edge.
            angles = np.roll(shape.get_angles(hull_.boundary), 1)
            if (angles < 0).any() | (angles > np.pi).any():
                raise NotImplementedError("negative angle or reflexive angle encountered. Your polygon likely has an incorrect winding direction.")
            # ring is always closed at the same point as it starts
            ordered_hull_points = GeoSeries(points_from_xy(*hull_.boundary.xy)[:-1])
            hull_locs, points_hull_locs = points[on_hull.values].sindex.query(ordered_hull_points, predicate='intersects')
            points_hull_ixs = points[on_hull.values].index[points_hull_locs]
            current_knob_on_hull = angles.argmin()
            current_knob_ix = points_hull_ixs[current_knob_on_hull]
            # for the points on the hull, get the ith observation's full dataset index
            trimmed.loc[current_knob_ix] = True
    return ~trimmed

def trimmed_hull(points, p=1, peeling=True, hull='convex', **hull_args):
    """
    Calculate the trimmed hull of an input point set that covers at least p% of the
    input pointset.

    Parameters
    ----------
    points : numpy.ndarray or GeoSeries
        input points intended to be pared
    p : float
        value between 0 and 1 indicating how much of the point set should be left
        after paring. Values closer to 1 indicate more points will be retained,
        while values closer to 0 indicate more points will be removed. Note that
        in the worst case, slightly less than p will be returned.
    peeling: bool
        whether to use the peeling heuristic. By default, the peeling heuristic is used.
            - For the default peeling heuristic, we build a hull for the candidate point set,
            then remove *all points on the hull*.
            - For the alternative paring heuristic, we build a hull for the candidate point
            set, then remove the point on the hull that is at the center of the smallest angle.
    hull: str or callable
        algorithm to use for the hull in each iteration. If "convex" (default), the
        convex hull will be used in each iteration. If 'concave', then the concave
        hull will be used. If a function is provided, then it must take a shapely
        multipoint object and return a polygon whose boundary intersects the input
        points.
    **hull_args: arguments passed straight to the hulling function.
    Returns
    -------
    a numpy array aligned with the input points that is "True" when a point remains
    in the dataset, and "False" when the point has been pared.

    Notes
    -----
    If convex=False, the concave hull algorithm provided by geopandas is used by default.
    If you use convex=False, you should also provide a "ratio" argument, which controls
    the percentage of the input dataset that is contained within the concave hull in
    each iteration. Generally, ratio should be larger than p if peeling=True.

    The peeling heuristic tends to maintain the shape of the original pointcloud, while the
    paring heuristic tends to return central coverage areas that are more circular.
    """
    if isinstance(points, GeoSeries):
        pass
    elif isinstance(points, GeoDataFrame):
        points = points.geometry
    else:
        if points.shape[1] == 2:
            points = GeoSeries(points_from_xy(*points.T))
        else:
            points = GeoSeries(points)
    mask = trim_pointset(points, p=p, peeling=peeling, hull=hull, hull_args=hull_args)
    if hull == 'convex':
        return points[mask.values].union_all().convex_hull
    elif hull == 'concave':
        return GeoSeries(points[mask.values].union_all()).concave_hull(**hull_args).item()
    elif callable(hull):
        return hull(points[mask.values], **hull_args)
    else:
        raise ValueError(f"hull must be either `convex', `concave', or a callable function. Recieved {hull}")


### SKYUM's ALGORITHM (DEPRECATED)

@euclidean_median.register
def _(points: GeoPandasBase) -> shapely.Point:
    coords = shapely.get_coordinates(points.geometry)
    return shapely.Point(euclidean_median(coords))


@singledispatch
def minimum_bounding_circle(points):
    """
    Implements Skyum (1990)'s algorithm for the minimum bounding circle in R^2.

    Store points clockwise.
    Find p in S that maximizes angle(prec(p), p, succ(p) THEN radius(prec(
    p), p, succ(p)). This is also called the lexicographic maximum, and is the last
    entry of a list of (radius, angle) in lexicographical order.

    * If angle(prec(p), p, succ(p)) <= 90 degrees, then finish.

    * If not, remove p from set.

    Parameters
    ----------
    points  : arraylike
        array representing a point pattern

    Returns
    -------
    circle
        minimum bounding circle

    Examples
    --------
    >>> import numpy as np
    >>> import geopandas as gpd

    Create an array of point coordinates.

    >>> coords = np.array(
    ...     [
    ...         [66.22, 32.54],
    ...         [22.52, 22.39],
    ...         [31.01, 81.21],
    ...         [9.47, 31.02],
    ...         [30.78, 60.10],
    ...         [75.21, 58.93],
    ...         [79.26, 7.68],
    ...         [8.23, 39.93],
    ...         [98.73, 77.17],
    ...         [89.78, 42.53],
    ...         [65.19, 92.08],
    ...     ]
    ... )

    Passing an array of coordinates returns an tuple of (x, y), radius.

    >>> minimum_bounding_circle(coords)
    ((55.244520477497474, 51.88135107645883), np.float64(50.304102155590726))

    Passing a GeoPandas object returns a shapely geometry.

    >>> geoms = gpd.GeoSeries.from_xy(*coords.T)
    >>> minimum_bounding_circle(geoms)
    <POLYGON ((105.549 51.881, 105.306 46.951, 104.582 42.068, 103.383 37.279, 1...>
    """
    try:
        points = np.asarray(points)
        return minimum_bounding_circle(points)
    except AttributeError as e:
        raise NotImplementedError from e


@minimum_bounding_circle.register
def _(points: np.ndarray) -> tuple[tuple[float, float], float]:
    try:
        from numba import njit

        HAS_NUMBA = True
    except ImportError:
        HAS_NUMBA = False
    points = hull(points)
    if not_clockwise(points):
        points = points[::-1]
        if not_clockwise(points):
            raise Exception("Points are neither clockwise nor counterclockwise")
    POINTS = copy.deepcopy(points)
    if HAS_NUMBA:  # noqa: SIM108
        circ = _skyum_numba(POINTS)[0]
    else:
        circ = _skyum_lists(POINTS)[0]
    return (circ[1], circ[2]), circ[0]


@minimum_bounding_circle.register
def _(points: GeoPandasBase) -> shapely.Polygon:
    coords = shapely.get_coordinates(points.geometry)
    (x, y), r = minimum_bounding_circle(coords)
    return shapely.Point(x, y).buffer(r)


def _skyum_lists(points):
    points = points.tolist()
    removed = []
    i = 0
    while True:
        angles = [
            _angle(
                _prec(p, points),
                p,
                _succ(p, points),
            )
            for p in points
        ]
        circles = [
            _circle(
                _prec(p, points),
                p,
                _succ(p, points),
            )
            for p in points
        ]
        radii = [c[0] for c in circles]
        lexord = np.lexsort((radii, angles))  # confusing as hell defaults...
        lexmax = lexord[-1]
        candidate = (
            _prec(points[lexmax], points),
            points[lexmax],
            _succ(points[lexmax], points),
        )
        if angles[lexmax] <= (np.pi / 2.0):
            # print("Constrained by points: {}".format(candidate))
            return _circle(*candidate), points, removed, candidate
        else:
            try:
                removed.append((points.pop(lexmax), i))
            except IndexError:
                raise Exception("Construction of Minimum Bounding Circle failed!")
        i += 1


try:
    from numba import njit, boolean

    HAS_NUMBA = True

    @njit(fastmath=True)
    def _skyum_numba(points):
        i = 0
        complete = False
        while not complete:
            complete, points, candidate, circle = _skyum_iteration(points)
            if complete:
                return circle, points, None, candidate

    @njit(fastmath=True)
    def _skyum_iteration(points):
        points = points.reshape(-1, 2)
        n = points.shape[0]
        angles = np.empty((n,))
        circles = np.empty((n, 3))
        for i in range(n):
            p = points[(i - 1) % n]
            q = points[i % n]
            r = points[(i + 1) % n]
            angles[i] = _angle(p, q, r)
            circles[i] = _circle(p, q, r)
        radii = circles[:, 0]
        # workaround for no lexsort in numba
        angle_argmax = angles.argmax()
        angle_max = angles[angle_argmax]
        # the maximum radius for the largest angle
        lexmax = (radii * (angles == angle_max)).argmax()

        if angles[lexmax] <= (np.pi / 2.0):
            return True, points, lexmax, circles[lexmax]
        else:
            mask = np.ones((n,), dtype=boolean)
            mask[lexmax] = False
            new_points = points[mask, :]
            return False, new_points, lexmax, circles[lexmax]


except ModuleNotFoundError:

    def njit(func, **kwargs):
        return func


@njit
def _angle(p, q, r):
    p = np.asarray(p)
    q = np.asarray(q)
    r = np.asarray(r)
    pq = p - q
    rq = r - q
    magnitudes = np.linalg.norm(pq) * np.linalg.norm(rq)
    return np.abs(np.arccos(np.dot(pq, rq) / magnitudes))


def _prec(p, l):
    """
    retrieve the predecessor of p in list l
    """
    pos = l.index(p)
    if pos - 1 < 0:
        return l[-1]
    else:
        return l[pos - 1]


def _succ(p, l):
    """
    retrieve the successor of p in list l
    """
    pos = l.index(p)
    if pos + 1 >= len(l):
        return l[0]
    else:
        return l[pos + 1]


@njit
def _euclidean_distance(px, py, qx, qy):
    return np.sqrt((px - qx) ** 2 + (py - qy) ** 2)


@njit
def _circle(p, q, r, dmetric=_euclidean_distance):
    """
    Returns (radius, (center_x, center_y)) of the circumscribed circle by the
    triangle pqr.

    note, this does not assume that p!=q!=r
    """
    px, py = p
    qx, qy = q
    rx, ry = r
    angle = np.abs(_angle(p, q, r))
    if np.abs(angle - np.pi) < 1e-5:  # angle is pi
        radius = _euclidean_distance(px, py, rx, ry) / 2.0
        center_x = (px + rx) / 2.0
        center_y = (py + ry) / 2.0
    elif np.abs(angle) < 1e-5:  # angle is zero
        radius = _euclidean_distance(px, py, qx, qy) / 2.0
        center_x = (px + qx) / 2.0
        center_y = (py + qy) / 2.0
    else:
        D = 2 * (px * (qy - ry) + qx * (ry - py) + rx * (py - qy))
        center_x = (
            (px**2 + py**2) * (qy - ry)
            + (qx**2 + qy**2) * (ry - py)
            + (rx**2 + ry**2) * (py - qy)
        ) / float(D)
        center_y = (
            (px**2 + py**2) * (rx - qx)
            + (qx**2 + qy**2) * (px - rx)
            + (rx**2 + ry**2) * (qx - px)
        ) / float(D)
        radius = _euclidean_distance(center_x, center_y, px, py)
    return radius, center_x, center_y
