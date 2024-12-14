"""
Centrographic measures for point patterns

- documentation

"""

__author__ = "Serge Rey sjsrey@gmail.com"

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
from libpysal.cg import is_clockwise
from numpy.typing import NDArray
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
             (n,2), (x,y) coordinates of a series of event points.

    Returns
    -------
    min_x  : float
             leftmost value of the vertices of minimum bounding rectangle.
    min_y  : float
             downmost value of the vertices of minimum bounding rectangle.
    max_x  : float
             rightmost value of the vertices of minimum bounding rectangle.
    max_y  : float
             upmost value of the vertices of minimum bounding rectangle.

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
    Compute the minimum rotated rectangle for an input point set.

    This is the smallest enclosing rectangle (possibly rotated)
    for the input point set. It is computed using Shapely.

    Parameters
    ----------
    points : numpy.ndarray
        A numpy array of shape (n_observations, 2) containing the point
        locations to compute the rotated rectangle
    return_angle : bool
        whether to return the angle (in degrees) of the angle between
        the horizontal axis of the rectanle and the first side (i.e. length).

    Returns
    -------
    an numpy.ndarray of shape (4, 2) containing the coordinates
    of the minimum rotated rectangle. If return_angle is True,
    also return the angle (in degrees) of the rotated rectangle.

    """
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
    points: arraylike
            (n,2), (x,y) coordinates of a series of event points.

    Returns
    -------
    _     : array
            (h,2), points defining the hull in counterclockwise order.
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
    points: arraylike
            (n,2), (x,y) coordinates of a series of event points.

    Returns
    -------
    _     : array
            (2,), (x,y) coordinates of the mean center.
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
    points  : arraylike
              (n,2), (x,y) coordinates of a series of event points.
    weights : arraylike
              a series of attribute values of length n.

    Returns
    -------
    _      : array
             (2,), (x,y) coordinates of the weighted mean center.
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
              (n,2), (x,y) coordinates of a series of event points.

    Returns
    -------
    _      : array
             (2,), (x,y) coordinates of the manhattan median.
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
def std_distance(points) -> NDArray[np.float64]:
    """
    Calculate standard distance of a point array.

    Parameters
    ----------
    points  : arraylike
              (n,2), (x,y) coordinates of a series of event points.

    Returns
    -------
    _      : float
             standard distance.
    """
    try:
        points = np.asarray(points)
        return std_distance(points)
    except AttributeError as e:
        raise NotImplementedError from e


@std_distance.register
def _(points: np.ndarray) -> NDArray[np.float64]:
    n, _ = points.shape
    m = points.mean(axis=0)
    return np.sqrt(((points * points).sum(axis=0) / n - m * m).sum())


@std_distance.register
def _(points: GeoPandasBase) -> NDArray[np.float64]:
    coords = shapely.get_coordinates(points.geometry)
    return std_distance(coords)


@singledispatch
def ellipse(points):
    """
    Calculate parameters of standard deviational ellipse for a point pattern.

    Parameters
    ----------
    points : arraylike
             (n,2), (x,y) coordinates of a series of event points.

    Returns
    -------
    _      : float
             semi-major axis.
    _      : float
             semi-minor axis.
    theta  : float
             clockwise rotation angle of the ellipse.

    Notes
    -----
    Implements approach from:

    https://www.icpsr.umich.edu/CrimeStat/files/CrimeStatChapter.4.pdf
    """
    try:
        points = np.asarray(points)
        return ellipse(points)
    except AttributeError as e:
        raise NotImplementedError from e


@ellipse.register
def _(points: np.ndarray) -> tuple[float, float, float]:
    n, _ = points.shape
    x = points[:, 0]
    y = points[:, 1]
    xd = x - x.mean()
    yd = y - y.mean()
    xss = (xd * xd).sum()
    yss = (yd * yd).sum()
    cv = (xd * yd).sum()
    num = (xss - yss) + np.sqrt((xss - yss) ** 2 + 4 * (cv) ** 2)
    den = 2 * cv
    theta = np.arctan(num / den)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    n_2 = n - 2
    sd_x = (2 * (xd * cos_theta - yd * sin_theta) ** 2).sum() / n_2
    sd_y = (2 * (xd * sin_theta - yd * cos_theta) ** 2).sum() / n_2
    return np.sqrt(sd_x), np.sqrt(sd_y), theta


@ellipse.register
def _(points: GeoPandasBase) -> shapely.Polygon:
    coords = shapely.get_coordinates(points.geometry)
    major, minor, rotation = ellipse(coords)
    centre = mean_center(points).buffer(1)
    scaled = shapely.affinity.scale(centre, major, minor)
    rotated = shapely.affinity.rotate(scaled, rotation, use_radians=True)
    return rotated


@singledispatch
def dtot(coord, points) -> float:
    """
    Sum of Euclidean distances between event points and a selected point.

    Parameters
    ----------
    coord   : arraylike
              (x,y) coordinates of a point.
    points  : arraylike
              (n,2), (x,y) coordinates of a series of event points.

    Returns
    -------
    d       : float
              sum of Euclidean distances.

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
    points: arraylike
            (n,2), (x,y) coordinates of a series of event points.

    Returns
    -------
    _     : array
            (2,), (x,y) coordinates of the Euclidean median.

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
    points  :   numpy.ndarray
        a numpy array of shape (n_observations, 2) to compute
        the minimum bounding circle

    Returns
    -------
    (x,y),radius for the minimum bounding circle.
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
