"""
Centrographic measures for point patterns

TODO

- testing
- documentation

"""

__author__ = "Serge Rey sjsrey@gmail.com"

__all__ = [
    "mbr",
    "hull",
    "mean_center",
    "weighted_mean_center",
    "manhattan_median",
    "std_distance",
    "euclidean_median",
    "ellipse",
    "minimum_rotated_rectangle",
    "minimum_bounding_rectangle",
    "skyum",
    "dtot",
    "_circle",
]


import sys
import numpy as np
import warnings
import copy

from math import pi as PI
from scipy.spatial import ConvexHull
from libpysal.cg import get_angle_between, Ray, is_clockwise
from scipy.spatial import distance as dist
from scipy.optimize import minimize

not_clockwise = lambda x: not is_clockwise(x)

MAXD = sys.float_info.max
MIND = sys.float_info.min


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
    points = np.asarray(points)
    min_x = min_y = MAXD
    max_x = max_y = MIND
    x, y = zip(*points)
    min_x = min(x)
    min_y = min(y)
    max_x = max(x)
    max_y = max(y)
    return min_x, min_y, max_x, max_y


def minimum_rotated_rectangle(points, return_angle=False):
    """
    Compute the minimum rotated rectangle for an input point set.

    This is the smallest enclosing rectangle (possibly rotated)
    for the input point set. It is computed using OpenCV2, so
    if that is not available, then this function will fail.

    Parameters
    ----------
    points : numpy.ndarray
        A numpy array of shape (n_observations, 2) containing the point
        locations to compute the rotated rectangle
    return_angle : bool
        whether to return the angle (in degrees) of the angle between
        the horizontal axis of the rectanle and the first side (i.e. length).
        Computed directly from cv2.minAreaRect.

    Returns
    -------
    an numpy.ndarray of shape (4, 2) containing the coordinates
    of the minimum rotated rectangle. If return_angle is True,
    also return the angle (in degrees) of the rotated rectangle.

    """
    points = np.asarray(points)
    try:
        from cv2 import minAreaRect, boxPoints
    except (ImportError, ModuleNotFoundError):
        raise ModuleNotFoundError("OpenCV2 is required to use this function. Please install it with `pip install opencv-contrib-python`")
    ((x, y), (w, h), angle) = rot_rect = minAreaRect(points.astype(np.float32))
    out_points = boxPoints(rot_rect)
    if return_angle:
        return (out_points, angle)
    return out_points


def mbr(points):
    warnings.warn(
        "This function will be deprecated in the next release of pointpats.",
        FutureWarning,
        stacklevel=2,
    )
    return minimum_bounding_rectangle(points)


mbr.__doc__ = minimum_bounding_rectangle.__doc__


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

    points = np.asarray(points)
    h = ConvexHull(points)
    return points[h.vertices]


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

    points = np.asarray(points)
    return points.mean(axis=0)


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

    points, weights = np.asarray(points), np.asarray(weights)
    w = weights * 1.0 / weights.sum()
    w.shape = (1, len(points))
    return np.dot(w, points)[0]


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

    points = np.asarray(points)
    if not len(points) % 2:
        s = "Manhattan Median is not unique for even point patterns."
        warnings.warn(s)
    return np.median(points, axis=0)


def std_distance(points):
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

    points = np.asarray(points)
    n, p = points.shape
    m = points.mean(axis=0)
    return np.sqrt(((points * points).sum(axis=0) / n - m * m).sum())


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

    points = np.asarray(points)
    n, k = points.shape
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


def dtot(coord, points):
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
    points = np.asarray(points)
    xd = points[:, 0] - coord[0]
    yd = points[:, 1] - coord[1]
    d = np.sqrt(xd * xd + yd * yd).sum()
    return d


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
    points = np.asarray(points)
    start = mean_center(points)
    res = minimize(dtot, start, args=(points,))
    return res["x"]


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
    (x,y),center for the minimum bounding circle.
    """
    points = hull(points)
    if not_clockwise(points):
        points = points[::-1]
        if not_clockwise(points):
            raise Exception("Points are neither clockwise nor counterclockwise")
    POINTS = copy.deepcopy(points)
    removed = []
    i = 0
    if HAS_NUMBA:
        circ = _skyum_numba(POINTS)[0]
    else:
        circ = _skyum_lists(POINTS)[0]
    return (circ[1], circ[2]), circ[0]


def skyum(points):
    warnings.warn(
        "This function will be deprecated in the next release of pointpats.",
        FutureWarning,
        stacklevel=2,
    )
    return minimum_bounding_circle(points)


skyum.__doc__ = (
    "WARNING: This function is deprecated in favor of minimum_bounding_circle\n"
    + minimum_bounding_circle.__doc__
)


def _skyum_lists(points):
    points = points.tolist()
    removed = []
    i = 0
    while True:
        angles = [_angle(_prec(p, points), p, _succ(p, points),) for p in points]
        circles = [_circle(_prec(p, points), p, _succ(p, points),) for p in points]
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

        candidate = (lexmax - 1) % n, lexmax, (lexmax + 1) % n
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
            (px ** 2 + py ** 2) * (qy - ry)
            + (qx ** 2 + qy ** 2) * (ry - py)
            + (rx ** 2 + ry ** 2) * (py - qy)
        ) / float(D)
        center_y = (
            (px ** 2 + py ** 2) * (rx - qx)
            + (qx ** 2 + qy ** 2) * (px - rx)
            + (rx ** 2 + ry ** 2) * (qx - px)
        ) / float(D)
        radius = _euclidean_distance(center_x, center_y, px, py)
    return radius, center_x, center_y
