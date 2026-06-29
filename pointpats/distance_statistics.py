import warnings
from collections import namedtuple

import geopandas
import numpy
import shapely
from joblib import Parallel, delayed
from scipy import interpolate, spatial

from .geometry import TREE_TYPES
from .geometry import area as _area
from .geometry import build_best_tree as _build_best_tree
from .geometry import k_neighbors as _k_neighbors
from .geometry import max_radius as _max_radius
from .geometry import prepare_hull as _prepare_hull
from .random import poisson

__all__ = [
    "f",
    "g",
    "k",
    "j",
    "l",
    "f_test",
    "g_test",
    "k_test",
    "j_test",
    "l_test",
    "KEstResult",
    "LEstResult",
]

# Sentinel used as the default for edge_correction in k() and l().
# Distinguishes "caller passed None (uncorrected)" from "caller passed nothing
# (compute all three spatstat-default corrections)".
_NOTSET = object()

KEstResult = namedtuple(
    "KEstResult", ("support", "theo", "border", "isotropic", "translate")
)
LEstResult = namedtuple(
    "LEstResult", ("support", "theo", "border", "isotropic", "translate")
)


def _prepare(coordinates, support, distances, metric, hull, edge_correction):
    """
    prepare the arguments to convert into a standard format
    1. cast the coordinates to a numpy array
    2. precomputed metrics must have distances provided
    3. metrics must be callable or string
    4. warn if distances are specified and metric is not default
    5. make distances a numpy.ndarray
    6. construct the support, accepting:
        - num_steps -> a linspace with len(support) == num_steps
                       from zero to a quarter of the bounding box's smallest side
        - (stop, ) -> a linspace with len(support) == 20
                 from zero to stop
        - (start, stop) -> a linspace with len(support) == 20
                           from start to stop
        - (start, stop, num_steps) -> a linspace with len(support) == num_steps
                                      from start to stop
        - numpy.ndarray -> passed through
    """
    # Throw early if edge correction is requested
    if edge_correction is not None:
        raise NotImplementedError("Edge correction is not currently implemented.")

    if isinstance(coordinates, geopandas.GeoDataFrame | geopandas.GeoSeries):
        coordinates = shapely.get_coordinates(coordinates.geometry)

    # cast to coordinate array
    if isinstance(coordinates, TREE_TYPES):
        tree = coordinates
        coordinates = tree.data
    else:
        coordinates = numpy.asarray(coordinates)
    hull = _prepare_hull(coordinates, hull)

    # evaluate distances
    if (distances is None) and metric == "precomputed":
        raise ValueError(
            "If metric =`precomputed` then distances must"
            " be provided as a (n,n) numpy array."
        )
    if not (isinstance(metric, str) or callable(metric)):
        raise TypeError(
            f"`metric` argument must be callable or a string. Recieved: {metric}"
        )
    if distances is not None and metric != "euclidean":
        warnings.warn(
            "Distances were provided. The specified metric will be ignored."
            " To use precomputed distances with a custom distance metric,"
            " do not specify a `metric` argument.",
            stacklevel=2,
        )
        metric = "euclidean"

    if support is None:
        support = 20

    if isinstance(support, int):  # if just n_steps, use the max nnd
        # this is O(n log n) for kdtrees & balltrees
        tmp_tree = _build_best_tree(coordinates, metric=metric)
        max_dist = _k_neighbors(tmp_tree, coordinates, 1)[0].max()
        support = numpy.linspace(0, max_dist, num=support)
    # otherwise, we need to build it using (start, stop, step) semantics
    elif isinstance(support, tuple):
        if len(support) == 1:  # assuming this is with zero implicit start
            support = numpy.linspace(0, support[0], num=20)  # default support n bins
        elif len(support) == 2:
            support = numpy.linspace(*support, num=20)  # default support n bins
        elif len(support) == 3:
            support = numpy.linspace(support[0], support[1], num=support[2])
    else:  # try to use it as is
        try:
            support = numpy.asarray(support)
        except:  # noqa: E722 - bare `except`
            raise TypeError(
                "`support` must be a tuple (either (start, stop, step), (start, stop) "
                "or (stop,)), an int describing the number of breaks to use to evalute "
                "the function, or an iterable containing the breaks to use to evaluate "
                f"the function. Received object of type {type(support)}: {support}"
            ) from None

    return coordinates, support, distances, metric, hull, edge_correction


def _hull_to_poly(hull_prepared):
    """Convert a prepared hull (bbox array, ConvexHull, or shapely geometry) to a
    shapely polygon, required for boundary-distance computation in erosion correction."""
    if isinstance(hull_prepared, shapely.Geometry):
        return hull_prepared
    if isinstance(hull_prepared, numpy.ndarray):
        return shapely.box(*hull_prepared)
    if isinstance(hull_prepared, spatial.ConvexHull):
        pts = hull_prepared.points[hull_prepared.vertices]
        return shapely.from_wkt(
            shapely.to_wkt(shapely.convex_hull(shapely.multipoints(pts)))
        )
    raise ValueError(
        "Edge correction with erosion requires a hull that can be converted to a "
        "shapely Polygon. Provide hull as a shapely Polygon, a bounding box array "
        "[xmin, ymin, xmax, ymax], or use hull='convex' or hull='alpha'."
    )


def _ripley_circle_weights(coordinates, poly, support, n_circle=36):
    """Per-point isotropic circle-sampling weights for Ripley's K correction.

    For each point i and radius r, w_i(r) = n_circle / n_inside, where n_inside
    is the count of equally-spaced circle points (radius r, centred at i) that
    fall inside *poly*.  Points whose full circle is inside the window get w=1.
    """
    n = len(coordinates)
    shapely_pts = shapely.points(coordinates[:, 0], coordinates[:, 1])
    dist_to_boundary = shapely.distance(shapely_pts, poly.boundary)

    angles = numpy.linspace(0, 2 * numpy.pi, n_circle, endpoint=False)
    cos_a = numpy.cos(angles)
    sin_a = numpy.sin(angles)

    weights = numpy.ones((n, len(support)))

    for j, r in enumerate(support):
        if r == 0:
            continue
        near = dist_to_boundary <= r
        if not near.any():
            continue
        idx = numpy.where(near)[0]
        # Circle sample points: shapes broadcast to (len(idx), n_circle)
        cx = coordinates[idx, 0:1] + r * cos_a
        cy = coordinates[idx, 1:2] + r * sin_a
        circle_pts = shapely.points(cx.ravel(), cy.ravel())
        inside = shapely.within(circle_pts, poly).reshape(len(idx), n_circle)
        n_inside = inside.sum(axis=1)
        # n_circle / n_inside; fall back to n_circle if no points inside (degenerate)
        weights[idx, j] = numpy.where(n_inside > 0, n_circle / n_inside, float(n_circle))

    return weights


def _ripley_analytic_weights(coordinates, poly, support):
    """Per-point analytic area-intersection weights for Ripley's K correction.

    For each point i and radius r, w_i(r) = π r² / area(circle(i,r) ∩ poly).
    Points whose full circle lies inside the window get w=1.
    """
    n = len(coordinates)
    shapely_pts = shapely.points(coordinates[:, 0], coordinates[:, 1])
    dist_to_boundary = shapely.distance(shapely_pts, poly.boundary)

    weights = numpy.ones((n, len(support)))

    for j, r in enumerate(support):
        if r == 0:
            continue
        near = dist_to_boundary <= r
        if not near.any():
            continue
        idx = numpy.where(near)[0]
        circle_area = numpy.pi * r * r
        circles = shapely.buffer(shapely_pts[idx], r)
        inter_areas = shapely.area(shapely.intersection(circles, poly))
        weights[idx, j] = numpy.where(inter_areas > 0, circle_area / inter_areas, circle_area)

    return weights


def _isotropic_weights(coordinates, poly, support):
    """Per-point exact arc-fraction weights for Ripley's isotropic K correction.

    For each point i and radius r, w_i(r) = 2πr / arc_inside, where arc_inside is
    the total length of the circle circumference (radius r, centred at i) that lies
    inside poly. Points whose full circle is inside the window get w=1.
    """
    n = len(coordinates)
    shapely_pts = shapely.points(coordinates[:, 0], coordinates[:, 1])
    dist_to_boundary = shapely.distance(shapely_pts, poly.boundary)

    weights = numpy.ones((n, len(support)))

    for j, r in enumerate(support):
        if r == 0:
            continue
        near = dist_to_boundary <= r
        if not near.any():
            continue
        idx = numpy.where(near)[0]
        circumference = 2 * numpy.pi * r
        circles = shapely.buffer(shapely_pts[idx], r)
        arc_inside = shapely.intersection(shapely.boundary(circles), poly)
        arc_lengths = shapely.length(arc_inside)
        weights[idx, j] = numpy.where(
            arc_lengths > 0, circumference / arc_lengths, circumference
        )

    return weights


def _translate_pair_weights(coordinates, poly, area):
    """Per-pair translation weights for the translation edge correction.

    For each unordered pair (i, j), the weight is area(W)² / area(W ∩ (W + h_ij))
    where h_ij = x_j - x_i.

    Returns a 1-D array of shape (n*(n-1)//2,) aligned with scipy pdist output.
    """
    n = len(coordinates)
    rows, cols = numpy.triu_indices(n, k=1)
    n_pairs = len(rows)
    translations = coordinates[cols] - coordinates[rows]  # (n_pairs, 2)

    exterior_coords = numpy.array(poly.exterior.coords)  # (m, 2) — closed ring
    shifted_exterior = exterior_coords[None] + translations[:, None]  # (n_pairs, m, 2)

    interior_rings = list(poly.interiors)
    if not interior_rings:
        shifted_polys = shapely.polygons(shapely.linearrings(shifted_exterior))
    else:
        all_shifted_holes = [
            numpy.array(ring.coords)[None] + translations[:, None]
            for ring in interior_rings
        ]
        shifted_polys = numpy.empty(n_pairs, dtype=object)
        for k_idx in range(n_pairs):
            shell = shapely.linearrings(shifted_exterior[k_idx])
            hole_rings = [shapely.linearrings(sh[k_idx]) for sh in all_shifted_holes]
            shifted_polys[k_idx] = shapely.polygons(shell, hole_rings)

    orig_array = numpy.full(n_pairs, poly)
    overlap_areas = shapely.area(shapely.intersection(orig_array, shifted_polys))
    return numpy.where(overlap_areas > 0, area * area / overlap_areas, area * area)


# ------------------------------------------------------------#
# Statistical Functions                                       #
# ------------------------------------------------------------#


def f(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
):
    """Ripley's F function

    The so-called "empty space" function, this is the cumulative density function of
    the distances from a random set of points to the known points in the pattern.

    Parameters
    ----------
    coordinates : geopandas object | numpy.ndarray of shape (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, p) or (p,)
        distances from every point in a random point set of size p
        to some point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.

    Returns
    -------
    a tuple containing the support values used to evalute the function
    and the values of the function at each distance value in the support.
    """
    coordinates, support, distances, metric, hull, _ = _prepare(
        coordinates, support, distances, metric, hull, edge_correction
    )
    if distances is not None:
        n = coordinates.shape[0]
        if distances.ndim == 2:
            k, p = distances.shape
            if k == p == n:
                warnings.warn(
                    f"A full distance matrix is not required for this function, and"
                    f" the intput matrix is a square {n},{n} matrix. Only the"
                    f" distances from p random points to their nearest neighbor within"
                    f" the pattern is required, as an {n},p matrix. Assuming the"
                    f" provided distance matrix has rows pertaining to input"
                    f" pattern and columns pertaining to the output points.",
                    stacklevel=2,
                )
                distances = distances.min(axis=0)
            elif k == n:
                distances = distances.min(axis=0)
            else:
                raise ValueError(
                    f"Distance matrix should have the same rows as the input"
                    f" coordinates with p columns, where n may be equal to p."
                    f" Recieved an {k},{p} distance matrix for {n} coordinates"
                )
        elif distances.ndim == 1:
            p = len(distances)
    else:
        # Do 1000 empties. Users can control this by computing their own
        # empty space distribution.
        n_empty_points = 1000

        randoms = poisson(hull=hull, size=(n_empty_points, 1))
        try:
            tree  # noqa: B018
        except NameError:
            tree = _build_best_tree(coordinates, metric)
        finally:
            distances, _ = tree.query(randoms, k=1)
            distances = distances.squeeze()

    counts, bins = numpy.histogram(distances, bins=support)
    fracs = numpy.cumsum(counts) / counts.sum()

    return bins, numpy.asarray([0, *fracs])


def g(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
):
    """Ripley's G function

    The G function is computed from the cumulative density function of the
    nearest neighbor distances between points in the pattern.

    Parameters
    ----------
    coordinates : geopandas object | numpy.ndarray of shape (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, n) or (n,)
        distances from every point in the point to another point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or None
        the study area geometry. Required when edge_correction='erosion'.
    edge_correction: None or 'erosion'
        edge correction method. When 'erosion', only points whose distance to the
        study window boundary exceeds r contribute to G(r). The support is
        automatically clipped to the erosion threshold radius returned by
        max_radius(..., method='erosion_threshold').

    Returns
    -------
    a tuple containing the support values used to evaluate the function
    and the values of the function at each distance value in the support.
    """
    if edge_correction not in (None, "erosion", True):
        raise ValueError(
            f"edge_correction must be None or 'erosion'. Got {edge_correction!r}"
        )
    use_erosion = edge_correction in ("erosion", True)

    # _prepare raises NotImplementedError for non-None edge_correction;
    # pass None and handle erosion ourselves after preparation.
    coordinates, support, distances, metric, hull_prepared, _ = _prepare(
        coordinates, support, distances, metric, hull, None
    )

    if distances is not None:
        if distances.ndim == 2:
            if distances.shape[0] == distances.shape[1] == coordinates.shape[0]:
                warnings.warn(
                    "The full distance matrix is not required for this function,"
                    " only the distance to the nearest neighbor within the pattern."
                    " Computing this and discarding the rest.",
                    stacklevel=2,
                )
                distances = distances.min(axis=1)
            else:
                k, p = distances.shape
                n = coordinates.shape[0]
                raise ValueError(
                    " Input distance matrix has an invalid shape: {k},{p}."
                    " Distances supplied can either be 2 dimensional"
                    " square matrices with the same number of rows"
                    f" as `coordinates` ({n}) or 1 dimensional and contain"
                    " the shortest distance from each point in "
                    " `coordinates` to some other point in coordinates."
                )
        elif distances.ndim == 1:
            if distances.shape[0] != coordinates.shape[0]:
                raise ValueError(
                    "Distances are not aligned with coordinates! Distance"
                    " matrix must be (n_coordinates, n_coordinates), but recieved"
                    f" {distances.shape} instead of ({coordinates.shape[0]},)"
                )
        else:
            raise ValueError(
                "Distances supplied can either be 2 dimensional"
                " square matrices with the same number of rows"
                " as `coordinates` or 1 dimensional and contain"
                " the shortest distance from each point in "
                " `coordinates` to some other point in coordinates."
                f" Input matrix was {distances.ndim} dimensional"
            )
        nnd = distances.squeeze()
    else:
        try:
            tree  # noqa: B018
        except NameError:
            tree = _build_best_tree(coordinates, metric)
        finally:
            distances, indices = _k_neighbors(tree, coordinates, k=1)
        nnd = distances.squeeze()

    if use_erosion:
        poly = _hull_to_poly(hull_prepared)

        # Clip the support to the erosion threshold radius so that at least
        # half the points remain as core (guard) points at the maximum r.
        max_r, _ = _max_radius(poly, points=coordinates, method="erosion_threshold")
        support = support[support <= max_r]
        if len(support) == 0:
            raise ValueError(
                "No support values remain after clipping to the erosion threshold "
                f"(max_radius={max_r:.4g}). Provide a support that starts below this value."
            )

        # Distance from each point to the study window boundary.
        shapely_pts = shapely.points(coordinates[:, 0], coordinates[:, 1])
        dist_to_boundary = shapely.distance(shapely_pts, poly.boundary)

        # Erosion estimator: G(r) = #{guard & NND ≤ r} / #{guard}
        # where guard(r) = {i : dist_to_boundary[i] > r}
        g_values = numpy.zeros(len(support))
        for i, r in enumerate(support):
            guard = dist_to_boundary > r
            n_guard = guard.sum()
            if n_guard > 0:
                g_values[i] = (guard & (nnd <= r)).sum() / n_guard

        return support, g_values

    counts, bins = numpy.histogram(nnd, bins=support)
    fracs = numpy.cumsum(counts) / counts.sum()
    return bins, numpy.asarray([0, *fracs])


def j(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    truncate=True,
):
    """Ripely's J function

    The so-called "spatial hazard" function, this is a function relating the
    F and G functions.

    Parameters
    ----------
    coordinates : geopandas object | numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: tuple of numpy.ndarray
        precomputed distances to use to evaluate the j function.
        The first must be of shape (n,n) or (n,) and is used in the g function.
        the second must be of shape (n,p) or (p,) (with p possibly equal to n)
        used in the f function.
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon
        the hull used to construct a random sample pattern for the f function.
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    truncate: bool (default: True)
        whether or not to truncate the results when the F function reaches one. If the
        F function is one but the G function is less than one, this function will return
        numpy.nan values.

    Returns
    -------
    a tuple containing the support values used to evalute the function
    and the values of the function at each distance value in the support.
    """
    if distances is not None:
        g_distances, f_distances = distances
    else:
        g_distances = f_distances = None
    fsupport, fstats = f(
        coordinates,
        support=support,
        distances=f_distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
    )

    gsupport, gstats = g(
        coordinates,
        support=support,
        distances=g_distances,
        metric=metric,
        edge_correction=edge_correction,
    )

    if isinstance(support, numpy.ndarray) and not numpy.allclose(gsupport, support):
        gfunction = interpolate.interp1d(gsupport, gstats, fill_value=1)
        gstats = gfunction(support)
        gsupport = support
    if not (numpy.allclose(gsupport, fsupport)):
        ffunction = interpolate.interp1d(fsupport, fstats, fill_value=1)
        fstats = ffunction(gsupport)
        fsupport = gsupport

    with numpy.errstate(invalid="ignore", divide="ignore"):
        hazard_ratio = (1 - gstats) / (1 - fstats)
    both_zero = (gstats == 1) & (fstats == 1)
    hazard_ratio[both_zero] = numpy.nan
    if truncate:
        result = _truncate(gsupport, hazard_ratio)
        if len(result[1]) != len(hazard_ratio):
            warnings.warn(
                f"requested {support} bins to evaluate the J function, but"
                f" it reaches infinity at d={result[0][-1]:.4f}, meaning only"
                f" {len(result[0])} bins will be used to characterize the J function.",
                stacklevel=2,
            )
        return result
    else:
        return gsupport, hazard_ratio


def k(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=_NOTSET,
    n_circle=36,
):
    """Ripley's K function

    This function counts the number of pairs of points that are closer than a given
    distance. As d increases, K approaches the number of point pairs.

    Parameters
    ----------
    coordinates : geopandas object | numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, n) or (n*(n-1)/2,)
        precomputed pairwise distances, either as a condensed upper-triangular
        vector (pdist format) or a full square matrix
    metric: str or callable
        distance metric to use when building the search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or None
        the study area geometry, used for intensity estimation and (when
        edge_correction is not None) for boundary-distance computation.
    edge_correction: None, 'border', 'isotropic', 'translate', 'erosion', 'ripley', or 'analytic'
        edge correction method. The three spatstat-default methods are:
        'border': reduced-sample (border) correction. Only points whose distance
            to the study window boundary exceeds r contribute as focal points.
            Alias for 'erosion'; the support is clipped to the erosion threshold.
        'isotropic': Ripley's exact isotropic correction. For each point i within r
            of the boundary, w_i(r) = 2πr / arc_inside, where arc_inside is the
            arc length of the circle of radius r centred at i that lies inside the
            window, computed exactly via shapely. Points fully inside get w_i = 1.
        'translate': translation correction (Ohser & Stoyan 1981). For each pair
            (i, j) with d_ij ≤ r, the weight is area(W)² / area(W ∩ (W + h_ij))
            where h_ij = x_j − x_i. Pairs whose translation keeps W fully inside
            get weight area(W) (reducing to the uncorrected estimator).
        Additional methods:
        'erosion': identical to 'border' (guard-point / eroded-window estimator).
        'ripley': approximate isotropic correction using n_circle equally-spaced
            test points sampled on the circle circumference (arc fraction estimated
            by counting points inside the window). Less exact than 'isotropic'.
        'analytic': area-ratio correction. w_i(r) = πr² / area(circle ∩ window).
            Uses the area fraction rather than the arc fraction.
    n_circle : int (default 36)
        Number of points to place on the test circle for the 'ripley' edge
        correction. Common choices are 36 (10° spacing) and 72 (5° spacing).
        Ignored when edge_correction is not 'ripley'.

    Returns
    -------
    a tuple containing the support values used to evaluate the function
    and the values of the function at each distance value in the support.
    """
    if edge_correction is _NOTSET:
        # Default: compute all three spatstat-default corrections and return a named tuple.
        coordinates_arr, support_arr, distances_out, metric, hull_prepared, _ = _prepare(
            coordinates, support, distances, metric, hull, None
        )
        poly = _hull_to_poly(hull_prepared)
        theo = numpy.pi * support_arr ** 2
        max_r, _ = _max_radius(poly, points=coordinates_arr, method="erosion_threshold")
        border_mask = support_arr <= max_r
        border_full = numpy.full(len(support_arr), numpy.nan)
        if border_mask.any():
            _, border_vals = k(
                coordinates_arr, support=support_arr[border_mask],
                distances=distances_out, metric=metric, hull=poly,
                edge_correction="border", n_circle=n_circle,
            )
            border_full[border_mask] = border_vals
        _, k_iso = k(
            coordinates_arr, support=support_arr, distances=distances_out,
            metric=metric, hull=poly, edge_correction="isotropic", n_circle=n_circle,
        )
        _, k_tra = k(
            coordinates_arr, support=support_arr, distances=distances_out,
            metric=metric, hull=poly, edge_correction="translate", n_circle=n_circle,
        )
        return KEstResult(
            support=support_arr, theo=theo,
            border=border_full, isotropic=k_iso, translate=k_tra,
        )

    _valid = (None, "border", "isotropic", "translate", "erosion", "ripley", "analytic", True)
    if edge_correction not in _valid:
        raise ValueError(
            f"edge_correction must be one of {_valid[:-1]}. Got {edge_correction!r}"
        )
    use_erosion = edge_correction in ("erosion", "border", True)
    use_ripley = edge_correction == "ripley"
    use_analytic = edge_correction == "analytic"
    use_isotropic = edge_correction == "isotropic"
    use_translate = edge_correction == "translate"

    coordinates, support, distances, metric, hull_prepared, _ = _prepare(
        coordinates, support, distances, metric, hull, None
    )
    n = coordinates.shape[0]
    upper_tri_n = n * (n - 1) * 0.5

    # Validate and normalise user-supplied distances (shape check only; no copy).
    upper_tri_distances = None
    if distances is not None:
        if distances.ndim == 1:
            if distances.shape[0] != upper_tri_n:
                raise ValueError(
                    f"Shape of inputted distances is not square, nor is the upper "
                    "triangular matrix matching the number of input points. The shape "
                    f"of the input matrix is {distances.shape}, but required shape "
                    f"is ({upper_tri_n},) or ({n},{n})"
                )
            upper_tri_distances = distances
        elif distances.shape[0] == distances.shape[1] == n:
            upper_tri_distances = distances[numpy.triu_indices_from(distances, k=1)]
        else:
            raise ValueError(
                f"Shape of inputted distances is not square, nor is the upper "
                "triangular matrix matching the number of input points. The shape "
                f"of the input matrix is {distances.shape}, but required shape "
                f"is ({upper_tri_n},) or ({n},{n})"
            )

    if use_erosion:
        poly = _hull_to_poly(hull_prepared)

        max_r, _ = _max_radius(poly, points=coordinates, method="erosion_threshold")
        support = support[support <= max_r]
        if len(support) == 0:
            raise ValueError(
                "No support values remain after clipping to the erosion threshold "
                f"(max_radius={max_r:.4g}). Provide a support that starts below this value."
            )

        shapely_pts = shapely.points(coordinates[:, 0], coordinates[:, 1])
        dist_to_boundary = shapely.distance(shapely_pts, poly.boundary)
        area = _area(poly)

        # Erosion estimator: K(r) = (A / (|guard(r)| × n)) × Σ_{i∈guard} #{j: d_ij < r}
        # where guard(r) = {i : dist_to_boundary[i] > r}
        k_values = numpy.zeros(len(support))

        if upper_tri_distances is not None:
            # User supplied precomputed distances: work from the condensed vector
            # without expanding it to a full n×n matrix.  Each undirected pair
            # (i, j) stored at condensed index k contributes once per guard
            # endpoint: guard[i] means i is a focal point counting j as a
            # neighbour, and guard[j] means j is a focal point counting i.
            rows, cols = numpy.triu_indices(n, k=1)
            for i, r in enumerate(support):
                guard = dist_to_boundary > r
                n_guard = int(guard.sum())
                if n_guard > 0:
                    within_r = upper_tri_distances < r
                    weight = guard[rows].astype(numpy.int8) + guard[cols].astype(numpy.int8)
                    k_values[i] = (area / (n_guard * n)) * int((weight * within_r).sum())
        else:
            # No precomputed distances: query a radius tree so we never build
            # the O(n²) condensed vector at all.
            tree = _build_best_tree(coordinates, metric)
            for i, r in enumerate(support):
                guard = dist_to_boundary > r
                n_guard = int(guard.sum())
                if n_guard > 0:
                    guard_coords = coordinates[guard]
                    if hasattr(tree, "query_radius"):  # sklearn KDTree / BallTree
                        counts = tree.query_radius(guard_coords, r, count_only=True)
                    else:  # scipy KDTree / Arc_KDTree
                        counts = numpy.asarray(
                            tree.query_ball_point(guard_coords, r, return_length=True)
                        )
                    # Each guard point matches itself; subtract to get neighbours only.
                    k_values[i] = (area / (n_guard * n)) * (int(counts.sum()) - n_guard)

        return support, k_values

    if use_ripley or use_analytic or use_isotropic:
        poly = _hull_to_poly(hull_prepared)
        area = _area(poly)
        if use_ripley:
            weights = _ripley_circle_weights(coordinates, poly, support, n_circle=n_circle)
        elif use_analytic:
            weights = _ripley_analytic_weights(coordinates, poly, support)
        else:
            weights = _isotropic_weights(coordinates, poly, support)

        k_values = numpy.zeros(len(support))

        if upper_tri_distances is not None:
            rows, cols = numpy.triu_indices(n, k=1)
            for j_idx, r in enumerate(support):
                if r == 0:
                    continue
                within_r = upper_tri_distances < r
                count_i = numpy.zeros(n, dtype=numpy.int64)
                numpy.add.at(count_i, rows[within_r], 1)
                numpy.add.at(count_i, cols[within_r], 1)
                k_values[j_idx] = (area / (n * n)) * (weights[:, j_idx] * count_i).sum()
        else:
            tree = _build_best_tree(coordinates, metric)
            for j_idx, r in enumerate(support):
                if r == 0:
                    continue
                if hasattr(tree, "query_radius"):  # sklearn KDTree / BallTree
                    count_i = tree.query_radius(coordinates, r, count_only=True) - 1
                else:  # scipy KDTree
                    count_i = numpy.array(
                        tree.query_ball_point(coordinates, r, return_length=True)
                    ) - 1
                k_values[j_idx] = (area / (n * n)) * (weights[:, j_idx] * count_i).sum()

        return support, k_values

    if use_translate:
        poly = _hull_to_poly(hull_prepared)
        area = _area(poly)
        if upper_tri_distances is None:
            upper_tri_distances = spatial.distance.pdist(coordinates, metric=metric)
        w_pairs = _translate_pair_weights(coordinates, poly, area)
        k_values = numpy.zeros(len(support))
        for j_idx, r in enumerate(support):
            within_r = upper_tri_distances < r
            k_values[j_idx] = (2.0 / (n * n)) * w_pairs[within_r].sum()
        return support, k_values

    # Non-erosion path: condensed pairwise distances.
    if upper_tri_distances is None:
        upper_tri_distances = spatial.distance.pdist(coordinates, metric=metric)
    n_pairs_less_than_d = (upper_tri_distances < support.reshape(-1, 1)).sum(axis=1)
    intensity = n / _area(hull_prepared)
    k_estimate = ((n_pairs_less_than_d * 2) / n) / intensity
    return support, k_estimate


def l(  # noqa: E743 - Ambiguous function name
    coordinates,
    support=None,
    permutations=9999,  # noqa: ARG001  -- Unused function argument
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=_NOTSET,
    linearized=False,
    n_circle=36,
):
    """Ripley's L function

    This is a scaled and shifted version of the K function that accounts for the K
    function's increasing expected value as distances increase. This means that the
    L function, for a completely random pattern, should be close to zero at all
    distance values in the support.

    Parameters
    ----------
    coordinates : geopandas object | numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, p) or (p,)
        distances from every point in a random point set of size p
        to some point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or None
        the study area geometry. Required when edge_correction is not None.
    edge_correction: None, 'border', 'isotropic', 'translate', 'erosion', 'ripley', or 'analytic'
        edge correction method passed through to the underlying K function.
    linearized : bool
        whether or not to subtract l from its expected value (support) at each
        distance bin. This centers the l function on zero for all distances.
        Proposed by Besag (1977)
    n_circle : int (default 36)
        Number of circle test points for the 'ripley' edge correction;
        passed through to k(). Common choices are 36 and 72.

    Returns
    -------
    a tuple containing the support values used to evaluate the function
    and the values of the function at each distance value in the support.
    """

    if edge_correction is _NOTSET:
        k_result = k(
            coordinates, support=support, distances=distances,
            metric=metric, hull=hull, n_circle=n_circle,
        )
        # k_result: KEstResult(support, theo, border, isotropic, translate)
        s = k_result.support
        l_theo = s  # sqrt(pi*r² / pi) = r

        def _sqrt_k(arr):
            return numpy.where(numpy.isnan(arr), numpy.nan, numpy.sqrt(arr / numpy.pi))

        l_bor = _sqrt_k(k_result.border)
        l_iso = _sqrt_k(k_result.isotropic)
        l_tra = _sqrt_k(k_result.translate)
        if linearized:
            return LEstResult(s, numpy.zeros_like(s), l_bor - s, l_iso - s, l_tra - s)
        return LEstResult(s, l_theo, l_bor, l_iso, l_tra)

    support, k_estimate = k(
        coordinates,
        support=support,
        distances=distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
        n_circle=n_circle,
    )

    _l = numpy.sqrt(k_estimate / numpy.pi)

    if linearized:
        return support, _l - support
    return support, _l


# ------------------------------------------------------------#
# Statistical Tests based on Ripley Functions                 #
# ------------------------------------------------------------#

FtestResult = namedtuple(
    "FtestResult", ("support", "statistic", "pvalue", "simulations")
)
GtestResult = namedtuple(
    "GtestResult", ("support", "statistic", "pvalue", "simulations")
)
JtestResult = namedtuple(
    "JtestResult", ("support", "statistic", "pvalue", "simulations")
)
KtestResult = namedtuple(
    "KtestResult", ("support", "statistic", "pvalue", "simulations")
)
LtestResult = namedtuple(
    "LtestResult", ("support", "statistic", "pvalue", "simulations")
)

_ripley_dispatch = {
    "F": (f, FtestResult),
    "G": (g, GtestResult),
    "J": (j, JtestResult),
    "K": (k, KtestResult),
    "L": (l, LtestResult),
}


def _ripley_test(
    calltype,
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    keep_simulations=False,
    n_simulations=9999,
    n_jobs=-1,
    **kwargs,
):
    if isinstance(coordinates, geopandas.GeoDataFrame | geopandas.GeoSeries):
        coordinates = shapely.get_coordinates(coordinates.geometry)

    stat_function, result_container = _ripley_dispatch.get(calltype)
    core_kwargs = {
        "support": support,
        "metric": metric,
        "edge_correction": edge_correction,
    }
    tree = _build_best_tree(coordinates, metric=metric)
    hull = _prepare_hull(coordinates, hull)
    empty_space_points = None

    if calltype in ("F", "J", "K", "L"):
        core_kwargs["hull"] = hull
        # amortize to avoid doing this every time
        empty_space_points = poisson(coordinates, size=(1000, 1))

        if distances is None:
            # Note: We now use the original coordinates' tree to calculate the observed
            # distance for the first time, as per the original logic flow.
            empty_space_distances, _ = _k_neighbors(tree, empty_space_points, k=1)

            if calltype == "F":
                distances = empty_space_distances.squeeze()
            else:  # calltype == 'J':
                n_distances, _ = _k_neighbors(tree, coordinates, k=1)
                distances = (n_distances.squeeze(), empty_space_distances.squeeze())
        else:
            pass
    core_kwargs.update(**kwargs)

    observed_support, observed_statistic = stat_function(
        coordinates, distances=distances, **core_kwargs
    )
    # The original function passed the tree, but the wrapper functions expect
    # coordinates. Corrected to pass coordinates, relying on stat_function to
    # manage the tree internally.

    core_kwargs["support"] = observed_support
    n_observations = coordinates.shape[0]

    # --- PARALLEL SIMULATION BLOCK ---
    if n_simulations <= 0:
        warnings.warn(
            "n_simulations must be positive. No simulations performed.",
            stacklevel=2,
        )
        simulations_array = numpy.empty((0, len(observed_support)))
    else:
        simulations_list = Parallel(n_jobs=n_jobs)(
            delayed(_run_one_ripley_simulation)(
                calltype,
                n_observations,
                hull,
                stat_function,
                metric,
                empty_space_points,
                core_kwargs,
                observed_support,
            )
            for _ in range(n_simulations)
        )
        simulations_array = numpy.array(simulations_list)
    # --- END PARALLEL SIMULATION BLOCK ---

    # --- VECTORIZED P-VALUE CALCULATION ---
    if simulations_array.shape[0] == 0:
        pvalues = numpy.nan * numpy.ones_like(observed_support)
    else:
        # Calculate how many simulations are as extreme as the observed statistic
        pvalues_count = (simulations_array >= observed_statistic).sum(axis=0)

        # Conservative p-value calculation
        pvalues = (pvalues_count + 1) / (n_simulations + 1)
        pvalues = numpy.minimum(pvalues, 1 - pvalues)
    # --- END P-VALUE CALCULATION ---

    return result_container(
        observed_support,
        observed_statistic,
        pvalues,
        simulations_array if keep_simulations else None,
    )


def f_test(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    keep_simulations=False,
    n_simulations=9999,
    n_jobs=-1,
):
    """Ripley's F function

    The so-called "empty space" function, this is the cumulative density function of
    the distances from a random set of points to the known points in the pattern. When
    the estimated statistic is larger than simulated values at a given distance, then
    the pattern is considered "dispersed" or "regular"

    Parameters
    ----------
    coordinates : geopandas object | numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, p) or (p,)
        distances from every point in a random point set of size p
        to some point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    keep_simulations: bool
        whether or not to keep the simulation envelopes. If so,
        will be returned as the result's simulations attribute
    n_simulations: int
        how many simulations to conduct, assuming that the reference pattern
        has complete spatial randomness.
    n_jobs : int (default: -1)
        The number of CPU cores to use for running the Monte Carlo simulations.
        Simulations are independent and can be run in parallel to significantly
        reduce execution time.

        * If ``n_jobs=-1``, all available CPU cores will be used.
        * If ``n_jobs=1``, the execution will be forced to run sequentially (serially),
          disabling parallel processing. This is often useful for debugging or
          testing purposes.
        * If ``n_jobs>1``, that specific number of cores will be used.

    Returns
    -------
    a named tuple with properties
    - support, the exact distance values used to evalute the statistic
    - statistic, the values of the statistic at each distance
    - pvalue, the percent of simulations that were as extreme as the observed value
    - simulations, the distribution of simulated statistics
        (shaped (n_simulations, n_support_points))
        or None if keep_simulations=False (which is the default)
    """

    return _ripley_test(
        "F",
        coordinates,
        support=support,
        distances=distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
        keep_simulations=keep_simulations,
        n_simulations=n_simulations,
        n_jobs=n_jobs,
    )


def g_test(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    keep_simulations=False,
    n_simulations=9999,
    n_jobs=-1,
):
    """Ripley's G function

    The G function is computed from the cumulative density function of the nearest
    neighbor distances between points in the pattern. When the G function is below
    the simulated values, it suggests dispersion.

    Parameters
    ----------
    coordinates : geopandas object | numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, p) or (p,)
        distances from every point in a random point set of size p
        to some point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    keep_simulations: bool
        whether or not to keep the simulation envelopes. If so,
        will be returned as the result's simulations attribute
    n_simulations: int
        how many simulations to conduct, assuming that the reference pattern
        has complete spatial randomness.
    n_jobs : int (default: -1)
        The number of CPU cores to use for running the Monte Carlo simulations.
        Simulations are independent and can be run in parallel to significantly
        reduce execution time.

        * If ``n_jobs=-1``, all available CPU cores will be used.
        * If ``n_jobs=1``, the execution will be forced to run sequentially (serially),
          disabling parallel processing. This is often useful for debugging or
          testing purposes.
        * If ``n_jobs>1``, that specific number of cores will be used.

    Returns
    -------
    a named tuple with properties
    - support, the exact distance values used to evalute the statistic
    - statistic, the values of the statistic at each distance
    - pvalue, the percent of simulations that were as extreme as the observed value
    - simulations, the distribution of simulated statistics
        (shaped (n_simulations, n_support_points))
        or None if keep_simulations=False (which is the default)
    """
    return _ripley_test(
        "G",
        coordinates,
        support=support,
        distances=distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
        keep_simulations=keep_simulations,
        n_simulations=n_simulations,
        n_jobs=n_jobs,
    )


def j_test(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    truncate=True,
    keep_simulations=False,
    n_simulations=9999,
    n_jobs=-1,
):
    """Ripley's J function

    The so-called "spatial hazard" function, this is a function relating the F and
    G functions. When the J function is consistently below 1, then it indicates
    clustering. When consistently above 1, it suggests dispersion.

    Parameters
    ----------
    coordinates : geopandas object | numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, p) or (p,)
        distances from every point in a random point set of size p
        to some point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    keep_simulations: bool
        whether or not to keep the simulation envelopes. If so,
        will be returned as the result's simulations attribute
    n_simulations: int
        how many simulations to conduct, assuming that the reference pattern
        has complete spatial randomness.
    n_jobs : int (default: -1)
        The number of CPU cores to use for running the Monte Carlo simulations.
        Simulations are independent and can be run in parallel to significantly
        reduce execution time.

        * If ``n_jobs=-1``, all available CPU cores will be used.
        * If ``n_jobs=1``, the execution will be forced to run sequentially (serially),
          disabling parallel processing. This is often useful for debugging or
          testing purposes.
        * If ``n_jobs>1``, that specific number of cores will be used.

    Returns
    -------
    a named tuple with properties
    - support, the exact distance values used to evalute the statistic
    - statistic, the values of the statistic at each distance
    - pvalue, the percent of simulations that were as extreme as the observed value
    - simulations, the distribution of simulated statistics
        (shaped (n_simulations, n_support_points))
        or None if keep_simulations=False (which is the default)
    """
    result = _ripley_test(
        "J",
        coordinates,
        support=support,
        distances=distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
        keep_simulations=keep_simulations,
        n_simulations=n_simulations,
        n_jobs=n_jobs,
        truncate=False,
    )
    if truncate:
        result_trunc = _truncate(*result)
        result_trunc = JtestResult(*result_trunc)
        if len(result_trunc.statistic) != len(result.statistic):
            warnings.warn(
                f"requested {support} bins to evaluate the J function, but"
                f" it reaches infinity at d={result[0][-1]:.4f}, meaning only"
                f" {len(result[0])} bins will be used to characterize the J function.",
                stacklevel=2,
            )
            return result_trunc

    else:
        return result


def k_test(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    keep_simulations=False,
    n_simulations=9999,
    n_jobs=-1,
):
    """Ripley's K function

    This function counts the number of pairs of points that are closer than a given
    distance. As d increases, K approaches the number of point pairs. When the K
    function is below simulated values, it suggests that the pattern is dispersed.

    Parameters
    ----------
    coordinates : geopandas object | numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, p) or (p,)
        distances from every point in a random point set of size p
        to some point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    keep_simulations: bool
        whether or not to keep the simulation envelopes. If so,
        will be returned as the result's simulations attribute
    n_simulations: int
        how many simulations to conduct, assuming that the reference pattern
        has complete spatial randomness.
    n_jobs : int (default: -1)
        The number of CPU cores to use for running the Monte Carlo simulations.
        Simulations are independent and can be run in parallel to significantly
        reduce execution time.

        * If ``n_jobs=-1``, all available CPU cores will be used.
        * If ``n_jobs=1``, the execution will be forced to run sequentially (serially),
          disabling parallel processing. This is often useful for debugging or
          testing purposes.
        * If ``n_jobs>1``, that specific number of cores will be used.

    Returns
    -------
    a named tuple with properties
    - support, the exact distance values used to evalute the statistic
    - statistic, the values of the statistic at each distance
    - pvalue, the percent of simulations that were as extreme as the observed value
    - simulations, the distribution of simulated statistics
        (shaped (n_simulations, n_support_points))
        or None if keep_simulations=False (which is the default)
    """
    return _ripley_test(
        "K",
        coordinates,
        support=support,
        distances=distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
        keep_simulations=keep_simulations,
        n_simulations=n_simulations,
        n_jobs=n_jobs,
    )


def l_test(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    linearized=False,
    keep_simulations=False,
    n_simulations=9999,
    n_jobs=-1,
):
    """Ripley's L function

    This is a scaled and shifted version of the K function that accounts for the K
    function's increasing expected value as distances increase. This means that the L
    function, for a completely random pattern, should be close to zero at all distance
    values in the support. When the L function is negative, this suggests dispersion.

    Parameters
    ----------
    coordinates : geopandas object | numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, p) or (p,)
        distances from every point in a random point set of size p
        to some point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    keep_simulations: bool
        whether or not to keep the simulation envelopes. If so,
        will be returned as the result's simulations attribute
    n_simulations: int
        how many simulations to conduct, assuming that the reference pattern
        has complete spatial randomness.
    n_jobs : int (default: -1)
        The number of CPU cores to use for running the Monte Carlo simulations.
        Simulations are independent and can be run in parallel to significantly
        reduce execution time.

        * If ``n_jobs=-1``, all available CPU cores will be used.
        * If ``n_jobs=1``, the execution will be forced to run sequentially (serially),
          disabling parallel processing. This is often useful for debugging or
          testing purposes.
        * If ``n_jobs>1``, that specific number of cores will be used.

    Returns
    -------
    a named tuple with properties
    - support, the exact distance values used to evalute the statistic
    - statistic, the values of the statistic at each distance
    - pvalue, the percent of simulations that were as extreme as the observed value
    - simulations, the distribution of simulated statistics
        (shaped (n_simulations, n_support_points)) or None if
        keep_simulations=False (which is the default)
    """
    return _ripley_test(
        "L",
        coordinates,
        support=support,
        distances=distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
        keep_simulations=keep_simulations,
        n_simulations=n_simulations,
        n_jobs=n_jobs,
        linearized=linearized,
    )


def _run_one_ripley_simulation(
    calltype,
    n_observations,
    hull,
    stat_function,
    metric,
    empty_space_points,
    core_kwargs,
    observed_support,
):
    """Run a single Monte Carlo simulation for a Ripley function test."""

    # 1. Generate a random point pattern (CSR)
    random_i = poisson(hull, size=n_observations)
    current_kwargs = core_kwargs.copy()
    current_kwargs["support"] = observed_support

    # 2. Prepare distances for F/J tests
    if calltype in ("F", "J"):
        random_tree = _build_best_tree(random_i, metric)
        empty_distances, _ = random_tree.query(empty_space_points, k=1)

        if calltype == "F":
            current_kwargs["distances"] = empty_distances.squeeze()
        else:  # calltype == 'J':
            n_distances, _ = _k_neighbors(random_tree, random_i, k=1)
            current_kwargs["distances"] = (
                n_distances.squeeze(),
                empty_distances.squeeze(),
            )

    # 3. Calculate the Ripley statistic for the simulated pattern
    # rep_support is ignored as we rely on observed_support
    _, simulations_i = stat_function(random_i, **current_kwargs)

    return simulations_i


def _truncate(support, realizations, *rest):
    is_invalid = numpy.isinf(realizations) | numpy.isnan(realizations)
    first_inv = is_invalid.argmax()
    if not is_invalid.any():
        return support, realizations, *rest
    elif first_inv < len(realizations):
        return (
            support[:first_inv],
            realizations[:first_inv],
            *[r[:first_inv] if r is not None else None for r in rest],
        )
