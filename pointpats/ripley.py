import numpy
import warnings
from scipy import spatial, interpolate
from functools import singledispatch
from collections import namedtuple
from libpysal.cg import alpha_shape_auto
from libpysal.cg.kdtree import Arc_KDTree
from .geometry import (
    area as _area,
    bbox as _bbox,
    contains as _contains,
    k_neighbors as _k_neighbors,
    build_best_tree as _build_best_tree,
    prepare_hull as _prepare_hull,
)

__all__ = [
    "simulate",
    "simulate_from",
    "f_function",
    "g_function",
    "k_function",
    "j_function",
    "l_function",
    "f_test",
    "g_test",
    "k_test",
    "j_test",
    "l_test",
]


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
        except:
            raise TypeError(
                "`support` must be a tuple (either (start, stop, step), (start, stop) or (stop,)),"
                " an int describing the number of breaks to use to evalute the function,"
                " or an iterable containing the breaks to use to evaluate the function."
                " Recieved object of type {}: {}".format(type(support), support)
            )

    return coordinates, support, distances, metric, hull, edge_correction


def _k_neighbors(tree, coordinates, k, **kwargs):
    """
    Query a kdtree for k neighbors, handling the self-neighbor case
    in the case of coincident points.
    """
    distances, indices = tree.query(coordinates, k=k + 1, **kwargs)
    n, ks = distances.shape
    assert ks == k + 1
    full_indices = numpy.arange(n)
    other_index_mask = indices != full_indices.reshape(n, 1)
    has_k_indices = other_index_mask.sum(axis=1) == (k + 1)
    other_index_mask[has_k_indices, -1] = False
    distances = distances[other_index_mask].reshape(n, k)
    indices = indices[other_index_mask].reshape(n, k)
    return distances, indices


# ------------------------------------------------------------#
# Simulators                                                 #
# ------------------------------------------------------------#


def simulate(hull, intensity=None, size=None):
    """
    Simulate from the given hull with a specified intensity or size.

    Hulls can be:
    - bounding boxes (numpy.ndarray with dim==1 and len == 4)
    - scipy.spatial.ConvexHull
    - shapely.geometry.Polygon
    - pygeos.Geometry

    If intensity is specified, size must be an integer reflecting
    the number of realizations.
    If the size is specified as a tuple, then the intensity is
    determined by the area of the hull.
    """
    if size is None:
        if intensity is not None:
            # if intensity is provided, assume
            # n_observations
            n_observations = int(_area(hull) * intensity)
        else:
            # default to 100 points
            n_observations = 100
        n_simulations = 1
        size = (n_observations, n_simulations)
    elif isinstance(size, tuple):
        if len(size) == 2 and intensity is None:
            n_observations, n_simulations = size
            intensity = n_observations / _area(hull)
        elif len(size) == 2 and intensity is not None:
            raise ValueError(
                "Either intensity or size as (n observations, n simulations)"
                " can be provided. Providing both creates statistical conflicts."
                " between the requested intensity and implied intensity by"
                " the number of observations and the area of the hull. If"
                " you want to specify the intensity, use the intensity argument"
                " and set size equal to the number of simulations."
            )
        else:
            raise ValueError(
                f"Intensity and size not understood. Provide size as a tuple"
                f" containing (number of observations, number of simulations)"
                f" with no specified intensity, or an intensity and size equal"
                f" to the number of simulations."
                f" Recieved: `intensity={intensity}, size={size}`"
            )
    elif isinstance(size, int):
        # assume int size with specified intensity means n_simulations at x intensity
        if intensity is not None:
            n_observations = int(intensity * _area(hull))
            n_simulations = size
        else:  # assume we have one replication at the specified number of points
            n_simulations = 1
            n_observations = size
            intensity = n_observations / _area(hull)
    else:
        raise ValueError(
            f"Intensity and size not understood. Provide size as a tuple"
            f" containing (number of observations, number of simulations)"
            f" with no specified intensity, or an intensity and size equal"
            f" to the number of simulations."
            f" Recieved: `intensity={intensity}, size={size}`"
        )
    result = numpy.empty((n_simulations, n_observations, 2))

    bbox = _bbox(hull)

    for i_replication in range(n_simulations):
        generating = True
        i_observation = 0
        while i_observation < n_observations:
            x, y = (
                numpy.random.uniform(bbox[0], bbox[2]),
                numpy.random.uniform(bbox[1], bbox[3]),
            )
            if _contains(hull, x, y):
                result[i_replication, i_observation] = (x, y)
                i_observation += 1
    return result.squeeze()


def simulate_from(coordinates, hull=None, size=None):
    """
    Simulate a pattern from the coordinates provided using a given assumption
    about the hull of the process.

    Note: will always assume the implicit intensity of the process.
    """
    try:
        coordinates = numpy.asarray(coordinates)
        assert coordinates.ndim == 2
    except:
        raise ValueError(
            "This function requires a numpy array for input."
            " If `coordinates` is a shape, use simulate()."
            " Otherwise, use the `hull` argument to specify"
            " which hull you intend to compute for the input"
            " coordinates."
        )
    if isinstance(size, int):
        n_observations = coordinates.shape[0]
        n_simulations = size
    elif isinstance(size, tuple):
        assert len(size) == 2, (
            f"`size` argument must be either an integer denoting the number"
            f" of simulations or a tuple containing "
            f" (n_simulated_observations, n_simulations). Instead, recieved"
            f" a tuple of length {len(size)}: {size}"
        )
        n_observations, n_simulations = size
    elif size is None:
        n_observations = coordinates.shape[0]
        n_simulations = 1
    hull = _prepare_hull(coordinates, hull)
    return simulate(hull, intensity=None, size=(n_observations, n_simulations))


# ------------------------------------------------------------#
# Statistical Functions                                       #
# ------------------------------------------------------------#


def f_function(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
):
    """
    coordinates : numpy.ndarray, (n,2)
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
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
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

        randoms = simulate(hull=hull, size=(n_empty_points, 1))
        try:
            tree
        except NameError:
            tree = _build_best_tree(coordinates, metric)
        finally:
            distances, _ = tree.query(randoms, k=1)
            distances = distances.squeeze()

    counts, bins = numpy.histogram(distances, bins=support)
    fracs = numpy.cumsum(counts) / counts.sum()

    return bins, numpy.asarray([0, *fracs])


def g_function(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    edge_correction=None,
):
    """
    coordinates : numpy.ndarray, (n,2)
        input coordinates to function
    support : tuple of length 1, 2, or 3, int, or numpy.ndarray
        tuple, encoding (stop,), (start, stop), or (start, stop, num)
        int, encoding number of equally-spaced intervals
        numpy.ndarray, used directly within numpy.histogram
    distances: numpy.ndarray, (n, n) or (n,)
        distances from every point in the point to another point in `coordinates`
    metric: str or callable
        distance metric to use when building search tree
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    """

    coordinates, support, distances, metric, *_ = _prepare(
        coordinates, support, distances, metric, None, edge_correction
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
                    " as `coordinates` ({n}) or 1 dimensional and contain"
                    " the shortest distance from each point in "
                    " `coordinates` to some other point in coordinates."
                )
        elif distances.ndim == 1:
            if distances.shape[0] != coordinates.shape[0]:
                raise ValueError(
                    f"Distances are not aligned with coordinates! Distance"
                    f" matrix must be (n_coordinates, n_coordinates), but recieved"
                    f" {distances.shape} instead of ({coordinates.shape[0]},)"
                )
        else:
            raise ValueError(
                "Distances supplied can either be 2 dimensional"
                " square matrices with the same number of rows"
                " as `coordinates` or 1 dimensional and contain"
                " the shortest distance from each point in "
                " `coordinates` to some other point in coordinates."
                " Input matrix was {distances.ndim} dimensioanl"
            )
    else:
        try:
            tree
        except NameError:
            tree = _build_best_tree(coordinates, metric)
        finally:
            distances, indices = _k_neighbors(tree, coordinates, k=1)

    counts, bins = numpy.histogram(distances.squeeze(), bins=support)
    fracs = numpy.cumsum(counts) / counts.sum()

    return bins, numpy.asarray([0, *fracs])


def j_function(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    truncate=True,
):
    """
    coordinates : numpy.ndarray, (n,2)
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
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern for the f function.
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    truncate: bool (default: True)
        whether or not to truncate the results when the F function reaches one. If the
        F function is one but the G function is less than one, this function will return
        numpy.nan values.
    """
    if distances is not None:
        g_distances, f_distances = distances
    else:
        g_distances = f_distances = None
    fsupport, fstats = f_function(
        coordinates,
        support=support,
        distances=f_distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
    )

    gsupport, gstats = g_function(
        coordinates,
        support=support,
        distances=g_distances,
        metric=metric,
        edge_correction=edge_correction,
    )

    if isinstance(support, numpy.ndarray):
        if not numpy.allclose(gsupport, support):
            gfunction = interpolate.interp1d(gsupport, gstats, fill_value=1)
            gstats = gfunction(support)
            gsupport = support
    if not (numpy.allclose(gsupport, fsupport)):
        ffunction = interpolate.interp1d(fsupport, fstats, fill_value=1)
        fstats = ffunction(gsupport)
        fsupport = gsupport

    with numpy.errstate(invalid="ignore", divide="ignore"):
        hazard_ratio = (1 - gstats) / (1 - fstats)
    if truncate:
        both_zero = (gstats == 1) & (fstats == 1)
        hazard_ratio[both_zero] = 1
        is_inf = numpy.isinf(hazard_ratio)
        first_inf = is_inf.argmax()
        if not is_inf.any():
            first_inf = len(hazard_ratio)
        if first_inf < len(hazard_ratio) and isinstance(support, int):
            warnings.warn(
                f"requested {support} bins to evaluate the J function, but"
                f" it reaches infinity at d={gsupport[first_inf]:.4f}, meaning only"
                f" {first_inf} bins will be used to characterize the J function.",
                stacklevel=2,
            )
    else:
        first_inf = len(gsupport) + 1
    return (gsupport[:first_inf], hazard_ratio[:first_inf])


def k_function(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    edge_correction=None,
):
    """
    coordinates : numpy.ndarray, (n,2)
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
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    """
    coordinates, support, distances, metric, hull, edge_correction = _prepare(
        coordinates, support, distances, metric, None, edge_correction
    )
    n = coordinates.shape[0]
    upper_tri_n = n * (n - 1) * 0.5
    if distances is not None:
        if distances.ndim == 1:
            if distances.shape[0] != upper_tri_n:
                raise ValueError(
                    f"Shape of inputted distances is not square, nor is the upper triangular"
                    f" matrix matching the number of input points. The shape of the input matrix"
                    f" is {distances.shape}, but required shape is ({upper_tri_n},) or ({n},{n})"
                )
            upper_tri_distances = distances
        elif distances.shape[0] == distances.shape[1] == n:
            upper_tri_distances = distances[numpy.triu_indices_from(distances, k=1)]
        else:
            raise ValueError(
                f"Shape of inputted distances is not square, nor is the upper triangular"
                f" matrix matching the number of input points. The shape of the input matrix"
                f" is {distances.shape}, but required shape is ({upper_tri_n},) or ({n},{n})"
            )
    else:
        upper_tri_distances = spatial.distance.pdist(coordinates, metric=metric)
    n_pairs_less_than_d = (upper_tri_distances < support.reshape(-1, 1)).sum(axis=1)
    intensity = n / _area(hull)
    k_estimate = ((n_pairs_less_than_d * 2) / n) / intensity
    return support, k_estimate


def l_function(
    coordinates,
    support=None,
    permutations=9999,
    distances=None,
    metric="euclidean",
    edge_correction=None,
    linearized=False,
):
    """
    coordinates : numpy.ndarray, (n,2)
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
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    linearized : bool
        whether or not to subtract l from its expected value (support) at each
        distance bin. This centers the l function on zero for all distances.
        Proposed by Besag (1977) #TODO: fix besag ref
    """

    support, k_estimate = k_function(
        coordinates,
        support=support,
        distances=distances,
        metric=metric,
        edge_correction=edge_correction,
    )

    l = numpy.sqrt(k_estimate / numpy.pi)

    if linearized:
        return support, l - support
    return support, l


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
    "F": (f_function, FtestResult),
    "G": (g_function, GtestResult),
    "J": (j_function, JtestResult),
    "K": (k_function, KtestResult),
    "L": (l_function, LtestResult),
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
    **kwargs,
):
    stat_function, result_container = _ripley_dispatch.get(calltype)
    core_kwargs = dict(
        support=support,
        metric=metric,
        edge_correction=edge_correction,
    )
    tree = _build_best_tree(coordinates, metric=metric)

    if calltype in ("F", "J"):  # these require simulations
        core_kwargs["hull"] = hull
        # amortize to avoid doing this every time
        empty_space_points = simulate_from(coordinates, size=(1000, 1))
        if distances is None:
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
        tree, distances=distances, **core_kwargs
    )
    core_kwargs["support"] = observed_support

    if keep_simulations:
        simulations = numpy.empty((len(observed_support), n_simulations)).T
    pvalues = numpy.ones_like(observed_support)
    for i_replication in range(n_simulations):
        random_i = simulate_from(tree.data)
        if calltype in ("F", "J"):
            random_tree = _build_best_tree(random_i, metric)
            empty_distances, _ = random_tree.query(empty_space_points, k=1)
            if calltype == "F":
                core_kwargs["distances"] = empty_distances.squeeze()
            else:  # calltype == 'J':
                n_distances, _ = _k_neighbors(random_tree, random_i, k=1)
                core_kwargs["distances"] = (
                    n_distances.squeeze(),
                    empty_distances.squeeze(),
                )
        rep_support, simulations_i = stat_function(random_i, **core_kwargs)
        pvalues += simulations_i >= observed_statistic
        if keep_simulations:
            simulations[i_replication] = simulations_i
    pvalues /= n_simulations + 1
    pvalues = numpy.minimum(pvalues, 1 - pvalues)
    return result_container(
        observed_support,
        observed_statistic,
        pvalues,
        simulations if keep_simulations else None,
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
):
    """
    coordinates : numpy.ndarray, (n,2)
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
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    keep_simulations: bool
        whether or not to keep the simulation envelopes. If so,
        will be returned as the result's simulations attribute
    n_simulations: int
        how many simulations to conduct, assuming that the reference pattern
        has complete spatial randomness.
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
):
    """
    coordinates : numpy.ndarray, (n,2)
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
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    keep_simulations: bool
        whether or not to keep the simulation envelopes. If so,
        will be returned as the result's simulations attribute
    n_simulations: int
        how many simulations to conduct, assuming that the reference pattern
        has complete spatial randomness.
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
):
    """
    coordinates : numpy.ndarray, (n,2)
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
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    keep_simulations: bool
        whether or not to keep the simulation envelopes. If so,
        will be returned as the result's simulations attribute
    n_simulations: int
        how many simulations to conduct, assuming that the reference pattern
        has complete spatial randomness.
    """
    return _ripley_test(
        "J",
        coordinates,
        support=support,
        distances=distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
        keep_simulations=keep_simulations,
        n_simulations=n_simulations,
        truncate=truncate,
    )


def k_test(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    keep_simulations=False,
    n_simulations=9999,
):
    """
    coordinates : numpy.ndarray, (n,2)
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
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    keep_simulations: bool
        whether or not to keep the simulation envelopes. If so,
        will be returned as the result's simulations attribute
    n_simulations: int
        how many simulations to conduct, assuming that the reference pattern
        has complete spatial randomness.
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
):
    """
    coordinates : numpy.ndarray, (n,2)
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
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon, or pygeos.Geometry
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    keep_simulations: bool
        whether or not to keep the simulation envelopes. If so,
        will be returned as the result's simulations attribute
    n_simulations: int
        how many simulations to conduct, assuming that the reference pattern
        has complete spatial randomness.
    """
    return _ripley_test(
        "L",
        coordinates,
        support=support,
        distances=distances,
        metric=metric,
        hull=hull,
        edge_correction=edge_correction,
        linearized=linearized,
        keep_simulations=keep_simulations,
        n_simulations=n_simulations,
    )
