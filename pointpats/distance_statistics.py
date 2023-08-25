import numpy
import warnings
from scipy import spatial, interpolate
from collections import namedtuple
from .geometry import (
    area as _area,
    k_neighbors as _k_neighbors,
    build_best_tree as _build_best_tree,
    prepare_hull as _prepare_hull,
    TREE_TYPES,
)
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
    """
    Ripley's F function

    The so-called "empty space" function, this is the cumulative density function of
    the distances from a random set of points to the known points in the pattern.

    Parameters
    ----------
    coordinates : numpy.ndarray of shape (n,2)
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
            tree
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
    edge_correction=None,
):
    """
    Ripley's G function

    The G function is computed from the cumulative density function of the nearest neighbor
    distances between points in the pattern.

    Parameters
    -----------
    coordinates : numpy.ndarray of shape (n,2)
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

    Returns
    -------
    a tuple containing the support values used to evalute the function
    and the values of the function at each distance value in the support.

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


def j(
    coordinates,
    support=None,
    distances=None,
    metric="euclidean",
    hull=None,
    edge_correction=None,
    truncate=True,
):
    """
    Ripely's J function

    The so-called "spatial hazard" function, this is a function relating the F and G functions.

    Parameters
    -----------
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
    edge_correction=None,
):
    """
    Ripley's K function

    This function counts the number of pairs of points that are closer than a given distance.
    As d increases, K approaches the number of point pairs.

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
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.

    Returns
    -------
    a tuple containing the support values used to evalute the function
    and the values of the function at each distance value in the support.
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


def l(
    coordinates,
    support=None,
    permutations=9999,
    distances=None,
    metric="euclidean",
    edge_correction=None,
    linearized=False,
):
    """
    Ripley's L function

    This is a scaled and shifted version of the K function that accounts for the K function's
    increasing expected value as distances increase. This means that the L function, for a
    completely random pattern, should be close to zero at all distance values in the support.

    Parameters
    ----------
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
    hull: bounding box, scipy.spatial.ConvexHull, shapely.geometry.Polygon
        the hull used to construct a random sample pattern, if distances is None
    edge_correction: bool or str
        whether or not to conduct edge correction. Not yet implemented.
    linearized : bool
        whether or not to subtract l from its expected value (support) at each
        distance bin. This centers the l function on zero for all distances.
        Proposed by Besag (1977)

    Returns
    -------
    a tuple containing the support values used to evalute the function
    and the values of the function at each distance value in the support.
    """

    support, k_estimate = k(
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
    **kwargs,
):
    stat_function, result_container = _ripley_dispatch.get(calltype)
    core_kwargs = dict(
        support=support,
        metric=metric,
        edge_correction=edge_correction,
    )
    tree = _build_best_tree(coordinates, metric=metric)
    hull = _prepare_hull(coordinates, hull)
    if calltype in ("F", "J"):  # these require simulations
        core_kwargs["hull"] = hull
        # amortize to avoid doing this every time
        empty_space_points = poisson(coordinates, size=(1000, 1))
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
    n_observations = coordinates.shape[0]

    if keep_simulations:
        simulations = numpy.empty((len(observed_support), n_simulations)).T
    pvalues = numpy.ones_like(observed_support)
    for i_replication in range(n_simulations):
        random_i = poisson(hull, size=n_observations)
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
    Ripley's F function

    The so-called "empty space" function, this is the cumulative density function of
    the distances from a random set of points to the known points in the pattern.

    When the estimated statistic is larger than simulated values at a given distance, then
    the pattern is considered "dispersed" or "regular"

    Parameters
    -----------
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

    Returns
    -------
    a named tuple with properties
    - support, the exact distance values used to evalute the statistic
    - statistic, the values of the statistic at each distance
    - pvalue, the percent of simulations that were as extreme as the observed value
    - simulations, the distribution of simulated statistics (shaped (n_simulations, n_support_points))
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
    Ripley's G function

    The G function is computed from the cumulative density function of the nearest neighbor
    distances between points in the pattern.

    When the G function is below the simulated values, it suggests dispersion.

    Parameters
    ----------
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

    Returns
    -------
    a named tuple with properties
    - support, the exact distance values used to evalute the statistic
    - statistic, the values of the statistic at each distance
    - pvalue, the percent of simulations that were as extreme as the observed value
    - simulations, the distribution of simulated statistics (shaped (n_simulations, n_support_points))
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
    Ripley's J function

    The so-called "spatial hazard" function, this is a function relating the F and G functions.

    When the J function is consistently below 1, then it indicates clustering.
    When consistently above 1, it suggests dispersion.

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

    Returns
    -------
    a named tuple with properties
    - support, the exact distance values used to evalute the statistic
    - statistic, the values of the statistic at each distance
    - pvalue, the percent of simulations that were as extreme as the observed value
    - simulations, the distribution of simulated statistics (shaped (n_simulations, n_support_points))
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
):
    """
    Ripley's K function

    This function counts the number of pairs of points that are closer than a given distance.
    As d increases, K approaches the number of point pairs.

    When the K function is below simulated values, it suggests that the pattern is dispersed.

    Parameters
    ----------
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

    Returns
    -------
    a named tuple with properties
    - support, the exact distance values used to evalute the statistic
    - statistic, the values of the statistic at each distance
    - pvalue, the percent of simulations that were as extreme as the observed value
    - simulations, the distribution of simulated statistics (shaped (n_simulations, n_support_points))
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
    Ripley's L function

    This is a scaled and shifted version of the K function that accounts for the K function's
    increasing expected value as distances increase. This means that the L function, for a
    completely random pattern, should be close to zero at all distance values in the support.

    When the L function is negative, this suggests dispersion.

    Parameters
    ----------
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

    Returns
    -------
    a named tuple with properties
    - support, the exact distance values used to evalute the statistic
    - statistic, the values of the statistic at each distance
    - pvalue, the percent of simulations that were as extreme as the observed value
    - simulations, the distribution of simulated statistics (shaped (n_simulations, n_support_points))
        or None if keep_simulations=False (which is the default)
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
