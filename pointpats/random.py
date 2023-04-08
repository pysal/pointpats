import numpy
from .geometry import (
    spatial,
    area as _area,
    centroid as _centroid,
    contains as _contains,
    bbox as _bbox,
    prepare_hull as _prepare_hull,
    HULL_TYPES,
)

# ------------------------------------------------------------ #
# Utilities                                                    #
# ------------------------------------------------------------ #


def parse_size_and_intensity(hull, intensity=None, size=None):
    """
    Given a hull, an intensity, and a size int/tuple, correctly
    compute the resulting missing quantities. Defaults to 100 points in one
    replication, meaning the intensity will be computed on the fly
    if nothing is provided.

    Parameters
    ----------
    hull : A geometry-like object
        This encodes the "space" in which to simulate the normal pattern. All points will
        lie within this hull. Supported values are:
        - a bounding box encoded in a numpy array as numpy.array([xmin, ymin, xmax, ymax])
        - an (N,2) array of points for which the bounding box will be computed & used
        - a shapely polygon/multipolygon
        - a scipy convexh hull
    intensity : float
        the number of observations per unit area in the hull to use. If provided,
        then the number of observations is determined using the intensity * area(hull) and
        the size is assumed to represent n_replications (if provided).
    size : tuple or int
        a tuple of (n_observations, n_replications), where the first number is the number
        of points to simulate in each replication and the second number is the number of
        total replications. So, (10, 4) indicates 10 points, 4 times.
        If an integer is provided and intensity is None, n_replications is assumed to be 1.
        If size is an integer and intensity is also provided, then size indicates n_replications,
        and the number of observations is computed on the fly using intensity and area.
    """
    if size is None:
        if intensity is not None:
            # if intensity is provided, assume
            # n_observations
            n_observations = int(_area(hull) * intensity)
        else:
            # default to 100 points
            n_observations = 100
            intensity = n_observations / _area(hull)
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
    return (n_observations, n_simulations, intensity)


# ------------------------------------------------------------ #
# Distributions                                                #
# ------------------------------------------------------------ #


def poisson(hull, intensity=None, size=None):
    """
    Simulate a poisson random point process with a specified intensity.

    Parameters
    ----------
    hull : A geometry-like object
        This encodes the "space" in which to simulate the normal pattern. All points will
        lie within this hull. Supported values are:
        - a bounding box encoded in a numpy array as numpy.array([xmin, ymin, xmax, ymax])
        - an (N,2) array of points for which the bounding box will be computed & used
        - a shapely polygon/multipolygon
        - a scipy convexh hull
    intensity : float
        the number of observations per unit area in the hull to use. If provided, then
        size must be an integer describing the number of replications to use.
    size : tuple or int
        a tuple of (n_observations, n_replications), where the first number is the number
        of points to simulate in each replication and the second number is the number of
        total replications. So, (10, 4) indicates 10 points, 4 times.
        If an integer is provided and intensity is None, n_replications is assumed to be 1.
        If size is an integer and intensity is also provided, then size indicates n_replications,
        and the number of observations is computed from the intensity.

    Returns
    --------
        :   numpy.ndarray
        either an (n_replications, n_observations, 2) or (n_observations,2) array containing
        the simulated realizations.
    """
    if isinstance(hull, numpy.ndarray):
        if hull.shape == (4,):
            hull = hull
        else:
            hull = _prepare_hull(hull)
    n_observations, n_simulations, intensity = parse_size_and_intensity(
        hull, intensity=intensity, size=size
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


def normal(hull, center=None, cov=None, size=None):
    """
    Simulate a multivariate random normal point cluster

    Parameters
    ----------
    hull : A geometry-like object
        This encodes the "space" in which to simulate the normal pattern. All points will
        lie within this hull. Supported values are:
        - a bounding box encoded in a numpy array as numpy.array([xmin, ymin, xmax, ymax])
        - an (N,2) array of points for which the bounding box will be computed & used
        - a shapely polygon/multipolygon
        - a scipy convexh hull
    center : iterable of shape (2, )
        A point where the simulations will be centered.
    cov : float or a numpy array of shape (2,2)
        either the standard deviation of an independent and identically distributed
        normal distribution, or a 2 by 2 covariance matrix expressing the covariance
        of the x and y for the distribution. Default is half of the width or height
        of the hull's bounding box, whichever is larger.
    size : tuple or int
        a tuple of (n_observations, n_replications), where the first number is the number
        of points to simulate in each replication and the second number is the number of
        total replications. So, (10, 4) indicates 10 points, 4 times.
        If an integer is provided, n_replications is assumed to be 1.

    Returns
    --------
        :   numpy.ndarray
        either an (n_replications, n_observations, 2) or (n_observations,2) array containing
        the simulated realizations.
    """
    if isinstance(hull, numpy.ndarray):
        if hull.shape == (4,):
            hull = hull
        else:
            hull = _prepare_hull(hull)
    if center is None:
        center = _centroid(hull)
    n_observations, n_simulations, intensity = parse_size_and_intensity(
        hull, intensity=None, size=size
    )
    if cov is None:
        bbox = _bbox(hull)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        cov = numpy.maximum(width / 2, height / 2) ** 2

    if isinstance(cov, (int, float)):
        sd = cov
        cov = numpy.eye(2) * sd
    elif isnstance(cov, numpy.ndarray):
        if cov.ndim == 2:
            assert cov.shape == (2, 2), "Bivariate covariance matrices must be 2 by 2"
        elif cov.ndim == 3:
            assert cov.shape[1:] == (
                2,
                2,
            ), "3-dimensional covariance matrices must have shape (n_simulations, 2,2)"
            assert (
                cov.shape[0] == n_simulations
            ), "3-dimensional covariance matrices must have shape (n_simulations, 2,2)"
    else:
        raise ValueError(
            "`cov` argument must be a float (signifying a standard deviation)"
            " or a 2 by 2 array expressing the covariance matrix of the "
            " bivariate normal distribution."
        )

    result = numpy.empty((n_simulations, n_observations, 2))

    bbox = _bbox(hull)

    for i_replication in range(n_simulations):
        generating = True
        i_observation = 0
        replication_cov = cov[i] if cov.ndim == 3 else cov
        replication_sd = numpy.diagonal(replication_cov) ** 0.5
        replication_cor = (1 / replication_sd) * replication_cov * (1 / replication_sd)

        while i_observation < n_observations:
            candidate = numpy.random.multivariate_normal((0, 0), replication_cor)
            x, y = center + candidate * replication_sd
            if _contains(hull, x, y):
                result[i_replication, i_observation] = (x, y)
                i_observation += 1
    return result.squeeze()


def cluster_poisson(
    hull, intensity=None, size=None, n_seeds=2, cluster_radius=None,
):
    """
    Simulate a cluster poisson random point process with a specified intensity & number of seeds.
    A cluster poisson process is a poisson process where the center of each "cluster" is
    itself distributed according to a spatial poisson process.

    Parameters
    ----------
    hull : A geometry-like object
        This encodes the "space" in which to simulate the normal pattern. All points will
        lie within this hull. Supported values are:
        - a bounding box encoded in a numpy array as numpy.array([xmin, ymin, xmax, ymax])
        - an (N,2) array of points for which the bounding box will be computed & used
        - a shapely polygon/multipolygon
        - a scipy convexh hull
    intensity : float
        the number of observations per unit area in the hull to use. If provided, then
        size must be an integer describing the number of replications to use.
    size : tuple or int
        a tuple of (n_observations, n_replications), where the first number is the number
        of points to simulate in each replication and the second number is the number of
        total replications. So, (10, 4) indicates 10 points, 4 times.
        If an integer is provided and intensity is None, n_replications is assumed to be 1.
        If size is an integer and intensity is also provided, then size indicates n_replications,
        and the number of observations is computed from the intensity.
    n_seeds : int
        the number of sub-clusters to use.
    cluster_radius : float or iterable
        the radius of each cluster. If a float, the same radius is used for all clusters.
        If an array, then there must be the same number of radii as clusters.
        If None, 50% of the minimum inter-point distance is used, which may fluctuate across
        replications.

    Returns
    --------
        :   numpy.ndarray
        either an (n_replications, n_observations, 2) or (n_observations,2) array containing
        the simulated realizations.
    """
    if isinstance(hull, numpy.ndarray):
        if hull.shape == (4,):
            hull = hull
        else:
            hull = _prepare_hull(hull)

    if isinstance(cluster_radius, numpy.ndarray):
        cluster_radii = cluster_radius.flatten()
        assert len(cluster_radii) == n_seeds, (
            f"number of radii provided ({len(cluster_radii)})"
            f"does not match number of clusters requested"
            f" ({n_seeds})."
        )
    elif isinstance(cluster_radius, (int, float)):
        cluster_radii = [cluster_radius] * n_seeds

    n_observations, n_simulations, intensity = parse_size_and_intensity(
        hull, intensity=intensity, size=size
    )

    result = numpy.empty((n_simulations, n_observations, 2))
    hull_area = _area(hull)
    for i_replication in range(n_simulations):
        seeds = poisson(hull, size=(n_seeds, n_simulations))
        if cluster_radius is None:
            # default cluster radius is one half the minimum distance between seeds
            cluster_radii = [spatial.distance.pdist(seeds).min() * 0.5] * n_seeds
        clusters = numpy.array_split(result[i_replication], n_seeds)
        for i_cluster, radius in enumerate(cluster_radii):
            seed = seeds[i_cluster]
            cluster_points = clusters[i_cluster]
            n_in_cluster = len(cluster_points)
            if n_in_cluster == 1:
                clusters[i_cluster] = seed
                continue
            if n_in_cluster < 1:
                raise Exception(
                    "There are too many clusters requested for the "
                    " inputted number of samples. Reduce `n_seeds` or"
                    " increase the number of sampled points."
                )
            candidates = _uniform_circle(
                n_in_cluster - 1, radius=radius, center=seed, hull=hull
            )
            clusters[i_cluster] = numpy.row_stack((seed, candidates))
        result[i_replication] = numpy.row_stack(clusters)
    return result.squeeze()


def cluster_normal(hull, cov=None, size=None, n_seeds=2):
    """
    Simulate a cluster poisson random point process with a specified intensity & number of seeds.
    A cluster poisson process is a poisson process where the center of each "cluster" is
    itself distributed according to a spatial poisson process.

    Parameters
    ----------
    hull : A geometry-like object
        This encodes the "space" in which to simulate the normal pattern. All points will
        lie within this hull. Supported values are:
        - a bounding box encoded in a numpy array as numpy.array([xmin, ymin, xmax, ymax])
        - an (N,2) array of points for which the bounding box will be computed & used
        - a shapely polygon/multipolygon
        - a scipy convexh hull
    cov : float, int, or numpy.ndarray of shape (2,2)
        The covariance structure for clusters. By default, this is the squared
        average distance between cluster seeds.
    size : tuple or int
        a tuple of (n_observations, n_replications), where the first number is the number
        of points to simulate in each replication and the second number is the number of
        total replications. So, (10, 4) indicates 10 points, 4 times.
        If an integer is provided and intensity is None, n_replications is assumed to be 1.
        If size is an integer and intensity is also provided, then size indicates n_replications,
        and the number of observations is computed from the intensity.
    n_seeds : int
        the number of sub-clusters to use.

    Returns
    --------
        :   numpy.ndarray
        either an (n_replications, n_observations, 2) or (n_observations,2) array containing
        the simulated realizations.
    """
    if isinstance(hull, numpy.ndarray):
        if hull.shape == (4,):
            hull = hull
        else:
            hull = _prepare_hull(hull)
    n_observations, n_simulations, intensity = parse_size_and_intensity(
        hull, intensity=None, size=size
    )
    result = numpy.empty((n_simulations, n_observations, 2))
    for i_replication in range(n_simulations):
        seeds = poisson(hull, size=(n_seeds, n_simulations))
        if cov is None:
            cov = spatial.distance.pdist(seeds).mean() ** 2
        clusters = numpy.array_split(result[i_replication], n_seeds)
        for i_cluster, seed in enumerate(seeds):
            cluster_points = clusters[i_cluster]
            n_in_cluster = len(cluster_points)
            if n_in_cluster == 1:
                clusters[i_cluster] = seed
                continue
            if n_in_cluster < 1:
                raise Exception(
                    "There are too many clusters requested for the "
                    " inputted number of samples. Reduce `n_seeds` or"
                    " increase the number of sampled points."
                )
            candidates = normal(hull, center=seed, cov=cov, size=n_in_cluster - 1)
            clusters[i_cluster] = numpy.row_stack((seed, candidates))
        result[i_replication] = numpy.row_stack(clusters)
    return result.squeeze()


def _uniform_circle(n, radius=1.0, center=(0.0, 0.0), burn=2, verbose=False, hull=None):
    """
    Generate n points within a circle of given radius.

    Parameters
    ----------
    n : int
        Number of points.
    radius : float
        Radius of the circle.
    center : tuple
        Coordinates of the center.
    burn : int
        number of coordinates to simulate at a time. This is the "chunk"
        size sent to numpy.random.uniform each iteration of the rejection sampler

    Returns
    -------
      : array
        (n, 2), coordinates of generated points

    """

    good = numpy.zeros((n, 2), float)
    c = 0
    center_x, center_y = center
    r = radius
    r2 = r * r
    it = 0
    while c < n:
        x = numpy.random.uniform(-r, r, (burn * n, 1))
        y = numpy.random.uniform(-r, r, (burn * n, 1))
        if hull is None:
            in_hull = True
        else:
            in_hull = numpy.asarray(
                [
                    _contains(hull, xi + center_x, yi + center_y)
                    for xi, yi in numpy.column_stack((x, y))
                ]
            ).reshape(-1, 1)
        ids, *_ = numpy.where(((x * x + y * y) <= r2) & in_hull)
        candidates = numpy.hstack((x, y))[ids]
        nc = candidates.shape[0]
        need = n - c
        if nc > need:  # more than we need
            good[c:] = candidates[:need]
        else:  # use them all and keep going
            good[c : c + nc] = candidates
        c += nc
        it += 1
    if verbose:
        print("Iterations: {}".format(it))
    return good + numpy.asarray(center)
