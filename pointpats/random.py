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


def _coerce_rng(rng=None):
    """
    Normalize RNG inputs to a numpy.random.Generator.

    Accepted:
      - None -> default_rng()
      - int / numpy integer -> default_rng(seed)
      - numpy.random.Generator -> as-is
      - numpy.random.RandomState -> wrapped
    """
    if rng is None:
        return numpy.random.default_rng()
    if isinstance(rng, (int, numpy.integer)):
        return numpy.random.default_rng(int(rng))
    if isinstance(rng, numpy.random.Generator):
        return rng
    if isinstance(rng, numpy.random.RandomState):
        return numpy.random.default_rng(rng)
    raise TypeError(
        "rng must be None, an int seed, numpy.random.Generator, or numpy.random.RandomState."
    )


def parse_size_and_intensity(hull, intensity=None, size=None):
    """
    Given a hull, an intensity, and a size int/tuple, correctly
    compute the resulting missing quantities. Defaults to 100 points in one
    replication, meaning the intensity will be computed on the fly
    if nothing is provided.
    """
    if size is None:
        if intensity is not None:
            n_observations = int(_area(hull) * intensity)
        else:
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
        if intensity is not None:
            n_observations = int(intensity * _area(hull))
            n_simulations = size
        else:
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


def poisson(hull, intensity=None, size=None, rng=None):
    """
    Simulate a poisson random point process with a specified intensity.

    Added
    -----
    rng : None | int | numpy.random.Generator | numpy.random.RandomState
        Controls randomness for reproducibility.
    """
    rng = _coerce_rng(rng)

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
        i_observation = 0
        while i_observation < n_observations:
            x = rng.uniform(bbox[0], bbox[2])
            y = rng.uniform(bbox[1], bbox[3])
            if _contains(hull, x, y):
                result[i_replication, i_observation] = (x, y)
                i_observation += 1

    return result.squeeze()


def normal(hull, center=None, cov=None, size=None, rng=None):
    """
    Simulate a multivariate random normal point cluster

    Added
    -----
    rng : None | int | numpy.random.Generator | numpy.random.RandomState
        Controls randomness for reproducibility.
    """
    rng = _coerce_rng(rng)

    if isinstance(hull, numpy.ndarray):
        if hull.shape == (4,):
            hull = hull
        else:
            hull = _prepare_hull(hull)

    if center is None:
        center = _centroid(hull)

    n_observations, n_simulations, _intensity = parse_size_and_intensity(
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
    elif isinstance(cov, numpy.ndarray):
        if cov.ndim == 2:
            assert cov.shape == (2, 2), "Bivariate covariance matrices must be 2 by 2"
        elif cov.ndim == 3:
            assert cov.shape[1:] == (2, 2), (
                "3-dimensional covariance matrices must have shape (n_simulations, 2,2)"
            )
            assert cov.shape[0] == n_simulations, (
                "3-dimensional covariance matrices must have shape (n_simulations, 2,2)"
            )
    else:
        raise ValueError(
            "`cov` argument must be a float (signifying a standard deviation)"
            " or a 2 by 2 array expressing the covariance matrix of the "
            " bivariate normal distribution."
        )

    result = numpy.empty((n_simulations, n_observations, 2))

    for i_replication in range(n_simulations):
        i_observation = 0

        replication_cov = cov[i_replication] if getattr(cov, "ndim", 0) == 3 else cov
        replication_sd = numpy.diagonal(replication_cov) ** 0.5
        replication_cor = (1 / replication_sd) * replication_cov * (1 / replication_sd)

        while i_observation < n_observations:
            candidate = rng.multivariate_normal((0, 0), replication_cor)
            x, y = center + candidate * replication_sd
            if _contains(hull, x, y):
                result[i_replication, i_observation] = (x, y)
                i_observation += 1

    return result.squeeze()


def cluster_poisson(
    hull,
    intensity=None,
    size=None,
    n_seeds=2,
    cluster_radius=None,
    rng=None,
):
    """
    Simulate a cluster poisson random point process with a specified intensity & number of seeds.

    Added
    -----
    rng : None | int | numpy.random.Generator | numpy.random.RandomState
        Controls randomness for reproducibility.
    """
    rng = _coerce_rng(rng)

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

    for i_replication in range(n_simulations):
        # IMPORTANT: we want seeds for *this* replication; request n_seeds points, 1 replication
        seeds = poisson(hull, size=n_seeds, rng=rng)

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
                n_in_cluster - 1, radius=radius, center=seed, hull=hull, rng=rng
            )
            clusters[i_cluster] = numpy.vstack((seed, candidates))

        result[i_replication] = numpy.vstack(clusters)

    return result.squeeze()


def cluster_normal(hull, cov=None, size=None, n_seeds=2, rng=None):
    """
    Simulate a cluster normal random point process.

    Added
    -----
    rng : None | int | numpy.random.Generator | numpy.random.RandomState
        Controls randomness for reproducibility.
    """
    rng = _coerce_rng(rng)

    if isinstance(hull, numpy.ndarray):
        if hull.shape == (4,):
            hull = hull
        else:
            hull = _prepare_hull(hull)

    n_observations, n_simulations, _intensity = parse_size_and_intensity(
        hull, intensity=None, size=size
    )

    result = numpy.empty((n_simulations, n_observations, 2))
    for i_replication in range(n_simulations):
        seeds = poisson(hull, size=n_seeds, rng=rng)
        if cov is None:
            cov_rep = spatial.distance.pdist(seeds).mean() ** 2
        else:
            cov_rep = cov

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
            candidates = normal(
                hull, center=seed, cov=cov_rep, size=n_in_cluster - 1, rng=rng
            )
            clusters[i_cluster] = numpy.vstack((seed, candidates))

        result[i_replication] = numpy.vstack(clusters)

    return result.squeeze()


def _uniform_circle(
    n, radius=1.0, center=(0.0, 0.0), burn=2, verbose=False, hull=None, rng=None
):
    """
    Generate n points within a circle of given radius.

    Added
    -----
    rng : None | int | numpy.random.Generator | numpy.random.RandomState
        Controls randomness for reproducibility.
    """
    rng = _coerce_rng(rng)

    good = numpy.zeros((n, 2), float)
    c = 0
    center_x, center_y = center
    r = radius
    r2 = r * r
    it = 0
    while c < n:
        x = rng.uniform(-r, r, (burn * n, 1))
        y = rng.uniform(-r, r, (burn * n, 1))

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
        if nc > need:
            good[c:] = candidates[:need]
        else:
            good[c : c + nc] = candidates
        c += nc
        it += 1

    if verbose:
        print("Iterations: {}".format(it))

    return good + numpy.asarray(center)
