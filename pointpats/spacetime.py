"""
Methods for identifying space-time interaction in spatio-temporal event
data.
"""
__author__ = (
    "Eli Knaap <eknaap@sdsu.edu>",
    "Nicholas Malizia <nmalizia@asu.edu>",
    "Sergio J. Rey <srey@sdsu.edu>",
    "Philip Stephens <philip.stephens@asu.edu",
)

__all__ = [
    "SpaceTimeEvents",
    "knox",
    "mantel",
    "jacquez",
    "modified_knox",
    "Knox",
    "KnoxLocal",
]

import os
from datetime import date
from warnings import warn

import geopandas as gpd
import libpysal as lps
import numpy as np
import pandas
import scipy.stats as stats
from libpysal import cg
from pandas.api.types import is_numeric_dtype
from scipy.spatial import KDTree
from scipy.stats import hypergeom, poisson
from shapely.geometry import LineString


class SpaceTimeEvents:
    """
    Method for reformatting event data stored in a shapefile for use in
    calculating metrics of spatio-temporal interaction.

    Parameters
    ----------
    path            : string
                      the path to the appropriate shapefile, including the
                      file name and extension
    time            : string
                      column header in the DBF file indicating the column
                      containing the time stamp.
    infer_timestamp : bool, optional
                      if the column containing the timestamp is formatted as
                      calendar dates, try to coerce them into Python datetime
                      objects (the default is False).

    Attributes
    ----------
    n               : int
                      number of events.
    x               : array
                      (n, 1), array of the x coordinates for the events.
    y               : array
                      (n, 1), array of the y coordinates for the events.
    t               : array
                      (n, 1), array of the temporal coordinates for the events.
    space           : array
                      (n, 2), array of the spatial coordinates (x,y) for the
                      events.
    time            : array
                      (n, 2), array of the temporal coordinates (t,1) for the
                      events, the second column is a vector of ones.

    Examples
    --------
    Read in the example shapefile data, ensuring to omit the file
    extension. In order to successfully create the event data the .dbf file
    associated with the shapefile should have a column of values that are a
    timestamp for the events. This timestamp may be a numerical value
    or a date. Date inference was added in version 1.6.

    >>> import libpysal as lps
    >>> path = lps.examples.get_path("burkitt.shp")
    >>> from pointpats import SpaceTimeEvents

    Create an instance of SpaceTimeEvents from a shapefile, where the
    temporal information is stored in a column named "T".

    >>> events = SpaceTimeEvents(path,'T')

    See how many events are in the instance.

    >>> events.n
    188

    Check the spatial coordinates of the first event.

    >>> events.space[0]
    array([300., 302.])

    Check the time of the first event.

    >>> events.t[0]
    array([413.])

    Calculate the time difference between the first two events.

    >>> events.t[1] - events.t[0]
    array([59.])

    New, in 1.6, date support:
    Now, create an instance of SpaceTimeEvents from a shapefile, where the
    temporal information is stored in a column named "DATE".

    >>> events = SpaceTimeEvents(path,'DATE')

    See how many events are in the instance.

    >>> events.n
    188

    Check the spatial coordinates of the first event.

    >>> events.space[0]
    array([300., 302.])

    Check the time of the first event. Note that this value is equivalent to
    413 days after January 1, 1900.

    >>> events.t[0][0]
    datetime.date(1901, 2, 16)

    Calculate the time difference between the first two events.

    >>> (events.t[1][0] - events.t[0][0]).days
    59
    """

    def __init__(self, path, time_col, infer_timestamp=False):
        shp = lps.io.open(path)
        head, tail = os.path.split(path)
        dbf_tail = tail.split(".")[0] + ".dbf"
        dbf = lps.io.open(lps.examples.get_path(dbf_tail))

        # extract the spatial coordinates from the shapefile
        x = [coords[0] for coords in shp]
        y = [coords[1] for coords in shp]

        self.n = n = len(shp)
        x = np.array(x)
        y = np.array(y)
        self.x = np.reshape(x, (n, 1))
        self.y = np.reshape(y, (n, 1))
        self.space = np.hstack((self.x, self.y))

        # extract the temporal information from the database
        if infer_timestamp:
            col = dbf.by_col(time_col)
            if isinstance(col[0], date):
                day1 = min(col)
                col = [(d - day1).days for d in col]
                t = np.array(col)
            else:
                print(
                    "Unable to parse your time column as Python datetime \
                      objects, proceeding as integers."
                )
                t = np.array(col)
        else:
            t = np.array(dbf.by_col(time_col))
        line = np.ones((n, 1))
        self.t = np.reshape(t, (n, 1))
        self.time = np.hstack((self.t, line))

        # close open objects
        dbf.close()
        shp.close()


def knox(s_coords, t_coords, delta, tau, permutations=99, debug=False):
    """
    Knox test for spatio-temporal interaction. :cite:`Knox:1964`

    Parameters
    ----------
    s_coords        : array
                      (n, 2), spatial coordinates.
    t_coords        : array
                      (n, 1), temporal coordinates.
    delta           : float
                      threshold for proximity in space.
    tau             : float
                      threshold for proximity in time.
    permutations    : int, optional
                      the number of permutations used to establish pseudo-
                      significance (the default is 99).
    debug           : bool, optional
                      if true, debugging information is printed (the default is
                      False).

    Returns
    -------
    knox_result     : dictionary
                      contains the statistic (stat) for the test and the
                      associated p-value (pvalue).
    stat            : float
                      value of the knox test for the dataset.
    pvalue          : float
                      pseudo p-value associated with the statistic.
    counts          : int
                      count of space time neighbors.

    Examples
    --------
    >>> import numpy as np
    >>> import libpysal as lps
    >>> from pointpats import SpaceTimeEvents, knox

    Read in the example data and create an instance of SpaceTimeEvents.

    >>> path = lps.examples.get_path("burkitt.shp")
    >>> events = SpaceTimeEvents(path,'T')

    Set the random seed generator. This is used by the permutation based
    inference to replicate the pseudo-significance of our example results -
    the end-user will normally omit this step.

    >>> np.random.seed(100)

    Run the Knox test with distance and time thresholds of 20 and 5,
    respectively. This counts the events that are closer than 20 units in
    space, and 5 units in time.

    >>> result = knox(events.space, events.t, delta=20, tau=5, permutations=99)

    Next, we examine the results. First, we call the statistic from the
    results dictionary. This reports that there are 13 events close
    in both space and time, according to our threshold definitions.

    >>> result['stat'] == 13
    True

    Next, we look at the pseudo-significance of this value, calculated by
    permuting the timestamps and rerunning the statistics. In this case,
    the results indicate there is likely no space-time interaction between
    the events.

    >>> print("%2.2f"%result['pvalue'])
    0.17
    """
    warn("This function is deprecated. Use Knox", DeprecationWarning, stacklevel=2)

    # Do a kdtree on space first as the number of ties (identical points) is
    # likely to be lower for space than time.

    kd_s = cg.KDTree(s_coords)
    neigh_s = kd_s.query_pairs(delta)
    tau2 = tau * tau
    ids = np.array(list(neigh_s))

    # For the neighboring pairs in space, determine which are also time
    # neighbors

    d_t = (t_coords[ids[:, 0]] - t_coords[ids[:, 1]]) ** 2
    n_st = sum(d_t <= tau2)

    knox_result = {"stat": n_st[0]}

    if permutations:
        joint = np.zeros((permutations, 1), int)
        for p in range(permutations):
            np.random.shuffle(t_coords)
            d_t = (t_coords[ids[:, 0]] - t_coords[ids[:, 1]]) ** 2
            joint[p] = np.sum(d_t <= tau2)

        larger = sum(joint >= n_st[0])
        if (permutations - larger) < larger:
            larger = permutations - larger
        p_sim = (larger + 1.0) / (permutations + 1.0)
        knox_result["pvalue"] = p_sim
    return knox_result


def mantel(
    s_coords, t_coords, permutations=99, scon=1.0, spow=-1.0, tcon=1.0, tpow=-1.0
):
    """
    Standardized Mantel test for spatio-temporal interaction. :cite:`Mantel:1967`

    Parameters
    ----------
    s_coords        : array
                      (n, 2), spatial coordinates.
    t_coords        : array
                      (n, 1), temporal coordinates.
    permutations    : int, optional
                      the number of permutations used to establish pseudo-
                      significance (the default is 99).
    scon            : float, optional
                      constant added to spatial distances (the default is 1.0).
    spow            : float, optional
                      value for power transformation for spatial distances
                      (the default is -1.0).
    tcon            : float, optional
                      constant added to temporal distances (the default is 1.0).
    tpow            : float, optional
                      value for power transformation for temporal distances
                      (the default is -1.0).

    Returns
    -------
    mantel_result   : dictionary
                      contains the statistic (stat) for the test and the
                      associated p-value (pvalue).
    stat            : float
                      value of the knox test for the dataset.
    pvalue          : float
                      pseudo p-value associated with the statistic.

    Examples
    --------
    >>> import numpy as np
    >>> import libpysal as lps
    >>> from pointpats import SpaceTimeEvents, mantel

    Read in the example data and create an instance of SpaceTimeEvents.

    >>> path = lps.examples.get_path("burkitt.shp")
    >>> events = SpaceTimeEvents(path,'T')

    Set the random seed generator. This is used by the permutation based
    inference to replicate the pseudo-significance of our example results -
    the end-user will normally omit this step.

    >>> np.random.seed(100)

    The standardized Mantel test is a measure of matrix correlation between
    the spatial and temporal distance matrices of the event dataset. The
    following example runs the standardized Mantel test without a constant
    or transformation; however, as recommended by :cite:`Mantel:1967`, these
    should be added by the user. This can be done by adjusting the constant
    and power parameters.

    >>> result = mantel(events.space, events.t, 99, scon=1.0, spow=-1.0, tcon=1.0, tpow=-1.0)

    Next, we examine the result of the test.

    >>> print("%6.6f"%result['stat'])
    0.048368

    Finally, we look at the pseudo-significance of this value, calculated by
    permuting the timestamps and rerunning the statistic for each of the 99
    permutations. According to these parameters, the results indicate
    space-time interaction between the events.

    >>> print("%2.2f"%result['pvalue'])
    0.01

    """

    t = t_coords
    s = s_coords
    n = len(t)

    # calculate the spatial and temporal distance matrices for the events
    distmat = cg.distance_matrix(s)
    timemat = cg.distance_matrix(t)

    # calculate the transformed standardized statistic
    timevec = (timemat[np.tril_indices(timemat.shape[0], k=-1)] + tcon) ** tpow
    distvec = (distmat[np.tril_indices(distmat.shape[0], k=-1)] + scon) ** spow
    stat = stats.pearsonr(timevec, distvec)[0].sum()

    # return the results (if no inference)
    if not permutations:
        return stat

    # loop for generating a random distribution to assess significance
    dist = []
    for _i in range(permutations):
        trand = _shuffle_matrix(timemat, np.arange(n))
        timevec = (trand[np.tril_indices(trand.shape[0], k=-1)] + tcon) ** tpow
        m = stats.pearsonr(timevec, distvec)[0].sum()
        dist.append(m)

    # establish the pseudo significance of the observed statistic
    distribution = np.array(dist)
    greater = np.ma.masked_greater_equal(distribution, stat)
    count = np.ma.count_masked(greater)
    pvalue = (count + 1.0) / (permutations + 1.0)

    # report the results
    mantel_result = {"stat": stat, "pvalue": pvalue}
    return mantel_result


def jacquez(s_coords, t_coords, k, permutations=99):
    """
    Jacquez k nearest neighbors test for spatio-temporal interaction.
    :cite:`Jacquez:1996`

    Parameters
    ----------
    s_coords        : array
                      (n, 2), spatial coordinates.
    t_coords        : array
                      (n, 1), temporal coordinates.
    k               : int
                      the number of nearest neighbors to be searched.
    permutations    : int, optional
                      the number of permutations used to establish pseudo-
                      significance (the default is 99).

    Returns
    -------
    jacquez_result  : dictionary
                      contains the statistic (stat) for the test and the
                      associated p-value (pvalue).
    stat            : float
                      value of the Jacquez k nearest neighbors test for the
                      dataset.
    pvalue          : float
                      p-value associated with the statistic (normally
                      distributed with k-1 df).

    Examples
    --------
    >>> import numpy as np
    >>> import libpysal as lps
    >>> from pointpats import SpaceTimeEvents, jacquez

    Read in the example data and create an instance of SpaceTimeEvents.

    >>> path = lps.examples.get_path("burkitt.shp")
    >>> events = SpaceTimeEvents(path,'T')

    The Jacquez test counts the number of events that are k nearest
    neighbors in both time and space. The following runs the Jacquez test
    on the example data and reports the resulting statistic. In this case,
    there are 12 instances where events are nearest neighbors in both space
    and time.
    # turning off as kdtree changes from scipy < 0.12 return 13

    >>> np.random.seed(100)
    >>> result = jacquez(events.space, events.t ,k=3,permutations=99)
    >>> print(result['stat'])
    12

    The significance of this can be assessed by calling the p-
    value from the results dictionary, as shown below. Again, no
    space-time interaction is observed.

    >>> result['pvalue'] < 0.01
    False

    """
    time = t_coords
    space = s_coords
    n = len(time)

    # calculate the nearest neighbors in space and time separately
    knnt = lps.weights.KNN.from_array(time, k)
    knns = lps.weights.KNN.from_array(space, k)

    nnt = knnt.neighbors
    nns = knns.neighbors
    knn_sum = 0

    # determine which events are nearest neighbors in both space and time
    for i in range(n):
        t_neighbors = nnt[i]
        s_neighbors = nns[i]
        check = set(t_neighbors)
        inter = check.intersection(s_neighbors)
        count = len(inter)
        knn_sum += count

    stat = knn_sum

    # return the results (if no inference)
    if not permutations:
        return stat

    # loop for generating a random distribution to assess significance
    dist = []
    for _p in range(permutations):
        j = 0
        trand = np.random.permutation(time)
        knnt = lps.weights.KNN.from_array(trand, k)
        nnt = knnt.neighbors
        for i in range(n):
            t_neighbors = nnt[i]
            s_neighbors = nns[i]
            check = set(t_neighbors)
            inter = check.intersection(s_neighbors)
            count = len(inter)
            j += count

        dist.append(j)

    # establish the pseudo significance of the observed statistic
    distribution = np.array(dist)
    greater = np.ma.masked_greater_equal(distribution, stat)
    count = np.ma.count_masked(greater)
    pvalue = (count + 1.0) / (permutations + 1.0)

    # report the results
    jacquez_result = {"stat": stat, "pvalue": pvalue}
    return jacquez_result


def modified_knox(s_coords, t_coords, delta, tau, permutations=99):
    """
    Baker's modified Knox test for spatio-temporal interaction.
    :cite:`Baker:2004`

    Parameters
    ----------
    s_coords        : array
                      (n, 2), spatial coordinates.
    t_coords        : array
                      (n, 1), temporal coordinates.
    delta           : float
                      threshold for proximity in space.
    tau             : float
                      threshold for proximity in time.
    permutations    : int, optional
                      the number of permutations used to establish pseudo-
                      significance (the default is 99).

    Returns
    -------
    modknox_result  : dictionary
                      contains the statistic (stat) for the test and the
                      associated p-value (pvalue).
    stat            : float
                      value of the modified knox test for the dataset.
    pvalue          : float
                      pseudo p-value associated with the statistic.

    Examples
    --------
    >>> import numpy as np
    >>> import libpysal as lps
    >>> from pointpats import SpaceTimeEvents, modified_knox

    Read in the example data and create an instance of SpaceTimeEvents.

    >>> path = lps.examples.get_path("burkitt.shp")
    >>> events = SpaceTimeEvents(path, 'T')

    Set the random seed generator. This is used by the permutation based
    inference to replicate the pseudo-significance of our example results -
    the end-user will normally omit this step.

    >>> np.random.seed(100)

    Run the modified Knox test with distance and time thresholds of 20 and 5,
    respectively. This counts the events that are closer than 20 units in
    space, and 5 units in time.

    >>> result = modified_knox(events.space, events.t, delta=20, tau=5, permutations=99)

    Next, we examine the results. First, we call the statistic from the
    results dictionary. This reports the difference between the observed
    and expected Knox statistic.

    >>> print("%2.8f" % result['stat'])
    2.81016043

    Next, we look at the pseudo-significance of this value, calculated by
    permuting the timestamps and rerunning the statistics. In this case,
    the results indicate there is likely no space-time interaction.

    >>> print("%2.2f" % result['pvalue'])
    0.11

    """
    s = s_coords
    t = t_coords
    n = len(t)

    # calculate the spatial and temporal distance matrices for the events
    sdistmat = cg.distance_matrix(s)
    tdistmat = cg.distance_matrix(t)

    # identify events within thresholds
    spacmat = np.ones((n, n))
    spacbin = sdistmat <= delta
    spacmat = spacmat * spacbin
    timemat = np.ones((n, n))
    timebin = tdistmat <= tau
    timemat = timemat * timebin

    # calculate the observed (original) statistic
    knoxmat = timemat * spacmat
    obsstat = knoxmat.sum() - n

    # calculate the expectated value
    ssumvec = np.reshape((spacbin.sum(axis=0) - 1), (n, 1))
    tsumvec = np.reshape((timebin.sum(axis=0) - 1), (n, 1))
    expstat = (ssumvec * tsumvec).sum()

    # calculate the modified stat
    stat = (obsstat - (expstat / (n - 1.0))) / 2.0

    # return results (if no inference)
    if not permutations:
        return stat
    distribution = []

    # loop for generating a random distribution to assess significance
    for _p in range(permutations):
        rtdistmat = _shuffle_matrix(tdistmat, list(range(n)))
        timemat = np.ones((n, n))
        timebin = rtdistmat <= tau
        timemat = timemat * timebin

        # calculate the observed knox again
        knoxmat = timemat * spacmat
        obsstat = knoxmat.sum() - n

        # calculate the expectated value again
        ssumvec = np.reshape((spacbin.sum(axis=0) - 1), (n, 1))
        tsumvec = np.reshape((timebin.sum(axis=0) - 1), (n, 1))
        expstat = (ssumvec * tsumvec).sum()

        # calculate the modified stat
        tempstat = (obsstat - (expstat / (n - 1.0))) / 2.0
        distribution.append(tempstat)

    # establish the pseudo significance of the observed statistic
    distribution = np.array(distribution)
    greater = np.ma.masked_greater_equal(distribution, stat)
    count = np.ma.count_masked(greater)
    pvalue = (count + 1.0) / (permutations + 1.0)

    # return results
    modknox_result = {"stat": stat, "pvalue": pvalue}
    return modknox_result


def _shuffle_matrix(X, ids):
    """
    Random permutation of rows and columns of a matrix

    Parameters
    ----------
    X   : array
          (k, k), array to be permuted.
    ids : array
          range (k, ).

    Returns
    -------
        : array
          (k, k) with rows and columns randomly shuffled.
    """
    np.random.shuffle(ids)
    return X[ids, :][:, ids]


def _knox(s_coords, t_coords, delta, tau, permutations=99, keep=False):
    """
    Parameters
    ==========

    s_coords: array-like
        spatial coordinates
    t_coords: array-like
        temporal coordinates
    delta: float
        distance threshold
    tau: float
        temporal threshold
    permutations: int
        number of permutations
    keep: bool
        return values from permutations (default False)


    Returns
    =======

    summary table observed
    summary table h0

    ns
    nt
    nst
    n
    p-value
    """

    n = s_coords.shape[0]

    stree = KDTree(s_coords)
    ttree = KDTree(t_coords)
    sneighbors = stree.query_ball_tree(stree, r=delta)
    sneighbors = [
        set(neighbors).difference([i]) for i, neighbors in enumerate(sneighbors)
    ]
    tneighbors = ttree.query_ball_tree(ttree, r=tau)
    tneighbors = [
        set(neighbors).difference([i]) for i, neighbors in enumerate(tneighbors)
    ]

    # number of spatial neighbor pairs
    ns = np.array([len(neighbors) for neighbors in sneighbors])  # by i

    NS = ns.sum() / 2  # total

    # number of temporal neigbor pairs
    nt = np.array([len(neighbors) for neighbors in tneighbors])
    NT = nt.sum() / 2

    # s-t neighbors (list of lists)
    stneighbors = [
        sneighbors_i.intersection(tneighbors_i)
        for sneighbors_i, tneighbors_i in zip(sneighbors, tneighbors)
    ]

    # number of spatio-temporal neigbor pairs
    nst = np.array([len(neighbors) for neighbors in stneighbors])
    NST = nst.sum() / 2

    all_pairs = []
    pairs = {}
    for i, neigh in enumerate(stneighbors):
        if len(neigh) > 0:
            all_pairs.extend([sorted((i, j)) for j in neigh])
    st_pairs = {tuple(l) for l in all_pairs}

    # ENST: expected number of spatio-temporal neighbors under HO
    pairs = n * (n - 1) / 2
    ENST = NS * NT / pairs

    # observed table
    observed = np.zeros((2, 2))

    NS_ = NS - NST  # spatial only
    NT_ = NT - NST  # temporal only

    observed[0, 0] = NST
    observed[0, 1] = NS_
    observed[1, 0] = NT_
    observed[1, 1] = pairs - NST - NS_ - NT_

    # expected table

    expected = np.zeros((2, 2))
    expected[0, 0] = ENST
    expected[0, 1] = NS - expected[0, 0]
    expected[1, 0] = NT - expected[0, 0]
    expected[1, 1] = pairs - expected.sum()

    p_value_poisson = 1 - poisson.cdf(NST, expected[0, 0])

    results = {}
    results["ns"] = ns.sum() / 2
    results["nt"] = nt.sum() / 2
    results["nst"] = nst.sum() / 2
    results["pairs"] = pairs
    results["expected"] = expected
    results["observed"] = observed
    results["p_value_poisson"] = p_value_poisson
    results["st_pairs"] = st_pairs
    results["sneighbors"] = sneighbors
    results["tneighbors"] = tneighbors
    results["stneighbors"] = stneighbors

    if permutations > 0:
        exceedence = 0
        n = len(sneighbors)
        ids = np.arange(n)
        if keep:
            ST = np.zeros(permutations)

        for perm in range(permutations):
            st = 0
            rids = np.random.permutation(ids)
            for i in range(n):
                ri = rids[i]
                tni = tneighbors[ri]
                rjs = [rids[j] for j in sneighbors[i]]
                sti = [j for j in rjs if j in tni]
                st += len(sti)
            st /= 2
            if st >= results["nst"]:
                exceedence += 1
            if keep:
                ST[perm] = st
        results["p_value_sim"] = (exceedence + 1) / (permutations + 1)
        results["exceedence"] = exceedence
        if keep:
            results["st_perm"] = ST

    return results


class Knox:
    """Global Knox statistic for space-time interactions

    Parameters
    ----------
    s_coords: array-like
        spatial coordinates of point events
    t_coords: array-like
        temporal coordinates of point events (floats or ints, not dateTime)
    delta: float
        spatial threshold defining distance below which pairs are spatial
        neighbors
    tau: float
        temporal threshold defining distance below which pairs are temporal
        neighbors
    permutations: int
        number of random permutations for inference
    keep: bool
        whether to store realized values of the statistic under permutations


    Attributes
    ----------
    s_coords: array-like
        spatial coordinates of point events
    t_coords: array-like
        temporal coordinates of point events (floats or ints, not dateTime)
    delta: float
        spatial threshold defining distance below which pairs are spatial
        neighbors
    tau: float
        temporal threshold defining distance below which pairs are temporal
        neighbors
    permutations: int
        number of random permutations for inference
    keep: bool
        whether to store realized values of the statistic under permutations
    nst: int
        number of space-time pairs
    p_poisson: float
        Analytical p-value under Poisson assumption
    p_sim: float
        Pseudo p-value based on random permutations
    expected: array
        Two-by-two array with expected counts under the null of no space-time
        interactions. [[NST, NS_], [NT_, N__]] where NST is the expected number
        of space-time pairs, NS_ is the expected number of spatial (but not also
        temporal) pairs, NT_ is the number of expected temporal (but not also
        spatial pairs), N__ is the number of pairs that are neighor spatial or
        temporal neighbors.
    observed: array
        Same structure as expected with the observed pair classifications
    sim: array
        Global statistics from permutations (if keep=True)

    Notes
    -----
    Technical details can be found in :cite:`Rogerson:2001`

    Examples
    --------
    >>> import libpysal
    >>> path = libpysal.examples.get_path('burkitt.shp')
    >>> import geopandas
    >>> df = geopandas.read_file(path)
    >>> from pointpats.spacetime import Knox
    >>> global_knox = Knox(df[['X', 'Y']], df[["T"]], delta=20, tau=5)
    >>> global_knox.statistic_
    13
    >>> global_knox.p_poisson
    0.14624558197140414
    >>> global_knox.observed
    array([[1.300e+01, 3.416e+03],
           [3.900e+01, 1.411e+04]])
    >>> global_knox.expected
    array([[1.01438161e+01, 3.41885618e+03],
           [4.18561839e+01, 1.41071438e+04]])
    >>> hasattr(global_knox, 'sim')
    False
    >>> import numpy
    >>> numpy.random.seed(12345)
    >>> global_knox = Knox(df[['X', 'Y']], df[["T"]], delta=20, tau=5, keep=True)
    >>> hasattr(global_knox, 'sim')
    True
    >>> global_knox.p_sim
    0.21
    """

    def __init__(self, s_coords, t_coords, delta, tau, permutations=99, keep=False):
        self.s_coords = s_coords
        self.t_coords = t_coords
        self.delta = delta
        self.tau = tau
        self.permutations = permutations
        self.keep = keep
        results = _knox(s_coords, t_coords, delta, tau, permutations, keep)
        self.nst = int(results["nst"])
        if permutations > 0:
            self.p_sim = results["p_value_sim"]
            if keep:
                self.sim = results["st_perm"]

        self.p_poisson = results["p_value_poisson"]
        self.observed = results["observed"]
        self.expected = results["expected"]
        self.statistic_ = self.nst

    @classmethod
    def from_dataframe(
        cls,
        dataframe,
        time_col: int,
        delta: int,
        tau: int,
        permutations: int = 99,
        keep: bool = False,
    ):
        """Compute a Knox statistic from a dataframe of Point observations

        Parameters
        ----------
        dataframe : geopandas.GeoDataFrame
            geodataframe holding observations. Should be in a projected coordinate
            system with geometries stored as Point
        time_col : str
            column in the dataframe storing the time values (integer coordinate)
            for each observation. For example if the observations are stored with
            a timestamp, the time_col should be converted to a series of integers
            representing, e.g. hours, days, seconds, etc.
        delta : int
            delta parameter defining the spatial neighbor threshold measured in the
            same units as the dataframe CRS
        tau : int
            tau parameter defining the temporal neihgbor threshold (in the units
            measured by `time_col`)
        permutations : int, optional
            permutations to use for computation inference, by default 99
        keep : bool
            whether to store realized values of the statistic under permutations

        Returns
        -------
        pointpats.spacetime.Knox
            a fitted Knox class

        """
        s_coords, t_coords = _spacetime_points_to_arrays(dataframe, time_col)

        return cls(s_coords, dataframe[[time_col]], delta, tau, permutations, keep)


def _knox_local(s_coords, t_coords, delta, tau, permutations=99, keep=False):
    """

    Parameters
    ----------
    s_coords: array (n,2)
        spatial coordinates
    t_coords: array (n,1)
        temporal coordinates
    delta: numeric
        spatial threshold distance for neighbor relation
    tau: numeric
        temporal threshold distance for neighbor relation
    permutations: int
        number of permutations for conditional randomization inference
    keep: bool
        whether to store local statistics from the permtuations

    """
    # think about passing in the global object as an option to avoid recomputing the trees
    res = _knox(s_coords, t_coords, delta, tau, permutations=permutations)
    sneighbors = {i: tuple(ns) for i, ns in enumerate(res["sneighbors"])}
    tneighbors = {i: tuple(nt) for i, nt in enumerate(res["tneighbors"])}

    n = len(s_coords)
    ids = np.arange(n)
    res["nsti"] = np.zeros(n)  # number of observed st_pairs for observation i
    res["nsi"] = [len(r) for r in res["sneighbors"]]
    res["nti"] = [len(r) for r in res["tneighbors"]]
    for pair in res["st_pairs"]:
        i, j = pair
        res["nsti"][i] += 1
        res["nsti"][j] += 1

    nsti = res["nsti"]
    nsi = res["nsi"]
    res["nti"]

    # rather than do n*permutations, we reuse the permutations
    # ensuring that each permutation is conditional on a focal unit i
    # for each of the permutations we loop over i and swap labels between the
    # label at index i in the current permutation and the label at the index
    # assigned i in the permutation.

    if permutations > 0:
        exceedence = np.zeros(n)
        if keep:
            STI = np.zeros((n, permutations))
        for perm in range(permutations):
            rids = np.random.permutation(ids)
            for i in range(n):
                rids_i = rids.copy()
                # set observed value of focal unit i
                # swap with value assigned to rids[i]
                # example
                # 0 1 2 (ids)
                # 2 0 1 (rids)
                # i=0
                # 0 2 1 (rids_i)
                # i=1
                # 2 1 0 (rids_i)
                # i=2
                # 1 0 2 (rids_i)

                rids_i[rids == i] = rids[i]
                rids_i[i] = i

                # calculate local stat
                rjs = [rids_i[j] for j in sneighbors[i]]
                tni = tneighbors[i]
                sti = [j for j in rjs if j in tni]
                count = len(sti)
                if count >= res["nsti"][i]:
                    exceedence[i] += 1
                if keep:
                    STI[i, perm] = count

        if keep:
            res["sti_perm"] = STI
        res["exceedence_pvalue"] = (exceedence + 1) / (permutations + 1)
        res["exceedences"] = exceedence

    # analytical inference
    ntjis = [len(r) for r in res["tneighbors"]]
    n1 = n - 1
    hg_pvalues = [
        1 - hypergeom.cdf(nsti[i] - 1, n1, ntjis, nsi[i]).mean() for i in range(n)
    ]
    res["hg_pvalues"] = np.array(hg_pvalues)

    # identification of hot spots

    adjlist = []
    for i, j in res["st_pairs"]:
        adjlist.append([i, j])
        adjlist.append([j, i])
    adjlist = pandas.DataFrame(data=adjlist, columns=["focal", "neighbor"])
    adjlist = adjlist.sort_values(by=["focal", "neighbor"])
    adjlist.reset_index(drop=True, inplace=True)

    adjlist["orientation"] = ""
    for index, row in adjlist.iterrows():
        focal = row["focal"]
        neighbor = row["neighbor"]
        ft = t_coords[focal]
        nt = t_coords[neighbor]
        if ft < nt:
            adjlist.iloc[index, 2] = "lead"
        elif ft > nt:
            adjlist.iloc[index, 2] = "lag"
        else:
            adjlist.iloc[index, 2] = "coincident"

    res["stadjlist"] = adjlist
    return res


class KnoxLocal:
    """Local Knox statistics for space-time interactions

    Parameters
    ----------
    s_coords: array (nx2)
        spatial coordinates of point events
    t_coords: array (nx1)
        temporal coordinates of point events (floats or ints, not dateTime)
    delta: float
        spatial threshold defining distance below which pairs are spatial
        neighbors
    tau: float
        temporal threshold defining distance below which pairs are temporal
        neighbors
    permutations: int
        number of random permutations for inference
    keep: bool
        whether to store realized values of the statistic under permutations
    conditional: bool
        whether to include conditional permutation inference
    crit: float
      signifcance level for local statistics
    crs: str (optional)
        coordinate reference system string for s_coords

    Attributes
    ----------
    s_coords: array (nx2)
        spatial coordinates of point events
    t_coords: array (nx1)
        temporal coordinates of point events (floats or ints, not dateTime)
    delta: float
        spatial threshold defining distance below which pairs are spatial
        neighbors
    tau: float
        temporal threshold defining distance below which pairs are temporal
        neighbors
    permutations: int
        number of random permutations for inference
    keep: bool
        whether to store realized values of the statistic under permutations
    nst: int
        number of space-time pairs (global)
    p_poisson: float
        Analytical p-value under Poisson assumption (global)
    p_sim: float
        Pseudo p-value based on random permutations (global)
    expected: array
        Two-by-two array with expected counts under the null of no space-time
        interactions. [[NST, NS_], [NT_, N__]] where NST is the expected number
        of space-time pairs, NS_ is the expected number of spatial (but not also
        temporal) pairs, NT_ is the number of expected temporal (but not also
        spatial pairs), N__ is the number of pairs that are neighor spatial or
        temporal neighbors. (global)
    observed: array
        Same structure as expected with the observed pair classifications (global)
    sim: array
        Global statistics from permutations (if keep=True and keep=True) (global)
    p_sims: array
        Local psuedo p-values from conditional permutations (if permutations>0)
    sims: array
        Local statistics from conditional permutations (if keep=True and
        permutations>0)
    nsti: array
        Local statistics
    p_hypergeom: array
        Analytical p-values based on hypergeometric distribution

    Notes
    -----
    Technical details can be found in :cite:`Rogerson:2001`. The conditional
    permutation inference is unique to pysal.pointpats.

    Examples
    -------
    >>> import libpysal
    >>> path = libpysal.examples.get_path('burkitt.shp')
    >>> import geopandas
    >>> df = geopandas.read_file(path)
    >>> from pointpats.spacetime import Knox
    >>> import numpy
    >>> numpy.random.seed(12345)
    >>> local_knox = KnoxLocal(df[['X', 'Y']], df[["T"]], delta=20, tau=5, keep=True)
    >>> local_knox.statistic_.shape
    (188,)
    >>> lres = local_knox
    >>> gt0ids = numpy.where(lres.nsti>0)
    >>> gt0ids # doctest: +NORMALIZE_WHITESPACE
    (array([ 25,  26,  30,  31,  35,  36,  41,  42,  46,  47,  51,  52, 102,
              103, 116, 118, 122, 123, 137, 138, 139, 140, 158, 159, 162, 163]),)
    >>> lres.nsti[gt0ids]
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1.])
    >>> lres.p_hypergeom[gt0ids]
    array([0.1348993 , 0.14220663, 0.07335085, 0.08400282, 0.1494317 ,
           0.21524073, 0.0175806 , 0.04599869, 0.17523687, 0.18209188,
           0.19111321, 0.16830444, 0.13734428, 0.14703242, 0.06796364,
           0.03192559, 0.13734428, 0.17523687, 0.12998154, 0.1933476 ,
           0.13244507, 0.13244507, 0.12502644, 0.14703242, 0.12502644,
           0.12998154])
    >>> lres.p_sims[gt0ids]
    array([0.3 , 0.33, 0.11, 0.17, 0.3 , 0.42, 0.06, 0.06, 0.33, 0.34, 0.36,
           0.38, 0.3 , 0.29, 0.41, 0.19, 0.31, 0.39, 0.18, 0.39, 0.48, 0.41,
           0.22, 0.41, 0.39, 0.32])
    """

    def __init__(
        self,
        s_coords,
        t_coords,
        delta,
        tau,
        permutations=99,
        keep=False,
        crit=0.05,
        crs=None,
        ids=None,
    ):
        if not isinstance(t_coords, np.ndarray):
            raise ValueError("t_coords  should be numpy.ndarray type")
        if not isinstance(s_coords, np.ndarray):
            raise ValueError("s_coords  should be numpy.ndarray type")
        n_s, k = s_coords.shape
        rangeids = list(range(n_s))
        if k < 2:
            raise ValueError("s_coords shape required to be nx2")
        n_t, k = t_coords.shape
        if n_s != n_t:
            raise ValueError("t_coords and s_coords need to be same length")
        if ids is not None:
            if len(ids) != n_s:
                raise ValueError("`ids` must have the same length as the inputs")
        else:
            ids = rangeids

        self._ids = ids
        self.s_coords = s_coords
        self.t_coords = t_coords
        self.delta = delta
        self.tau = tau
        self.permutations = permutations
        self.keep = keep
        self.crit = crit
        results = _knox_local(s_coords, t_coords, delta, tau, permutations, keep)
        self.adjlist = results["stadjlist"]
        self.nst = int(results["nst"])
        if permutations > 0:
            self.p_sim = results["p_value_sim"]
            if keep:
                self.sim = results["sti_perm"]

        self.p_poisson = results["p_value_poisson"]
        self.observed = results["observed"]
        self.expected = results["expected"]
        self.p_hypergeom = results["hg_pvalues"]
        if permutations > 0:
            self.p_sims = results["exceedence_pvalue"]
            if keep:
                self.sims = results["sti_perm"]
        self.nsti = results["nsti"]
        # self.hotspots = results["hotspots"]
        self._crs = crs
        self.statistic_ = self.nsti
        self._id_map = dict(zip(rangeids, self._ids))
        self.adjlist["focal"] = self.adjlist["focal"].replace(self._id_map)
        self.adjlist["neighbor"] = self.adjlist["neighbor"].replace(self._id_map)
        # reconstruct df
        geom = gpd.points_from_xy(self.s_coords[:, 0], self.s_coords[:, 1])
        _gdf = gpd.GeoDataFrame(geometry=geom, crs=self._crs, index=self._ids)
        _gdf["time"] = self.t_coords
        if permutations > 0:
            _gdf["p_sim"] = self.p_sims
        _gdf["p_hypergeom"] = self.p_hypergeom
        self._gdf_static = _gdf

    @property
    def _gdf(self):
        return self._gdf_static.copy()

    @classmethod
    def from_dataframe(
        cls,
        dataframe,
        time_col: str,
        delta: int,
        tau: int,
        permutations: int = 99,
        keep: bool = False,
    ):
        """Compute a set of local Knox statistics from a dataframe of Point observations

        Parameters
        ----------
        dataframe : geopandas.GeoDataFrame
            dataframe holding observations. Should be in a projected coordinate system
            with geometries stored as Points
        time_col : str
            column in the dataframe storing the time values (integer coordinate)
            for each observation. For example if the observations are stored with
            a timestamp, the time_col should be converted to a series of integers
            representing, e.g. hours, days, seconds, etc.
        delta : int
            delta parameter defining the spatial neighbor threshold measured in the
            same units as the dataframe CRS
        tau : int
            tau parameter defining the temporal neihgbor threshold (in the units
            measured by `time_col`)
        permutations : int, optional
            permutations to use for computational inference, by default 99
        keep : bool
            whether to store realized values of the statistic under permutations

        Returns
        -------
        pointpats.spacetime.LocalKnox
            a fitted KnoxLocal class

        """
        s_coords, t_coords = _spacetime_points_to_arrays(dataframe, time_col)

        return cls(
            s_coords,
            t_coords,
            delta,
            tau,
            permutations,
            keep,
            crs=dataframe.crs,
            ids=dataframe.index.values,
        )

    def hotspots(self, crit=0.05, inference="permutation"):
        """Table of significant space-time clusters that define local hotspots.

        Parameters
        ----------
        crit : float, optional
            critical value for statistical inference, by default 0.05
        inference : str, optional
            whether p-values should use permutation or analutical inference, by default
            "permutation"

        Returns
        -------
        pandas.DataFrame
            dataframe of significant hotspots

        Raises
        ------
        ValueError
            if `inference` is not in {'permutation', 'analytic'}
        """
        if inference == "permutation":
            if not hasattr(self, "p_sim"):
                warn(
                    "Pseudo-p values not availalable. Permutation-based p-values require "
                    "fitting the KnoxLocal class using `permutations` set to a large "
                    "number. Using analytic p-values instead"
                )
                col = "p_hypergeom"
            else:
                col = "p_sim"
        elif inference == "analytic":
            col = "p_hypergeom"
        else:
            raise ValueError("inference must be either `permutation` or `analytic`")
        # determine hot spots
        pdf_sig = self._gdf[self._gdf[col] <= crit][[col, "time"]].rename(
            columns={col: "pvalue", "time": "focal_time"}
        )
        pdf_sig = pdf_sig.merge(
            self.adjlist, how="inner", left_index=True, right_on="focal"
        ).reset_index(drop=True)

        return pdf_sig.copy()

    def plot(
        self,
        colors: dict = {"focal": "red", "neighbor": "yellow", "nonsig": "grey"},
        crit: float = 0.05,
        inference: str = "permutation",
        point_kwargs: dict = None,
        plot_edges: bool = True,
        edge_color: str = "black",
        edge_kwargs: dict = None,
        ax=None,
    ):
        """plot hotspots

        Parameters
        ----------
        colors : dict, optional
            mapping of colors to hotspot values, by default
            {"focal": "red", "neighbor": "yellow", "nonsig": "grey"}
        crit : float, optional
            critical value for assessing statistical sgifnicance, by default 0.05
        inference : str, optional
            whether to use permutation or analytic inference, by default "permutation"
        point_kwargs : dict, optional
            additional keyword arguments passsed to point plot, by default None
        plot_edges : bool, optional
            whether to plot edges connecting members of the same hotspot subgraph,
            by default True
        edge_color : str, optional
            color of edges when plot_edges is True, by default 'black'
        edge_kwargs : dict, optional
            additional keyword arguments passsed to edge plot, by default None
        ax : matplotlib.axes.Axes, optional
            axes object on which to create the plot, by default None

        Returns
        -------
        matplotlib.axes.Axes
            plot of local space-time hotspots
        """

        if point_kwargs is None:
            point_kwargs = dict()
        if edge_kwargs is None:
            edge_kwargs = dict()
        g = self._gdf.copy()

        g["color"] = colors["nonsig"]
        g["pvalue"] = self.p_hypergeom
        if inference == "permutation":
            if not hasattr(self, "p_sims"):
                warn(
                    "Pseudo-p values not availalable. Permutation-based p-values require "
                    "fitting the KnoxLocal class using `permutations` set to a large "
                    "number. Using analytic p-values instead"
                )
                g["pvalue"] = self.p_hypergeom
            else:
                g["pvalue"] = self.p_sims
        elif inference == "analytic":
            g["pvalue"] = self.p_hypergeom
        else:
            raise ValueError("inference must be either `permutation` or `analytic`")

        mask = g[g.pvalue <= crit].index.values
        neighbors = self.adjlist[self.adjlist.focal.isin(mask)].neighbor.unique()
        g.loc[neighbors, "color"] = colors["neighbor"]
        g.loc[g.pvalue <= crit, "color"] = colors["focal"]
        m = g[g.color == colors["nonsig"]].plot(
            color=colors["nonsig"], ax=ax, **point_kwargs
        )
        g[g.color == colors["neighbor"]].plot(
            ax=m, color=colors["neighbor"], **point_kwargs
        )
        g[g.color == colors["focal"]].plot(ax=m, color=colors["focal"], **point_kwargs)

        if plot_edges:

            # edges between hotspot and st-neighbors
            ghs = self.hotspots(crit=crit, inference=inference)
            ghs = ghs.dropna()
            origins = g.loc[ghs.focal].geometry
            destinations = g.loc[ghs.neighbor].geometry
            ods = zip(origins, destinations)
            lines = gpd.GeoSeries([LineString(od) for od in ods], crs=g.crs)
            lines.plot(ax=m, color=edge_color, **edge_kwargs)

        return m

    def explore(
        self,
        crit: float = 0.05,
        inference: str = "permutation",
        radius: int = 5,
        style_kwds: dict = None,
        tiles: str = "CartoDB Positron",
        plot_edges: bool = True,
        edge_weight: int = 2,
        edge_color: str = "black",
        colors: dict = {"focal": "red", "neighbor": "yellow", "nonsig": "grey"},
    ):
        """Interactive plotting for space-time hotspots.

        Parameters
        ----------
        crit : float, optional
            critical value for statistical inference, by default 0.05
        inference : str, optional
            which p-value to use for determining hotspots. Either "permutation" or
            "analytic", by default "permutation"
        radius : int, optional
            radius of the circlemarker plotted by folium, passed to
            geopandas.GeoDataFrame.explore style_kwds as a convenience. Ignored if
            `style_kwds` is passed directly, by default 5
        style_kwds : dict, optional
            additional style kewords passed to GeoDataFrame.explore, by default None
        tiles : str, optional
            tileset passed to GeoDataFrame.explore `tiles` argument, by default
            "CartoDB Positron"
        plot_edges : bool, optional
            Whether to include lines drawn between members of a singnificant hotspot, by
            default True
        edge_weight : int, optional
            line thickness when `plot_edges=True`, by default 2
        edge_color : str, optional
            color of line when `plot_edges=True`, by default "black"
        colors : dict, optional
            mapping of observation type to color,
            by default {"focal": "red", "neighbor": "yellow", "nonsig": "grey"}

        Returns
        -------
        folium.Map
            an interactive map showing locally-significant spacetime hotspots
        """
        if style_kwds is None:
            style_kwds = {"radius": radius}
        g = self._gdf.copy()

        g["color"] = colors["nonsig"]

        if inference == "permutation":
            if not hasattr(self, "p_sims"):
                warn(
                    "Pseudo-p values not availalable. Permutation-based p-values require "
                    "fitting the KnoxLocal class using `permutations` set to a large "
                    "number. Using analytic p-values instead"
                )
                g["pvalue"] = self.p_hypergeom
            else:
                g["pvalue"] = self.p_sims
        elif inference == "analytic":
            g["pvalue"] = self.p_hypergeom
        else:
            raise ValueError("inference must be either `permutation` or `analytic`")

        mask = g[g.pvalue <= crit].index.values
        neighbors = self.adjlist[self.adjlist.focal.isin(mask)].neighbor.unique()

        # this is clunky, but enforces plotting order so significance is prioritized
        g.loc[neighbors, "color"] = colors["neighbor"]
        g.loc[g.pvalue <= crit, "color"] = colors["focal"]
        nbs = self.adjlist.groupby("focal").agg(list)["neighbor"]
        g = g.reset_index().merge(nbs, left_on="index", right_index=True, how="left")

        m = g[g.color == colors["nonsig"]].explore(
            color="grey", style_kwds=style_kwds, tiles=tiles
        )
        blues = g[g.color == colors["neighbor"]]
        if blues.shape[0] == 0:
            warn("empty neighbor set.")
        else:
            m = blues.explore(m=m, color=colors["neighbor"], style_kwds=style_kwds)
        m = g[g.color == colors["focal"]].explore(
            m=m, color=colors["focal"], style_kwds=style_kwds
        )

        if plot_edges:

            # edges between hotspot and st-neighbors
            g = g.set_index("index")
            ghs = self.hotspots(crit=crit, inference=inference)
            ghs = ghs.dropna()
            origins = g.loc[ghs.focal].geometry
            destinations = g.loc[ghs.neighbor].geometry
            ods = zip(origins, destinations)
            lines = gpd.GeoSeries([LineString(od) for od in ods], crs=g.crs)
            lines.explore(m=m, color=edge_color, style_kwds={"weight": edge_weight})

        return m

    def _gdfhs(self, crit=0.05, inference="permutation"):
        # merge df with self.hotspots
        return gpd.GeoDataFrame(
            self._gdf.merge(
                self.hotspots(crit=crit, inference=inference),
                left_index=True,
                right_on="focal",
            )
        )


def _spacetime_points_to_arrays(dataframe, time_col):
    """convert long-form geodataframe into arrays for kdtree

    Parameters
    ----------
    dataframe : geopandas.GeoDataFrame
        geodataframe with point geometries
    time_col : str
        name of the column on dataframe that stores time values

    Returns
    -------
    tuple
        two numpy arrays holding spatial coodinates s_coords (n,2)
        and temporal coordinates t_coords (n,1)

    """
    if dataframe.crs is None:
        warn(
            "There is no CRS set on the dataframe. The KDTree will assume coordinates "
            "are stored in Euclidean distances"
        )
    else:
        if dataframe.crs.is_geographic:
            raise ValueError(
                "The input dataframe must be in a projected coordinate system."
            )

    assert dataframe.geom_type.unique().tolist() == [
        "Point"
    ], "The Knox statistic is only defined for Point geometries"

    # kdtree wont operate on datetime
    if is_numeric_dtype(dataframe[time_col].dtype) is False:
        raise ValueError(
            "The time values must be stored as "
            f"a numeric dtype but the column {time_col} is stored as "
            f"{dataframe[time_col].dtype}"
        )

    s_coords = np.vstack((dataframe.geometry.x.values, dataframe.geometry.y.values)).T
    t_coords = np.vstack(dataframe[time_col].values)

    return s_coords, t_coords
