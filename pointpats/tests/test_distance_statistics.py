import numpy
from scipy import spatial
from pointpats import distance_statistics as ripley, geometry, random
from libpysal.cg import alpha_shape_auto
import shapely
import warnings
import pytest

points = numpy.asarray(
    [
        [66.22, 32.54],
        [22.52, 22.39],
        [31.01, 81.21],
        [9.47, 31.02],
        [30.78, 60.10],
        [75.21, 58.93],
        [79.26, 7.68],
        [8.23, 39.93],
        [98.73, 77.17],
        [89.78, 42.53],
        [65.19, 92.08],
        [54.46, 8.48],
    ]
)

tree = spatial.KDTree(points)

chull = spatial.ConvexHull(points)
ashape = alpha_shape_auto(points)

bbox = numpy.asarray((*points.min(axis=0), *points.max(axis=0)))

support = numpy.linspace(0, 100, num=15)

d_self = spatial.distance.pdist(points)
D_self = spatial.distance.squareform(d_self)
try:
    numpy.random.seed(2478879)
    random_pattern = random.poisson(bbox, size=500)
    D_other = spatial.distance.cdist(points, random_pattern)
except:
    # will cause failures in all ripley tests later from NameErrors about D_other
    # If D_other is missing, then test_simulate should also fail.
    pass


def test_primitives():
    area_bbox = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    assert area_bbox == geometry.area(bbox)
    area_chull = chull.volume
    assert area_chull == geometry.area(chull)
    area_pgon = geometry.area(ashape)
    assert area_pgon == ashape.area
    assert area_pgon == geometry.area(ashape)
    point_in = list(ashape.centroid.coords)[0]
    point_out = (100, 100)

    assert geometry.contains(chull, *point_in)
    assert geometry.contains(ashape, *point_in)
    assert geometry.contains(ashape, *point_in)
    assert geometry.contains(bbox, *point_in)

    assert not (geometry.contains(chull, *point_out))
    assert not (geometry.contains(ashape, *point_out))
    assert not (geometry.contains(ashape, *point_out))
    assert not (geometry.contains(bbox, *point_out))

    numpy.testing.assert_array_equal(bbox, geometry.bbox(bbox))
    numpy.testing.assert_array_equal(bbox, geometry.bbox(ashape))
    numpy.testing.assert_array_equal(bbox, geometry.bbox(ashape))
    numpy.testing.assert_array_equal(bbox, geometry.bbox(chull))
    numpy.testing.assert_array_equal(bbox, geometry.bbox(points))


def test_tree_functions():
    kdtree = ripley._build_best_tree(points, "euclidean")
    balltree = ripley._build_best_tree(points, "haversine")
    try:
        failtree = ripley._build_best_tree(points, "notametric")
    except KeyError:
        pass
    except:
        raise AssertionError("Failed to raise an error for _build_best_tree")

    with pytest.warns(UserWarning):
        mytree = ripley._build_best_tree(points, lambda u, v: numpy.var(u - v))

    # check that neighbors are not returned as a self-neighbor
    # for self-neighbor queries
    distances, indices = ripley._k_neighbors(kdtree, points, k=1)
    assert (indices.squeeze() != numpy.arange(points.shape[0])).all()
    distances, indices = ripley._k_neighbors(balltree, points, k=1)
    assert (indices.squeeze() != numpy.arange(points.shape[0])).all()
    distances, indices = ripley._k_neighbors(mytree, points, k=1)
    assert (indices.squeeze() != numpy.arange(points.shape[0])).all()


def test_prepare():
    tmp_bbox = ripley._prepare_hull(points, "bbox")
    numpy.testing.assert_array_equal(bbox, tmp_bbox)

    tmp_bbox = ripley._prepare_hull(points, None)
    numpy.testing.assert_array_equal(bbox, tmp_bbox)

    tmp_bbox = ripley._prepare_hull(points, bbox)
    assert tmp_bbox is bbox  # pass-through with no modification

    tmp_ashape = ripley._prepare_hull(points, "alpha")
    assert tmp_ashape.equals(ashape)

    tmp_ashape = ripley._prepare_hull(points, "α")
    assert tmp_ashape.equals(ashape)

    tmp_ashape = ripley._prepare_hull(points, ashape)
    assert tmp_ashape is ashape  # pass-through with no modification

    tmp_ashape = ripley._prepare_hull(points, ashape)
    assert shapely.equals(tmp_ashape, ashape)

    tmp_chull = ripley._prepare_hull(points, chull)
    assert tmp_chull is chull  # pass-through with no modification

    tmp_chull = ripley._prepare_hull(points, "convex")
    numpy.testing.assert_allclose(tmp_chull.equations, chull.equations)

    # --------------------------------------------------------------------------
    # Now, check the prepare generally
    # check edge correction raise

    try:
        ripley._prepare(points, None, None, "euclidean", ashape, "ripley")
        raise AssertionError()
    except NotImplementedError:
        pass
    except AssertionError:
        raise AssertionError("Did not raise an error when edge correction is set")

    # check tree gets converted into data with no tree
    out = ripley._prepare(tree, None, None, "euclidean", ashape, None)
    numpy.testing.assert_array_equal(points, out[0])

    # check three distance metrics
    out = ripley._prepare(tree, None, None, "euclidean", ashape, None)[3]
    assert out == "euclidean"
    out = ripley._prepare(tree, None, None, "haversine", ashape, None)[3]
    assert out == "haversine"
    test_func = lambda u, v: numpy.var(u - v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ripley._prepare(tree, None, None, test_func, ashape, None)[3]
        assert out is test_func

    # check precomputed failure
    try:
        out = ripley._prepare(tree, None, None, "precomputed", ashape, None)
        raise AssertionError()
    except ValueError:
        pass
    except AssertionError:
        raise AssertionError(
            'Did not raise when metric="precomputed" but' " no distances provided"
        )

    # check support setting will:
    # give 20 breaks from 0 to max dist if none
    out = ripley._prepare(tree, None, None, "euclidean", ashape, None)[1]
    assert len(out) == 20
    assert out.min() == 0
    numpy.testing.assert_allclose(out.max(), 34.631242)
    numpy.testing.assert_allclose(out.min(), 0)
    out = ripley._prepare(tree, 30, None, "euclidean", ashape, None)[1]
    assert len(out) == 30
    numpy.testing.assert_allclose(out.max(), 34.631242)
    numpy.testing.assert_allclose(out.min(), 0)
    # give tuple correctly for 1, 2, and 3-length tuples
    out = ripley._prepare(tree, (4,), None, "euclidean", ashape, None)[1]
    assert out.max() == 4
    out = ripley._prepare(tree, (2, 10), None, "euclidean", ashape, None)[1]
    assert out.max() == 10
    assert out.min() == 2
    out = ripley._prepare(tree, (2, 10, 5), None, "euclidean", ashape, None)[1]
    assert out.max() == 10
    assert out.min() == 2
    assert len(out) == 5
    # passthrough support
    out = ripley._prepare(tree, numpy.arange(40), None, "euclidean", ashape, None)[1]
    assert len(out) == 40
    assert (out == numpy.arange(40)).all()


def test_simulate():
    assert random.poisson(ashape).shape == (100, 2)
    assert random.poisson(chull).shape == (100, 2)
    assert random.poisson(bbox).shape == (100, 2)

    assert random.poisson(ashape, intensity=1e-2).shape == (50, 2)
    assert random.poisson(chull, intensity=1e-2).shape == (52, 2)
    assert random.poisson(bbox, intensity=1e-2).shape == (76, 2)

    assert random.poisson(ashape, size=90).shape == (90, 2)
    assert random.poisson(chull, intensity=1e-2).shape == (52, 2)
    assert random.poisson(bbox, intensity=1e-2, size=3).shape == (3, 76, 2)
    assert random.poisson(bbox, intensity=None, size=(10, 4)).shape == (4, 10, 2)

    # still need to check the other simulators
    # normal
    # cluster poisson
    # cluster normal


def test_f():
    # -------------------------------------------------------------------------#
    # Check f function has consistent performance

    nn_other = D_other.min(axis=0)
    n_obs_at_dist, histogram_support = numpy.histogram(nn_other, bins=support)
    manual_f = numpy.asarray([0, *numpy.cumsum(n_obs_at_dist) / n_obs_at_dist.sum()])
    numpy.random.seed(2478879)
    f_test = ripley.f_test(points, support=support, distances=D_other, n_simulations=99)

    numpy.testing.assert_allclose(support, f_test.support)
    numpy.testing.assert_allclose(manual_f, f_test.statistic)
    numpy.testing.assert_allclose(
        f_test.pvalue < 0.05, [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    )
    assert f_test.simulations is None

    f_test = ripley.f_test(
        points,
        support=support,
        distances=D_other,
        n_simulations=99,
        keep_simulations=True,
    )
    assert f_test.simulations.shape == (99, 15)


def test_g():
    # -------------------------------------------------------------------------#
    # Check f function works, has statistical results that are consistent

    nn_self = (D_self + numpy.eye(points.shape[0]) * 10000).min(axis=0)
    n_obs_at_dist, histogram_support = numpy.histogram(nn_self, bins=support)
    numpy.random.seed(2478879)
    manual_g = numpy.asarray([0, *numpy.cumsum(n_obs_at_dist) / n_obs_at_dist.sum()])
    g_test = ripley.g_test(points, support=support, n_simulations=99)

    numpy.testing.assert_allclose(support, g_test.support)
    numpy.testing.assert_allclose(manual_g, g_test.statistic)
    numpy.testing.assert_allclose(
        g_test.pvalue < 0.05, [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    assert g_test.simulations is None

    g_test = ripley.g_test(
        points, support=support, n_simulations=99, keep_simulations=True
    )
    assert g_test.simulations.shape == (99, 15)


def test_j():
    # -------------------------------------------------------------------------#
    # Check j function works, matches manual, is truncated correctly

    numpy.random.seed(2478879)
    j_test = ripley.j_test(points, support=support, n_simulations=99, truncate=True)
    numpy.random.seed(2478879)
    j_test_fullno = ripley.j_test(
        points, support=support, n_simulations=0, truncate=False
    )
    numpy.testing.assert_array_equal(j_test.support[:4], support[:4])
    numpy.testing.assert_array_equal(j_test_fullno.support, support)
    numpy.random.seed(2478879)
    _, f_tmp = ripley.f(points, support=support)
    _, g_tmp = ripley.g(points, support=support)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        manual_j = (1 - g_tmp) / (1 - f_tmp)
    assert numpy.isnan(manual_j[-1])
    assert len(manual_j) > len(j_test.support)
    assert len(manual_j) == len(j_test_fullno.support)

    numpy.testing.assert_allclose(j_test.statistic, manual_j[:4], atol=0.1, rtol=0.05)


def test_k():
    # -------------------------------------------------------------------------#
    # Check K function works, matches a manual, slower explicit computation

    k_test = ripley.k_test(points, support=support)
    n = points.shape[0]
    intensity = n / ripley._area(bbox)
    manual_unscaled_k = numpy.asarray(
        [(d_self < d).sum() for d in support], dtype=float
    )
    numpy.testing.assert_allclose(
        k_test.statistic, manual_unscaled_k * 2 / n / intensity
    )


def test_l():
    # -------------------------------------------------------------------------#
    # Check L Function works, can be linearized, and has the right value
    _, k = ripley.k(points, support=support)
    l_test = ripley.l_test(points, support=support, n_simulations=0)
    l_test_lin = ripley.l_test(
        points, support=support, n_simulations=0, linearized=True
    )

    numpy.testing.assert_allclose(l_test.statistic, numpy.sqrt(k / numpy.pi))
    numpy.testing.assert_allclose(
        l_test_lin.statistic, numpy.sqrt(k / numpy.pi) - l_test.support
    )
