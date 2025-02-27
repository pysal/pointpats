import numpy as np
import shapely
import pytest
import geopandas as gpd

from pointpats import centrography

from libpysal.common import RTOL

# points from Ebdon, D. (1985) Statistics for Geographers.  Second Edition
points = np.array([(1, 2), (2, 3), (2, 2), (2, 1), (3, 4), (3, 3), (3, 1), (4, 4)])


geoms = gpd.GeoSeries.from_xy(*points.T)
sequence = points.tolist()

dispatch_types = pytest.mark.parametrize(
    "points", [points, geoms, sequence], ids=["ndarray", "geoseries", "list"]
)


@pytest.mark.skipif(
    shapely.geos_version < (3, 12, 0),
    reason="Requires GEOS 3.12.0 to use correct algorithm",
)
@dispatch_types
def test_minimum_rotated_rectangle(points):
    mrr = centrography.minimum_rotated_rectangle(points)
    known = np.array([[2.5, 0.5], [5.0, 3.0], [3.5, 4.5], [1.0, 2.0]])

    if isinstance(points, gpd.GeoSeries):
        assert shapely.Polygon(known).normalize().equals_exact(mrr.normalize(), 1e-5)
    else:
        for i in range(5):
            success = np.allclose(mrr, np.roll(known, i, axis=0))
            if success:
                break
        if not success:
            raise AssertionError(
                f"Minimum Rotated Rectangle cannot be"
                f"aligned with correct answer:"
                f"\ncomputed {mrr}\nknown: {known}"
            )


@pytest.mark.skipif(
    shapely.geos_version < (3, 12, 0),
    reason="Requires GEOS 3.12.0 to use correct algorithm",
)
@dispatch_types
def test_minimum_rotated_rectangle_angle(points):
    _, angle = centrography.minimum_rotated_rectangle(points, return_angle=True)
    np.testing.assert_allclose(angle, 45.0, RTOL)


@dispatch_types
def test_minimum_bounding_rectangle(points):
    res = centrography.minimum_bounding_rectangle(points)
    if isinstance(points, gpd.GeoSeries):
        assert shapely.box(1, 1, 4, 4).normalize().equals_exact(res.normalize(), 1e-5)
    else:
        min_x, min_y, max_x, max_y = res
        np.testing.assert_allclose(min_x, 1, RTOL)
        np.testing.assert_allclose(min_y, 1, RTOL)
        np.testing.assert_allclose(max_x, 4, RTOL)
        np.testing.assert_allclose(max_y, 4, RTOL)


@dispatch_types
def test_hull(points):
    hull = centrography.hull(points)
    exp = np.array(
        [
            [1, 2],
            [2, 1],
            [3, 1],
            [4, 4],
            [3, 4],
        ]
    )
    if isinstance(points, gpd.GeoSeries):
        assert shapely.Polygon(exp).normalize().equals_exact(hull.normalize(), 1e-5)
    else:
        np.testing.assert_array_equal(hull, exp)


@dispatch_types
def test_mean_center(points):
    exp = np.array([2.5, 2.5])
    res = centrography.mean_center(points)
    if isinstance(points, gpd.GeoSeries):
        exp = shapely.Point(exp)
        assert exp.equals_exact(res, 1e-5)
    else:
        np.testing.assert_array_almost_equal(res, exp)


@dispatch_types
def test_std_distance(points):
    std = centrography.std_distance(points)
    np.testing.assert_allclose(std, 1.4142135623730951, RTOL)


@dispatch_types
def test_ellipse(points):
    res = centrography.ellipse(points)
    if isinstance(points, gpd.GeoSeries):
        assert shapely.Point(2.5, 2.5).equals_exact(res.centroid, 1e-5)
        np.testing.assert_allclose(res.area, 2.6006886199741195, RTOL)
    else:
        np.testing.assert_allclose(res[0], 0.6640655130520275, RTOL)
        np.testing.assert_allclose(res[1], 1.2486060204784164, RTOL)
        np.testing.assert_allclose(res[2], -0.5535743588970453, RTOL)


@dispatch_types
def test_euclidean_median(points):
    euclidean = centrography.euclidean_median(points)
    res = np.array([2.34998979, 2.48670953])
    if isinstance(points, gpd.GeoSeries):
        assert isinstance(euclidean, shapely.Point)
        euclidean = shapely.get_coordinates(euclidean).flatten()
    np.testing.assert_array_almost_equal(euclidean, res, decimal=3)


@dispatch_types
def test_minimum_bounding_circle(points):
    res = centrography.minimum_bounding_circle(points)
    x = 2.642857142857143
    y = 2.7857142857142856
    r = 1.821078397711709
    if isinstance(points, gpd.GeoSeries):
        assert shapely.Point(x, y).equals_exact(res.centroid, 1e-5)
        np.testing.assert_allclose(np.sqrt(res.area / np.pi), r, 0.1)
    else:
        np.testing.assert_allclose(res[0][0], x, RTOL)
        np.testing.assert_allclose(res[0][1], y, RTOL)
        np.testing.assert_allclose(res[1], r, RTOL)


@dispatch_types
def test_weighted_mean_center(points):
    res = centrography.weighted_mean_center(points, np.tile(np.array([1, 2]), 4))
    exp = np.array([2.5833333, 2.5833333])
    if isinstance(points, gpd.GeoSeries):
        exp = shapely.Point(exp)
        assert exp.equals_exact(res, 1e-5)
    else:
        np.testing.assert_array_almost_equal(res, exp)


@dispatch_types
def test_manhattan_median(points):
    with pytest.warns(UserWarning, match="Manhattan Median is not unique for even"):
        res = centrography.manhattan_median(points)
    exp = np.array([2.5, 2.5])
    if isinstance(points, gpd.GeoSeries):
        exp = shapely.Point(exp)
        assert exp.equals_exact(res, 1e-5)
    else:
        np.testing.assert_array_almost_equal(res, exp)


@dispatch_types
def test_dtot(points):
    coord = [20, 30]
    if isinstance(points, gpd.GeoSeries):
        coord = shapely.Point(*coord)
    res = centrography.dtot(coord, points)
    np.testing.assert_allclose(res, 260.8212494704881, RTOL)


def test_ellipse_2():
    p2 = np.array([[3, 4], [3, 9], [4, 7], [5, 5], [6, 3], [7, 1], [7, 6]])
    sx, sy, theta = centrography.ellipse(p2)
    np.testing.assert_allclose(sx, 2.6726124191242437, RTOL)
    np.testing.assert_allclose(sy, 1.1952286093343936, RTOL)
    np.testing.assert_allclose(theta, -1.1071487177940904)
