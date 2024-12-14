import numpy as np
import shapely
import pytest
import geopandas as gpd

from pointpats import centrography

from libpysal.common import RTOL

points = np.array(
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
geoms =  gpd.GeoSeries.from_xy(*points.T)
sequence = points.tolist()

dispatch_types = pytest.mark.parametrize("points", [points, geoms, sequence], ids=["ndarray", "geoseries", "list"])

@pytest.mark.skipif(
        shapely.geos_version < (3, 12, 0),
        reason="Requires GEOS 3.12.0 to use correct algorithm"
)
@dispatch_types
def test_minimum_rotated_rectangle(points):
    mrr = centrography.minimum_rotated_rectangle(points)
    known = np.array(
        [
            [36.40165, 104.61744],
            [4.087286, 30.417522],
            [75.59908, -0.726158],
            [107.913445, 73.47376251220703],
        ]
    )
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
        reason="Requires GEOS 3.12.0 to use correct algorithm"
)
@dispatch_types
def test_minimum_rotated_rectangle_angle(points):
    _, angle = centrography.minimum_rotated_rectangle(points, return_angle=True)
    np.testing.assert_allclose(angle, 66.46667861350298, RTOL)


@dispatch_types
def test_minimum_bounding_rectangle(points):
    res = centrography.minimum_bounding_rectangle(points)
    if isinstance(points, gpd.GeoSeries):
        assert shapely.box(
            8.2300000000000004,
            7.6799999999999997,
            98.730000000000004,
            92.079999999999998
            ).normalize().equals_exact(res.normalize(), 1e-5)
    else:
        min_x, min_y, max_x, max_y = res
        np.testing.assert_allclose(min_x, 8.2300000000000004, RTOL)
        np.testing.assert_allclose(min_y, 7.6799999999999997, RTOL)
        np.testing.assert_allclose(max_x, 98.730000000000004, RTOL)
        np.testing.assert_allclose(max_y, 92.079999999999998, RTOL)

@dispatch_types
def test_hull(points):
    hull = centrography.hull(points)
    exp = np.array(
        [
            [31.01, 81.21],
            [8.23, 39.93],
            [9.47, 31.02],
            [22.52, 22.39],
            [54.46, 8.48],
            [79.26, 7.68],
            [89.78, 42.53],
            [98.73, 77.17],
            [65.19, 92.08],
        ]
    )
    if isinstance(points, gpd.GeoSeries):
        assert shapely.Polygon(exp).normalize().equals_exact(hull.normalize(), 1e-5)
    else:
        np.testing.assert_array_equal(hull, exp)

@dispatch_types
def test_mean_center(points):
    exp = np.array([52.57166667, 46.17166667])
    res = centrography.mean_center(points)
    if isinstance(points, gpd.GeoSeries):
        exp = shapely.Point(exp)
        assert exp.equals_exact(res, 1e-5)
    else:
        np.testing.assert_array_almost_equal(res, exp)

@dispatch_types
def test_std_distance(points):
    std = centrography.std_distance(points)
    np.testing.assert_allclose(std, 40.149806489086714, RTOL)

@dispatch_types
def test_ellipse(points):
    res = centrography.ellipse(points)
    if isinstance(points, gpd.GeoSeries):
        assert shapely.Point(52.571666666666836, 46.17166666666682).equals_exact(res.centroid, 1e-5)
        np.testing.assert_allclose(res.area, 5313.537950951353, RTOL)
    else:
        np.testing.assert_allclose(res[0], 39.623867886462982, RTOL)
        np.testing.assert_allclose(res[1], 42.753818949026815, RTOL)
        np.testing.assert_allclose(res[2], 1.1039268428650906, RTOL)

@dispatch_types
def test_euclidean_median(points):
    euclidean = centrography.euclidean_median(points)
    res = np.array([54.16770671, 44.4242589])
    if isinstance(points, gpd.GeoSeries):
        assert isinstance(euclidean, shapely.Point)
        euclidean = shapely.get_coordinates(euclidean).flatten()
    np.testing.assert_array_almost_equal(euclidean, res, decimal=3)

@dispatch_types
def test_minimum_bounding_circle(points):
    res = centrography.minimum_bounding_circle(points)
    x = 55.244520477497474
    y = 51.88135107645883
    r = 50.304102155590726
    if isinstance(points, gpd.GeoSeries):
        assert shapely.Point(x, y).equals_exact(res.centroid, 1e-5)
        np.testing.assert_allclose(np.sqrt(res.area /np.pi), r, 0.1)
    else:
        np.testing.assert_allclose(res[0][0], x, RTOL)
        np.testing.assert_allclose(res[0][1], y, RTOL)
        np.testing.assert_allclose(res[1], r, RTOL)

@dispatch_types
def test_weighted_mean_center(points):
    res = centrography.weighted_mean_center(points, np.tile(np.array([1, 2, 3]), 4))
    exp = np.array([53.18333333333333, 50.83916666666667])
    if isinstance(points, gpd.GeoSeries):
        exp = shapely.Point(exp)
        assert exp.equals_exact(res, 1e-5)
    else:
        np.testing.assert_array_almost_equal(res, exp)

@dispatch_types
def test_manhattan_median(points):
    with pytest.warns(UserWarning, match="Manhattan Median is not unique for even"):
        res = centrography.manhattan_median(points)
    exp = np.array([59.825, 41.230])
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
    np.testing.assert_allclose(res, 570.3801950846755, RTOL)