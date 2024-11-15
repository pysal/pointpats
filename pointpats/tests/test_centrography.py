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
def test_centrography_mar(points):
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

@dispatch_types
def test_centrography_mbr(points):
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
def test_centrography_hull(points):
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
def test_centrography_mean_center(points):
    exp = np.array([52.57166667, 46.17166667])
    res = centrography.mean_center(points)
    if isinstance(points, gpd.GeoSeries):
        exp = shapely.Point(exp)
        assert exp.equals_exact(res, 1e-5)
    else:
        np.testing.assert_array_almost_equal(res, exp)

@dispatch_types
def test_centrography_std_distance(points):
    std = centrography.std_distance(points)
    np.testing.assert_allclose(std, 40.149806489086714, RTOL)

@dispatch_types
def test_centrography_ellipse(points):
    res = centrography.ellipse(points)
    if isinstance(points, gpd.GeoSeries):
        assert shapely.Point(52.571666666666836, 46.17166666666682).equals_exact(res.centroid, 1e-5)
        np.testing.assert_allclose(res.area, 5313.537950951353, RTOL)
    else:
        np.testing.assert_allclose(res[0], 39.623867886462982, RTOL)
        np.testing.assert_allclose(res[1], 42.753818949026815, RTOL)
        np.testing.assert_allclose(res[2], 1.1039268428650906, RTOL)

@dispatch_types
def test_centrography_euclidean_median(points):
    euclidean = centrography.euclidean_median(points)
    res = np.array([54.16770671, 44.4242589])
    np.testing.assert_array_almost_equal(euclidean, res, decimal=3)
