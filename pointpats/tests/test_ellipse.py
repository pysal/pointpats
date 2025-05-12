import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import Point
from pointpats import ellipse

@pytest.fixture
def sample_coords():
    return np.array([
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
    ])

@pytest.fixture
def sample_points():
    seed = 65647437836358831880808032086803839626
    rng = np.random.default_rng(seed)
    points = rng.integers(0, 100, (50, 2))
    return points

@pytest.fixture
def sample_weights():
    return np.arange(1, 12)

@pytest.fixture
def sample_geoseries(sample_coords):
    return gpd.GeoSeries([Point(x, y) for x, y in sample_coords])

def test_ellipse_numpy_no_weights(sample_coords):
    major, minor, rotation = ellipse(sample_coords)
    assert isinstance(major, float)
    assert isinstance(minor, float)
    assert isinstance(rotation, float)
    assert major > 0
    assert minor > 0
    assert -np.pi <= rotation <= np.pi

def test_ellipse_numpy_with_weights(sample_coords, sample_weights):
    major, minor, rotation = ellipse(sample_coords, weights=sample_weights)
    assert isinstance(major, float)
    assert isinstance(minor, float)
    assert isinstance(rotation, float)

def test_ellipse_list_input(sample_coords):
    coords_list = sample_coords.tolist()
    major, minor, rotation = ellipse(coords_list)
    assert isinstance(major, float)
    assert isinstance(minor, float)
    assert isinstance(rotation, float)

def test_ellipse_list_with_weights(sample_coords, sample_weights):
    coords_list = sample_coords.tolist()
    weights_list = sample_weights.tolist()
    major, minor, rotation = ellipse(coords_list, weights=weights_list)
    assert isinstance(major, float)
    assert isinstance(minor, float)
    assert isinstance(rotation, float)

def test_ellipse_geoseries(sample_geoseries):
    ellipse_ = ellipse(sample_geoseries)
    assert ellipse_.geom_type == "Polygon"
    assert ellipse_.is_valid

def test_ellipse_geoseries_with_weights(sample_geoseries, sample_weights):
    ellipse_ = ellipse(sample_geoseries, weights=sample_weights)
    assert ellipse_.geom_type == "Polygon"
    assert ellipse_.is_valid

def test_invalid_method(sample_coords):
    with pytest.raises(ValueError, match="`method` must be either 'crimestat' or 'yuill'"):
        ellipse(sample_coords, method="invalidmethod")

def test_weights_length_mismatch(sample_coords):
    wrong_weights = np.arange(5)
    with pytest.raises(ValueError):
        ellipse(sample_coords, weights=wrong_weights)

def test_unweighted(sample_points):
    result = ellipse(sample_points)
    expected = (np.float64(43.85494662229593),
                np.float64(36.28453973005919),
                np.float64(-0.9362557045365753))
    assert isinstance(result, tuple)
    assert len(result) == 3
    for r, e in zip(result, expected):
        assert isinstance(r, np.float64)
        np.testing.assert_almost_equal(r, e, decimal=8)

        
def test_weighted(sample_points):
    points = sample_points
    points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(points[:,0], points[:,1]))

    result = ellipse(points_gdf, weights=sample_points[:,1]).area
    expected = 4085.3662956683897
    np.testing.assert_almost_equal(result, expected, decimal=8)

def test_params(sample_points):
    result = ellipse(sample_points)
    expected = (np.float64(43.85494662229593),
                np.float64(36.28453973005919),
                np.float64(-0.9362557045365753))
    assert isinstance(result, tuple)
    assert len(result) == 3
    for r, e in zip(result, expected):
        assert isinstance(r, np.float64)
        np.testing.assert_almost_equal(r, e, decimal=8)

    result = ellipse(sample_points, method='yuill')
    expected = (np.float64(43.85494662229593),
                np.float64(36.28453973005919),
                np.float64(-0.9362557045365753))
    assert isinstance(result, tuple)
    assert len(result) == 3
    for r, e in zip(result, expected):
        assert isinstance(r, np.float64)
        np.testing.assert_almost_equal(r, e, decimal=8)

    result = ellipse(sample_points, method='yuill', crimestatCorr=False)
    expected = (np.float64(31.01013014519952),
                np.float64(25.65704409535755),
                np.float64(-0.9362557045365753))
    assert len(result) == 3
    for r, e in zip(result, expected):
        assert isinstance(r, np.float64)
        np.testing.assert_almost_equal(r, e, decimal=8)

    result = ellipse(sample_points, method='yuill', degfreedCorr=False)
    expected = (np.float64(42.96889676864705),
                np.float64(35.551443156155464),
                np.float64(-0.9362557045365753))
    assert len(result) == 3
    for r, e in zip(result, expected):
        assert isinstance(r, np.float64)
        np.testing.assert_almost_equal(r, e, decimal=8)


    result = ellipse(sample_points, method='yuill',
                     crimestatCorr=False,
                     degfreedCorr=False)
    expected = (np.float64(30.383598285215058),
                np.float64(25.138666536685605),
                np.float64(-0.9362557045365753))
    assert len(result) == 3
    for r, e in zip(result, expected):
        assert isinstance(r, np.float64)
        np.testing.assert_almost_equal(r, e, decimal=8)
