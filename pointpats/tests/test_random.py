import numpy as np
import pytest
from shapely.geometry import box
from pointpats.random import (poisson, normal, cluster_poisson, cluster_normal,
                              strauss, _pairwise_count_kdtree)


@pytest.fixture
def hull():
    return box(0, 0, 1, 1)

@pytest.fixture
def rng():
    return np.random.default_rng(42)


# Use a square as a simple hull
square_hull = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])

def test_poisson_output_shape():
    result = poisson(square_hull, intensity=1, size=2, rng=42)
    assert result.shape == (2, 100, 2)  # default is 100 points

def test_normal_output_with_custom_cov():
    cov = np.array([[1, 0.5], [0.5, 2]])
    result = normal(square_hull, cov=cov, size=(20, 3), rng=42)
    assert result.shape == (3, 20, 2)

def test_all_points_within_hull_poisson():
    result = poisson(square_hull, size=(10, 2), rng=42)
    for sim in result:
        for x, y in sim:
            assert 0 <= x <= 10 and 0 <= y <= 10

def test_seed_consistency():
    a = poisson(square_hull, intensity=1, size=2, rng=42)
    b = poisson(square_hull, intensity=1, size=2, rng=42)
    np.testing.assert_allclose(a, b)

def test_cluster_poisson_shapes():
    result = cluster_poisson(square_hull, size=(20, 2), n_seeds=4, rng=123)
    assert result.shape == (2, 20, 2)

def test_value_error_on_conflicting_inputs():
    with pytest.raises(ValueError):
        poisson(square_hull, intensity=1.0, size=(10, 2))

def test_cluster_normal_output():
    result = cluster_normal(square_hull, size=(50, 3), n_seeds=5, rng=100)
    assert result.shape == (3, 50, 2)


def test_pairwise_count_kdtree_basic():
    # Create a square of 4 points within 1 unit distance of each other
    points = np.array([
        [0.0, 0.0],
        [0.0, 0.5],
        [0.5, 0.0],
        [0.5, 0.5]
    ])
    r = 0.75  # should connect all pairs within a 0.75 radius
    count = _pairwise_count_kdtree(points, r)
    assert count == 6  # 4 points, 6 unique pairs

def test_strauss_basic_box():
    hull = np.array([0.0, 0.0, 1.0, 1.0])  # bounding box
    output = strauss(
        hull,
        intensity=5,
        gamma=0.5,
        r=0.1,
        n_iter=100,
        rng=42,
        max_iter=5
    )
    assert isinstance(output, np.ndarray)
    assert output.shape == (5, 2)
    output = strauss(
        hull,
        size=(5, 3),
        gamma=0.5,
        r=0.1,
        n_iter=100,
        rng=42,
        max_iter=5
        )
    assert isinstance(output, np.ndarray)
    assert output.shape == (3, 5, 2)


def test_strauss_invalid_pattern_raises():
    hull = np.array([0.0, 0.0, 1.0, 1.0])  # bounding box
    with pytest.raises(RuntimeError):
        # Use extreme inhibition to force rejection
        strauss(hull, size=(100, 1), gamma=0.0001, r=0.9, max_iter=2, rng=999)
