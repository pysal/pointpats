import numpy as np
import pytest
from pointpats.random import poisson, normal, cluster_poisson, cluster_normal

# Use a square as a simple hull
square_hull = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])

def test_poisson_output_shape():
    result = poisson(square_hull, intensity=1, size=2, seed=42)
    assert result.shape == (2, 100, 2)  # default is 100 points

def test_normal_output_with_custom_cov():
    cov = np.array([[1, 0.5], [0.5, 2]])
    result = normal(square_hull, cov=cov, size=(20, 3), seed=42)
    assert result.shape == (3, 20, 2)

def test_all_points_within_hull_poisson():
    result = poisson(square_hull, size=(10, 2), seed=42)
    for sim in result:
        for x, y in sim:
            assert 0 <= x <= 10 and 0 <= y <= 10

def test_seed_consistency():
    a = poisson(square_hull, intensity=1, size=2, seed=42)
    b = poisson(square_hull, intensity=1, size=2, seed=42)
    np.testing.assert_allclose(a, b)

def test_cluster_poisson_shapes():
    result = cluster_poisson(square_hull, size=(20, 2), n_seeds=4, seed=123)
    assert result.shape == (2, 20, 2)

def test_value_error_on_conflicting_inputs():
    with pytest.raises(ValueError):
        poisson(square_hull, intensity=1.0, size=(10, 2))

def test_cluster_normal_output():
    result = cluster_normal(square_hull, size=(50, 3), n_seeds=5, seed=100)
    assert result.shape == (3, 50, 2)
