import numpy as np

from pointpats.random import halton, sobol


# -----------------------------------------------------------------------------
# Sobol tests
# -----------------------------------------------------------------------------
def test_sobol_shape():
    hull = np.array([0, 0, 1, 1])
    points = sobol(hull, size=10)
    assert points.shape == (10, 2)

def test_sobol_multiple_replications():
    hull = np.array([0, 0, 1, 1])
    points = sobol(hull, size=(10, 4))
    assert points.shape == (4, 10, 2)

def test_sobol_reproducible():
    hull = np.array([0, 0, 1, 1])
    p1 = sobol(hull, size=100, seed=123)
    p2 = sobol(hull, size=100, seed=123)
    np.testing.assert_allclose(p1, p2)

# -----------------------------------------------------------------------------
# Halton tests
# -----------------------------------------------------------------------------
def test_halton_shape():
    hull = np.array([0, 0, 1, 1])
    points = halton(hull, size=10)
    assert points.shape == (10, 2)

def test_halton_multiple_replications():
    hull = np.array([0, 0, 1, 1])
    points = halton(hull, size=(10, 4))
    assert points.shape == (4, 10, 2)

def test_halton_reproducible():
    hull = np.array([0, 0, 1, 1])
    p1 = halton(hull, size=100, seed=123)
    p2 = halton(hull, size=100, seed=123)
    np.testing.assert_allclose(p1, p2)
