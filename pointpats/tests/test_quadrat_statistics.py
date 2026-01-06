import pytest
import numpy as np
import scipy.stats
from shapely.geometry import box, Point, Polygon
import matplotlib.pyplot as plt
from unittest.mock import MagicMock


from pointpats.quadrat_statistics import (
    _as_points_array,
    _compute_mbb,
    RectangleM,
    HexagonM,
    QStatistic,
    _coerce_rng,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def random_points():
    """Generates 100 random points in a 100x100 square."""
    return np.random.default_rng(42).uniform(0, 100, size=(100, 2))


@pytest.fixture
def grid_window():
    """A standard bounding box window."""
    return box(0, 0, 100, 100)


# -----------------------------------------------------------------------------
# Helper Function Tests
# -----------------------------------------------------------------------------


def test_as_points_array_validation():
    # Valid input
    pts = [[0, 0], [1.0, 1.0]]
    normalized = _as_points_array(pts)
    assert normalized.shape == (2, 2)
    assert normalized.dtype == float

    # Invalid shapes
    with pytest.raises(ValueError, match="must be a \(n, 2\) array-like"):
        _as_points_array([1, 2, 3])

    # Empty input
    with pytest.raises(ValueError, match="points is empty"):
        _as_points_array(np.array([]).reshape(0, 2))


def test_compute_mbb():
    pts = np.array([[10, 20], [50, 5], [0, 30]])
    mbb = _compute_mbb(pts)
    # Expected: [xmin, ymin, xmax, ymax]
    np.testing.assert_array_equal(mbb, [0.0, 5.0, 50.0, 30.0])


def test_coerce_rng():
    # Int seed
    rng = _coerce_rng(42)
    assert isinstance(rng, np.random.Generator)

    # Generator
    gen = np.random.default_rng(10)
    assert _coerce_rng(gen) is gen

    # Invalid type
    with pytest.raises(TypeError):
        _coerce_rng("invalid")


# -----------------------------------------------------------------------------
# RectangleM Tests
# -----------------------------------------------------------------------------


def test_rectangle_m_init():
    pts = np.array([[0, 0], [10, 10]])
    # 2x2 grid
    rect = RectangleM(pts, count_column=2, count_row=2)
    assert rect.count_column == 2
    assert rect.count_row == 2
    assert rect.rectangle_width == 5.0
    assert rect.rectangle_height == 5.0
    assert rect.num == 4


def test_rectangle_m_counts():
    # Place 4 points in 4 distinct quadrants of a 10x10 space
    pts = np.array([[2, 2], [8, 2], [2, 8], [8, 8]])
    rect = RectangleM(pts, count_column=2, count_row=2)
    counts = rect.point_location_sta()
    # Each cell should have exactly 1 point
    assert all(count == 1 for count in counts.values())
    assert len(counts) == 4


# -----------------------------------------------------------------------------
# HexagonM Tests
# -----------------------------------------------------------------------------


def test_hexagon_m_init():
    pts = np.array([[0, 0], [50, 50]])
    hex_grid = HexagonM(pts, lh=10)
    assert hex_grid.h_length == 10.0
    assert hex_grid.num > 0

    counts = hex_grid.point_location_sta()
    assert sum(counts.values()) == 2


# -----------------------------------------------------------------------------
# QStatistic Tests
# -----------------------------------------------------------------------------


def test_qstatistic_basic(random_points):
    q = QStatistic(random_points, shape="rectangle", nx=2, ny=2)

    # Chi2 properties
    assert hasattr(q, "chi2")
    assert hasattr(q, "chi2_pvalue")
    assert q.df == 3  # (2*2) - 1
    assert len(q.chi2_contrib) == 4


def test_qstatistic_simulation(random_points, grid_window):
    # Mock the poisson module since it's an external dependency in your code
    # If .random.poisson is available, this will run a real simulation
    try:
        q = QStatistic(random_points, realizations=5, window=grid_window, rng=42)
        assert len(q.chi2_realizations) == 5
        assert 0 <= q.chi2_r_pvalue <= 1
    except ImportError:
        pytest.skip("External .random.poisson dependency not found")


def test_qstatistic_invalid_shape():
    with pytest.raises(ValueError, match="shape must be either"):
        QStatistic([[0, 0]], shape="triangle")


# -----------------------------------------------------------------------------
# Plotting Tests (Smoke Tests)
# -----------------------------------------------------------------------------


def test_plotting_logic(random_points):
    """Verify plotting calls don't crash and return an Axes object."""
    q = QStatistic(random_points, nx=3, ny=3)

    # Test counts plot
    ax = q.plot(show="counts")
    assert isinstance(ax, plt.Axes)
    plt.close()

    # Test chi2 plot
    ax_chi = q.plot(show="chi2")
    assert isinstance(ax_chi, plt.Axes)
    plt.close()

    with pytest.raises(ValueError):
        q.plot(show="invalid_option")


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------


def test_points_on_boundary():
    # MBB is [0, 0, 10, 10]
    pts = np.array([[0, 0], [10, 10]])
    rect = RectangleM(pts, count_column=2, count_row=2)
    counts = rect.point_location_sta()
    # Ensure points on the max boundary are handled by the index-1 logic in the code
    assert sum(counts.values()) == 2
