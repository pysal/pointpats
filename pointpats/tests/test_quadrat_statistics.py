import importlib
import math

import numpy as np
import pytest
import scipy.stats
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon, box


import pointpats.quadrat_statistics as m


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def pts_simple():
    # Four points spanning a 2x2 square (mbb = [0,0,2,2])
    return np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.5, 1.5],
            [2.0, 2.0],  # on max boundary
        ],
        dtype=float,
    )


@pytest.fixture
def pts_random_small():
    rng = np.random.default_rng(123)
    return rng.random((50, 2), dtype=float) * 100.0


@pytest.fixture
def window_poly():
    # Simple rectangular polygon window
    return box(0.0, 0.0, 10.0, 5.0)


@pytest.fixture
def window_poly_with_hole():
    outer = box(0.0, 0.0, 10.0, 10.0)
    hole = box(4.0, 4.0, 6.0, 6.0)
    return Polygon(outer.exterior.coords, holes=[hole.exterior.coords])


@pytest.fixture
def window_multipoly():
    a = box(0.0, 0.0, 1.0, 1.0)
    b = box(2.0, 2.0, 3.0, 3.0)
    return MultiPolygon([a, b])


# -----------------------------------------------------------------------------
# Helper tests
# -----------------------------------------------------------------------------
def test_as_points_array_accepts_arraylike():
    pts = m._as_points_array([[0, 1], [2, 3.5]])
    assert isinstance(pts, np.ndarray)
    assert pts.shape == (2, 2)
    assert pts.dtype.kind in ("f", "i")  # will often be float, but allow int input


def test_as_points_array_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"points must be a \(n, 2\)"):
        m._as_points_array([1, 2, 3])

    with pytest.raises(ValueError, match=r"points must be a \(n, 2\)"):
        m._as_points_array(np.zeros((3, 3)))


def test_as_points_array_rejects_empty():
    with pytest.raises(ValueError, match="points is empty"):
        m._as_points_array(np.empty((0, 2)))


def test_compute_mbb():
    pts = np.array([[1, 2], [3, 4], [-1, 10]], dtype=float)
    mbb = m._compute_mbb(pts)
    assert mbb.shape == (4,)
    assert np.allclose(mbb, [-1.0, 2.0, 3.0, 10.0])


def test_ensure_window_none_uses_box():
    mbb = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    w = m._ensure_window(None, mbb)
    assert w.equals(box(0.0, 1.0, 2.0, 3.0))


def test_coerce_rng_variants():
    g0 = m._coerce_rng(None)
    assert isinstance(g0, np.random.Generator)

    g1 = m._coerce_rng(123)
    assert isinstance(g1, np.random.Generator)

    g2_in = np.random.default_rng(456)
    g2 = m._coerce_rng(g2_in)
    assert g2 is g2_in

    rs = np.random.RandomState(789)
    g3 = m._coerce_rng(rs)
    assert isinstance(g3, np.random.Generator)

    with pytest.raises(TypeError, match="rng must be None"):
        m._coerce_rng("nope")


def test_window_to_paths_polygon_and_hole(window_poly_with_hole):
    paths = m._window_to_paths(window_poly_with_hole)
    # exterior + 1 hole -> 2 paths
    assert len(paths) == 2
    for x, y in paths:
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert x.shape == y.shape
        assert x.ndim == 1


def test_window_to_paths_multipolygon(window_multipoly):
    paths = m._window_to_paths(window_multipoly)
    assert len(paths) == 2  # two polygons -> two exterior rings


@pytest.mark.skipif(
    pytest.importorskip("geopandas", reason="geopandas not installed") is None,
    reason="geopandas not installed",
)
def test_as_points_array_geoseries_points_and_multipoints():
    import geopandas as gpd

    s = gpd.GeoSeries([Point(0, 0), MultiPoint([Point(1, 1), Point(2, 2)]), None])
    pts = m._as_points_array(s)
    assert pts.shape == (3, 2)
    assert np.allclose(pts, np.array([[0, 0], [1, 1], [2, 2]], dtype=float))


@pytest.mark.skipif(
    pytest.importorskip("geopandas", reason="geopandas not installed") is None,
    reason="geopandas not installed",
)
def test_as_points_array_geoseries_rejects_non_point_geoms():
    import geopandas as gpd

    poly = box(0, 0, 1, 1)
    s = gpd.GeoSeries([poly])
    with pytest.raises(TypeError, match="GeoSeries must contain Point geometries"):
        m._as_points_array(s)


@pytest.mark.skipif(
    pytest.importorskip("geopandas", reason="geopandas not installed") is None,
    reason="geopandas not installed",
)
def test_as_points_array_geoseries_all_empty_errors():
    import geopandas as gpd

    s = gpd.GeoSeries([None, Point()])
    # Point() is empty in shapely
    with pytest.raises(ValueError, match="no non-empty Point geometries"):
        m._as_points_array(s)


# -----------------------------------------------------------------------------
# RectangleM tests
# -----------------------------------------------------------------------------
def test_rectangle_grid_default_dimensions(pts_simple):
    r = m.RectangleM(pts_simple, count_column=2, count_row=2)
    assert r.count_column == 2
    assert r.count_row == 2
    assert r.num == 4
    assert np.allclose(r.mbb, [0.0, 0.0, 2.0, 2.0])
    # cell size should be 1x1
    assert math.isclose(r.rectangle_width, 1.0)
    assert math.isclose(r.rectangle_height, 1.0)


def test_rectangle_grid_width_height_override(pts_simple):
    # range_x=2, range_y=2, width=0.75 -> ceil(2/0.75)=3 columns
    r = m.RectangleM(pts_simple, rectangle_width=0.75, rectangle_height=1.0)
    assert r.count_column == 3
    assert r.count_row == 2
    assert r.num == 6


def test_rectangle_point_location_counts_sum_to_n(pts_random_small):
    r = m.RectangleM(pts_random_small, count_column=5, count_row=4)
    d = r.point_location_sta()
    assert len(d) == r.num
    assert sum(d.values()) == pts_random_small.shape[0]


def test_rectangle_boundary_point_in_last_cell(pts_simple):
    # The point [2,2] lies on max boundary; code clamps index==count to last cell
    r = m.RectangleM(pts_simple, count_column=2, count_row=2)
    d = r.point_location_sta()
    # bottom-left cell id 0 contains [0,0] and [0.5,0.5]
    assert d[0] == 2
    # top-right cell id 3 contains [1.5,1.5] and [2,2]
    assert d[3] == 2


def test_rectangle_plot_counts_smoke(pts_simple):
    r = m.RectangleM(pts_simple, count_column=2, count_row=2)
    ax, cell_ids = r.plot(show="counts")
    assert ax is not None
    assert len(cell_ids) == r.num


def test_rectangle_plot_chi2_requires_contrib(pts_simple):
    r = m.RectangleM(pts_simple, count_column=2, count_row=2)
    with pytest.raises(ValueError, match="chi2_contrib must be provided"):
        r.plot(show="chi2", chi2_contrib=None)


def test_rectangle_plot_chi2_length_mismatch(pts_simple):
    r = m.RectangleM(pts_simple, count_column=2, count_row=2)
    with pytest.raises(ValueError, match="length must match number of plotted cells"):
        r.plot(show="chi2", chi2_contrib=np.zeros(r.num - 1))


def test_rectangle_plot_invalid_show(pts_simple):
    r = m.RectangleM(pts_simple, count_column=2, count_row=2)
    with pytest.raises(ValueError, match='show must be "counts" or "chi2"'):
        r.plot(show="nope")


# -----------------------------------------------------------------------------
# HexagonM tests
# -----------------------------------------------------------------------------
def test_hexagon_grid_basic_invariants(pts_random_small):
    h = m.HexagonM(pts_random_small, lh=10.0)
    assert h.h_length == 10.0
    assert h.semi_height > 0
    assert h.count_column >= 1
    assert h.count_row_even >= 1
    assert h.count_row_odd >= 1
    assert h.num >= 1


def test_hexagon_point_location_counts_sum_to_n(pts_random_small):
    h = m.HexagonM(pts_random_small, lh=10.0)
    d = h.point_location_sta()
    assert sum(d.values()) == pts_random_small.shape[0]
    # dict keys correspond to the number of cells included
    assert len(d) >= 1


def test_hexagon_plot_counts_smoke(pts_simple):
    h = m.HexagonM(pts_simple, lh=1.0)
    ax, cell_ids = h.plot(show="counts")
    assert ax is not None
    assert len(cell_ids) == len(h.point_location_sta())


def test_hexagon_plot_chi2_requires_contrib(pts_simple):
    h = m.HexagonM(pts_simple, lh=1.0)
    with pytest.raises(ValueError, match="chi2_contrib must be provided"):
        h.plot(show="chi2", chi2_contrib=None)


def test_hexagon_plot_chi2_length_mismatch(pts_simple):
    h = m.HexagonM(pts_simple, lh=1.0)
    k = len(h.point_location_sta())
    with pytest.raises(ValueError, match="length must match number of plotted cells"):
        h.plot(show="chi2", chi2_contrib=np.zeros(k - 1))


def test_hexagon_plot_invalid_show(pts_simple):
    h = m.HexagonM(pts_simple, lh=1.0)
    with pytest.raises(ValueError, match='show must be "counts" or "chi2"'):
        h.plot(show="nope")


# -----------------------------------------------------------------------------
# QStatistic tests (analytical)
# -----------------------------------------------------------------------------
def test_qstatistic_rectangle_matches_scipy(pts_simple):
    qs = m.QStatistic(pts_simple, shape="rectangle", nx=2, ny=2, realizations=0)
    d = qs.mr.point_location_sta()
    obs = np.asarray(list(d.values()), dtype=float)

    chi2_ref, p_ref = scipy.stats.chisquare(obs)
    assert math.isclose(qs.chi2, chi2_ref, rel_tol=1e-12, abs_tol=0.0)
    assert math.isclose(qs.chi2_pvalue, p_ref, rel_tol=1e-12, abs_tol=0.0)
    assert qs.df == obs.size - 1
    assert len(qs.cell_ids) == obs.size


def test_qstatistic_hexagon_basic(pts_random_small):
    qs = m.QStatistic(pts_random_small, shape="hexagon", lh=10.0, realizations=0)
    assert qs.shape == "hexagon"
    assert qs.df == len(qs.cell_ids) - 1
    assert np.isfinite(qs.chi2)
    assert 0.0 <= qs.chi2_pvalue <= 1.0


def test_qstatistic_contrib_formula(pts_simple):
    qs = m.QStatistic(pts_simple, shape="rectangle", nx=2, ny=2, realizations=0)
    d = qs.mr.point_location_sta()
    obs = np.asarray(list(d.values()), dtype=float)
    expected = obs.mean()
    contrib_ref = (
        (obs - expected) ** 2 / expected if expected > 0 else np.full_like(obs, np.nan)
    )
    assert np.allclose(qs.chi2_contrib, contrib_ref, equal_nan=True)


def test_qstatistic_invalid_shape_raises(pts_simple):
    with pytest.raises(
        ValueError, match='shape must be either "rectangle" or "hexagon"'
    ):
        m.QStatistic(pts_simple, shape="triangle")


def test_qstatistic_plot_counts_smoke(pts_simple):
    qs = m.QStatistic(pts_simple, shape="rectangle", nx=2, ny=2, realizations=0)
    ax = qs.plot(show="counts")
    assert ax is not None


def test_qstatistic_plot_chi2_smoke(pts_simple):
    qs = m.QStatistic(pts_simple, shape="rectangle", nx=2, ny=2, realizations=0)
    ax = qs.plot(show="chi2")
    assert ax is not None


def test_qstatistic_plot_invalid_show(pts_simple):
    qs = m.QStatistic(pts_simple, shape="rectangle", nx=2, ny=2, realizations=0)
    with pytest.raises(ValueError, match='show must be either "counts" or "chi2"'):
        qs.plot(show="nope")


# -----------------------------------------------------------------------------
# QStatistic tests (simulation branch via poisson stub)
# -----------------------------------------------------------------------------
class _PoissonReturn:
    """Mimic an object with a `.points` attribute."""

    def __init__(self, points):
        self.points = points


def test_qstatistic_simulation_branch_calls_poisson_and_sets_outputs(
    monkeypatch, pts_simple, window_poly
):
    # Make points fit inside the window for clarity
    pts = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 2.0], [4.0, 4.0]], dtype=float)

    calls = {}

    def poisson_stub(window, intensity, realizations, rng=None):
        # record call args for assertions
        calls["window"] = window
        calls["intensity"] = intensity
        calls["realizations"] = realizations
        calls["rng_is_generator"] = isinstance(rng, np.random.Generator)

        # deterministic realizations: identical copies (chi2_sim == chi2_obs)
        return [_PoissonReturn(pts.copy()) for _ in range(realizations)]

    # Patch the imported poisson symbol in the module under test
    monkeypatch.setattr(m, "poisson", poisson_stub)

    qs = m.QStatistic(
        pts,
        shape="rectangle",
        nx=2,
        ny=2,
        realizations=5,
        window=window_poly,
        rng=123,
    )

    assert calls["realizations"] == 5
    assert calls["rng_is_generator"] is True
    assert math.isclose(
        calls["intensity"], pts.shape[0] / window_poly.area, rel_tol=1e-12
    )

    assert hasattr(qs, "chi2_realizations")
    assert qs.chi2_realizations.shape == (5,)
    # Because stub returns same points each time, simulated chi2 equals observed chi2.
    assert np.allclose(qs.chi2_realizations, qs.chi2)

    # With chi2_sim == chi2_obs, #{>=} == R, so p = (R+1)/(R+1) = 1
    assert math.isclose(qs.chi2_r_pvalue, 1.0, rel_tol=0.0, abs_tol=0.0)


def test_qstatistic_simulation_accepts_realizations_object_without_points_attr(
    monkeypatch, pts_simple, window_poly
):
    pts = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 2.0], [4.0, 4.0]], dtype=float)

    def poisson_stub(window, intensity, realizations, rng=None):
        # Return raw arrays (code path uses getattr(ri, "points", ri))
        return [pts.copy() for _ in range(realizations)]

    monkeypatch.setattr(m, "poisson", poisson_stub)

    qs = m.QStatistic(
        pts,
        shape="rectangle",
        nx=2,
        ny=2,
        realizations=3,
        window=window_poly,
        rng=np.random.default_rng(0),
    )
    assert qs.chi2_realizations.shape == (3,)
    assert 0.0 <= qs.chi2_r_pvalue <= 1.0
