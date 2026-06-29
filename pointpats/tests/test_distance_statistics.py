import numpy as np
import pytest
import shapely
from scipy import spatial
from shapely.geometry import box

from pointpats import g, k, l
from pointpats.distance_statistics import KEstResult, LEstResult
from pointpats.geometry import max_radius
from pointpats.random import (
    _pairwise_count_kdtree,
    cluster_normal,
    cluster_poisson,
    normal,
    poisson,
)


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
    with pytest.warns(
        RuntimeWarning,
        match="covariance is not symmetric positive-semidefinite.",
    ):
        result = normal(square_hull, cov=cov, size=(20, 3), rng=42)
    assert result.shape == (3, 20, 2)


def test_all_points_within_hull_poisson():
    result = poisson(square_hull, size=(10, 2), rng=42)
    for sim in result:
        for x, y in sim:
            assert 0 <= x <= 10 and 0 <= y <= 10


def test_poisson_seed_consistency():
    a = poisson(square_hull, intensity=1, size=2, rng=42)
    b = poisson(square_hull, intensity=1, size=2, rng=42)
    np.testing.assert_allclose(a, b)


def test_cluster_poisson_shapes():
    result = cluster_poisson(square_hull, size=(20, 2), n_seeds=4, rng=123)
    assert result.shape == (2, 20, 2)


def test_cluster_poisson_seed_consistency():
    a = cluster_poisson(square_hull, size=(20, 2), n_seeds=4, rng=123)
    b = cluster_poisson(square_hull, size=(20, 2), n_seeds=4, rng=123)
    np.testing.assert_allclose(a, b)


def test_value_error_on_conflicting_inputs():
    with pytest.raises(ValueError, match="Either intensity or size as"):
        poisson(square_hull, intensity=1.0, size=(10, 2))


def test_cluster_normal_output():
    result = cluster_normal(square_hull, size=(50, 3), n_seeds=5, rng=100)
    assert result.shape == (3, 50, 2)


def test_pairwise_count_kdtree_basic():
    # Create a square of 4 points within 1 unit distance of each other
    points = np.array([[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]])
    r = 0.75  # should connect all pairs within a 0.75 radius
    count = _pairwise_count_kdtree(points, r)
    assert count == 6  # 4 points, 6 unique pairs


# ---------------------------------------------------------------------------
# Ripley's G function tests
# ---------------------------------------------------------------------------


@pytest.fixture
def coords_and_poly():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (100, 2))
    poly = box(0, 0, 10, 10)
    return coords, poly


class TestGStandard:
    def test_output_shape_consistent(self, coords_and_poly):
        coords, _ = coords_and_poly
        bins, fracs = g(coords)
        assert bins.shape == fracs.shape

    def test_fracs_starts_at_zero(self, coords_and_poly):
        coords, _ = coords_and_poly
        _, fracs = g(coords)
        assert fracs[0] == 0.0

    def test_fracs_ends_at_one(self, coords_and_poly):
        coords, _ = coords_and_poly
        _, fracs = g(coords)
        assert fracs[-1] == pytest.approx(1.0)

    def test_monotone_non_decreasing(self, coords_and_poly):
        coords, _ = coords_and_poly
        _, fracs = g(coords)
        assert np.all(np.diff(fracs) >= 0)

    def test_custom_support_length(self, coords_and_poly):
        coords, _ = coords_and_poly
        bins, fracs = g(coords, support=50)
        assert len(bins) == 50
        assert len(fracs) == 50

    def test_precomputed_nnd_1d(self, coords_and_poly):
        coords, _ = coords_and_poly
        from scipy.spatial import KDTree

        tree = KDTree(coords)
        dists, _ = tree.query(coords, k=2)
        nnd = dists[:, 1]
        bins, fracs = g(coords, distances=nnd)
        assert fracs[-1] == pytest.approx(1.0)

    def test_invalid_edge_correction_raises(self, coords_and_poly):
        coords, _ = coords_and_poly
        with pytest.raises(ValueError, match="edge_correction must be None or 'erosion'"):
            g(coords, edge_correction="ripley")


class TestGErosion:
    def test_output_shape_consistent(self, coords_and_poly):
        coords, poly = coords_and_poly
        support, gvals = g(coords, hull=poly, edge_correction="erosion")
        assert support.shape == gvals.shape

    def test_starts_at_zero(self, coords_and_poly):
        coords, poly = coords_and_poly
        _, gvals = g(coords, hull=poly, edge_correction="erosion")
        assert gvals[0] == 0.0

    def test_values_bounded_in_unit_interval(self, coords_and_poly):
        coords, poly = coords_and_poly
        _, gvals = g(coords, hull=poly, edge_correction="erosion")
        assert np.all(gvals >= 0.0)
        assert np.all(gvals <= 1.0)

    def test_support_clipped_to_erosion_threshold(self, coords_and_poly):
        coords, poly = coords_and_poly
        support, _ = g(coords, hull=poly, edge_correction="erosion")
        max_r, _ = max_radius(poly, points=coords, method="erosion_threshold")
        assert support[-1] <= max_r + 1e-12

    def test_support_clipped_when_extended_beyond_threshold(self, coords_and_poly):
        # Provide a wide support that extends past the erosion threshold; the
        # returned support should be truncated at max_r.
        coords, poly = coords_and_poly
        max_r, _ = max_radius(poly, points=coords, method="erosion_threshold")
        wide_support = np.linspace(0, max_r * 2, 40)
        eroded_support, _ = g(coords, hull=poly, support=wide_support, edge_correction="erosion")
        assert eroded_support[-1] <= max_r + 1e-12
        assert len(eroded_support) < len(wide_support)

    def test_with_shapely_polygon_hull(self, coords_and_poly):
        coords, poly = coords_and_poly
        support, gvals = g(coords, hull=poly, edge_correction="erosion")
        assert len(support) > 0
        assert len(gvals) > 0

    def test_with_bbox_array_hull(self, coords_and_poly):
        coords, _ = coords_and_poly
        bbox = np.array([0.0, 0.0, 10.0, 10.0])
        support, gvals = g(coords, hull=bbox, edge_correction="erosion")
        assert len(support) > 0
        assert np.all(gvals >= 0.0)

    def test_with_convex_hull(self, coords_and_poly):
        coords, _ = coords_and_poly
        ch = spatial.ConvexHull(coords)
        support, gvals = g(coords, hull=ch, edge_correction="erosion")
        assert len(support) > 0
        assert np.all(gvals >= 0.0)

    def test_no_explicit_hull_defaults_to_bbox(self, coords_and_poly):
        # Without hull, _prepare_hull returns a bbox — erosion still runs.
        coords, _ = coords_and_poly
        support, gvals = g(coords, edge_correction="erosion")
        assert len(support) > 0
        assert gvals[0] == 0.0

    def test_true_alias_matches_erosion_string(self, coords_and_poly):
        coords, poly = coords_and_poly
        s1, v1 = g(coords, hull=poly, edge_correction="erosion")
        s2, v2 = g(coords, hull=poly, edge_correction=True)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(v1, v2)

    def test_guard_semantics_exclude_boundary_points(self):
        # Place points: one deep inside, one exactly on the boundary.
        # At a radius larger than the boundary point's distance to the edge,
        # the boundary point should be excluded from the guard set.
        interior = np.array([[5.0, 5.0]])  # far from any edge
        boundary_adj = np.array([[0.05, 5.0]])  # 0.05 from left edge
        coords = np.vstack([interior, boundary_adj])
        poly = box(0, 0, 10, 10)

        shapely_pts = shapely.points(coords[:, 0], coords[:, 1])
        dtb = shapely.distance(shapely_pts, poly.boundary)
        # At r = 0.1, boundary_adj (dtb ≈ 0.05) is NOT in the guard set.
        r = 0.1
        guard = dtb > r
        assert guard[0]  # interior point is a guard point
        assert not guard[1]  # boundary-adjacent point is excluded


# ---------------------------------------------------------------------------
# Ripley's K function tests
# ---------------------------------------------------------------------------


class TestKStandard:
    """Tests for the uncorrected estimator (edge_correction=None)."""

    def test_output_is_tuple_of_two_arrays(self, coords_and_poly):
        coords, _ = coords_and_poly
        support, kvals = k(coords, edge_correction=None)
        assert isinstance(support, np.ndarray)
        assert isinstance(kvals, np.ndarray)
        assert support.shape == kvals.shape

    def test_starts_at_zero(self, coords_and_poly):
        coords, _ = coords_and_poly
        support, kvals = k(coords, edge_correction=None)
        assert kvals[0] == pytest.approx(0.0)

    def test_non_decreasing(self, coords_and_poly):
        coords, _ = coords_and_poly
        _, kvals = k(coords, edge_correction=None)
        assert np.all(np.diff(kvals) >= 0)

    def test_custom_support_length(self, coords_and_poly):
        coords, _ = coords_and_poly
        support, kvals = k(coords, support=30, edge_correction=None)
        assert len(support) == 30

    def test_precomputed_square_distances(self, coords_and_poly):
        coords, _ = coords_and_poly
        full_dists = spatial.distance.cdist(coords, coords)
        support, kvals = k(coords, distances=full_dists, edge_correction=None)
        _, kvals_ref = k(coords, edge_correction=None)
        np.testing.assert_allclose(kvals, kvals_ref)

    def test_precomputed_condensed_distances(self, coords_and_poly):
        coords, _ = coords_and_poly
        pdist = spatial.distance.pdist(coords)
        support, kvals = k(coords, distances=pdist, edge_correction=None)
        _, kvals_ref = k(coords, edge_correction=None)
        np.testing.assert_allclose(kvals, kvals_ref)

    def test_invalid_edge_correction_raises(self, coords_and_poly):
        coords, _ = coords_and_poly
        with pytest.raises(ValueError, match="edge_correction must be one of"):
            k(coords, edge_correction="invalid")


class TestKErosion:
    def test_output_shape_consistent(self, coords_and_poly):
        coords, poly = coords_and_poly
        support, kvals = k(coords, hull=poly, edge_correction="erosion")
        assert support.shape == kvals.shape

    def test_starts_at_zero(self, coords_and_poly):
        coords, poly = coords_and_poly
        _, kvals = k(coords, hull=poly, edge_correction="erosion")
        assert kvals[0] == pytest.approx(0.0)

    def test_values_non_negative(self, coords_and_poly):
        coords, poly = coords_and_poly
        _, kvals = k(coords, hull=poly, edge_correction="erosion")
        assert np.all(kvals >= 0.0)

    def test_support_clipped_to_erosion_threshold(self, coords_and_poly):
        coords, poly = coords_and_poly
        support, _ = k(coords, hull=poly, edge_correction="erosion")
        max_r, _ = max_radius(poly, points=coords, method="erosion_threshold")
        assert support[-1] <= max_r + 1e-12

    def test_support_clipped_when_extended_beyond_threshold(self, coords_and_poly):
        coords, poly = coords_and_poly
        max_r, _ = max_radius(poly, points=coords, method="erosion_threshold")
        wide_support = np.linspace(0, max_r * 2, 40)
        eroded_support, _ = k(coords, hull=poly, support=wide_support, edge_correction="erosion")
        assert eroded_support[-1] <= max_r + 1e-12
        assert len(eroded_support) < len(wide_support)

    def test_with_shapely_polygon_hull(self, coords_and_poly):
        coords, poly = coords_and_poly
        support, kvals = k(coords, hull=poly, edge_correction="erosion")
        assert len(support) > 0 and len(kvals) > 0

    def test_with_bbox_array_hull(self, coords_and_poly):
        coords, _ = coords_and_poly
        bbox = np.array([0.0, 0.0, 10.0, 10.0])
        support, kvals = k(coords, hull=bbox, edge_correction="erosion")
        assert np.all(kvals >= 0.0)

    def test_with_convex_hull(self, coords_and_poly):
        coords, _ = coords_and_poly
        ch = spatial.ConvexHull(coords)
        support, kvals = k(coords, hull=ch, edge_correction="erosion")
        assert len(support) > 0 and np.all(kvals >= 0.0)

    def test_no_explicit_hull_defaults_to_bbox(self, coords_and_poly):
        coords, _ = coords_and_poly
        support, kvals = k(coords, edge_correction="erosion")
        assert len(support) > 0 and kvals[0] == pytest.approx(0.0)

    def test_true_alias_matches_erosion_string(self, coords_and_poly):
        coords, poly = coords_and_poly
        s1, v1 = k(coords, hull=poly, edge_correction="erosion")
        s2, v2 = k(coords, hull=poly, edge_correction=True)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(v1, v2)

    def test_guard_excludes_focal_points_near_boundary(self, coords_and_poly):
        # At r slightly above 0.05, a point 0.05 from the boundary is excluded.
        coords, poly = coords_and_poly
        shapely_pts = shapely.points(coords[:, 0], coords[:, 1])
        dtb = shapely.distance(shapely_pts, poly.boundary)
        r = dtb.min() + 1e-6  # just above the closest point's distance to boundary
        guard = dtb > r
        assert guard.sum() < len(coords)  # at least one point excluded

    def test_csr_k_approx_pi_r_squared(self):
        # For a dense CSR pattern, K(r) ≈ π r² under the erosion estimator.
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 20, (300, 2))
        poly = shapely.box(0, 0, 20, 20)
        support, kvals = k(coords, hull=poly, support=15, edge_correction="erosion")
        expected = np.pi * support**2
        # Allow ±30% relative tolerance — erosion clips edge, so compare only interior.
        mid = len(support) // 2
        np.testing.assert_allclose(kvals[1:mid], expected[1:mid], rtol=0.30)


# ---------------------------------------------------------------------------
# Ripley's L function (erosion via K) tests
# ---------------------------------------------------------------------------


class TestLErosion:
    def test_l_is_sqrt_k_over_pi(self, coords_and_poly):
        coords, poly = coords_and_poly
        s_k, kvals = k(coords, hull=poly, edge_correction="erosion")
        s_l, lvals = l(coords, hull=poly, edge_correction="erosion")
        np.testing.assert_array_equal(s_k, s_l)
        np.testing.assert_allclose(lvals, np.sqrt(kvals / np.pi))

    def test_support_clipped_same_as_k(self, coords_and_poly):
        coords, poly = coords_and_poly
        s_k, _ = k(coords, hull=poly, edge_correction="erosion")
        s_l, _ = l(coords, hull=poly, edge_correction="erosion")
        np.testing.assert_array_equal(s_k, s_l)

    def test_linearized_centers_on_zero_for_csr(self):
        rng = np.random.default_rng(7)
        coords = rng.uniform(0, 20, (300, 2))
        poly = shapely.box(0, 0, 20, 20)
        support, lvals = l(coords, hull=poly, edge_correction="erosion", linearized=True)
        # Linearized L(r) - r should be near 0 for CSR (allow ±1 unit tolerance).
        assert np.abs(lvals).max() < 1.5


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 'border' correction (alias for 'erosion')
# ---------------------------------------------------------------------------


class TestKBorder:
    def test_border_equals_erosion(self, coords_and_poly):
        coords, poly = coords_and_poly
        from pointpats.geometry import max_radius

        max_r, _ = max_radius(poly, points=coords, method="erosion_threshold")
        support = np.linspace(0, max_r * 0.9, 8)
        _, k_bor = k(coords, hull=poly, support=support, edge_correction="border")
        _, k_ero = k(coords, hull=poly, support=support, edge_correction="erosion")
        np.testing.assert_array_equal(k_bor, k_ero)

    def test_output_shape_consistent(self, coords_and_poly):
        coords, poly = coords_and_poly
        support, kvals = k(coords, hull=poly, edge_correction="border")
        assert support.shape == kvals.shape

    def test_values_non_negative(self, coords_and_poly):
        coords, poly = coords_and_poly
        _, kvals = k(coords, hull=poly, edge_correction="border")
        assert np.all(kvals >= 0.0)


# ---------------------------------------------------------------------------
# 'isotropic' correction (exact arc-fraction)
# ---------------------------------------------------------------------------


class TestKIsotropic:
    def test_output_shape_consistent(self, coords_and_poly):
        coords, poly = coords_and_poly
        support, kvals = k(coords, hull=poly, edge_correction="isotropic")
        assert support.shape == kvals.shape

    def test_starts_at_zero(self, coords_and_poly):
        coords, poly = coords_and_poly
        _, kvals = k(coords, hull=poly, edge_correction="isotropic")
        assert kvals[0] == pytest.approx(0.0)

    def test_values_non_negative(self, coords_and_poly):
        coords, poly = coords_and_poly
        _, kvals = k(coords, hull=poly, edge_correction="isotropic")
        assert np.all(kvals >= 0.0)

    def test_non_decreasing(self, coords_and_poly):
        coords, poly = coords_and_poly
        _, kvals = k(coords, hull=poly, edge_correction="isotropic")
        assert np.all(np.diff(kvals) >= 0)

    def test_corrected_geq_uncorrected(self, coords_and_poly):
        coords, poly = coords_and_poly
        support = np.linspace(0, 4, 10)
        _, k_raw = k(coords, hull=poly, support=support, edge_correction=None)
        _, k_iso = k(coords, hull=poly, support=support, edge_correction="isotropic")
        assert np.all(k_iso >= k_raw - 1e-10)

    def test_interior_points_match_uncorrected(self):
        coords = np.array([[5.0, 5.0], [5.5, 5.0], [5.0, 5.5]])
        poly = shapely.box(0, 0, 10, 10)
        support = np.array([0.0, 0.6, 1.0])
        _, k_raw = k(coords, hull=poly, support=support, edge_correction=None)
        _, k_iso = k(coords, hull=poly, support=support, edge_correction="isotropic")
        np.testing.assert_allclose(k_raw, k_iso)

    def test_deterministic(self, coords_and_poly):
        coords, poly = coords_and_poly
        support = np.linspace(0, 4, 10)
        _, k1 = k(coords, hull=poly, support=support, edge_correction="isotropic")
        _, k2 = k(coords, hull=poly, support=support, edge_correction="isotropic")
        np.testing.assert_array_equal(k1, k2)

    def test_precomputed_distances_match_tree(self, coords_and_poly):
        coords, poly = coords_and_poly
        pdist = spatial.distance.pdist(coords)
        support = np.linspace(0, 4, 10)
        _, k_tree = k(coords, hull=poly, support=support, edge_correction="isotropic")
        _, k_pd = k(
            coords, hull=poly, support=support, distances=pdist, edge_correction="isotropic"
        )
        np.testing.assert_allclose(k_tree, k_pd, rtol=1e-10)

    def test_csr_k_approx_pi_r_squared(self):
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 20, (300, 2))
        poly = shapely.box(0, 0, 20, 20)
        support, kvals = k(coords, hull=poly, support=15, edge_correction="isotropic")
        expected = np.pi * support**2
        mid = len(support) // 2
        np.testing.assert_allclose(kvals[1:mid], expected[1:mid], rtol=0.30)

    def test_l_is_sqrt_k_over_pi(self, coords_and_poly):
        coords, poly = coords_and_poly
        support = np.linspace(0, 4, 10)
        _, kvals = k(coords, hull=poly, support=support, edge_correction="isotropic")
        _, lvals = l(coords, hull=poly, support=support, edge_correction="isotropic")
        np.testing.assert_allclose(lvals, np.sqrt(kvals / np.pi))


# ---------------------------------------------------------------------------
# 'translate' correction (translation / Ohser-Stoyan)
# ---------------------------------------------------------------------------


class TestKTranslate:
    def test_output_shape_consistent(self, coords_and_poly):
        coords, poly = coords_and_poly
        support, kvals = k(coords, hull=poly, edge_correction="translate")
        assert support.shape == kvals.shape

    def test_starts_at_zero(self, coords_and_poly):
        coords, poly = coords_and_poly
        _, kvals = k(coords, hull=poly, edge_correction="translate")
        assert kvals[0] == pytest.approx(0.0)

    def test_values_non_negative(self, coords_and_poly):
        coords, poly = coords_and_poly
        _, kvals = k(coords, hull=poly, edge_correction="translate")
        assert np.all(kvals >= 0.0)

    def test_non_decreasing(self, coords_and_poly):
        coords, poly = coords_and_poly
        _, kvals = k(coords, hull=poly, edge_correction="translate")
        assert np.all(np.diff(kvals) >= 0)

    def test_corrected_geq_uncorrected(self, coords_and_poly):
        # Overlap area ≤ window area, so w_ij ≥ area, and K_translate ≥ K_uncorrected.
        coords, poly = coords_and_poly
        support = np.linspace(0, 4, 10)
        _, k_raw = k(coords, hull=poly, support=support, edge_correction=None)
        _, k_tra = k(coords, hull=poly, support=support, edge_correction="translate")
        assert np.all(k_tra >= k_raw - 1e-10)

    def test_precomputed_distances_match_computed(self, coords_and_poly):
        coords, poly = coords_and_poly
        pdist = spatial.distance.pdist(coords)
        support = np.linspace(0, 4, 10)
        _, k_tree = k(coords, hull=poly, support=support, edge_correction="translate")
        _, k_pd = k(
            coords, hull=poly, support=support, distances=pdist, edge_correction="translate"
        )
        np.testing.assert_allclose(k_tree, k_pd, rtol=1e-10)

    def test_with_bbox_array_hull(self, coords_and_poly):
        coords, _ = coords_and_poly
        bbox = np.array([0.0, 0.0, 10.0, 10.0])
        _, kvals = k(coords, hull=bbox, edge_correction="translate")
        assert np.all(kvals >= 0.0)

    def test_csr_k_approx_pi_r_squared(self):
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 20, (300, 2))
        poly = shapely.box(0, 0, 20, 20)
        support, kvals = k(coords, hull=poly, support=15, edge_correction="translate")
        expected = np.pi * support**2
        mid = len(support) // 2
        np.testing.assert_allclose(kvals[1:mid], expected[1:mid], rtol=0.30)

    def test_l_is_sqrt_k_over_pi(self, coords_and_poly):
        coords, poly = coords_and_poly
        support = np.linspace(0, 4, 10)
        _, kvals = k(coords, hull=poly, support=support, edge_correction="translate")
        _, lvals = l(coords, hull=poly, support=support, edge_correction="translate")
        np.testing.assert_allclose(lvals, np.sqrt(kvals / np.pi))


class TestKDefault:
    """Tests for k() called with no edge_correction — the all-three-corrections default."""

    @pytest.fixture
    def default_result(self, coords_and_poly):
        coords, poly = coords_and_poly
        max_r, _ = max_radius(poly, points=coords, method="erosion_threshold")
        support = np.linspace(0, max_r * 1.5, 20)
        return k(coords, hull=poly, support=support), support

    def test_returns_kestresult(self, coords_and_poly):
        coords, poly = coords_and_poly
        result = k(coords, hull=poly)
        assert isinstance(result, KEstResult)

    def test_fields_are_arrays(self, default_result):
        result, _ = default_result
        for field in ("support", "theo", "isotropic", "translate"):
            assert isinstance(getattr(result, field), np.ndarray)
        assert isinstance(result.border, np.ndarray)

    def test_support_length_consistent(self, default_result):
        result, support = default_result
        assert len(result.support) == len(support)
        assert len(result.theo) == len(support)
        assert len(result.border) == len(support)
        assert len(result.isotropic) == len(support)
        assert len(result.translate) == len(support)

    def test_theo_equals_pi_r_squared(self, default_result):
        result, _ = default_result
        np.testing.assert_allclose(result.theo, np.pi * result.support ** 2)

    def test_border_nan_beyond_erosion_threshold(self, coords_and_poly):
        coords, poly = coords_and_poly
        max_r, _ = max_radius(poly, points=coords, method="erosion_threshold")
        support = np.linspace(0, max_r * 1.5, 20)
        result = k(coords, hull=poly, support=support)
        # Support values beyond max_r should be NaN in border
        beyond = support > max_r
        assert np.all(np.isnan(result.border[beyond]))
        # Support values within max_r should be finite
        within = support <= max_r
        assert np.all(np.isfinite(result.border[within]))

    def test_iso_and_translate_fully_defined(self, default_result):
        result, _ = default_result
        assert not np.isnan(result.isotropic).any()
        assert not np.isnan(result.translate).any()

    def test_all_corrections_non_negative(self, default_result):
        result, _ = default_result
        assert np.all(result.theo >= 0)
        assert np.all(result.isotropic >= 0)
        assert np.all(result.translate >= 0)
        within_r = ~np.isnan(result.border)
        assert np.all(result.border[within_r] >= 0)

    def test_border_matches_explicit_border_call(self, coords_and_poly):
        coords, poly = coords_and_poly
        max_r, _ = max_radius(poly, points=coords, method="erosion_threshold")
        support = np.linspace(0, max_r * 0.9, 10)
        result = k(coords, hull=poly, support=support)
        _, k_bor = k(coords, hull=poly, support=support, edge_correction="border")
        np.testing.assert_allclose(result.border, k_bor)

    def test_iso_matches_explicit_isotropic_call(self, coords_and_poly):
        coords, poly = coords_and_poly
        max_r, _ = max_radius(poly, points=coords, method="erosion_threshold")
        support = np.linspace(0, max_r * 0.9, 10)
        result = k(coords, hull=poly, support=support)
        _, k_iso = k(coords, hull=poly, support=support, edge_correction="isotropic")
        np.testing.assert_allclose(result.isotropic, k_iso)

    def test_translate_matches_explicit_translate_call(self, coords_and_poly):
        coords, poly = coords_and_poly
        max_r, _ = max_radius(poly, points=coords, method="erosion_threshold")
        support = np.linspace(0, max_r * 0.9, 10)
        result = k(coords, hull=poly, support=support)
        _, k_tra = k(coords, hull=poly, support=support, edge_correction="translate")
        np.testing.assert_allclose(result.translate, k_tra)


class TestLDefault:
    """Tests for l() called with no edge_correction — the all-three-corrections default."""

    @pytest.fixture
    def default_result(self, coords_and_poly):
        coords, poly = coords_and_poly
        max_r, _ = max_radius(poly, points=coords, method="erosion_threshold")
        support = np.linspace(0, max_r * 1.5, 20)
        return l(coords, hull=poly, support=support), support

    def test_returns_lestresult(self, coords_and_poly):
        coords, poly = coords_and_poly
        result = l(coords, hull=poly)
        assert isinstance(result, LEstResult)

    def test_theo_equals_support(self, default_result):
        result, _ = default_result
        np.testing.assert_allclose(result.theo, result.support)

    def test_l_iso_is_sqrt_k_over_pi(self, coords_and_poly):
        coords, poly = coords_and_poly
        max_r, _ = max_radius(poly, points=coords, method="erosion_threshold")
        support = np.linspace(0, max_r * 0.9, 10)
        k_result = k(coords, hull=poly, support=support)
        l_result = l(coords, hull=poly, support=support)
        np.testing.assert_allclose(l_result.isotropic, np.sqrt(k_result.isotropic / np.pi))

    def test_l_translate_is_sqrt_k_over_pi(self, coords_and_poly):
        coords, poly = coords_and_poly
        max_r, _ = max_radius(poly, points=coords, method="erosion_threshold")
        support = np.linspace(0, max_r * 0.9, 10)
        k_result = k(coords, hull=poly, support=support)
        l_result = l(coords, hull=poly, support=support)
        np.testing.assert_allclose(l_result.translate, np.sqrt(k_result.translate / np.pi))

    def test_linearized_theo_is_zero(self, coords_and_poly):
        coords, poly = coords_and_poly
        result = l(coords, hull=poly, linearized=True)
        assert isinstance(result, LEstResult)
        np.testing.assert_allclose(result.theo, 0.0)

    def test_border_nan_beyond_erosion_threshold(self, coords_and_poly):
        coords, poly = coords_and_poly
        max_r, _ = max_radius(poly, points=coords, method="erosion_threshold")
        support = np.linspace(0, max_r * 1.5, 20)
        result = l(coords, hull=poly, support=support)
        beyond = support > max_r
        assert np.all(np.isnan(result.border[beyond]))
