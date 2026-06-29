import numpy as np
import pytest
import shapely
from scipy import spatial
from shapely.geometry import box

from pointpats import g, k, l
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
    def test_output_is_tuple_of_two_arrays(self, coords_and_poly):
        coords, _ = coords_and_poly
        support, kvals = k(coords)
        assert isinstance(support, np.ndarray)
        assert isinstance(kvals, np.ndarray)
        assert support.shape == kvals.shape

    def test_starts_at_zero(self, coords_and_poly):
        coords, _ = coords_and_poly
        support, kvals = k(coords)
        assert kvals[0] == pytest.approx(0.0)

    def test_non_decreasing(self, coords_and_poly):
        coords, _ = coords_and_poly
        _, kvals = k(coords)
        assert np.all(np.diff(kvals) >= 0)

    def test_custom_support_length(self, coords_and_poly):
        coords, _ = coords_and_poly
        support, kvals = k(coords, support=30)
        assert len(support) == 30

    def test_precomputed_square_distances(self, coords_and_poly):
        coords, _ = coords_and_poly
        full_dists = spatial.distance.cdist(coords, coords)
        support, kvals = k(coords, distances=full_dists)
        _, kvals_ref = k(coords)
        # same support values → same K estimates
        np.testing.assert_allclose(kvals, kvals_ref)

    def test_precomputed_condensed_distances(self, coords_and_poly):
        coords, _ = coords_and_poly
        pdist = spatial.distance.pdist(coords)
        support, kvals = k(coords, distances=pdist)
        _, kvals_ref = k(coords)
        np.testing.assert_allclose(kvals, kvals_ref)

    def test_invalid_edge_correction_raises(self, coords_and_poly):
        coords, _ = coords_and_poly
        with pytest.raises(ValueError, match="edge_correction must be None"):
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
# Ripley's K function — 'ripley' isotropic circle-sampling edge correction
# ---------------------------------------------------------------------------


class TestKRipley:
    def test_output_shape_consistent(self, coords_and_poly):
        coords, poly = coords_and_poly
        support, kvals = k(coords, hull=poly, edge_correction="ripley")
        assert support.shape == kvals.shape

    def test_starts_at_zero(self, coords_and_poly):
        coords, poly = coords_and_poly
        _, kvals = k(coords, hull=poly, edge_correction="ripley")
        assert kvals[0] == pytest.approx(0.0)

    def test_values_non_negative(self, coords_and_poly):
        coords, poly = coords_and_poly
        _, kvals = k(coords, hull=poly, edge_correction="ripley")
        assert np.all(kvals >= 0.0)

    def test_non_decreasing(self, coords_and_poly):
        coords, poly = coords_and_poly
        _, kvals = k(coords, hull=poly, edge_correction="ripley")
        assert np.all(np.diff(kvals) >= 0)

    def test_corrected_geq_uncorrected(self, coords_and_poly):
        # Weights >= 1, so K_ripley >= K_uncorrected at every support point.
        coords, poly = coords_and_poly
        support = np.linspace(0, 4, 10)
        _, k_raw = k(coords, hull=poly, support=support)
        _, k_rip = k(coords, hull=poly, support=support, edge_correction="ripley")
        assert np.all(k_rip >= k_raw - 1e-10)

    def test_interior_points_match_uncorrected(self):
        # Points far from every edge (>support max) → all weights = 1 → same as no correction.
        coords = np.array([[5.0, 5.0], [5.5, 5.0], [5.0, 5.5]])
        poly = shapely.box(0, 0, 10, 10)
        support = np.array([0.0, 0.6, 1.0])
        _, k_raw = k(coords, hull=poly, support=support)
        _, k_rip = k(coords, hull=poly, support=support, edge_correction="ripley")
        np.testing.assert_allclose(k_raw, k_rip)

    def test_n_circle_72_close_to_36(self, coords_and_poly):
        coords, poly = coords_and_poly
        support = np.linspace(0, 4, 10)
        _, k36 = k(coords, hull=poly, support=support, edge_correction="ripley", n_circle=36)
        _, k72 = k(coords, hull=poly, support=support, edge_correction="ripley", n_circle=72)
        np.testing.assert_allclose(k36, k72, rtol=0.05)

    def test_precomputed_condensed_distances_match_tree(self, coords_and_poly):
        coords, poly = coords_and_poly
        pdist = spatial.distance.pdist(coords)
        support = np.linspace(0, 4, 10)
        _, k_tree = k(coords, hull=poly, support=support, edge_correction="ripley")
        _, k_pdist = k(
            coords, hull=poly, support=support, distances=pdist, edge_correction="ripley"
        )
        np.testing.assert_allclose(k_tree, k_pdist, rtol=1e-10)

    def test_with_bbox_array_hull(self, coords_and_poly):
        coords, _ = coords_and_poly
        bbox = np.array([0.0, 0.0, 10.0, 10.0])
        _, kvals = k(coords, hull=bbox, edge_correction="ripley")
        assert np.all(kvals >= 0.0)

    def test_csr_k_approx_pi_r_squared(self):
        rng = np.random.default_rng(0)
        coords = rng.uniform(0, 20, (300, 2))
        poly = shapely.box(0, 0, 20, 20)
        support, kvals = k(coords, hull=poly, support=15, edge_correction="ripley")
        expected = np.pi * support**2
        mid = len(support) // 2
        np.testing.assert_allclose(kvals[1:mid], expected[1:mid], rtol=0.30)


# ---------------------------------------------------------------------------
# Ripley's L function — 'ripley' isotropic circle-sampling edge correction
# ---------------------------------------------------------------------------


class TestLRipley:
    def test_l_is_sqrt_k_over_pi(self, coords_and_poly):
        coords, poly = coords_and_poly
        support = np.linspace(0, 4, 10)
        s_k, kvals = k(coords, hull=poly, support=support, edge_correction="ripley")
        s_l, lvals = l(coords, hull=poly, support=support, edge_correction="ripley")
        np.testing.assert_array_equal(s_k, s_l)
        np.testing.assert_allclose(lvals, np.sqrt(kvals / np.pi))

    def test_n_circle_passed_through(self, coords_and_poly):
        coords, poly = coords_and_poly
        support = np.linspace(0, 4, 10)
        _, l36 = l(coords, hull=poly, support=support, edge_correction="ripley", n_circle=36)
        _, l72 = l(coords, hull=poly, support=support, edge_correction="ripley", n_circle=72)
        np.testing.assert_allclose(l36, l72, rtol=0.05)

    def test_linearized_centers_on_zero_for_csr(self):
        rng = np.random.default_rng(3)
        coords = rng.uniform(0, 20, (300, 2))
        poly = shapely.box(0, 0, 20, 20)
        support, lvals = l(coords, hull=poly, edge_correction="ripley", linearized=True)
        assert np.abs(lvals).max() < 1.5
