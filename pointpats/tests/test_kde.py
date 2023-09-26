import numpy as np
import pytest

from pointpats import plot_density

matplotlib = pytest.importorskip("matplotlib")
statsmodels = pytest.importorskip("statsmodels")
KDEpy = pytest.importorskip("KDEpy")


class TestDensity:
    def setup_method(self):
        self.points = np.array(
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

    def test_default(self):
        ax = plot_density(self.points, 10)
        contourset = ax.collections[0]
        assert len(contourset.collections) == 12
        assert not contourset.filled
        np.testing.assert_array_equal(contourset.get_linewidths(), np.array([1.5] * 12))
        np.testing.assert_array_equal(
            contourset.get_edgecolor()[5],
            np.array([0.143343, 0.522773, 0.556295, 1.0]),
        )

    def test_bandwidth(self):
        ax = plot_density(self.points, 1)
        contourset = ax.collections[0]
        assert len(contourset.collections) == 10

    def test_resolution(self):
        ax = plot_density(self.points, 10, resolution=200)
        contourset = ax.collections[0]
        collections = contourset.collections
        assert len(collections) == 12

    def test_margin(self):
        ax = plot_density(self.points, 10, margin=0.3)
        contourset = ax.collections[0]
        collections = contourset.collections
        assert len(collections) == 12

    def test_kdepy(self):
        ax = plot_density(self.points, 10, kernel="gaussian")
        contourset = ax.collections[0]
        collections = contourset.collections
        assert len(collections) == 12
        np.testing.assert_array_equal(contourset.get_linewidths(), np.array([1.5] * 12))

    def test_levels(self):
        ax = plot_density(self.points, 10, levels=5)
        contourset = ax.collections[0]
        collections = contourset.collections
        assert len(collections) == 7

    def test_fill(self):
        ax = plot_density(self.points, 10, fill=True)
        contourset = ax.collections[0]
        assert contourset.get_edgecolor().shape == (0, 4)
        assert contourset.filled
        np.testing.assert_array_equal(
            contourset.get_facecolor()[0],
            np.array([0.279566, 0.067836, 0.391917, 1.0]),
        )

    def test_geopandas(self):
        geopandas = pytest.importorskip("geopandas")

        gs = geopandas.GeoSeries.from_xy(*self.points.T)
        ax = plot_density(gs, 10)
        contourset = ax.collections[0]
        collections = contourset.collections
        assert len(collections) == 12
        np.testing.assert_array_equal(contourset.get_linewidths(), np.array([1.5] * 12))

    def test_kwargs(self):
        ax = plot_density(
            self.points, 10, cmap="magma", linewidths=0.5, linestyles="-."
        )
        contourset = ax.collections[0]
        collections = contourset.collections
        assert len(collections) == 12
        np.testing.assert_array_equal(contourset.get_linewidths(), np.array([0.5] * 12))

        np.testing.assert_array_equal(
            contourset.get_edgecolor()[5],
            np.array([0.639216, 0.189921, 0.49415, 1.0]),
        )
