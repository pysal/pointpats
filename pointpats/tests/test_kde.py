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
        qm = plot_density(self.points, 10)
        collections = list(qm.collections)
        assert len(collections) == 12
        for col in collections:
            assert col.get_linewidths() == np.array(1.5)
        np.testing.assert_array_equal(
            collections[5].get_edgecolor(),
            np.array([[0.143343, 0.522773, 0.556295, 1.0]]),
        )

    def test_bandwidth(self):
        qm = plot_density(self.points, 1)
        collections = list(qm.collections)
        assert len(collections) == 10

    def test_resolution(self):
        qm = plot_density(self.points, 10, resolution=200)
        collections = list(qm.collections)
        assert len(collections) == 12

    def test_margin(self):
        qm = plot_density(self.points, 10, margin=.3)
        collections = list(qm.collections)
        assert len(collections) == 12

    def test_kdepy(self):
        qm = plot_density(self.points, 10, kernel="gaussian")
        collections = list(qm.collections)
        assert len(collections) == 12
        for col in collections:
            assert col.get_linewidths() == np.array(1.5)

    def test_levels(self):
        qm = plot_density(self.points, 10, levels=5)
        collections = list(qm.collections)
        assert len(collections) == 7

    def test_fill(self):
        qm = plot_density(self.points, 10, fill=True)
        collections = list(qm.collections)
        assert collections[0].get_edgecolor().shape == (0, 4)
        np.testing.assert_array_equal(
            collections[0].get_facecolor(),
            np.array([[0.279566, 0.067836, 0.391917, 1.0]]),
        )

    def test_geopandas(self):
        geopandas = pytest.importorskip("geopandas")

        gs = geopandas.GeoSeries.from_xy(*self.points.T)
        qm = plot_density(gs, 10)
        collections = list(qm.collections)
        assert len(collections) == 12
        for col in collections:
            assert col.get_linewidths() == np.array(1.5)

    def test_kwargs(self):
        qm = plot_density(
            self.points, 10, cmap="magma", linewidths=0.5, linestyles="-."
        )
        collections = list(qm.collections)
        assert len(collections) == 12
        for col in collections:
            assert col.get_linewidths() == np.array(0.5)
        np.testing.assert_array_equal(
            collections[5].get_edgecolor(),
            np.array([[0.639216, 0.189921, 0.49415, 1.0]]),
        )
