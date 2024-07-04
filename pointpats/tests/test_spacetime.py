from warnings import warn

import geopandas as gpd
import libpysal as lps
import numpy
from pytest import approx
import pytest
import matplotlib.pyplot as plt

from pointpats import (
    Knox,
    KnoxLocal,
    SpaceTimeEvents,
    jacquez,
    knox,
    mantel,
    modified_knox,
)


class TestKnox:
    def setup_method(self):
        path = lps.examples.get_path("burkitt.shp")
        self.gdf = gpd.read_file(path)

    def test_knox(self):
        global_knox = Knox(self.gdf[["X", "Y"]], self.gdf[["T"]], delta=20, tau=5)
        assert global_knox.statistic_ == 13
        assert global_knox.p_poisson == 0.14624558197140414
        assert hasattr(global_knox, "sim") == False
        numpy.testing.assert_array_equal(
            global_knox.observed, [[1.300e01, 3.416e03], [3.900e01, 1.411e04]]
        )
        numpy.testing.assert_allclose(
            global_knox.expected,
            [[1.01438161e01, 3.41885618e03], [4.1856139e01, 1.41071438e04]],
            rtol=1e-5,
            atol=0,
        )

        numpy.random.seed(12345)
        global_knox = Knox(
            self.gdf[["X", "Y"]], self.gdf[["T"]], delta=20, tau=5, keep=True
        )
        assert global_knox.statistic_ == 13
        assert hasattr(global_knox, "sim") == True
        assert global_knox.p_sim == 0.21

    def test_knox_from_gdf(self):
        gdf = self.gdf.copy()
        # not technically the correct CRS...
        gdf.crs = 21096
        global_knox = Knox.from_dataframe(gdf, time_col="T", delta=20, tau=5)
        assert global_knox.statistic_ == 13
        assert global_knox.p_poisson == 0.14624558197140414
        assert hasattr(global_knox, "sim") == False
        numpy.testing.assert_array_equal(
            global_knox.observed, [[1.300e01, 3.416e03], [3.900e01, 1.411e04]]
        )
        numpy.testing.assert_allclose(
            global_knox.expected,
            [[1.01438161e01, 3.41885618e03], [4.1856139e01, 1.41071438e04]],
            rtol=1e-5,
            atol=0,
        )

        # no CRS should raise a warning
        global_knox = Knox.from_dataframe(self.gdf, time_col="T", delta=20, tau=5)
        numpy.testing.assert_allclose(
            global_knox.expected,
            [[1.01438161e01, 3.41885618e03], [4.1856139e01, 1.41071438e04]],
            rtol=1e-5,
            atol=0,
        )

        # unprojected coords
        try:
            gdf.crs = 4326
            global_knox = Knox.from_dataframe(gdf, time_col="T", delta=20, tau=5)
        except ValueError:
            warn("successfully caught crs error")
            pass

        # non-numeric type for time
        try:
            gdf["T"] = gdf["T"].astype("O")
            global_knox = Knox.from_dataframe(gdf, time_col="T", delta=20, tau=5)
        except ValueError:
            warn("successfully caught dtype error")
            pass

        numpy.random.seed(12345)
        global_knox = Knox(
            self.gdf[["X", "Y"]], self.gdf[["T"]], delta=20, tau=5, keep=True
        )
        assert global_knox.statistic_ == 13
        assert hasattr(global_knox, "sim") == True
        assert global_knox.p_sim == 0.21


class TestKnoxLocal:
    def setup_method(self):
        path = lps.examples.get_path("burkitt.shp")
        self.gdf = gpd.read_file(path)

    def test_knox_local(self):
        numpy.random.seed(12345)
        local_knox = KnoxLocal(
            self.gdf[["X", "Y"]].values,
            self.gdf[["T"]].values,
            delta=20,
            tau=5,
            keep=True,
        )
        assert local_knox.statistic_.shape == (188,)
        lres = local_knox
        gt0ids = numpy.where(lres.nsti > 0)
        numpy.testing.assert_array_equal(
            gt0ids,
            [
                [
                    25,
                    26,
                    30,
                    31,
                    35,
                    36,
                    41,
                    42,
                    46,
                    47,
                    51,
                    52,
                    102,
                    103,
                    116,
                    118,
                    122,
                    123,
                    137,
                    138,
                    139,
                    140,
                    158,
                    159,
                    162,
                    163,
                ]
            ],
        )
        numpy.testing.assert_allclose(
            lres.p_hypergeom[gt0ids],
            [
                0.1348993,
                0.14220663,
                0.07335085,
                0.08400282,
                0.1494317,
                0.21524073,
                0.0175806,
                0.04599869,
                0.17523687,
                0.18209188,
                0.19111321,
                0.16830444,
                0.13734428,
                0.14703242,
                0.06796364,
                0.03192559,
                0.13734428,
                0.17523687,
                0.12998154,
                0.1933476,
                0.13244507,
                0.13244507,
                0.12502644,
                0.14703242,
                0.12502644,
                0.12998154,
            ],
            rtol=1e-5,
            atol=0,
        )
        numpy.testing.assert_array_equal(
            lres.p_sims[gt0ids],
            [
                0.3,
                0.33,
                0.11,
                0.17,
                0.3,
                0.42,
                0.06,
                0.06,
                0.33,
                0.34,
                0.36,
                0.38,
                0.3,
                0.29,
                0.41,
                0.19,
                0.31,
                0.39,
                0.18,
                0.39,
                0.48,
                0.41,
                0.22,
                0.41,
                0.39,
                0.32,
            ],
        )

    def test_knox_local_from_gdf(self):
        gdf = self.gdf
        gdf.crs = 21096
        numpy.random.seed(12345)
        local_knox = KnoxLocal.from_dataframe(
            gdf, time_col="T", delta=20, tau=5, keep=True
        )
        assert local_knox.statistic_.shape == (188,)
        lres = local_knox
        gt0ids = numpy.where(lres.nsti > 0)
        numpy.testing.assert_array_equal(
            gt0ids,
            [
                [
                    25,
                    26,
                    30,
                    31,
                    35,
                    36,
                    41,
                    42,
                    46,
                    47,
                    51,
                    52,
                    102,
                    103,
                    116,
                    118,
                    122,
                    123,
                    137,
                    138,
                    139,
                    140,
                    158,
                    159,
                    162,
                    163,
                ]
            ],
        )
        numpy.testing.assert_allclose(
            lres.p_hypergeom[gt0ids],
            [
                0.1348993,
                0.14220663,
                0.07335085,
                0.08400282,
                0.1494317,
                0.21524073,
                0.0175806,
                0.04599869,
                0.17523687,
                0.18209188,
                0.19111321,
                0.16830444,
                0.13734428,
                0.14703242,
                0.06796364,
                0.03192559,
                0.13734428,
                0.17523687,
                0.12998154,
                0.1933476,
                0.13244507,
                0.13244507,
                0.12502644,
                0.14703242,
                0.12502644,
                0.12998154,
            ],
            rtol=1e-5,
            atol=0,
        )
        numpy.testing.assert_array_equal(
            lres.p_sims[gt0ids],
            [
                0.3,
                0.33,
                0.11,
                0.17,
                0.3,
                0.42,
                0.06,
                0.06,
                0.33,
                0.34,
                0.36,
                0.38,
                0.3,
                0.29,
                0.41,
                0.19,
                0.31,
                0.39,
                0.18,
                0.39,
                0.48,
                0.41,
                0.22,
                0.41,
                0.39,
                0.32,
            ],
        )

    def test_explore(self):
        gdf = self.gdf.copy()
        gdf.crs = 21096
        numpy.random.seed(12345)
        m = KnoxLocal.from_dataframe(
            gdf, time_col="T", delta=20, tau=5, keep=True
        ).explore()
        numpy.testing.assert_array_equal(
            m.get_bounds(),
            [
                [-0.0005034046601185694, 28.514258651567],
                [0.0008675512091255166, 28.514975377735905],
            ],
        )
        # old folium returns 5, new folium returns 3
        assert len(m.to_dict()["children"]) >= 3


    def test_hotspots_without_neighbors(self):
        gdf = self.gdf.copy()
        gdf = gdf.set_crs(21096)
        numpy.random.seed(1)
        knox = KnoxLocal.from_dataframe(
            gdf, time_col="T", delta=20, tau=5, 
        ).hotspots(keep_neighbors=False, inference='analytic')
        assert knox.shape == (3,7)

    def test_hotspots_with_neighbors(self):
        gdf = self.gdf.copy()
        gdf = gdf.set_crs(21096)
        knox = KnoxLocal.from_dataframe(
            gdf, time_col="T", delta=20, tau=5, 
        ).hotspots(keep_neighbors=True, inference='analytic')
        assert knox.shape == (4,7)

    @pytest.mark.mpl_image_compare
    def test_plot(self):
        gdf = self.gdf.copy()
        gdf.crs = 21096
        fig, ax2 = plt.subplots(figsize=(30,18))
        lk = KnoxLocal.from_dataframe(
            gdf, time_col="T", delta=20, tau=5, keep=True)
        lk.plot(inference='analytic', ax=ax2)
        assert fig



# old tests refactored to pytest


class TestSpaceTimeEvents:
    def setup_method(self):
        path = lps.examples.get_path("burkitt.shp")
        self.events = SpaceTimeEvents(path, "T")

    def test_space_time_events(self):
        assert self.events.n == 188

    def test_knox(self):
        result = knox(self.events.space, self.events.t, delta=20, tau=5, permutations=1)
        assert result["stat"] == 13.0

    def test_mantel(self):
        result = mantel(
            self.events.space,
            self.events.time,
            1,
            scon=0.0,
            spow=1.0,
            tcon=0.0,
            tpow=1.0,
        )

        assert result["stat"] == approx(0.014154, rel=1e-4)

    def test_jacquez(self):
        result = jacquez(self.events.space, self.events.t, k=3, permutations=1)

        assert result["stat"] == 12

    def test_modified_knox(self):
        result = modified_knox(
            self.events.space, self.events.t, delta=20, tau=5, permutations=1
        )

        assert result["stat"] == approx(2.810160, rel=1e-4)
