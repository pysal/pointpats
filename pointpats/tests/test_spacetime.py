import pytest
import numpy
import libpysal as lps
import geopandas as gpd
from pointpats import (SpaceTimeEvents, knox, mantel,
                       jacquez, modified_knox, Knox, KnoxLocal)
import scipy
from pytest import approx


class TestKnox:
    def setup_method(self):
        path = lps.examples.get_path("burkitt.shp")
        self.gdf = gpd.read_file(path)

    def test_knox(self):
        global_knox = Knox(self.gdf[['X', 'Y']],
                           self.gdf[['T']],
                           delta=20,
                           tau=5)
        assert global_knox.statistic_ == 13
        assert global_knox.p_poisson == 0.14624558197140414
        assert hasattr(global_knox, 'sim') == False
        numpy.testing.assert_array_equal(global_knox.observed,
                                         [[1.300e+01, 3.416e+03],
                                          [3.900e+01, 1.411e+04]])
        numpy.testing.assert_allclose(global_knox.expected,
                                      [[1.01438161e+01, 3.41885618e+03],
                                       [4.1856139e+01, 1.41071438e+04]],
                                      rtol=1e-5,atol=0)


        numpy.random.seed(12345)
        global_knox = Knox(self.gdf[['X', 'Y']],
                           self.gdf[['T']],
                           delta=20,
                           tau=5,
                           keep=True)
        assert global_knox.statistic_ == 13
        assert hasattr(global_knox, 'sim') == True
        assert global_knox.p_sim == 0.21


class TestKnoxLocal:
    def setup_method(self):
        path = lps.examples.get_path("burkitt.shp")
        self.gdf = gpd.read_file(path)

    def test_knox_local(self):
        numpy.random.seed(12345)
        local_knox = KnoxLocal(self.gdf[['X', 'Y']],
                               self.gdf[["T"]],
                               delta=20, tau=5, keep=True)
        assert local_knox.statistic_.shape == (188,)
        lres = local_knox
        gt0ids = numpy.where(lres.nsti>0)
        numpy.testing.assert_array_equal(gt0ids,
                                         [[25,  26,  30,  31,  35,  36,  41,
                                          42,  46,  47,  51,  52, 102, 103,
                                          116, 118, 122, 123, 137, 138, 139,
                                          140, 158, 159, 162, 163]])
        numpy.testing.assert_allclose(lres.p_hypergeom[gt0ids],
                                 [0.1348993 , 0.14220663, 0.07335085,
                                   0.08400282, 0.1494317 , 0.21524073,
                                   0.0175806 , 0.04599869, 0.17523687,
                                   0.18209188, 0.19111321, 0.16830444,
                                   0.13734428, 0.14703242, 0.06796364,
                                   0.03192559, 0.13734428, 0.17523687,
                                   0.12998154, 0.1933476 , 0.13244507,
                                   0.13244507, 0.12502644, 0.14703242,
                                   0.12502644, 0.12998154],
                                   rtol=1e-5,atol=0)
        numpy.testing.assert_array_equal(lres.p_sims[gt0ids],
                                         [0.3 , 0.33, 0.11, 0.17, 0.3 , 0.42,
                                          0.06, 0.06, 0.33, 0.34, 0.36, 0.38,
                                          0.3 , 0.29, 0.41, 0.19, 0.31, 0.39,
                                          0.18, 0.39, 0.48, 0.41, 0.22, 0.41,
                                          0.39, 0.32])



# old tests refactored to pytest

class TestSpaceTimeEvents:
    def setup_method(self):
        path = lps.examples.get_path("burkitt.shp")
        self.events = SpaceTimeEvents(path, "T")


    def test_space_time_events(self):
        assert self.events.n == 188


    def test_knox(self):
        result = knox(self.events.space, self.events.t,
                      delta=20, tau=5, permutations=1)
        assert result['stat'] == 13.0


    def test_mantel(self):
        result = mantel(self.events.space,
                         self.events.time,
                         1,
                         scon=0.0,
                         spow=1.0,
                         tcon=0.0,
                         tpow=1.0,
                        )

        assert result['stat'] == approx(0.014154, rel=1e-4)


    def test_jacquez(self):
        result = jacquez(self.events.space,
                         self.events.t,
                         k=3,
                         permutations=1)

        assert result['stat'] == 12


    def test_modified_knox(self):
        result = modified_knox(self.events.space,
                               self.events.t,
                               delta=20,
                               tau=5,
                               permutations=1)

        assert result['stat'] == approx(2.810160, rel=1e-4)

