import unittest
import numpy as np
import geopandas

from ..quadrat_statistics import *
from ..pointpattern import PointPattern

from libpysal.common import RTOL, ATOL


class TestQuadratStatistics(unittest.TestCase):
    def setUp(self):
        self.points = [
            [94., 93.], [80., 95.], [79., 90.], [78., 92.], [76., 92.], [66., 93.], [64., 90.], [27., 70.], [58., 88.],
            [57., 92.], [53., 92.], [50., 90.], [49., 90.], [32., 90.], [31., 87.], [22., 87.], [21., 87.], [21., 86.],
            [22., 81.], [23., 83.], [27., 85.], [27., 84.], [27., 83.], [27., 82.], [30., 84.], [31., 84.], [31., 84.],
            [32., 83.], [33., 81.], [32., 79.], [32., 76.], [33., 77.], [34., 86.], [34., 84.], [38., 82.], [39., 81.],
            [40., 80.], [41., 83.], [43., 75.], [44., 81.], [46., 81.], [47., 82.], [47., 81.], [48., 80.], [48., 81.],
            [50., 85.], [51., 84.], [52., 83.], [55., 85.], [57., 88.], [57., 81.], [60., 87.], [69., 80.], [71., 82.],
            [72., 81.], [74., 82.], [75., 81.], [77., 88.], [80., 88.], [82., 77.], [66., 62.], [64., 71.], [59., 63.],
            [55., 64.], [53., 68.], [52., 59.], [51., 61.], [50., 75.], [50., 74.], [45., 61.], [44., 60.], [43., 59.],
            [42., 61.], [39., 71.], [37., 67.], [35., 70.], [31., 68.], [30., 71.], [29., 61.], [26., 69.], [24., 68.],
            [7., 52.], [11., 53.], [34., 50.], [36., 47.], [37., 45.], [37., 56.], [38., 55.], [38., 50.], [39., 52.],
            [41., 52.], [47., 49.], [50., 57.], [52., 56.], [53., 55.], [56., 57.], [69., 52.], [69., 50.], [71., 51.],
            [71., 51.], [73., 48.], [74., 48.], [75., 46.], [75., 46.], [86., 51.], [87., 51.], [87., 52.], [90., 52.],
            [91., 51.], [87., 42.], [81., 39.], [80., 43.], [79., 37.], [78., 38.], [75., 44.], [73., 41.], [71., 44.],
            [68., 29.], [62., 33.], [61., 35.], [60., 34.], [58., 36.], [54., 30.], [52., 38.], [52., 36.], [47., 37.],
            [46., 36.], [45., 33.], [36., 32.], [22., 39.], [21., 38.], [22., 35.], [21., 36.], [22., 30.], [19., 29.],
            [17., 40.], [14., 41.], [13., 36.], [10., 34.], [7., 37.], [2., 39.], [21., 16.], [22., 14.], [29., 17.],
            [30., 25.], [32., 26.], [39., 28.], [40., 26.], [40., 26.], [42., 25.], [43., 24.], [43., 16.], [48., 16.],
            [51., 25.], [52., 26.], [57., 27.], [60., 22.], [63., 24.], [64., 23.], [64., 27.], [71., 25.], [50., 10.],
            [48., 12.], [45., 14.], [33., 8.], [31., 7.], [32., 6.], [31., 8.]
        ]
        self.pp = PointPattern(self.points)

    def test_QStatistic(self):
        q_r = QStatistic(self.pp, shape="rectangle", nx=3, ny=3)
        np.testing.assert_allclose(q_r.chi2, 33.1071428571, RTOL)
        np.testing.assert_allclose(q_r.chi2_pvalue, 5.89097854516e-05, ATOL)
        assert q_r.df == 8

        q_r = QStatistic(self.pp, shape="rectangle",
                         rectangle_height = 29.7, rectangle_width = 30.7)
        np.testing.assert_allclose(q_r.chi2, 33.1071428571, RTOL)
        np.testing.assert_allclose(q_r.chi2_pvalue, 5.89097854516e-05, ATOL)
        assert q_r.df == 8

        q_r = QStatistic(self.pp, shape="hexagon", lh=10)
        np.testing.assert_allclose(q_r.chi2, 195.0, RTOL)
        np.testing.assert_allclose(q_r.chi2_pvalue, 6.3759506952e-22, RTOL)
        assert q_r.df == 41

    def test_RectangleM1(self):
        rm = RectangleM(self.pp, count_column = 3, count_row = 3)
        rm2 = RectangleM(self.pp, rectangle_height = 29.7, rectangle_width = 30.7)
        np.testing.assert_array_equal(list(rm.point_location_sta().values()),
                                      [12, 22, 4, 11, 26, 22, 22, 33, 16])
        np.testing.assert_array_equal(list(rm2.point_location_sta().values()),
                                      [12, 22, 4, 11, 26, 22, 22, 33, 16])
    def test_RectangleM2(self):
        hm = HexagonM(self.pp, lh = 10)
        np.testing.assert_array_equal(list(hm.point_location_sta().values()),
                                      [0, 2, 4, 5, 0, 0, 0, 0, 9, 6, 10, 7, 3, 0, 2, 2, 3, 7, 4,
                                       13, 1, 1, 1, 4, 11, 3, 0, 4, 0, 5, 15, 15, 3, 10, 0, 0,
                                       0, 9, 0, 7, 1, 1])
    def test_geoseries(self):
        pp_array = np.array(self.points)
        pts = geopandas.GeoSeries.from_xy(x=pp_array[:, 0], y=pp_array[:, 1])
        q_r = QStatistic(pts, shape="rectangle", nx=3, ny=3)
        np.testing.assert_allclose(q_r.chi2, 33.1071428571, RTOL)
        np.testing.assert_allclose(q_r.chi2_pvalue, 5.89097854516e-05, ATOL)
        assert q_r.df == 8