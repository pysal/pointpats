import unittest
import numpy as np
import geopandas

from ..quadrat_statistics import *
from ..pointpattern import PointPattern

from libpysal.common import RTOL, ATOL


class TestQuadratStatistics(unittest.TestCase):
    def setUp(self):
        self.points = np.array(
            [
                [94.0, 93.0],
                [80.0, 95.0],
                [79.0, 90.0],
                [78.0, 92.0],
                [76.0, 92.0],
                [66.0, 93.0],
                [64.0, 90.0],
                [27.0, 70.0],
                [58.0, 88.0],
                [57.0, 92.0],
                [53.0, 92.0],
                [50.0, 90.0],
                [49.0, 90.0],
                [32.0, 90.0],
                [31.0, 87.0],
                [22.0, 87.0],
                [21.0, 87.0],
                [21.0, 86.0],
                [22.0, 81.0],
                [23.0, 83.0],
                [27.0, 85.0],
                [27.0, 84.0],
                [27.0, 83.0],
                [27.0, 82.0],
                [30.0, 84.0],
                [31.0, 84.0],
                [31.0, 84.0],
                [32.0, 83.0],
                [33.0, 81.0],
                [32.0, 79.0],
                [32.0, 76.0],
                [33.0, 77.0],
                [34.0, 86.0],
                [34.0, 84.0],
                [38.0, 82.0],
                [39.0, 81.0],
                [40.0, 80.0],
                [41.0, 83.0],
                [43.0, 75.0],
                [44.0, 81.0],
                [46.0, 81.0],
                [47.0, 82.0],
                [47.0, 81.0],
                [48.0, 80.0],
                [48.0, 81.0],
                [50.0, 85.0],
                [51.0, 84.0],
                [52.0, 83.0],
                [55.0, 85.0],
                [57.0, 88.0],
                [57.0, 81.0],
                [60.0, 87.0],
                [69.0, 80.0],
                [71.0, 82.0],
                [72.0, 81.0],
                [74.0, 82.0],
                [75.0, 81.0],
                [77.0, 88.0],
                [80.0, 88.0],
                [82.0, 77.0],
                [66.0, 62.0],
                [64.0, 71.0],
                [59.0, 63.0],
                [55.0, 64.0],
                [53.0, 68.0],
                [52.0, 59.0],
                [51.0, 61.0],
                [50.0, 75.0],
                [50.0, 74.0],
                [45.0, 61.0],
                [44.0, 60.0],
                [43.0, 59.0],
                [42.0, 61.0],
                [39.0, 71.0],
                [37.0, 67.0],
                [35.0, 70.0],
                [31.0, 68.0],
                [30.0, 71.0],
                [29.0, 61.0],
                [26.0, 69.0],
                [24.0, 68.0],
                [7.0, 52.0],
                [11.0, 53.0],
                [34.0, 50.0],
                [36.0, 47.0],
                [37.0, 45.0],
                [37.0, 56.0],
                [38.0, 55.0],
                [38.0, 50.0],
                [39.0, 52.0],
                [41.0, 52.0],
                [47.0, 49.0],
                [50.0, 57.0],
                [52.0, 56.0],
                [53.0, 55.0],
                [56.0, 57.0],
                [69.0, 52.0],
                [69.0, 50.0],
                [71.0, 51.0],
                [71.0, 51.0],
                [73.0, 48.0],
                [74.0, 48.0],
                [75.0, 46.0],
                [75.0, 46.0],
                [86.0, 51.0],
                [87.0, 51.0],
                [87.0, 52.0],
                [90.0, 52.0],
                [91.0, 51.0],
                [87.0, 42.0],
                [81.0, 39.0],
                [80.0, 43.0],
                [79.0, 37.0],
                [78.0, 38.0],
                [75.0, 44.0],
                [73.0, 41.0],
                [71.0, 44.0],
                [68.0, 29.0],
                [62.0, 33.0],
                [61.0, 35.0],
                [60.0, 34.0],
                [58.0, 36.0],
                [54.0, 30.0],
                [52.0, 38.0],
                [52.0, 36.0],
                [47.0, 37.0],
                [46.0, 36.0],
                [45.0, 33.0],
                [36.0, 32.0],
                [22.0, 39.0],
                [21.0, 38.0],
                [22.0, 35.0],
                [21.0, 36.0],
                [22.0, 30.0],
                [19.0, 29.0],
                [17.0, 40.0],
                [14.0, 41.0],
                [13.0, 36.0],
                [10.0, 34.0],
                [7.0, 37.0],
                [2.0, 39.0],
                [21.0, 16.0],
                [22.0, 14.0],
                [29.0, 17.0],
                [30.0, 25.0],
                [32.0, 26.0],
                [39.0, 28.0],
                [40.0, 26.0],
                [40.0, 26.0],
                [42.0, 25.0],
                [43.0, 24.0],
                [43.0, 16.0],
                [48.0, 16.0],
                [51.0, 25.0],
                [52.0, 26.0],
                [57.0, 27.0],
                [60.0, 22.0],
                [63.0, 24.0],
                [64.0, 23.0],
                [64.0, 27.0],
                [71.0, 25.0],
                [50.0, 10.0],
                [48.0, 12.0],
                [45.0, 14.0],
                [33.0, 8.0],
                [31.0, 7.0],
                [32.0, 6.0],
                [31.0, 8.0],
            ]
        )
        self.pp = self.points

    def test_QStatistic(self):
        q_r = QStatistic(self.pp, shape="rectangle", nx=3, ny=3)
        np.testing.assert_allclose(q_r.chi2, 33.1071428571, RTOL)
        np.testing.assert_allclose(q_r.chi2_pvalue, 5.89097854516e-05, ATOL)
        assert q_r.df == 8

        q_r = QStatistic(
            self.pp, shape="rectangle", rectangle_height=29.7, rectangle_width=30.7
        )
        np.testing.assert_allclose(q_r.chi2, 33.1071428571, RTOL)
        np.testing.assert_allclose(q_r.chi2_pvalue, 5.89097854516e-05, ATOL)
        assert q_r.df == 8

        q_r = QStatistic(self.pp, shape="hexagon", lh=10)
        np.testing.assert_allclose(q_r.chi2, 195.0, RTOL)
        np.testing.assert_allclose(q_r.chi2_pvalue, 6.3759506952e-22, RTOL)
        assert q_r.df == 41

    def test_RectangleM1(self):
        rm = RectangleM(self.pp, count_column=3, count_row=3)
        rm2 = RectangleM(self.pp, rectangle_height=29.7, rectangle_width=30.7)
        np.testing.assert_array_equal(
            list(rm.point_location_sta().values()), [12, 22, 4, 11, 26, 22, 22, 33, 16]
        )
        np.testing.assert_array_equal(
            list(rm2.point_location_sta().values()), [12, 22, 4, 11, 26, 22, 22, 33, 16]
        )

    def test_RectangleM2(self):
        hm = HexagonM(self.pp, lh=10)
        np.testing.assert_array_equal(
            list(hm.point_location_sta().values()),
            [
                0,
                2,
                4,
                5,
                0,
                0,
                0,
                0,
                9,
                6,
                10,
                7,
                3,
                0,
                2,
                2,
                3,
                7,
                4,
                13,
                1,
                1,
                1,
                4,
                11,
                3,
                0,
                4,
                0,
                5,
                15,
                15,
                3,
                10,
                0,
                0,
                0,
                9,
                0,
                7,
                1,
                1,
            ],
        )

    def test_geoseries(self):
        pp_array = np.array(self.points)
        pts = geopandas.GeoSeries.from_xy(x=pp_array[:, 0], y=pp_array[:, 1])
        q_r = QStatistic(pts, shape="rectangle", nx=3, ny=3)
        np.testing.assert_allclose(q_r.chi2, 33.1071428571, RTOL)
        np.testing.assert_allclose(q_r.chi2_pvalue, 5.89097854516e-05, ATOL)
        assert q_r.df == 8
