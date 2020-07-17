import unittest
import libpysal as lps
from pointpats import SpaceTimeEvents, knox, mantel, jacquez, modified_knox
import scipy


class SpaceTimeEvents_Tester(unittest.TestCase):
    def setUp(self):
        self.path = lps.examples.get_path("burkitt.shp")

    def test_SpaceTimeEvents(self):
        events = SpaceTimeEvents(self.path, "T")
        self.assertEquals(events.n, 188)
        self.assertEquals(list(events.space[0]), [300.0, 302.0])
        self.assertEquals(list(events.t[0]), [413])


class Knox_Tester(unittest.TestCase):
    def setUp(self):
        path = lps.examples.get_path("burkitt.shp")
        self.events = SpaceTimeEvents(path, "T")

    def test_knox(self):
        result = knox(self.events.space, self.events.t, delta=20, tau=5, permutations=1)
        self.assertEquals(result["stat"], 13.0)


class Mantel_Tester(unittest.TestCase):
    def setUp(self):
        path = lps.examples.get_path("burkitt.shp")
        self.events = SpaceTimeEvents(path, "T")

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
        self.assertAlmostEquals(result["stat"], 0.014154, 6)


class Jacquez_Tester(unittest.TestCase):
    def setUp(self):
        path = lps.examples.get_path("burkitt.shp")
        self.events = SpaceTimeEvents(path, "T")

    def test_jacquez(self):
        result = jacquez(self.events.space,
                self.events.t, k=3, permutations=1)
        self.assertEquals(result['stat'], 12)

class ModifiedKnox_Tester(unittest.TestCase):
    def setUp(self):
        path = lps.examples.get_path("burkitt.shp")
        self.events = SpaceTimeEvents(path, "T")

    def test_modified_knox(self):
        result = modified_knox(
            self.events.space, self.events.t, delta=20, tau=5, permutations=1
        )
        self.assertAlmostEquals(result["stat"], 2.810160, 6)


suite = unittest.TestSuite()
test_classes = [
    SpaceTimeEvents_Tester,
    Knox_Tester,
    Mantel_Tester,
    Jacquez_Tester,
    ModifiedKnox_Tester,
]
for i in test_classes:
    a = unittest.TestLoader().loadTestsFromTestCase(i)
    suite.addTest(a)

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite)
