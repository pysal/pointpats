"""
Planar Point Pattern Class
"""

import numpy as np
import sys
from libpysal.cg import KDTree
from .centrography import hull
from .window import as_window, poly_from_bbox
from .util import cached_property
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

# NEW IMPORT (SAFE)
from .distance import EuclideanDistanceBackend

__author__ = "Serge Rey sjsrey@gmail.com"
__all__ = ["PointPattern"]

if sys.version_info[0] > 2:
    xrange = range


class PointPattern(object):
    """
    Planar Point Pattern Class 2-D.
    """

    def __init__(
        self,
        points,
        window=None,
        names=None,
        coord_names=None,
        distance_backend=None,  # NEW (OPTIONAL)
    ):
        # first two series in df are x, y unless coord_names and names are specified

        self.df = pd.DataFrame(points)
        n, p = self.df.shape
        self._n_marks = p - 2

        if coord_names is None:
            if names is not None:
                coord_names = names[:2]
            else:
                coord_names = ["x", "y"]

        if names is None:
            col_names = coord_names.copy()
            if p > 2:
                for m in range(2, p):
                    col_names.append("mark_{}".format(m - 2))
            coord_names = coord_names[:2]
        else:
            col_names = names

        self.coord_names = coord_names
        self._x, self._y = coord_names
        self.df.columns = col_names
        self.points = self.df.loc[:, [self._x, self._y]]
        self._n, self._p = self.points.shape

        if window is None:
            self.set_window(as_window(poly_from_bbox(self.mbb)))
        else:
            self.set_window(window)

        #  ADD DISTANCE BACKEND *AFTER* ORIGINAL INITIALIZATION
        if distance_backend is None:
            self.distance_backend = EuclideanDistanceBackend()
        else:
            self.distance_backend = distance_backend

        self._facade()

    def __len__(self):
        return len(self.df)

    def __contains__(self, n):
        name = self.df.columns.values.tolist()
        return ((self.df[name[0]] == n[0]) & (self.df[name[1]] == n[1])).any()

    def set_window(self, window):
        try:
            self._window = window
        except Exception:
            print("not a valid Window object")

    def get_window(self):
        if not hasattr(self, "_window") or self._window is None:
            self.set_window(as_window(poly_from_bbox(self.mbb)))
        return self._window

    window = property(get_window, set_window)

    def summary(self):
        print("Point Pattern")
        print("{} points".format(self.n))
        print("Bounding rectangle [({},{}), ({},{})]".format(*self.mbb))
        print("Area of window: {}".format(self.window.area))
        print("Intensity estimate for window: {}".format(self.lambda_window))
        print(self.head())

    def add_marks(self, marks, mark_names=None):
        if mark_names is None:
            nm = range(len(marks))
            mark_names = ["mark_{}".format(self._n_marks + 1 + j) for j in nm]
        for name, mark in zip(mark_names, marks):
            self.df[name] = mark
            self._n_marks += 1

    def plot(self, window=False, title="Point Pattern", hull=False, get_ax=False):
        fig, ax = plt.subplots()
        plt.plot(self.df[self._x], self.df[self._y], ".")
        plt.title(title)

        if window:
            patches = []
            for part in self.window.parts:
                p = Polygon(np.asarray(part))
                patches.append(p)
            ax.add_collection(
                PatchCollection(patches, facecolor="none", edgecolor="k", alpha=0.3)
            )

        if hull:
            patches = []
            p = Polygon(self.hull)
            patches.append(p)
            ax.add_collection(
                PatchCollection(patches, facecolor="none", edgecolor="k", alpha=0.3)
            )

        ax.set_aspect("equal")
        if get_ax:
            return ax

    def _mbb(self):
        mins = self.points.min(axis=0)
        maxs = self.points.max(axis=0)
        return np.hstack((mins, maxs))

    mbb = cached_property(_mbb)

    def _mbb_area(self):
        return np.prod(self.mbb[[2, 3]] - self.mbb[[0, 1]])

    mbb_area = cached_property(_mbb_area)

    def _n(self):
        return self.points.shape[0]

    n = cached_property(_n)

    def _rot(self):
        w, s, e, n = self.mbb
        return 0.25 * min(e - w, n - s)

    rot = cached_property(_rot)

    def _lambda_mbb(self):
        return self.n / self.mbb_area

    lambda_mbb = cached_property(_lambda_mbb)

    def _hull(self):
        return hull(self.points)

    hull = cached_property(_hull)

    def _lambda_window(self):
        return self.n / self.window.area

    lambda_window = cached_property(_lambda_window)

    def _hull_area(self):
        h = self.hull
        if not np.all(h[0] == h[-1]):
            h = np.vstack((h, h[0]))
        s = h[:-1, 0] * h[1:, 1] - h[1:, 0] * h[:-1, 1]
        return s.sum() / 2.0

    hull_area = cached_property(_hull_area)

    def _lambda_hull(self):
        return self.n / self.hull_area

    lambda_hull = cached_property(_lambda_hull)

    def _build_tree(self):
        return KDTree(self.points)

    tree = cached_property(_build_tree)

    def knn(self, k=1):
        if k < 1:
            raise ValueError("k must be at least 1")
        nn = self.tree.query(self.tree.data, k=k + 1)
        return nn[1][:, 1:], nn[0][:, 1:]

    def _nn_sum(self):
        ids, nnd = self.knn(1)
        return nnd

    nnd = cached_property(_nn_sum)

    def _min_nnd(self):
        return self.nnd.min()

    min_nnd = cached_property(_min_nnd)

    def _max_nnd(self):
        return self.nnd.max()

    max_nnd = cached_property(_max_nnd)

    def _mean_nnd(self):
        return self.nnd.mean()

    mean_nnd = cached_property(_mean_nnd)

    def find_pairs(self, r):
        return self.tree.query_pairs(r)

    def knn_other(self, other, k=1):
        if k < 1:
            raise ValueError("k must be at least 1")
        try:
            nn = self.tree.query(np.asarray(other.points), k=k)
        except Exception:
            nn = self.tree.query(np.asarray(other), k=k)
        return nn[1], nn[0]

    def explode(self, mark):
        uv = np.unique(self.df[mark])
        pps = [self.df[self.df[mark] == v] for v in uv]
        names = self.df.columns.values.tolist()
        cnames = self.coord_names
        return [PointPattern(pp, names=names, coord_names=cnames) for pp in pps]

    def unique(self):
        names = self.df.columns.values.tolist()
        coord_names = self.coord_names
        window = self.set_window
        unique_df = self.df.drop_duplicates()
        return PointPattern(
            unique_df, names=names, coord_names=coord_names, window=window
        )

    def superimpose(self, point_pattern):
        names_pp1 = self.df.columns.values.tolist()
        cnames_pp1 = self.coord_names
        names_pp2 = point_pattern.df.columns.values.tolist()
        cnames_pp2 = point_pattern.coord_names

        if names_pp1 != names_pp2 or cnames_pp1 != cnames_pp2:
            raise TypeError(
                "Both point patterns should have similar attributes and spatial coordinates"
            )

        pp = pd.concat((self.df, point_pattern.df))
        pp = pp.drop_duplicates()
        return PointPattern(pp, names=names_pp1, coord_names=cnames_pp1)

    def flip_coordinates(self):
        self._x, self._y = self._y, self._x

    def _facade(self):
        self.head = self.df.head
        self.tail = self.df.tail