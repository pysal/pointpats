"""
Distance backends for point pattern analysis.

This module defines an abstraction layer for distance computation
used throughout pointpats. By default, Euclidean distances are used,
but alternative backends (e.g., network-based distances) can be
implemented by subclassing DistanceBackend.
"""

import abc
from scipy.spatial.distance import cdist


class DistanceBackend(abc.ABC):
    """Abstract base class for distance computation."""

    @abc.abstractmethod
    def pairwise(self, points):
        """
        Compute pairwise distances between points.

        Parameters
        ----------
        points : array-like, shape (n, d)

        Returns
        -------
        distances : ndarray, shape (n, n)
        """
        raise NotImplementedError

    def within(self, points, r):
        """
        Boolean matrix of distances <= r.

        Parameters
        ----------
        points : array-like
        r : float
        """
        return self.pairwise(points) <= r


class EuclideanDistanceBackend(DistanceBackend):
    """Default Euclidean distance backend."""

    def pairwise(self, points):
        return cdist(points, points)
