"""
Quadrat statistics for planar point patterns

- First argument everywhere is a NumPy ndarray of point coordinates (n, 2).
- MBB is computed internally from the array.
- Optional `window` (shapely geometry). If omitted, uses the MBB rectangle.
- Optional `rng` for reproducible simulation-based inference.
- Rectangle and hexagon grids supported.
- Plotting is self-contained (matplotlib only): scatters points, draws window, draws grid,
  annotates counts, and optionally annotates per-cell chi-square contributions.

"""

__author__ = "Serge Rey, Wei Kang, Hu Shao (refactor by ChatGPT)"
__all__ = ["RectangleM", "HexagonM", "QStatistic"]

import math
from typing import Optional

import numpy as np
import scipy.stats
from shapely.geometry import Polygon, MultiPolygon, box
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from .random import poisson


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _as_points_array(points) -> np.ndarray:
    """
    Normalize input into an (n, 2) float ndarray.

    Accepts:
      - array-like shaped (n, 2)
      - GeoPandas GeoSeries / GeometryArray of Points (and optionally MultiPoints)

    Returns:
      - np.ndarray of shape (n, 2)

    Raises:
      - ValueError for empty inputs or non-2D coordinate inputs
      - TypeError for unsupported geometry types
    """
    # ---- GeoPandas / Shapely path (only if those objects are present) ----
    try:
        import geopandas as gpd  # optional dependency
        from shapely.geometry import Point, MultiPoint
    except Exception:
        gpd = None
        Point = MultiPoint = None

    if gpd is not None and isinstance(
        points, (gpd.GeoSeries, getattr(gpd.array, "GeometryArray", ()))
    ):
        # Convert GeoSeries/GeometryArray to a flat list of coordinate pairs
        geoms = list(points)

        if len(geoms) == 0:
            raise ValueError("points is empty; cannot compute mbb.")

        coords = []
        for geom in geoms:
            if geom is None or (hasattr(geom, "is_empty") and geom.is_empty):
                continue

            if Point is not None and isinstance(geom, Point):
                coords.append((geom.x, geom.y))
            elif MultiPoint is not None and isinstance(geom, MultiPoint):
                coords.extend([(p.x, p.y) for p in geom.geoms])
            else:
                raise TypeError(
                    "GeoSeries must contain Point geometries (optionally MultiPoint). "
                    f"Got geometry type: {getattr(geom, 'geom_type', type(geom).__name__)}"
                )

        if len(coords) == 0:
            raise ValueError(
                "points has no non-empty Point geometries; cannot compute mbb."
            )

        pts = np.asarray(coords, dtype=float)
        # guaranteed (n,2) here
        return pts

    # ---- Generic array-like path ----
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be a (n, 2) array-like. Got shape {pts.shape}.")
    if pts.size == 0:
        raise ValueError("points is empty; cannot compute mbb.")
    return pts


def _compute_mbb(points: np.ndarray) -> np.ndarray:
    x_min = float(np.min(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    x_max = float(np.max(points[:, 0]))
    y_max = float(np.max(points[:, 1]))
    return np.array([x_min, y_min, x_max, y_max], dtype=float)


def _ensure_window(window, mbb):
    if window is None:
        return box(mbb[0], mbb[1], mbb[2], mbb[3])
    return window


def _coerce_rng(rng: Optional[object]):
    """
    Accepts:
      - None -> creates a fresh default_rng()
      - int  -> seeded default_rng(int)
      - np.random.Generator -> returned as-is
      - np.random.RandomState -> wrapped via default_rng(RandomState)
    """
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.RandomState):
        return np.random.default_rng(rng)
    raise TypeError(
        "rng must be None, an int seed, numpy.random.Generator, or numpy.random.RandomState."
    )


def _window_to_paths(window_geom):
    """Return list of (x, y) arrays for plotting boundary(ies)."""
    paths = []
    if window_geom is None:
        return paths

    if isinstance(window_geom, Polygon):
        x, y = window_geom.exterior.xy
        paths.append((np.asarray(x), np.asarray(y)))
        for ring in window_geom.interiors:
            x, y = ring.xy
            paths.append((np.asarray(x), np.asarray(y)))
        return paths

    if isinstance(window_geom, MultiPolygon):
        for poly in window_geom.geoms:
            paths.extend(_window_to_paths(poly))
        return paths

    if hasattr(window_geom, "geoms"):
        for g in window_geom.geoms:
            paths.extend(_window_to_paths(g))
    return paths


def _scatter_points(ax, points):
    ax.scatter(points[:, 0], points[:, 1], s=10)


def _draw_window(ax, window_geom):
    for x, y in _window_to_paths(window_geom):
        ax.plot(x, y, lw=1.5)


def _cell_center_from_bounds(x0, y0, x1, y1):
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


# -----------------------------------------------------------------------------
# Rectangle grid
# -----------------------------------------------------------------------------
class RectangleM:
    """
    Rectangle grid structure for quadrat-based method.
    """

    def __init__(
        self,
        points,
        count_column=3,
        count_row=3,
        rectangle_width=0,
        rectangle_height=0,
        window=None,
    ):
        self.points = _as_points_array(points)
        self.mbb = _compute_mbb(self.points)
        self.window = _ensure_window(window, self.mbb)

        x_range = self.mbb[2] - self.mbb[0]
        y_range = self.mbb[3] - self.mbb[1]

        if rectangle_width and rectangle_height:
            self.rectangle_width = float(rectangle_width)
            self.rectangle_height = float(rectangle_height)
            self.count_column = int(math.ceil(x_range / self.rectangle_width))
            self.count_row = int(math.ceil(y_range / self.rectangle_height))
        else:
            self.count_column = int(count_column)
            self.count_row = int(count_row)
            self.rectangle_width = x_range / float(self.count_column)
            self.rectangle_height = y_range / float(self.count_row)

        self.num = self.count_column * self.count_row

    def point_location_sta(self):
        dict_id_count = {
            j + i * self.count_column: 0
            for i in range(self.count_row)
            for j in range(self.count_column)
        }

        x_min, y_min = self.mbb[0], self.mbb[1]
        for point in self.points:
            index_x = (point[0] - x_min) // self.rectangle_width
            index_y = (point[1] - y_min) // self.rectangle_height

            if index_x == self.count_column:
                index_x -= 1
            if index_y == self.count_row:
                index_y -= 1

            cell_id = int(index_y) * self.count_column + int(index_x)
            dict_id_count[cell_id] += 1

        return dict_id_count

    def _rect_cell_bounds(self, ix, iy):
        x0 = self.mbb[0] + ix * self.rectangle_width
        y0 = self.mbb[1] + iy * self.rectangle_height
        x1 = x0 + self.rectangle_width
        y1 = y0 + self.rectangle_height
        return x0, y0, x1, y1

    def _rect_cell_polygon(self, ix, iy):
        x0, y0, x1, y1 = self._rect_cell_bounds(ix, iy)
        return box(x0, y0, x1, y1)

    def _iter_cells(self):
        for iy in range(self.count_row):
            for ix in range(self.count_column):
                cell_id = ix + iy * self.count_column
                poly = self._rect_cell_polygon(ix, iy)
                x0, y0, x1, y1 = self._rect_cell_bounds(ix, iy)
                cx, cy = _cell_center_from_bounds(x0, y0, x1, y1)
                yield cell_id, poly, (cx, cy)

    def plot(
        self,
        title="Quadrat Count",
        show="counts",
        chi2_contrib=None,
    ):
        fig, ax = plt.subplots()
        ax.set_title(title)

        _scatter_points(ax, self.points)
        _draw_window(ax, self.window)

        dict_id_count = self.point_location_sta()

        patches = []
        ann_xy = []
        cell_ids = []

        for cell_id, poly, (cx, cy) in self._iter_cells():
            patches.append(MplPolygon(np.asarray(poly.exterior.coords), closed=True))
            ann_xy.append((cx, cy))
            cell_ids.append(cell_id)

        pc = PatchCollection(patches, linewidths=1.0, edgecolor="red", facecolor="none")
        ax.add_collection(pc)

        if show == "counts":
            for (cx, cy), cell_id in zip(ann_xy, cell_ids):
                ax.text(
                    cx, cy, str(dict_id_count.get(cell_id, 0)), ha="center", va="center"
                )

        elif show == "chi2":
            if chi2_contrib is None:
                raise ValueError('chi2_contrib must be provided when show="chi2".')
            chi2_contrib = np.asarray(chi2_contrib, dtype=float)
            if chi2_contrib.shape[0] != len(cell_ids):
                raise ValueError(
                    "chi2_contrib length must match number of plotted cells "
                    f"({len(cell_ids)}). Got {chi2_contrib.shape[0]}."
                )
            pc.set_array(chi2_contrib)
            pc.set_alpha(0.6)
            pc.set_facecolor(None)
            plt.colorbar(pc, ax=ax, label="Chi-square contribution")

            for (cx, cy), v in zip(ann_xy, chi2_contrib):
                ax.text(cx, cy, f"{v:.2f}", ha="center", va="center")

        else:
            raise ValueError('show must be "counts" or "chi2".')

        ax.set_aspect("equal", adjustable="box")
        return ax, cell_ids


# -----------------------------------------------------------------------------
# Hexagon grid
# -----------------------------------------------------------------------------
class HexagonM:
    """
    Hexagon grid structure for quadrat-based method.
    """

    def __init__(self, points, lh, window=None):
        self.points = _as_points_array(points)
        self.h_length = float(lh)
        self.mbb = _compute_mbb(self.points)
        self.window = _ensure_window(window, self.mbb)

        range_x = self.mbb[2] - self.mbb[0]
        range_y = self.mbb[3] - self.mbb[1]

        self.count_column = 1
        if self.h_length / 2.0 < range_x:
            temp = math.ceil((range_x - self.h_length / 2.0) / (1.5 * self.h_length))
            self.count_column += int(temp)

        self.semi_height = self.h_length * math.cos(math.pi / 6.0)
        self.count_row_even = 1
        if self.semi_height < range_y:
            temp = math.ceil((range_y - self.semi_height) / (2.0 * self.semi_height))
            self.count_row_even += int(temp)

        self.count_row_odd = int(math.ceil(range_y / (2.0 * self.semi_height)))

        self.num = self.count_row_odd * (
            (self.count_column // 2) + (self.count_column % 2)
        ) + self.count_row_even * (self.count_column // 2)

    def point_location_sta(self):
        semi_cell_length = self.h_length / 2.0
        dict_id_count = {}

        for i in range(self.count_row_even):
            for j in range(self.count_column):
                if (
                    self.count_row_even != self.count_row_odd
                    and i == self.count_row_even - 1
                    and j % 2 == 1
                ):
                    continue
                dict_id_count[j + i * self.count_column] = 0

        x_min = self.mbb[0]
        y_min = self.mbb[1]

        for point in self.points:
            intercept_degree_x = (point[0] - x_min) // semi_cell_length

            possible_y_index_even = int(
                (point[1] + self.semi_height - y_min) / (2.0 * self.semi_height)
            )
            possible_y_index_odd = int((point[1] - y_min) / (2.0 * self.semi_height))

            if intercept_degree_x % 3 != 1:
                center_index_x = int((intercept_degree_x + 1) // 3)
                center_index_y = possible_y_index_odd
                if center_index_x % 2 == 0:
                    center_index_y = possible_y_index_even
                dict_id_count[center_index_x + center_index_y * self.count_column] += 1
            else:
                center_index_x = int(intercept_degree_x // 3)
                center_x = center_index_x * semi_cell_length * 3.0 + x_min

                center_index_y = possible_y_index_odd
                center_y = (center_index_y * 2.0 + 1.0) * self.semi_height + y_min

                if center_index_x % 2 == 0:
                    center_index_y = possible_y_index_even
                    center_y = center_index_y * 2.0 * self.semi_height + y_min

                if point[1] > center_y:
                    x0, y0 = center_x + self.h_length, center_y
                    x1, y1 = center_x + semi_cell_length, center_y + self.semi_height
                    indicator = -(
                        point[1]
                        - (
                            (y0 - y1) / (x0 - x1) * point[0]
                            + (x0 * y1 - x1 * y0) / (x0 - x1)
                        )
                    )
                else:
                    x0, y0 = center_x + semi_cell_length, center_y - self.semi_height
                    x1, y1 = center_x + self.h_length, center_y
                    indicator = point[1] - (
                        (y0 - y1) / (x0 - x1) * point[0]
                        + (x0 * y1 - x1 * y0) / (x0 - x1)
                    )

                if indicator <= 0:
                    center_index_x += 1
                    center_index_y = possible_y_index_odd
                    if center_index_x % 2 == 0:
                        center_index_y = possible_y_index_even

                dict_id_count[center_index_x + center_index_y * self.count_column] += 1

        return dict_id_count

    def _hex_vertices(self, cx, cy):
        sh = self.semi_height
        lh = self.h_length
        return np.array(
            [
                [cx + lh, cy],
                [cx + lh / 2.0, cy + sh],
                [cx - lh / 2.0, cy + sh],
                [cx - lh, cy],
                [cx - lh / 2.0, cy - sh],
                [cx + lh / 2.0, cy - sh],
                [cx + lh, cy],
            ],
            dtype=float,
        )

    def _iter_cells(self):
        dict_id_count = self.point_location_sta()

        x_min = self.mbb[0]
        y_min = self.mbb[1]

        for cell_id in dict_id_count.keys():
            ix = cell_id % self.count_column
            iy = cell_id // self.count_column

            cx = ix * self.h_length / 2.0 * 3.0 + x_min
            cy = iy * self.semi_height * 2.0 + y_min
            if ix % 2 == 1:
                cy = (iy * 2.0 + 1.0) * self.semi_height + y_min

            verts = self._hex_vertices(cx, cy)
            poly = Polygon(verts[:-1])

            yield cell_id, poly, (cx, cy)

    def plot(
        self,
        title="Quadrat Count",
        show="counts",
        chi2_contrib=None,
    ):
        fig, ax = plt.subplots()
        ax.set_title(title)

        _scatter_points(ax, self.points)
        _draw_window(ax, self.window)

        dict_id_count = self.point_location_sta()

        patches = []
        ann_xy = []
        cell_ids = []

        for cell_id, poly, (cx, cy) in self._iter_cells():
            patches.append(MplPolygon(np.asarray(poly.exterior.coords), closed=True))
            ann_xy.append((cx, cy))
            cell_ids.append(cell_id)

        pc = PatchCollection(patches, linewidths=1.0, edgecolor="red", facecolor="none")
        ax.add_collection(pc)

        if show == "counts":
            for (cx, cy), cell_id in zip(ann_xy, cell_ids):
                ax.text(
                    cx, cy, str(dict_id_count.get(cell_id, 0)), ha="center", va="center"
                )

        elif show == "chi2":
            if chi2_contrib is None:
                raise ValueError('chi2_contrib must be provided when show="chi2".')
            chi2_contrib = np.asarray(chi2_contrib, dtype=float)
            if chi2_contrib.shape[0] != len(cell_ids):
                raise ValueError(
                    "chi2_contrib length must match number of plotted cells "
                    f"({len(cell_ids)}). Got {chi2_contrib.shape[0]}."
                )
            pc.set_array(chi2_contrib)
            pc.set_alpha(0.6)
            pc.set_facecolor(None)
            plt.colorbar(pc, ax=ax, label="Chi-square contribution")

            for (cx, cy), v in zip(ann_xy, chi2_contrib):
                ax.text(cx, cy, f"{v:.2f}", ha="center", va="center")

        else:
            raise ValueError('show must be "counts" or "chi2".')

        ax.set_aspect("equal", adjustable="box")
        return ax, cell_ids


# -----------------------------------------------------------------------------
# Quadrat statistic
# -----------------------------------------------------------------------------
class QStatistic:
    """Pearson chi-square quadrat test for complete spatial randomness (CSR).

    This class partitions the study region into quadrats (rectangles or hexagons),
    counts the number of events falling in each quadrat, and evaluates departure
    from CSR using a Pearson chi-square statistic under an equal-intensity CSR
    null model.

    The primary outputs are the observed chi-square statistic and its analytical
    p-value (from the chi-square distribution). Optionally, a simulation-based
    (Monte Carlo) p-value can be computed by generating CSR realizations within
    the study window and recomputing the chi-square statistic for each
    realization.

    Parameters
    ----------
    points : array_like
        Event coordinates as an (n, 2) array-like of floats (x, y).
    shape : {"rectangle", "hexagon"}, default="rectangle"
        Quadrat tessellation type.
    nx : int, default=3
        Number of columns when `shape="rectangle"` and `rectangle_width` /
        `rectangle_height` are not provided.
    ny : int, default=3
        Number of rows when `shape="rectangle"` and `rectangle_width` /
        `rectangle_height` are not provided.
    rectangle_width : float, default=0
        Target rectangle width. Must be provided together with
        `rectangle_height`. When both are provided and non-zero, `nx` and `ny`
        are ignored and the grid dimensions are computed from the point-set MBB.
    rectangle_height : float, default=0
        Target rectangle height. Must be provided together with
        `rectangle_width`. When both are provided and non-zero, `nx` and `ny`
        are ignored and the grid dimensions are computed from the point-set MBB.
    lh : float, default=10
        Hexagon side length when `shape="hexagon"`.
    realizations : int, default=0
        Number of CSR simulations used to compute a Monte Carlo p-value.
        If 0, no simulation-based inference is performed.
    window : shapely geometry, optional
        Study window (e.g., Polygon or MultiPolygon). If None, the window is
        taken to be the axis-aligned MBB rectangle of `points`.
    rng : None | int | numpy.random.Generator | numpy.random.RandomState, optional
        Random number generator control for reproducible CSR simulations when
        `realizations > 0`.

        - None: create a new ``numpy.random.default_rng()``
        - int: use as a seed for ``numpy.random.default_rng(seed)``
        - Generator: used as-is
        - RandomState: wrapped via ``numpy.random.default_rng(RandomState)``

    Attributes
    ----------
    points : numpy.ndarray
        (n, 2) array of event coordinates.
    mbb : numpy.ndarray
        Bounding box of `points` as ``[xmin, ymin, xmax, ymax]``.
    window : shapely geometry
        Study window used for simulation and plotting.
    shape : str
        Quadrat tessellation type: "rectangle" or "hexagon".
    rng : numpy.random.Generator
        RNG used for CSR simulations (always stored as a Generator).
    mr : RectangleM or HexagonM
        Tessellation manager instance.
    cell_ids : list of int
        Cell identifiers included in the test statistic (all grid cells).
    chi2 : float
        Observed Pearson chi-square statistic.
    df : int
        Degrees of freedom for the analytical chi-square reference distribution,
        equal to ``k - 1`` where ``k`` is the number of included cells.
    chi2_pvalue : float
        Analytical p-value from the chi-square distribution.
    chi2_contrib : numpy.ndarray
        Per-cell chi-square contributions aligned with `cell_ids`, computed as
        ``(O - E)^2 / E`` with ``E = mean(O)`` over included cells.
    chi2_realizations : numpy.ndarray
        Simulated chi-square statistics for CSR realizations. Present only when
        `realizations > 0`.
    chi2_r_pvalue : float
        Monte Carlo p-value based on `chi2_realizations`, computed using the
        standard +1 correction:
        ``( #{T_sim >= T_obs} + 1 ) / (realizations + 1)``.
        Present only when `realizations > 0`.

    Notes
    -----
    - The analytical test uses ``scipy.stats.chisquare`` with expected
      counts equal across cells (i.e., ``E = mean(O)``). This
      corresponds to a homogeneous CSR null with equal-area cells. For
      irregular windows, a fully-correct analytical reference would
      area-weight expectations.

    - The simulation-based null generates CSR realizations within `window` with
      intensity ``n / area(window)``. Reproducibility depends on passing `rng`
      through to the underlying CSR generator (``poisson(..., rng=...)``).

    See Also
    --------
    RectangleM : Rectangular tessellation over the point-set MBB.
    HexagonM : Hexagonal tessellation over the point-set MBB.

    """

    def __init__(
        self,
        points,
        shape="rectangle",
        nx=3,
        ny=3,
        rectangle_width=0,
        rectangle_height=0,
        lh=10,
        realizations=0,
        window=None,
        rng=None,
    ):
        self.points = _as_points_array(points)
        self.mbb = _compute_mbb(self.points)
        self.window = _ensure_window(window, self.mbb)
        self.shape = shape
        self.rng = _coerce_rng(rng)

        if shape == "rectangle":
            self.mr = RectangleM(
                self.points,
                count_column=nx,
                count_row=ny,
                rectangle_width=rectangle_width,
                rectangle_height=rectangle_height,
                window=self.window,
            )
        elif shape == "hexagon":
            self.mr = HexagonM(self.points, lh, window=self.window)
        else:
            raise ValueError('shape must be either "rectangle" or "hexagon".')
        dict_id_count = self.mr.point_location_sta()

        obs_counts = np.asarray(list(dict_id_count.values()), dtype=float)
        self.cell_ids = list(dict_id_count.keys())

        self.chi2, self.chi2_pvalue = scipy.stats.chisquare(obs_counts)
        self.df = obs_counts.size - 1

        expected = obs_counts.mean() if obs_counts.size else np.nan
        if obs_counts.size and expected > 0:
            self.chi2_contrib = (obs_counts - expected) ** 2 / expected
        else:
            self.chi2_contrib = np.full_like(obs_counts, np.nan, dtype=float)

        # simulation-based inference under CSR (reproducible via rng)
        if realizations and realizations > 0:
            n = self.points.shape[0]
            intensity = n / float(self.window.area)

            reals = poisson(self.window, intensity, realizations, rng=self.rng)

            chi2_realizations = []
            for i in range(realizations):
                ri = reals[i]
                pts_i = getattr(ri, "points", ri)
                pts_i = _as_points_array(pts_i)

                if shape == "rectangle":
                    mr_temp = RectangleM(
                        pts_i,
                        count_column=self.mr.count_column,
                        count_row=self.mr.count_row,
                        window=self.window,
                    )
                else:
                    mr_temp = HexagonM(pts_i, self.mr.h_length, window=self.window)

                dtemp = mr_temp.point_location_sta()
                sim_counts = np.asarray(list(dtemp.values()), dtype=float)

                chi2_sim, _p = scipy.stats.chisquare(sim_counts)
                chi2_realizations.append(chi2_sim)

            self.chi2_realizations = np.asarray(chi2_realizations, dtype=float)
            self.chi2_r_pvalue = (np.sum(self.chi2_realizations >= self.chi2) + 1.0) / (
                realizations + 1.0
            )

    def plot(self, title="Quadrat Count", show="counts"):
        if show == "counts":
            ax, _cell_ids = self.mr.plot(title=title, show="counts")
            return ax

        if show == "chi2":
            # get plotted ids
            _, plotted_ids = self.mr.plot(title=title, show="counts")
            plt.close()

            contrib_map = {cid: v for cid, v in zip(self.cell_ids, self.chi2_contrib)}
            plotted_contrib = np.asarray(
                [contrib_map.get(cid, np.nan) for cid in plotted_ids], dtype=float
            )

            ax, _ = self.mr.plot(title=title, show="chi2", chi2_contrib=plotted_contrib)
            return ax

        raise ValueError('show must be either "counts" or "chi2".')
