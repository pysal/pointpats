import numpy as np


def plot_density(
    data, bandwidth, resolution=100, levels=10, fill=False, margin=0.1, **kwargs
):
    """Plot kernel density of a given point pattern

    This uses s``tatsmodels.nonparametric.KDEMultivariate`` class to create
    KDE and matplotlib's ``contour`` or ``contourf` function to plot the
    density.

    If MultiPoints are given, each point is treated as a separate
    observation.

    Parameters
    ----------
    data : array or geopandas object
        Array with a shape (2, n) containing coordinates of points
        or a geopandas object with (Multi)Point geometry. Assumes
        projected coordinates, geographical coordinates (latitude, longitude)
        are not supported.
    bandwidth : float
        bandwidth in the units of CRS in which data is
    resolution : int | tuple(int, int), optional
        resolution of the grid used to evaluate the probability density
        function. If tuple, each dimension of the grid is specified separately.
        By default 100
    levels : int or array-like, optional
        Determines the number and positions of the contour lines / regions.
        See the documentation of ``matplotlib.pyplot.contour`` for details., by default 10
    fill : bool, optional
        Fill the area between contour lines, by default False
    margin : float, optional
        The factor of the margin by which the extent of the data will be expanded when
        creating the grid. 0.1 means 10% on each side, by default 0.1.

    Returns
    -------
    matplotlib.pyplot.QuadContourSet
        plot
    """
    try:
        import statsmodels.api as sm
    except ImportError as err:
        raise ImportError("statsmodels is required for `plot_density`") from err

    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("matplotlib is required for `plot_density`") from err

    if isinstance(data, np.ndarray):
        x = data[:, 0]
        y = data[:, 1]
    else:  # geopandas
        if not data.geom_type.str.contains("Point").all():
            raise ValueError(
                "data contain non-point geometries. "
                "Only (Multi)Points are supported."
            )
        coords = data.get_coordinates()
        x = coords.x.values
        y = coords.y.values

    dens_u = sm.nonparametric.KDEMultivariate(
        data=[x, y],
        var_type="cc",
        bw=[bandwidth, bandwidth],
    )

    xmax = x.max()
    xmin = x.min()
    ymax = y.max()
    ymin = y.min()

    # get margin to go beyond the extent to avoid cutting of countour lines
    x_margin = (xmax - xmin) * margin
    y_margin = (ymax - ymin) * margin

    if isinstance(resolution, tuple):
        x_res, y_res = resolution
    elif isinstance(resolution, (float, int)):
        x_res = resolution
        y_res = resolution
    elif resolution is None:
        x_res = 100
        y_res = 100
    else:
        raise ValueError("Unsupported option for `resolution`.")

    # create mesh for predicting KDE on with more space around the points
    x_mesh, y_mesh = np.meshgrid(
        np.linspace(xmin - x_margin, xmax + x_margin, x_res),
        np.linspace(ymin - y_margin, ymax + y_margin, y_res),
    )

    # get the prediction
    pred = dens_u.pdf(np.vstack([x_mesh.flatten(), y_mesh.flatten()]).T)

    if fill:
        return plt.contourf(
            x_mesh, y_mesh, pred.reshape(x_mesh.shape), levels=levels, **kwargs
        )
    else:
        return plt.contour(
            x_mesh, y_mesh, pred.reshape(x_mesh.shape), levels=levels, **kwargs
        )
