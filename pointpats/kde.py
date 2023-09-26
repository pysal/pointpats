import numpy as np


def plot_density(
    data,
    bandwidth,
    kernel=None,
    resolution=100,
    levels=10,
    fill=False,
    margin=0.1,
    ax=None,
    figsize=None,
    **kwargs,
):
    """Plot kernel density of a given point pattern

    The KDE can be done either using :class:`statsmodels.nonparametric.KDEMultivariate`,
    which is used when ``kernel=None``, or using :class:`KDEpy.FFTKDE` when kernel is
    set. :class:`~KDEpy.FFTKDE` tends to be generally faster in most cases but may need
    different than ``"gaussian"`` kernel to resolve in some cases. For small data of up
    to 10 000 points, the difference is not noticeable. For larger data, specify
    ``bandwidth`` to enforce the use of :class:`~KDEpy.FFTKDE`. Note that while being
    faster, :class:`~KDEpy.FFTKDE` may in some case result in erroneous KDE.

    KDE is plotted using matplotlib's :meth:`~matplotlib.pyplot.contour` or
    :meth:`~matplotlib.pyplot.contourf` function to plot the density.

    If MultiPoints are given, each point is treated as separate observation.

    Parameters
    ----------
    data : array or geopandas object
        Array with a shape (2, n) containing coordinates of points
        or a geopandas object with (Multi)Point geometry. Assumes
        projected coordinates, geographical coordinates (latitude, longitude)
        are not supported.
    bandwidth : float
        bandwidth in the units of CRS in which data is
    kernel : str | None, optional
        The kernel function. If None, defaults to the Gaussian kernel and statsmodels
        implementation. If set, uses KDEpy implementation. See
        :meth:`KDEpy.FFTKDE._available_kernels.keys()` for choices.
    resolution : int | tuple(int, int), optional
        resolution of the grid used to evaluate the probability density
        function. If tuple, each dimension of the grid is specified separately.
        By default 100
    levels : int or array-like, optional
        Determines the number and positions of the contour lines / regions.
        See the documentation of :meth:`~matplotlib.pyplot.contour` for details.
        By default 10
    fill : bool, optional
        Fill the area between contour lines, by default False
    margin : float, optional
        The factor of the margin by which the extent of the data will be expanded when
        creating the grid. 0.1 means 10% on each side, by default 0.1. Only used
        with the ``statsmodels`` implementation.
    ax : matplotlib.axes.Axes (default None)
        axes on which to draw the plot
    figsize : tuple of integers (default None)
        Size of the resulting ``matplotlib.figure.Figure``. If the argument
        ``ax`` is given explicitly, ``figsize`` is ignored.
    **kwargs
        Keyword arguments passed to :meth:`~matplotlib.pyplot.contour` or
        :meth:`~matplotlib.pyplot.contourf` used for further
        styling of the plot, for example ``cmap``, ``linewidths``, ``linestyles``,
        or `alpha`. See the documentation of :meth:`~matplotlib.pyplot.contour` for
        details.

    Returns
    -------
    matplotlib.axes.Axes
        matplotlib axes instance with the contour plot
    """
    if kernel is None:
        try:
            import statsmodels.api as sm
        except ImportError as err:
            raise ImportError(
                "statsmodels is required for `plot_density` when kernel"
                "is not specified."
            ) from err

        engine = "sm"
    else:
        try:
            from KDEpy import FFTKDE
        except ImportError as err:
            raise ImportError(
                "KDEpy is required for `plot_density` when kernel is not None."
            ) from err

        engine = "kdepy"

    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError("matplotlib is required for `plot_density`") from err

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.set_aspect("equal") # bandwidth is fixed, hence aspect shall be equal

    if isinstance(data, np.ndarray):
        pass
    else:  # geopandas
        if not data.geom_type.str.contains("Point").all():
            raise ValueError(
                "data contain non-point geometries. "
                "Only (Multi)Points are supported."
            )
        data = data.get_coordinates().values

    if engine == "sm":
        dens_u = sm.nonparametric.KDEMultivariate(
            data=[data[:, 0], data[:, 1]],
            var_type="cc",
            bw=[bandwidth, bandwidth],
        )

        xmax = data[:, 0].max()
        xmin = data[:, 0].min()
        ymax = data[:, 1].max()
        ymin = data[:, 1].min()

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
        z = pred.reshape(x_mesh.shape)

    else:
        kde = FFTKDE(bw=bandwidth, kernel=kernel)
        grid, points = kde.fit(data).evaluate(resolution)
        x_mesh, y_mesh = np.unique(grid[:, 0]), np.unique(grid[:, 1])
        z = points.reshape(resolution, resolution).T

    if fill:
        ax.contourf(x_mesh, y_mesh, z, levels=levels, **kwargs)
    else:
        ax.contour(x_mesh, y_mesh, z, levels=levels, **kwargs)

    return ax
