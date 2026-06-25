import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pytRIBS.shared.aux import Aux


class Viz:
    "Framework class for Results Class"
    @staticmethod
    def plot_water_balance(waterbalance, saved_fig=None):
        """
        Plots water balance components and saves the figure if a filename is provided.

        This function creates a bar plot of water balance components, including precipitation (`nP`), runoff (`nQ`),
        evapotranspiration (`nET`), and changes in storage (`dS`). It displays labels for the difference between
        precipitation and the sum of other components. The plot is saved to a file if `saved_fig` is provided.

        Parameters
        ----------
        waterbalance : pd.DataFrame
            DataFrame containing water balance components with columns:
            - `nP`: Precipitation
            - `nQ`: Runoff
            - `nET`: Evapotranspiration
            - `dS`: Change in storage

        saved_fig : str, optional
            Filename to save the figure. If not provided, the figure is not saved.

        Returns
        -------
        tuple
            A tuple containing the `matplotlib.figure.Figure` and `matplotlib.axes.Axes` objects for the plot.

        Notes
        -----
        - The plot includes a stacked bar chart of `nQ`, `nET`, and `dS` with different colors.
        - Labels indicate the net difference between `nP` and the sum of `dS`, `nQ`, and `nET`.
        - The plot will automatically format the x-axis dates and display mean difference in the plot.
        """

        # plt.style.use('bmh')
        barwidth = 0.25
        fig, ax = plt.subplots()

        ax.bar(np.arange(len(waterbalance)) + barwidth, waterbalance['nP'], align='center', width=barwidth,
               color='grey', label='nP')
        rects = ax.patches

        # Make some labels.
        labels = ["%.0f" % (p - waterbalance) for p, waterbalance in
                  zip(waterbalance['nP'], waterbalance['dS'] + waterbalance['nQ'] + waterbalance['nET'])]
        netdiff = [p - waterbalance for p, waterbalance in
                   zip(waterbalance['nP'], waterbalance['dS'] + waterbalance['nQ'] + waterbalance['nET'])]

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
            )

        #ax.text(len(waterbalance.index), max(waterbalance.nP), "mean difference: " + "%.0f" % np.mean(netdiff))

        waterbalance.plot.bar(ax=ax, y=["nQ", "nET", "dS"], stacked=True, width=barwidth,
                              color=['tab:blue', 'tab:red', 'tab:cyan'])
        ax.legend(bbox_to_anchor=(1.35, 0.85), loc='center right',
                  labels=["Precip.", "Runoff", "Evapo. Trans.", r"$\Delta$ Storage"])
        ax.set_ylabel(r"Water Flux & $\Delta$ Storage (mm)")
        ax.set_xticks(range(len(waterbalance.index)), waterbalance.index.strftime("%Y-%m"), rotation=45)
        fig.autofmt_xdate()
        plt.show()

        if saved_fig is not None:
            plt.savefig(saved_fig, bbox_inches='tight')

        return fig, ax

    def plot_hydrograph(self, observed=None, start=None, end=None, resample=None,
                        ax=None, saved_fig=None):
        """
        Quick-look plot of simulated outlet streamflow, optionally against observations.

        Plots the outlet discharge (``Qstrm_m3s``) from the ``.qout`` file versus time. This is a
        fast "did my run do something sane?" look, not a publication figure; it returns the
        ``Axes`` so it can be styled further or composed into a larger figure.

        Parameters
        ----------
        observed : pandas.Series or pandas.DataFrame, optional
            Observed discharge with a datetime index, in the same units as the simulation
            (m^3/s). A one-column DataFrame is accepted and squeezed to a Series. Any unit
            conversion or loading is the caller's responsibility.
        start, end : str or datetime-like, optional
            Restrict the plot to this window (e.g. a single storm event). Applied to both the
            simulated and observed series.
        resample : str, optional
            Pandas offset alias (e.g. ``"5min"``, ``"1h"``) to resample both series to a common
            frequency (mean) before plotting. Useful when the simulation and observations are
            recorded at different intervals. If ``None``, each series is plotted as-is.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. If ``None``, a new figure and axes are created.
        saved_fig : str, optional
            If provided, the figure is saved to this path.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the hydrograph was drawn on (use ``ax.figure`` for the figure).
        """
        sim = self.get_qout_results().set_index('Time')['Qstrm_m3s']

        obs = observed
        if obs is not None:
            if hasattr(obs, 'columns'):  # DataFrame
                if obs.shape[1] != 1:
                    raise ValueError("observed DataFrame must have a single discharge column; "
                                     "pass a Series or a one-column DataFrame.")
                obs = obs.iloc[:, 0]
            obs = obs.copy()

        if resample is not None:
            sim = sim.resample(resample).mean()
            if obs is not None:
                obs = obs.resample(resample).mean()

        if start is not None or end is not None:
            sim = sim.loc[start:end]
            if obs is not None:
                obs = obs.loc[start:end]

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        ax.plot(sim.index, sim.values, label='Simulated (tRIBS)', color='#1f77b4', linewidth=2)
        if obs is not None:
            ax.plot(obs.index, obs.values, label='Observed', color='black', marker='o',
                    markersize=4, linestyle='--', linewidth=1)
            ax.legend()

        ax.set_ylabel('Streamflow ($m^3/s$)')
        ax.set_xlabel('Date')
        ax.set_title('Simulated vs. Observed Streamflow at Outlet' if obs is not None
                     else 'Simulated Streamflow at Outlet')

        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.grid(True, linestyle=':', alpha=0.7)

        if saved_fig is not None:
            ax.figure.savefig(saved_fig, bbox_inches='tight')

        return ax

    def plot_mrf(self, var, rainfall='MAP_mm_hr', start=None, end=None, invert=False,
                 ax=None, saved_fig=None):
        """
        Quick-look plot of a basin-averaged (``.mrf``) variable with the rainfall hyetograph.

        Plots one or more basin-averaged time series on the primary axis, and always overlays the
        basin-averaged rainfall as an inverted hyetograph on a twin top axis (the common
        rainfall-runoff layout). Because ``.mrf`` files carry many variables whose names depend on
        the run configuration, the column name(s) must be given explicitly.

        Parameters
        ----------
        var : str or list of str
            Column name(s) in the ``.mrf`` table to plot on the primary axis (e.g. ``"MDGW_mm"``).
            A list draws several series on the same axis.
        rainfall : str, optional
            Column name of the basin-averaged rainfall plotted as the top hyetograph. Defaults to
            ``"MAP_mm_hr"``. If the column is absent, the hyetograph is skipped with a warning.
        start, end : str or datetime-like, optional
            Restrict the plot to this window.
        invert : bool, optional
            Invert the primary y-axis. Useful for depth-like variables (e.g. depth to water table),
            where increasing depth should point downward. Default ``False``.
        ax : matplotlib.axes.Axes, optional
            Primary axes to draw into. If ``None``, a new figure and axes are created.
        saved_fig : str, optional
            If provided, the figure is saved to this path.

        Returns
        -------
        matplotlib.axes.Axes
            The primary axes (``ax.figure.axes`` also exposes the rainfall twin axis).
        """
        if self.mrf.get('mrf') is None:
            self.get_mrf_results()
        mrf = self.mrf['mrf'].set_index('Time')

        varlist = [var] if isinstance(var, str) else list(var)
        missing = [v for v in varlist if v not in mrf.columns]
        if missing:
            raise ValueError(f"Column(s) {missing} not found in the mrf output. "
                             f"Available columns: {list(mrf.columns)}")

        mrf = mrf.loc[start:end]

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))

        for v in varlist:
            ax.plot(mrf.index, mrf[v], linewidth=2, label=v)
        ax.set_xlabel('Date')
        ax.set_ylabel(varlist[0] if len(varlist) == 1 else 'Value')
        if invert:
            ax.invert_yaxis()

        # Rainfall hyetograph on an inverted twin axis, scaled to the top ~third of the figure.
        rain_handles, rain_labels = [], []
        if rainfall in mrf.columns:
            ax2 = ax.twinx()
            bar_width = ((mrf.index[1] - mrf.index[0]).total_seconds() / 86400.0
                         if len(mrf.index) > 1 else 0.02)
            ax2.bar(mrf.index, mrf[rainfall], width=bar_width, color='#1f77b4', alpha=0.6,
                    label='Rainfall')
            ax2.set_ylabel('Rainfall (mm/hr)', color='#1f77b4')
            ax2.tick_params(axis='y', labelcolor='#1f77b4')
            max_rain = mrf[rainfall].max()
            if max_rain and max_rain > 0:
                ax2.set_ylim(max_rain * 3, 0)
            rain_handles, rain_labels = ax2.get_legend_handles_labels()
        elif rainfall is not None:
            print(f"Warning: rainfall column '{rainfall}' not found in mrf; skipping hyetograph.")

        var_handles, var_labels = ax.get_legend_handles_labels()
        ax.legend(var_handles + rain_handles, var_labels + rain_labels, loc='center right')

        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.grid(True, linestyle=':', alpha=0.7)

        if saved_fig is not None:
            ax.figure.savefig(saved_fig, bbox_inches='tight')

        return ax

    def plot_element(self, node_id, var, start=None, end=None, ax=None, saved_fig=None):
        """
        Quick-look plot of one or more variables from a single element's ``.pixel`` output.

        Loads the per-node pixel results (if not already loaded) and plots the requested
        variable(s) versus time for the given node. Because ``.pixel`` files carry many variables,
        the column name(s) must be given explicitly. A list of variables is drawn as stacked,
        x-aligned panels (one per variable).

        Parameters
        ----------
        node_id : int
            Node ID of the pixel to plot. Must be present in ``self.element``.
        var : str or list of str
            Column name(s) in the ``.pixel`` table (e.g. ``"Nwt_mm"``, ``"Rain_mm_h"``).
        start, end : str or datetime-like, optional
            Restrict the plot to this window.
        ax : matplotlib.axes.Axes or list of Axes, optional
            Axes to draw into. For a single variable, one ``Axes``; for a list of variables, a
            matching list of stacked axes. If ``None``, axes are created.
        saved_fig : str, optional
            If provided, the figure is saved to this path.

        Returns
        -------
        matplotlib.axes.Axes or tuple
            For a single variable, the ``Axes``. For a list, ``(figure, axes)``.
        """
        if not self.element:
            self.get_element_results()
        if node_id not in self.element:
            raise ValueError(f"Node {node_id} not found. Available nodes: "
                             f"{sorted(self.element.keys())}")

        pixel = self.element[node_id]['pixel'].set_index('Time')

        single = isinstance(var, str)
        varlist = [var] if single else list(var)
        missing = [v for v in varlist if v not in pixel.columns]
        if missing:
            raise ValueError(f"Column(s) {missing} not found in pixel output for node {node_id}. "
                             f"Available columns: {list(pixel.columns)}")

        pixel = pixel.loc[start:end]

        if ax is None:
            _, axarr = plt.subplots(len(varlist), 1, figsize=(12, 3 * len(varlist)), sharex=True)
            axes = list(np.atleast_1d(axarr))
        else:
            axes = list(np.atleast_1d(ax))
            if len(axes) != len(varlist):
                raise ValueError(f"Got {len(axes)} axes for {len(varlist)} variable(s).")

        for a, v in zip(axes, varlist):
            a.plot(pixel.index, pixel[v], linewidth=1, color='tab:blue')
            a.set_ylabel(v)
            a.grid(True, linestyle=':', alpha=0.7)

        axes[0].set_title(f'Pixel results — Node {node_id}', fontweight='bold')
        axes[-1].set_xlabel('Date')
        locator = mdates.AutoDateLocator()
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        if saved_fig is not None:
            axes[0].figure.savefig(saved_fig, bbox_inches='tight')

        return axes[0] if single else (axes[0].figure, axes)

    def create_animation(self, outfile, df_dict, frames, var, fps=4, vlims=None, nan_color='gray',
                         nan_edge_color='red', cmap='viridis'):
        """
        Create and save an animation based on a dictionary of DataFrames of tRIBS dynamic files.

        Parameters:
            outfile (str): The file path for saving the animation, format is determined from file extension (.mp4,.gif,.avi,.html).
            df_dict (dict): A dictionary where keys represent animation frames and values are DataFrames to be plotted.
            frames (iterable): Iterable containing keys from df_dict representing the frames to include in the animation.
            var (str): The column name in DataFrames to be plotted.
            fps (int, optional): Frames per second for the animation (default is 4).
            vlims (tuple, optional): Tuple containing minimum and maximum values for color normalization (default is None).
            nan_color (str, optional): Color for NaN values in the plot (default is 'gray').
            nan_edge_color (str, optional): Edge color for NaN values in the plot (default is 'red').

        Returns:
            None

        Raises:
            ValueError: If outfile is not a valid file path or frames is empty.

        Notes:
            - This method creates an animation by iterating over frames specified in the frames parameter.
            - Each frame corresponds to a key in the df_dict dictionary, and the corresponding DataFrame is plotted.
            - NaN values in the DataFrame are flagged with the specified nan_color and nan_edge_color.
            - The animation format is dependent on the outfile extension with the specified frames per second (fps).

        Example:
            # Assuming instance is an instance of the class containing create_animation method
            instance.create_animation("animation.gif", df_dict, frames=['0','1','2','3'], var="ET", fps=10)
        """

        def update_plot(key, ax, cax):
            """
            Update the plot for each frame in the animation.

            Parameters:
                key: The key representing the current frame in df_dict.
                ax: The main axes object for the plot.
                cax: The colorbar axes object.
                df_dict: A dictionary containing DataFrames for each frame.
                results_class: An instance of the class containing the voronoi attribute.
                var: The variable to be plotted from DataFrames.
                vlims: Tuple containing minimum and maximum values for color normalization.

            Returns:
                None
            """
            ax.clear()

            df = df_dict[key]
            gdf = self.voronoi.copy()
            gdf = gdf.merge(df, on="ID", how="inner")

            if vlims is not None:
                gdf.plot(ax=ax, column=var, legend=True, cmap=cmap, vmin=min(vlims), vmax=max(vlims), cax=cax)
            else:
                gdf.plot(ax=ax, column=var, legend=True, cmap=cmap, cax=cax)

            # flag and plot nans
            if len(gdf[gdf[var].isnull()]) != 0:
                gdf[gdf[var].isnull()].plot(ax=ax, color=nan_color, edgecolor=nan_edge_color)

            ax.set_title(f'{var}: {key}')

            plt.axis('off')
            plt.xticks([])  # Remove x-axis ticks and labels
            plt.yticks([])  # Remove y-axis ticks and labels

        # Create a figure and axis
        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # Create the animation
        animation = FuncAnimation(fig, update_plot, frames=frames, fargs=(ax, cax),
                                  repeat=False)

        # To save the animation as a GIF
        animation.save(outfile, fps=fps)

        plt.show()
    @staticmethod
    def discrete_colormap(N, base_cmap=None):
        cmap = Aux.discrete_cmap(N, base_cmap)
        return cmap