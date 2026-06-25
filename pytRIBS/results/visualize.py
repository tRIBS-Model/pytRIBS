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
                  labels=["Precip.", "Runoff", "Evapo. Trans.", "$\Delta$ Storage"])
        ax.set_ylabel("Water Flux & $\Delta$ Storage (mm)")
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