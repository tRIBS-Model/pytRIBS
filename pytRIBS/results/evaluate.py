import numpy as np
import pandas as pd
from scipy.stats import pearsonr


class Evaluate:
    """
    A collection of static methods for evaluating the performance of simulated data against observed data.

    """

    @staticmethod
    def align_series(simulated, observed, freq, start=None, end=None, interpolate=False):
        """
        Resample a simulated and an observed series onto a common interval and pair them.

        Both series are resampled to ``freq`` (mean), optionally restricted to ``[start, end]``, and
        paired on the common timestamps so the result has no gaps. The returned uniform time series
        is the building block for computing standard metrics (:meth:`compute_metrics`) or any
        event-based statistics (peak, time-to-peak, volume) the caller wants to derive themselves.

        Parameters
        ----------
        simulated, observed : pandas.Series
            Datetime-indexed series in the same units. ``observed`` may have any (even irregular)
            cadence; both are resampled to ``freq``.
        freq : str
            Resampling interval as a pandas offset alias (e.g. ``"15min"``, ``"h"``, ``"D"``).
            Required; choose it to match the observation cadence, since the comparison cannot be
            finer than the real data.
        start, end : str or datetime-like, optional
            Restrict the result to this window.
        interpolate : bool or str, optional
            Fill gaps in the *observed* series after resampling (before pairing), so a window that
            falls between sparse reports is still covered. ``True`` uses linear interpolation; a
            string selects a pandas interpolation method (e.g. ``"time"``, ``"cubic"``). Done on the
            full series before clipping. Default ``False``.

        Returns
        -------
        pandas.DataFrame
            Columns ``observed`` and ``simulated`` on the common ``freq`` grid, with unpaired
            timestamps dropped.

        Notes
        -----
        Dropping unpaired timestamps yields a practically uniform series within a dense window, but
        a genuine interior gap in either input leaves a step in the spacing. Use ``interpolate`` to
        fill observed gaps if you need the grid kept regular across them.
        """
        sim = simulated.resample(freq).mean()
        obs = observed.resample(freq).mean()
        if interpolate:
            method = interpolate if isinstance(interpolate, str) else 'linear'
            obs = obs.interpolate(method=method)

        aligned = pd.DataFrame({'observed': obs, 'simulated': sim})
        return aligned.loc[start:end].dropna()

    def align_streamflow(self, observed, freq, start=None, end=None, interpolate=False):
        """
        Align simulated outlet streamflow with observed discharge.

        Pulls the simulated outlet discharge (``Qstrm_m3s``) from the ``.qout`` file and pairs it
        with ``observed`` via :meth:`align_series` (see it for the parameters). Both must be in the
        same units (m^3/s).

        Returns
        -------
        pandas.DataFrame
            Columns ``observed`` and ``simulated`` (m^3/s) on the common ``freq`` grid.
        """
        sim = self.get_qout_results().set_index('Time')['Qstrm_m3s']
        observed = self._observed_series(observed)
        return Evaluate.align_series(sim, observed, freq, start, end, interpolate)

    def align_swe(self, observed, freq, node_id=None, start=None, end=None, interpolate=False):
        """
        Align simulated snow water equivalent with observed SWE.

        Pulls simulated SWE in mm (basin-average from the ``.mrf`` if ``node_id`` is ``None``, else
        that node's point SWE from its ``.pixel``) via :meth:`get_swe_series` and pairs it with
        ``observed`` via :meth:`align_series` (see it for the parameters). ``observed`` must be in
        mm.

        Returns
        -------
        pandas.DataFrame
            Columns ``observed`` and ``simulated`` (mm) on the common ``freq`` grid.
        """
        sim = self.get_swe_series(node_id)
        observed = self._observed_series(observed)
        return Evaluate.align_series(sim, observed, freq, start, end, interpolate)

    @staticmethod
    def compute_metrics(observed, simulated):
        """
        Compute the standard goodness-of-fit metrics for a paired observed/simulated series.

        Convenience wrapper that runs :meth:`nash_sutcliffe`, :meth:`kling_gupta_efficiency`,
        :meth:`percent_bias`, and :meth:`root_mean_squared_error` on an aligned pair (e.g. a column
        pair from :meth:`align_series`).

        Parameters
        ----------
        observed, simulated : array-like
            Paired, equal-length observed and simulated values (no NaN).

        Returns
        -------
        dict
            ``{'NSE': ..., 'KGE': ..., 'PBIAS': ..., 'RMSE': ...}``.
        """
        observed = np.asarray(observed, dtype=float)
        simulated = np.asarray(simulated, dtype=float)
        return {
            'NSE': Evaluate.nash_sutcliffe(observed, simulated),
            'KGE': Evaluate.kling_gupta_efficiency(observed, simulated),
            'PBIAS': Evaluate.percent_bias(observed, simulated),
            'RMSE': Evaluate.root_mean_squared_error(observed, simulated),
        }
    @staticmethod
    def nash_sutcliffe(observed, simulated):
        """
        Calculate the Nash-Sutcliffe efficiency coefficient.

        The Nash-Sutcliffe efficiency (NSE) is a normalized statistic that determines the relative magnitude of the residual variance
        compared to the measured data variance. It ranges from -∞ to 1, with 1 indicating a perfect match between observed and simulated values.

        Parameters
        ----------
        observed : numpy.ndarray
            Array of observed data values.
        simulated : numpy.ndarray
            Array of simulated data values.

        Returns
        -------
        float
            The Nash-Sutcliffe efficiency coefficient.
        """
        return 1 - (np.sum((observed - simulated) ** 2) / np.sum((observed - np.mean(observed)) ** 2))

    @staticmethod
    def percent_bias(observed, simulated):
        """
        Calculate the percent bias.

        The percent bias (PBIAS) measures the average tendency of the simulated data to be larger or smaller than the observed data.
        Positive values indicate model underestimation, while negative values indicate model overestimation.

        Parameters
        ----------
        observed : numpy.ndarray
            Array of observed data values.
        simulated : numpy.ndarray
            Array of simulated data values.

        Returns
        -------
        float
            The percent bias.
        """
        return 100 * np.sum(observed - simulated) / np.sum(observed)

    @staticmethod
    def root_mean_squared_error(observed, simulated):
        """
        Calculate the root mean squared error (RMSE).

        RMSE measures the square root of the average squared differences between observed and simulated values. It provides a measure
        of how well the model predicts the observed data.

        Parameters
        ----------
        observed : numpy.ndarray
            Array of observed data values.
        simulated : numpy.ndarray
            Array of simulated data values.

        Returns
        -------
        float
            The root mean squared error.
        """
        return np.sqrt(np.mean((observed - simulated) ** 2))

    @staticmethod
    def kling_gupta_efficiency(observed, simulated):
        """
        Calculate the Kling-Gupta efficiency (KGE).

        The Kling-Gupta efficiency is a metric that evaluates model performance based on correlation, variability, and bias. It ranges
        from -∞ to 1, with 1 indicating perfect model performance.

        Parameters
        ----------
        observed : numpy.ndarray
            Array of observed data values.
        simulated : numpy.ndarray
            Array of simulated data values.

        Returns
        -------
        float
            The Kling-Gupta efficiency coefficient.
        """
        r, _ = pearsonr(observed, simulated)
        alpha = np.std(simulated) / np.std(observed)
        beta = np.mean(simulated) / np.mean(observed)
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        return kge