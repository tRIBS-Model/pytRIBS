import os
import copy
import glob
import datetime

import pandas as pd


class SpinupProcessor:
    """
    Framework for the spin-up workflows of the Model Class. See classes.py.

    Hydrologic spin-up establishes a physically reasonable initial state (water table and the
    overlying unsaturated soil column) before a "real" simulation. In tRIBS v6.0.0 this is done
    with the reworked hotstart/restart machinery, which writes the *full* state-variable model state
    rather than just the groundwater-table grid. The methods here build a derived spin-up model from a
    configured :class:`~pytRIBS.classes.Model`, generate (or validate) its forcing, and wire the
    real run to consume the restart file the spin-up produces.

    Two ways to drive the spin-up are provided:

    * :meth:`create_spinup_repeat`: tile a representative window of an existing point record a
      fixed number of times to build a long synthetic forcing series. Use when the available
      record is short (the classic "repeat one year for ten years" approach).
    * :meth:`create_spinup_observed`: use a real, continuous point record as is over a user
      specified window. tRIBS v6.0.0 seeks the forcing file by date, so no new forcing files are
      written i.e. pytRIBS only validates that the record actually covers the requested window.

    Both methods return a *new* ``Model`` configured to write a restart file; the user runs that
    model themselves, then calls :meth:`apply_hotstart` to point the real model at the result.

    Notes
    -----
    * Only point-station forcing (rain gauges, ``RAINSOURCE = 2``; weather stations,
      ``METDATAOPTION = 1``) is supported. Gridded forcing raises ``NotImplementedError``.
    * A restart file is tied to the mesh/node structure, so the spin-up and the real run must use
      the same mesh and parameterization.
    """

    # tRIBS STARTDATE format: MM/DD/YYYY/HH/MM
    _DATE_FMT = "%m/%d/%Y/%H/%M"

    # helper functions

    @staticmethod
    def _to_datetime(value):
        """Coerce a tRIBS STARTDATE string (``MM/DD/YYYY/HH/MM``), ``datetime``, or anything
        :func:`pandas.to_datetime` understands into a ``datetime.datetime``."""
        if isinstance(value, datetime.datetime):
            return value
        if isinstance(value, str):
            return datetime.datetime.strptime(value, SpinupProcessor._DATE_FMT)
        return pd.to_datetime(value).to_pydatetime()

    @staticmethod
    def _format_startdate(dt):
        """Format a ``datetime`` as a tRIBS STARTDATE string (``MM/DD/YYYY/HH/MM``)."""
        return dt.strftime(SpinupProcessor._DATE_FMT)

    @staticmethod
    def _ensure_dir(path):
        """Create ``path`` (and parents) if it does not exist. tRIBS does not create its own
        output directories, so the spin-up workflow must make them up front."""
        if path:
            os.makedirs(path, exist_ok=True)

    def _check_point_forcing(self):
        """Raise if the model is configured for gridded (rather than point) forcing, which the
        spin-up workflow does not yet support."""
        if int(float(self.options["rainsource"]["value"])) == 1:
            raise NotImplementedError(
                "Spin-up currently supports point rain gauges only (RAINSOURCE: 2). "
                "Gridded rainfall (RAINSOURCE: 1) is not yet supported.")
        if int(float(self.options["metdataoption"]["value"])) == 2:
            raise NotImplementedError(
                "Spin-up currently supports point weather stations only (METDATAOPTION: 1). "
                "Gridded meteorological data (METDATAOPTION: 2) is not yet supported.")

    @staticmethod
    def _timestamp_step_minutes(interval_minutes):
        """Spacing between *distinct* timestamps in a tRIBS forcing file, in minutes.

        tRIBS precip (``.mdf``) and met data files carry hourly timestamps with no minute field;
        sub-hourly data (e.g. ``RAININTRVL`` 0.25 or ``METSTEP`` 15) is stored as several rows under
        the same hour, read in order. So distinct timestamps are never spaced finer than one hour,
        regardless of the data interval.
        """
        return max(float(interval_minutes), 60.0)

    @staticmethod
    def check_forcing_coverage(df, start, end, step_minutes):
        """
        Summarize how well a forcing record covers a requested window.

        Parameters
        ----------
        df : pandas.DataFrame
            Station record with a ``date`` column (as returned by ``read_precip_station`` or
            ``read_met_station``).
        start, end : datetime.datetime or str
            Requested window (inclusive). Strings are parsed as ``MM/DD/YYYY/HH/MM``.
        step_minutes : float
            Expected spacing between *distinct* timestamps, in minutes. Because tRIBS forcing files
            are hourly-stamped (sub-hourly data is stored as multiple rows per hour), this should
            never be finer than 60; use :meth:`_timestamp_step_minutes`. Used to count interior gaps.

        Returns
        -------
        dict
            Keys: ``data_start``, ``data_end`` (the record's extent); ``covers_span`` (bool, the
            record brackets the full window); ``n_missing_steps`` (count of expected timestamps
            absent inside the window); ``n_gaps`` (count of contiguous missing runs); and
            ``missing_fraction`` (fraction of expected timestamps missing).

        Notes
        -----
        ``covers_span`` is judged on extent only; interior gaps are reported but do not flip it,
        because tRIBS tolerates short gaps via its fill conventions.
        """
        start = SpinupProcessor._to_datetime(start)
        end = SpinupProcessor._to_datetime(end)

        dates = pd.to_datetime(df["date"]).sort_values().reset_index(drop=True)
        data_start = dates.iloc[0].to_pydatetime()
        data_end = dates.iloc[-1].to_pydatetime()

        expected = pd.date_range(start=start, end=end, freq=pd.Timedelta(minutes=step_minutes))
        missing_mask = ~expected.isin(pd.DatetimeIndex(dates))
        n_missing = int(missing_mask.sum())

        # number of contiguous runs of missing timestamps
        n_gaps = 0
        prev = False
        for flag in missing_mask:
            if flag and not prev:
                n_gaps += 1
            prev = bool(flag)

        return {
            "data_start": data_start,
            "data_end": data_end,
            "covers_span": (data_start <= start) and (data_end >= end),
            "n_missing_steps": n_missing,
            "n_gaps": n_gaps,
            "missing_fraction": (n_missing / len(expected)) if len(expected) else 0.0,
        }

    @staticmethod
    def _configure_restart(model, runtime_hours, restart_dir):
        """Configure ``model`` to write a single restart (hotstart) file at the end of its run.

        Restart-output timing is independent of how forcing is read, so both spin-up modes share
        this wiring. ``RESTARTINTRVL`` is set equal to ``RUNTIME`` so the snapshot lands at the
        final timestep.
        """
        SpinupProcessor._ensure_dir(restart_dir)
        model.options["restartmode"]["value"] = 1          # write restart files
        model.options["restartintrvl"]["value"] = runtime_hours
        model.options["restartdir"]["value"] = restart_dir

    # tRIBS names restart files "<RESTARTDIR>/tRIBS_Rstrt_<HOURS>", where <HOURS> is the elapsed
    # simulation time truncated to an int and zero-padded to a minimum width of 5 (tSimul.cpp).
    _RESTART_PREFIX = "tRIBS_Rstrt_"

    @classmethod
    def _expected_restart_file(cls, restart_dir, hours):
        """Predicted path of the restart file written at ``hours`` of simulation time. Used for
        messaging; :meth:`apply_hotstart` still discovers the actual file, since the embedded hour
        can differ by a step if the timestep does not divide ``RESTARTINTRVL`` evenly."""
        return os.path.join(restart_dir, f"{cls._RESTART_PREFIX}{int(hours):05d}")

    @classmethod
    def _restart_hours(cls, path):
        """Parse the elapsed-hour count embedded in a restart filename, or -1 if it does not match
        the tRIBS naming convention."""
        suffix = os.path.basename(path).rsplit(cls._RESTART_PREFIX, 1)[-1]
        try:
            return int(suffix)
        except ValueError:
            return -1

    def _resolve_spinup_dir(self, spinup_dir):
        """Resolve and create the directory that holds the spin-up's generic model *output*
        (``OUTFILENAME``). The restart file goes to ``restart_dir`` and tiled forcing goes next to
        the original forcing (see :meth:`_forcing_subdir`), both kept separate from here.

        When ``spinup_dir`` is ``None`` it defaults to a ``spinup`` subfolder of the model's
        results directory (derived from ``OUTFILENAME``), matching ``Project.directories['spinup']``
        (``results/spinup``). A ``Project`` already creates that folder, but tRIBS does not create
        directories, so it is created here as well for standalone use.
        """
        if spinup_dir is None:
            out = self.options["outfilename"]["value"]
            if not out:
                raise ValueError(
                    "spinup_dir was not given and OUTFILENAME is unset, so the spin-up directory "
                    "cannot be derived. Provide spinup_dir explicitly.")
            spinup_dir = os.path.join(os.path.dirname(out), "spinup")
        self._ensure_dir(spinup_dir)
        return spinup_dir

    def _forcing_subdir(self, sdf_path):
        """Return (creating if needed) a ``spinup`` subfolder next to the original station forcing
        file. Generated spin-up forcing goes here so it sits with the real forcing under
        ``data/model/met/`` and never overwrites it."""
        forcing_dir = os.path.join(os.path.dirname(sdf_path), "spinup")
        self._ensure_dir(forcing_dir)
        return forcing_dir

    @staticmethod
    def _redirect_output(model, spinup_dir, default_base="spinup"):
        """Point ``OUTFILENAME`` into ``spinup_dir`` so the spin-up does not overwrite the real
        run's output, preserving the original output base name."""
        current = model.options["outfilename"]["value"]
        base = os.path.basename(current) if current else default_base
        model.options["outfilename"]["value"] = os.path.join(spinup_dir, base)

    @staticmethod
    def _tile_record(df, start, end, n_repeats):
        """Tile the half-open window ``[start, end)`` of ``df`` ``n_repeats`` times onto a
        contiguous synthetic timeline, re-stamping each copy by a fixed block duration.

        A fixed duration (rather than a calendar year) is used so leap days and variable month
        lengths cannot desync the repeats.
        """
        start = SpinupProcessor._to_datetime(start)
        end = SpinupProcessor._to_datetime(end)
        block_delta = pd.Timedelta(seconds=(end - start).total_seconds())

        window = df[(df["date"] >= start) & (df["date"] < end)].copy()

        tiles = []
        for i in range(n_repeats):
            tile = window.copy()
            tile["date"] = tile["date"] + i * block_delta
            tiles.append(tile)

        return pd.concat(tiles, ignore_index=True)

    # repeat mode

    def create_spinup_repeat(self, repeat_start, repeat_end, n_repeats, spinup_dir=None,
                             restart_dir="restart", input_file="spinup.in"):
        """
        Build a spin-up model by tiling a representative window of the point forcing.

        A half-open window ``[repeat_start, repeat_end)`` of each rain-gauge and weather-station
        record is repeated ``n_repeats`` times onto a contiguous synthetic timeline and written to
        ``spinup_dir``. The returned model is configured to start at ``repeat_start``, run for
        ``n_repeats`` block lengths, and write a single restart file at the end.

        Parameters
        ----------
        repeat_start, repeat_end : datetime.datetime or str
            Bounds of the window to repeat. Strings are parsed as ``MM/DD/YYYY/HH/MM``. The window
            must fall within every active station's record.
        n_repeats : int
            Number of times to tile the window (e.g. 10 to spin up ~10 years from a 1-year window).
        spinup_dir : str, optional
            Directory for the spin-up's model output, restart file, and ``.in`` file. Defaults to a
            ``spinup`` subfolder of the model's results directory (``results/spinup``, matching
            ``Project.directories['spinup']``). Created if it does not exist (tRIBS does not create
            directories). Tiled forcing is written separately to a ``spinup`` subfolder next to the
            original station forcing (under ``data/model/met/``), so the real forcing is never
            overwritten.
        restart_dir : str, optional
            Directory for the restart (hotstart) file tRIBS writes. Defaults to a ``restart`` folder
            in the current working directory (matching ``Project.directories['restart']``), kept
            separate from the generic model output. Created if it does not exist.
        input_file : str, optional
            Path to write the spin-up ``.in`` file. Defaults to ``spinup.in`` in the current working
            directory, alongside your main input file. Spin-up *outputs* still go to ``spinup_dir``.

        Returns
        -------
        Model
            A new ``Model`` (the original is left untouched) configured for the spin-up run. Run it
            yourself, then call :meth:`apply_hotstart` to wire the real model to its restart file.
        """
        self._check_point_forcing()

        if not isinstance(n_repeats, int) or n_repeats < 1:
            raise ValueError("n_repeats must be a positive integer.")

        start_dt = self._to_datetime(repeat_start)
        end_dt = self._to_datetime(repeat_end)
        if end_dt <= start_dt:
            raise ValueError("repeat_end must be after repeat_start.")

        block_hours = (end_dt - start_dt).total_seconds() / 3600.0
        runtime = n_repeats * block_hours

        spinup_dir = self._resolve_spinup_dir(spinup_dir)
        spin = copy.deepcopy(self)

        # --- rain gauges ---
        if int(float(self.options["rainsource"]["value"])) == 2:
            gauges = self.read_precip_sdf()
            if gauges:
                step_min = self._timestamp_step_minutes(
                    float(self.options["rainintrvl"]["value"]) * 60.0)
                forcing_dir = self._forcing_subdir(self.options["gaugestations"]["value"])
                new_gauges = self._tile_stations(
                    gauges, self.read_precip_station, self.write_precip_station,
                    start_dt, end_dt, n_repeats, forcing_dir, step_min, "rain gauge")
                sdf_path = os.path.join(
                    forcing_dir, self._sdf_name(self.options["gaugestations"]["value"], "precip"))
                self.write_precip_sdf(new_gauges, sdf_path)
                spin.options["gaugestations"]["value"] = sdf_path

        # --- weather stations ---
        if int(float(self.options["metdataoption"]["value"])) == 1:
            stations = self.read_met_sdf()
            if stations:
                step_min = self._timestamp_step_minutes(float(self.options["metstep"]["value"]))
                forcing_dir = self._forcing_subdir(self.options["hydrometstations"]["value"])
                new_stations = self._tile_stations(
                    stations, self.read_met_station, self.write_met_station,
                    start_dt, end_dt, n_repeats, forcing_dir, step_min, "weather station")
                sdf_path = os.path.join(
                    forcing_dir, self._sdf_name(self.options["hydrometstations"]["value"], "met"))
                self.write_met_sdf(new_stations, sdf_path)
                spin.options["hydrometstations"]["value"] = sdf_path

        # --- run configuration ---
        spin.options["startdate"]["value"] = self._format_startdate(start_dt)
        spin.options["runtime"]["value"] = runtime
        self._redirect_output(spin, spinup_dir)
        self._configure_restart(spin, runtime, restart_dir)

        in_path = input_file
        spin.write_input_file(in_path)

        print(f"Spin-up (repeat) model written to {in_path}\n"
              f"  forcing tiled {n_repeats}x over {block_hours:.1f} h "
              f"(RUNTIME: {runtime:.1f} h), written next to the original forcing\n"
              f"  output  -> {spinup_dir}\n"
              f"  restart -> {self._expected_restart_file(restart_dir, runtime)}\n"
              "Run this model, then call apply_hotstart(spinup) to wire your real run to it.")

        return spin

    def _tile_stations(self, stations, reader, writer, start_dt, end_dt, n_repeats,
                       out_dir, step_min, label):
        """Tile every station in ``stations``, write the tiled records to ``out_dir``, and return
        an updated station list whose ``file_path`` entries point at the new files."""
        # The repeat window is half-open [start, end), so the record only needs to reach the last
        # tiled timestamp, end - one step (the window end is the *start* of the next cycle).
        last_needed = end_dt - datetime.timedelta(minutes=step_min)
        new_stations = []
        for station in stations:
            df = reader(station["file_path"])
            cov = self.check_forcing_coverage(df, start_dt, last_needed, step_min)
            if not cov["covers_span"]:
                raise ValueError(
                    f"{label} {station['station_id']} ({station['file_path']}) does not cover the "
                    f"repeat window {self._format_startdate(start_dt)}-"
                    f"{self._format_startdate(end_dt)}; its record spans "
                    f"{cov['data_start']} to {cov['data_end']}.")
            if cov["n_gaps"]:
                print(f"Warning: {label} {station['station_id']} has {cov['n_gaps']} gap(s) "
                      f"({cov['missing_fraction'] * 100:.1f}% of steps) inside the repeat window.")

            tiled = self._tile_record(df, start_dt, end_dt, n_repeats)
            out_path = os.path.join(out_dir, os.path.basename(station["file_path"]))
            writer(tiled.copy(), out_path)

            updated = dict(station)
            updated["file_path"] = out_path
            new_stations.append(updated)
        return new_stations

    @staticmethod
    def _sdf_name(current_path, default_stem):
        """Reuse the original ``.sdf`` basename, falling back to ``<default_stem>.sdf``."""
        if current_path:
            return os.path.basename(current_path)
        return f"{default_stem}.sdf"

    # observed data mode

    def create_spinup_observed(self, start, end, spinup_dir=None, restart_dir="restart",
                               input_file="spinup.in"):
        """
        Build a spin-up model from a real, continuous point record over ``[start, end]``.

        Because tRIBS v6.0.0 seeks the forcing file by date, the model's existing forcing files are
        used as-is -- no new forcing is written. pytRIBS only validates that every active station's
        record brackets the requested window. If any record is too short, the shortfall is reported
        and ``None`` is returned.

        Parameters
        ----------
        start, end : datetime.datetime or str
            Spin-up window (inclusive). Strings are parsed as ``MM/DD/YYYY/HH/MM``.
        spinup_dir : str, optional
            Directory for the model output, restart file, and ``.in`` file. Defaults to a ``spinup``
            subfolder of the model's results directory (``results/spinup``, matching
            ``Project.directories['spinup']``). Created if it does not exist (tRIBS does not create
            directories). No forcing files are written here.
        restart_dir : str, optional
            Directory for the restart (hotstart) file tRIBS writes. Defaults to a ``restart`` folder
            in the current working directory (matching ``Project.directories['restart']``), kept
            separate from the generic model output. Created if it does not exist.
        input_file : str, optional
            Path to write the spin-up ``.in`` file. Defaults to ``spinup.in`` in the current working
            directory, alongside your main input file. Spin-up *outputs* still go to ``spinup_dir``.

        Returns
        -------
        Model or None
            A new ``Model`` configured for the spin-up run, or ``None`` if the existing forcing does
            not cover ``[start, end]``. Run the returned model yourself, then call
            :meth:`apply_hotstart` to wire the real model to its restart file.
        """
        self._check_point_forcing()

        start_dt = self._to_datetime(start)
        end_dt = self._to_datetime(end)
        if end_dt <= start_dt:
            raise ValueError("end must be after start.")

        runtime = (end_dt - start_dt).total_seconds() / 3600.0

        # Validate every active station covers the window before changing anything.
        checks = []
        if int(float(self.options["rainsource"]["value"])) == 2:
            checks.append(("rain gauge", self.read_precip_sdf(), self.read_precip_station,
                           self._timestamp_step_minutes(
                               float(self.options["rainintrvl"]["value"]) * 60.0)))
        if int(float(self.options["metdataoption"]["value"])) == 1:
            checks.append(("weather station", self.read_met_sdf(), self.read_met_station,
                           self._timestamp_step_minutes(float(self.options["metstep"]["value"]))))

        covered = True
        for label, stations, reader, step_min in checks:
            for station in stations or []:
                df = reader(station["file_path"])
                cov = self.check_forcing_coverage(df, start_dt, end_dt, step_min)
                if not cov["covers_span"]:
                    covered = False
                    have_h = (min(cov["data_end"], end_dt) - max(cov["data_start"], start_dt))
                    have_h = max(have_h.total_seconds() / 3600.0, 0.0)
                    print(f"{label} {station['station_id']} ({station['file_path']}) is too short: "
                          f"record spans {cov['data_start']} to {cov['data_end']}, but the spin-up "
                          f"window {self._format_startdate(start_dt)}-"
                          f"{self._format_startdate(end_dt)} needs {runtime:.0f} h "
                          f"(only ~{have_h:.0f} h available).")
                elif cov["n_gaps"]:
                    print(f"Warning: {label} {station['station_id']} has {cov['n_gaps']} gap(s) "
                          f"({cov['missing_fraction'] * 100:.1f}% of steps) inside the window.")

        if not covered:
            print("Spin-up not created: provide a longer record or a shorter window.")
            return None

        spinup_dir = self._resolve_spinup_dir(spinup_dir)

        spin = copy.deepcopy(self)
        spin.options["startdate"]["value"] = self._format_startdate(start_dt)
        spin.options["runtime"]["value"] = runtime
        self._redirect_output(spin, spinup_dir)
        self._configure_restart(spin, runtime, restart_dir)

        in_path = input_file
        spin.write_input_file(in_path)

        print(f"Spin-up (observed) model written to {in_path}\n"
              f"  window {self._format_startdate(start_dt)}-{self._format_startdate(end_dt)} "
              f"(RUNTIME: {runtime:.1f} h), using existing forcing\n"
              f"  output  -> {spinup_dir}\n"
              f"  restart -> {self._expected_restart_file(restart_dir, runtime)}\n"
              "Run this model, then call apply_hotstart(spinup) to wire your real run to it.")

        return spin

    def apply_hotstart(self, spinup, restart_file=None):
        """
        Wire this (real) model to start from a spin-up's restart file.

        Sets ``RESTARTMODE`` to 2 (read only) and ``RESTARTFILE`` to the spin-up's restart output.
        Call this after running the spin-up model returned by :meth:`create_spinup_repeat` or
        :meth:`create_spinup_observed`.

        Parameters
        ----------
        spinup : Model
            The spin-up model whose ``RESTARTDIR`` holds the restart file.
        restart_file : str, optional
            Explicit path to the restart file. If omitted, the ``tRIBS_Rstrt_*`` file in the
            spin-up's ``RESTARTDIR`` whose embedded hour count is closest to that spin-up's
            ``RUNTIME`` is used. Matching on this spin-up's run length (rather than the largest hour)
            means a different spin-up's restart file sharing the same ``RESTARTDIR`` is not picked up.

        Raises
        ------
        RuntimeError
            If no restart file is found (e.g. the spin-up has not been run yet).
        """
        if restart_file is None:
            restart_dir = spinup.options["restartdir"]["value"]
            # Match only tRIBS restart files so generic model output in the same tree is never
            # mistaken for a restart.
            candidates = [p for p in glob.glob(
                os.path.join(restart_dir, f"{self._RESTART_PREFIX}*")) if os.path.isfile(p)]
            if not candidates:
                raise RuntimeError(
                    f"No restart file ({self._RESTART_PREFIX}*) found in {restart_dir}. "
                    "Run the spin-up model first.")
            # Each spin-up writes its final dump at its own RUNTIME, so pick the file whose embedded
            # hour is closest to this spin-up's RUNTIME. This selects the right file even when
            # several spin-ups (e.g. a repeat and an observed run) share one RESTARTDIR.
            runtime = float(spinup.options["runtime"]["value"])
            restart_file = min(candidates, key=lambda p: abs(self._restart_hours(p) - runtime))

        self.options["restartmode"]["value"] = 2           # read restart file only
        self.options["restartfile"]["value"] = restart_file
        print(f"Real model set to hotstart from {restart_file} (RESTARTMODE: 2).")
